"""Protocol checks for reusing the existing four-seed Mava paired experiment."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "experiments/mava_jumanji"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("mava_arec_lambda_sweep", HERE / "run_arec_lambda_sweep.py")
sweep = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(sweep)
CONFIG = sweep.paired.read_json(sweep.paired.CONFIG_PATH)


def _original_experiment(tmp_path):
    mava = tmp_path / "Mava"
    baseline_script = mava / "mava/systems/ppo/anakin/rec_mappo.py"
    baseline_script.parent.mkdir(parents=True)
    baseline_script.write_text("# pinned test fixture\n")
    source = tmp_path / "formal_4seed_v1"
    jobs = sweep.paired.make_jobs(CONFIG, sweep.TASKS, sweep.SEEDS, False)
    original = {
        "protocol": "mava-jumanji-rec-mappo-arec-paired-optimal-v1",
        "smoke": False,
        "tasks": list(sweep.TASKS),
        "seeds": list(sweep.SEEDS),
        "jobs": jobs,
        "arec": {
            "coef": 1e-4, "q_steps": 4, "q_lr": 1e-3, "fisher_ridge": 1e-3,
        },
        "benchmark_config_sha256": sweep.paired.digest(sweep.paired.CONFIG_PATH),
        "baseline_script_sha256": sweep.paired.digest(baseline_script),
        "arec_script_sha256": sweep.paired.digest(sweep.paired.AREC_SCRIPT),
        "arec_config_sha256": sweep.paired.digest(sweep.paired.AREC_CONFIG),
        "mava_commit": CONFIG["mava_commit"],
        "mava_root": str(mava),
    }
    sweep.paired.write_json(source / "experiment_manifest.json", original)
    for job in jobs:
        sweep.paired.write_json(sweep.paired.status_path(source, job), {
            "status": "completed", "exit_code": 0,
        })
        metric = source / "runs" / job["name"] / "json" / "attempt1" / "metrics.json"
        sweep.paired.write_json(metric, {
            "step_122": {
                "step_count": sweep.total_steps(job),
                "mean_episode_return": 0.5,
            }
        })
    return source, mava, jobs


def test_grid_reuses_exactly_sixteen_and_launches_only_thirty_two(tmp_path):
    source, mava, _ = _original_experiment(tmp_path)
    reused = sweep.validate_reused_run(source, CONFIG)
    planned = sweep.build_manifest(CONFIG, source, tmp_path / "sweep", mava, reused)
    assert len(reused) == 16
    assert sum(row["condition"] == "none" for row in reused) == 8
    assert sum(row["coefficient"] == 1e-4 for row in reused) == 8
    assert [group["coefficient"] for group in planned["new_groups"]] == [
        3e-6, 1e-5, 3e-5, 3e-4,
    ]
    assert sum(len(group["jobs"]) for group in planned["new_groups"]) == 32
    assert all(job["condition"] == "arec" for group in planned["new_groups"] for job in group["jobs"])
    assert {job["seed"] for group in planned["new_groups"] for job in group["jobs"]} == set(range(1, 5))
    assert all(row["metric_sha256"] for row in reused)


def test_reuse_refuses_incomplete_or_changed_source(tmp_path):
    source, _, jobs = _original_experiment(tmp_path)
    sweep.paired.write_json(sweep.paired.status_path(source, jobs[0]), {
        "status": "failed", "exit_code": 1,
    })
    with pytest.raises(RuntimeError, match="not complete"):
        sweep.validate_reused_run(source, CONFIG)
    sweep.paired.write_json(sweep.paired.status_path(source, jobs[0]), {
        "status": "completed", "exit_code": 0,
    })
    manifest = json.loads((source / "experiment_manifest.json").read_text())
    manifest["arec"]["q_steps"] = 8
    sweep.paired.write_json(source / "experiment_manifest.json", manifest)
    with pytest.raises(RuntimeError, match="arec differs"):
        sweep.validate_reused_run(source, CONFIG)


def test_child_commands_preserve_optimal_config_and_only_train_arec(tmp_path):
    source, mava, _ = _original_experiment(tmp_path)
    planned = sweep.build_manifest(
        CONFIG, source, tmp_path / "sweep", mava,
        sweep.validate_reused_run(source, CONFIG),
    )
    args = argparse.Namespace(
        mava_root=mava, gpus="0,1,2,3", max_runs_per_gpu=1,
        retry_failed=False, dry_run=False,
    )
    for group in planned["new_groups"]:
        command = sweep.child_command(args, group)
        assert "--conditions" in command
        assert command[command.index("--conditions") + 1] == "arec"
        assert command[command.index("--seeds") + 1] == "1-4"
        assert command[command.index("--max-runs-per-gpu") + 1] == "1"
        assert command[command.index("--arec-coef") + 1] == str(group["coefficient"])
    sweep.validate_manifest(tmp_path / "nonexistent.json", planned)
    sweep.paired.write_json(tmp_path / "existing.json", {**planned, "q_steps": 8})
    with pytest.raises(RuntimeError, match="differs"):
        sweep.validate_manifest(tmp_path / "existing.json", planned)


def test_launcher_dispatches_four_arec_only_children_and_preserves_source(tmp_path, monkeypatch):
    source, mava, _ = _original_experiment(tmp_path)
    original_manifest = (source / "experiment_manifest.json").read_bytes()
    commands = []
    monkeypatch.setattr(sweep.paired, "validate_mava", lambda *_: None)
    monkeypatch.setattr(
        sweep.subprocess, "run",
        lambda command, **_: (commands.append(command), SimpleNamespace(returncode=0))[1],
    )
    run_root = tmp_path / "sweep"
    args = argparse.Namespace(
        mava_root=mava, source_root=source, run_root=run_root,
        gpus="0,1,2,3", max_runs_per_gpu=1,
        retry_failed=False, dry_run=False,
    )
    assert sweep.run(args) == 0
    assert len(commands) == 4
    assert all(command[command.index("--conditions") + 1] == "arec" for command in commands)
    assert (source / "experiment_manifest.json").read_bytes() == original_manifest
    assert len(sweep.paired.read_json(run_root / "experiment_manifest.json")["new_groups"]) == 4
