"""Protocol checks for paired RWARE MAPPO/LN-MSE/linear-CKA training."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parents[1] / "experiments/mava_jumanji"
sys.path.insert(0, str(HERE))
import run_rware_alignment as protocol  # noqa: E402


CONFIG = protocol.paired.read_json(protocol.paired.CONFIG_PATH)


def args(tmp_path: Path, *, smoke: bool = False):
    mava = tmp_path / "Mava"
    baseline = mava / "mava/systems/ppo/anakin/rec_mappo.py"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text("# pinned test baseline\n")
    protocol.paired.write_json(tmp_path / "source/experiment_manifest.json", {})
    return SimpleNamespace(
        mava_root=mava, source_root=tmp_path / "source",
        run_root=tmp_path / "alignment", mse_coef=0.01, cka_coef=0.03,
        epsilon=1e-8, smoke=smoke, gpus="0,1,2,3",
        max_runs_per_gpu=1, dry_run=False, retry_failed=False,
    )


def test_formal_manifest_has_eight_seed_matched_runs_and_four_reused_baselines(tmp_path):
    options = args(tmp_path)
    baseline_rows = [{"seed": seed, "metric_sha256": str(seed)} for seed in (1, 2, 3, 4)]
    manifest = protocol.build_manifest(options, CONFIG, baseline_rows)
    assert len(manifest["jobs"]) == 8
    assert len(manifest["reused_baselines"]) == 4
    assert [job["condition"] for job in manifest["jobs"]] == ["mse", "cka"] * 4
    assert [job["seed"] for job in manifest["jobs"]] == [1, 1, 2, 2, 3, 3, 4, 4]
    assert manifest["coefficients"] == {"mse": 0.01, "cka": 0.03}
    for job in manifest["jobs"]:
        original = protocol.paired.make_jobs(
            CONFIG, (protocol.TASK,), (job["seed"],), False, ("none",)
        )[0]
        assert job["shared_overrides"] == original["shared_overrides"]


def test_worker_changes_only_distance_and_coefficient(tmp_path):
    options = args(tmp_path)
    manifest = protocol.build_manifest(options, CONFIG, [])
    for job in manifest["jobs"]:
        command = protocol.worker_command(options.mava_root, options.run_root, job, manifest)
        assert command[2].endswith("rec_mappo_alignment.py")
        overrides = dict(token.split("=", 1) for token in command[3:])
        assert overrides["align.distance"] == protocol.DISTANCES[job["condition"]]
        assert float(overrides["align.coef"]) == manifest["coefficients"][job["condition"]]
        assert overrides["system.seed"] == str(job["seed"])
        assert overrides["env"] == "rware"
        assert overrides["env/scenario"] == "large-8ag"
        assert "arec.coef" not in overrides


def test_smoke_uses_same_rware_scenario_but_reduced_budget(tmp_path):
    options = args(tmp_path, smoke=True)
    manifest = protocol.build_manifest(options, CONFIG, [])
    assert len(manifest["jobs"]) == 2
    assert manifest["seeds"] == [1]
    assert manifest["reused_baselines"] == []
    assert all(job["shared_overrides"]["env/scenario"] == "large-8ag" for job in manifest["jobs"])
    assert all(job["shared_overrides"]["system.num_updates"] == 2 for job in manifest["jobs"])


def test_only_complete_pinned_mapppo_baselines_can_be_reused(tmp_path):
    options = args(tmp_path)
    source = options.source_root
    jobs = protocol.paired.make_jobs(CONFIG, (protocol.TASK,), protocol.SEEDS, False, ("none",))
    protocol.paired.write_json(source / "experiment_manifest.json", {
        "protocol": protocol.SOURCE_PROTOCOL,
        "smoke": False,
        "mava_commit": CONFIG["mava_commit"],
        "benchmark_config_sha256": protocol.paired.digest(protocol.paired.CONFIG_PATH),
        "baseline_script_sha256": protocol.paired.digest(
            options.mava_root / "mava/systems/ppo/anakin/rec_mappo.py"
        ),
        "jobs": jobs,
    })
    for job in jobs:
        protocol.paired.write_json(protocol.paired.status_path(source, job), {
            "status": "completed", "exit_code": 0,
        })
        stride = protocol.total_steps(job) // job["shared_overrides"]["arch.num_evaluation"]
        metrics = {
            "RobotWarehouse": {
                "large-8ag": {
                    "rec_mappo": {
                        f"seed_{job['seed']}": {
                            f"step_{index}": {
                                "step_count": stride * index,
                                "mean_episode_return": 0.1,
                            }
                            for index in range(1, job["shared_overrides"]["arch.num_evaluation"] + 1)
                        }
                    }
                }
            }
        }
        protocol.paired.write_json(
            source / "runs" / job["name"] / "json/metrics.json", metrics
        )
    baselines = protocol.validated_baselines(source, options.mava_root, CONFIG)
    assert [row["seed"] for row in baselines] == list(protocol.SEEDS)
    assert all(row["metric_sha256"] for row in baselines)
    protocol.paired.write_json(protocol.paired.status_path(source, jobs[0]), {
        "status": "failed", "exit_code": 1,
    })
    with pytest.raises(RuntimeError, match="has not completed"):
        protocol.validated_baselines(source, options.mava_root, CONFIG)


def test_alignment_worker_requires_its_own_full_evaluation_log(tmp_path):
    options = args(tmp_path, smoke=True)
    job = protocol.build_manifest(options, CONFIG, [])["jobs"][0]
    run_dir = tmp_path / "worker"
    metric = run_dir / "json/metrics.json"
    assert protocol.complete_metric(run_dir, job) is None
    protocol.paired.write_json(metric, {
        "RobotWarehouse": {"large-8ag": {"rec_mappo_ln_mse": {
            "seed_1": {"step_1": {
                "step_count": protocol.total_steps(job),
                "mean_episode_return": 0.2,
            }}
        }}}
    })
    assert protocol.complete_metric(run_dir, job) == metric


def test_reject_invalid_coefficient_before_launch(tmp_path, monkeypatch):
    options = args(tmp_path, smoke=True)
    options.mse_coef = 0.0
    with pytest.raises(ValueError, match="finite positive"):
        protocol.run(options)
