"""The replacement study reuses only finished HAPPO/MAPPO runs."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.harl_dexhands import monitor, run_paper_madpo_matrix as launcher
from experiments.harl_dexhands.protocol import task_matrix


REPO_ROOT = Path(__file__).resolve().parents[1]
HARL_ROOT = REPO_ROOT / "third_party" / "HARL"
CONFIG = HARL_ROOT / launcher.OFFICIAL_CONFIG


def make_legacy(tmp_path: Path, *, complete_all_baselines: bool = False) -> Path:
    root = tmp_path / "legacy"
    root.mkdir()
    config_hash = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    grid = (3e-5, 1e-4, 3e-4)
    tasks = task_matrix(
        ("happo", "mappo", "madpo"),
        (1, 2, 3, 4),
        conditions=("none", "arec"),
        arec_coefs=grid,
    )
    manifest = {
        "study_spec": {
            "algorithms": ["happo", "mappo", "madpo"],
            "conditions": ["none", "arec"],
            "seeds": [1, 2, 3, 4],
            "arec_coefs": list(grid),
            "arec_q_steps": 4,
            "arec_q_lr": 0.001,
            "arec_fisher_ridge": 0.001,
            "num_env_steps": 50_000_000,
            "n_rollout_threads": None,
            "official_config_sha256": config_hash,
        },
        "runs": [
            {
                "run_name": task.name,
                "algorithm": task.algorithm,
                "seed": task.seed,
                "condition": task.condition,
            }
            for task in tasks
        ],
    }
    (root / "experiment_manifest.json").write_text(json.dumps(manifest))
    (root / "status").mkdir()
    (root / "metrics").mkdir()
    for task in tasks:
        if task.algorithm == "madpo" or (task.seed != 1 and not complete_all_baselines):
            continue
        (root / "status" / f"{task.name}.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "run_name": task.name,
                    "total_env_steps": 50_000_000,
                }
            )
        )
        (root / "metrics" / f"{task.name}.jsonl").write_text("{}\n")
    return root


def test_replacement_manifest_reuses_eight_and_sets_paper_madpo_budget(tmp_path):
    old = make_legacy(tmp_path)
    new = tmp_path / "new"
    result = subprocess.run(
        [
            sys.executable,
            str(launcher.__file__),
            "--legacy-run-root",
            str(old),
            "--run-root",
            str(new),
            "--gpus",
            "0",
            "--wandb-mode",
            "disabled",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "TOTAL=48 REUSED=8" in result.stdout
    manifest = json.loads((new / "experiment_manifest.json").read_text())
    assert len(manifest["runs"]) == 48
    assert sum(run["algorithm"] == "madpo" for run in manifest["runs"]) == 16
    assert {
        run["num_env_steps"] for run in manifest["runs"] if run["algorithm"] == "madpo"
    } == {40_000_000}
    assert {
        run["num_env_steps"] for run in manifest["runs"] if run["algorithm"] != "madpo"
    } == {50_000_000}
    assert all(
        "reuse_completed_from" in run
        for run in manifest["runs"]
        if run["algorithm"] != "madpo"
    )
    assert all(
        "reuse_completed_from" not in run
        for run in manifest["runs"]
        if run["algorithm"] == "madpo"
    )
    assert (
        sum(row["status"] == "completed" for row in monitor.load_manifest_rows(new))
        == 8
    )
    assert "madpo-div1000-w0p2-sig500-k4000" in result.stdout


def test_failed_job_does_not_stop_seed_and_is_retried_after_peers(
    tmp_path, monkeypatch
):
    old = make_legacy(tmp_path, complete_all_baselines=True)
    new = tmp_path / "new"
    calls = []
    failed_once = set()

    def fake_run(command, **_kwargs):
        name = command[command.index("--run-name") + 1]
        calls.append(name)
        if "seed1" in name and name not in failed_once and not failed_once:
            failed_once.add(name)
            return SimpleNamespace(returncode=1)
        path = new / "status" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"status": "completed", "run_name": name}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(launcher.__file__),
            "--legacy-run-root",
            str(old),
            "--run-root",
            str(new),
            "--gpus",
            "0",
            "--max-runs-per-gpu",
            "1",
            "--wandb-mode",
            "disabled",
        ],
    )
    launcher.main()
    assert len(calls) == 17  # 16 paper MADPO runs, one retried once
    first = next(iter(failed_once))
    first_seed2 = next(i for i, name in enumerate(calls) if "seed2" in name)
    assert calls.count(first) == 2
    assert calls.index(first) < calls.index(first, 1) < first_seed2
    assert all("seed1" in name for name in calls[:first_seed2])
    assert (
        sum(row["status"] == "completed" for row in monitor.load_manifest_rows(new))
        == 48
    )
    assert (new / "failed_attempts" / first / "attempt_01" / "status.json").is_file()


def test_only_missing_baselines_are_launched_and_resume_is_idempotent(
    tmp_path, monkeypatch
):
    old = make_legacy(tmp_path)
    new = tmp_path / "new"
    calls = []

    def fake_run(command, **_kwargs):
        name = command[command.index("--run-name") + 1]
        calls.append(name)
        path = new / "status" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"status": "completed", "run_name": name}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(launcher.__file__),
            "--legacy-run-root",
            str(old),
            "--run-root",
            str(new),
            "--gpus",
            "0",
            "--max-runs-per-gpu",
            "1",
            "--wandb-mode",
            "disabled",
        ],
    )
    launcher.main()
    assert len(calls) == 40
    assert not any(
        "seed1" in name and ("happo" in name or "mappo" in name) for name in calls
    )
    assert sum("madpo" in name for name in calls) == 16
    assert (
        sum(row["status"] == "completed" for row in monitor.load_manifest_rows(new))
        == 48
    )
    launcher.main()
    assert len(calls) == 40


def test_worker_stop_requires_verified_identity_and_uses_sigterm(tmp_path, monkeypatch):
    old = tmp_path / "legacy"
    old.mkdir()
    status = {"status": "running", "pid": 123}
    checks = iter(["0", "0", None])
    signals = []
    monkeypatch.setattr(launcher, "running_gpu", lambda *_args, **_kwargs: next(checks))
    monkeypatch.setattr(
        launcher.os, "kill", lambda pid, sig: signals.append((pid, sig))
    )
    assert launcher.stop_legacy_worker("happo-seed1", status, old, ("0",))
    assert signals == [(123, launcher.signal.SIGTERM)]


def test_launcher_process_matching_is_exact(tmp_path):
    old = tmp_path / "study"
    other = tmp_path / "other"
    proc = tmp_path / "proc"
    (proc / "100").mkdir(parents=True)
    (proc / "101").mkdir()
    (proc / "100" / "cmdline").write_bytes(
        b"python\0experiments/harl_dexhands/run_matrix.py\0--run-root\0"
        + str(old).encode()
        + b"\0"
    )
    (proc / "101" / "cmdline").write_bytes(
        b"python\0experiments/harl_dexhands/run_matrix.py\0--run-root\0"
        + str(other).encode()
        + b"\0"
    )
    assert launcher.legacy_launcher_pids(old, proc_root=proc) == [100]
