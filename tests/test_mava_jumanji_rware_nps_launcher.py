"""Protocol checks for the independent-actor RWARE MAPPO comparison."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parents[1] / "experiments/mava_jumanji"
sys.path.insert(0, str(HERE))
import run_rware_nps as protocol  # noqa: E402


CONFIG = protocol.paired.read_json(protocol.paired.CONFIG_PATH)


def options(tmp_path: Path, *, smoke: bool = False):
    mava_root = tmp_path / "Mava"
    baseline = mava_root / "mava/systems/ppo/anakin/rec_mappo.py"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text("# pinned MAPPO test source\n")
    return SimpleNamespace(
        mava_root=mava_root, run_root=tmp_path / "nps",
        mse_coef=0.01, cka_coef=0.03, arec_coef=1e-4,
        q_steps=4, q_lr=1e-3, fisher_ridge=1e-3, epsilon=1e-8,
        smoke=smoke, gpus="0,1,2,3", max_runs_per_gpu=1,
        dry_run=False, retry_failed=False,
    )


def test_formal_creates_16_new_seed_matched_runs_including_nps_baseline(tmp_path):
    manifest = protocol.build_manifest(options(tmp_path), CONFIG)
    assert manifest["actor_parameter_sharing"] is False
    assert manifest["seeds"] == [1, 2, 3, 4]
    assert len(manifest["jobs"]) == 16
    assert [(job["seed"], job["condition"]) for job in manifest["jobs"]] == [
        (seed, method) for seed in range(1, 5) for method in protocol.METHODS
    ]
    for job in manifest["jobs"]:
        optimal = protocol.paired.make_jobs(
            CONFIG, (protocol.TASK,), (job["seed"],), False, ("none",)
        )[0]
        assert job["shared_overrides"] == optimal["shared_overrides"]


def test_none_has_exact_zero_auxiliary_coefficient(tmp_path):
    opts = options(tmp_path)
    manifest = protocol.build_manifest(opts, CONFIG)
    assert manifest["coefficients"] == {
        "none": 0.0, "mse": 0.01, "cka": 0.03, "arec": 1e-4,
    }
    for job in manifest["jobs"]:
        command = protocol.worker_command(opts.mava_root, opts.run_root, job, manifest)
        overrides = dict(token.split("=", 1) for token in command[3:])
        if job["condition"] == "arec":
            assert command[2].endswith("rec_mappo_arec_nps.py")
            assert float(overrides["arec.coef"]) == 1e-4
            assert overrides["arec.q_steps"] == "4"
            assert float(overrides["arec.q_lr"]) == 1e-3
            assert float(overrides["arec.fisher_ridge"]) == 1e-3
            assert "align.distance" not in overrides
        else:
            assert command[2].endswith("rec_mappo_nps.py")
            assert overrides["align.distance"] == protocol.DISTANCES[job["condition"]]
            assert float(overrides["align.coef"]) == manifest["coefficients"][job["condition"]]
        assert overrides["system.seed"] == str(job["seed"])
        assert overrides["env/scenario"] == "large-8ag"


def test_smoke_keeps_rware_scenario_and_reduces_budget(tmp_path):
    manifest = protocol.build_manifest(options(tmp_path, smoke=True), CONFIG)
    assert len(manifest["jobs"]) == 4
    assert all(job["seed"] == 1 for job in manifest["jobs"])
    assert all(job["shared_overrides"]["system.num_updates"] == 2 for job in manifest["jobs"])


def test_each_condition_requires_its_own_complete_evaluation_log(tmp_path):
    manifest = protocol.build_manifest(options(tmp_path, smoke=True), CONFIG)
    for job in manifest["jobs"]:
        run_dir = tmp_path / job["name"]
        metric = run_dir / "json/metrics.json"
        assert protocol.complete_metric(run_dir, job) is None
        protocol.paired.write_json(metric, {
            "RobotWarehouse": {"large-8ag": {
                protocol.ALGORITHM_NAMES[job["condition"]]: {
                    "seed_1": {"step_1": {
                        "step_count": protocol.total_steps(job),
                        "mean_episode_return": 0.2,
                    }}
                }
            }}
        })
        assert protocol.complete_metric(run_dir, job) == metric


def test_invalid_coefficients_rejected_before_launch(tmp_path):
    opts = options(tmp_path, smoke=True)
    opts.mse_coef = 0.0
    with pytest.raises(ValueError, match="finite positive"):
        protocol.run(opts)
