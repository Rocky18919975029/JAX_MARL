import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.analyze_smax_nps_baseline_sweep import summarize
from scripts.run_smax_actor_score_recovery_sweep import (
    SweepRun,
    command as old_sweep_command,
)
from scripts.run_smax_nps_baseline_sweep import (
    INCUMBENT,
    PROTOCOL,
    command,
    completed,
    run_matrix,
)


def test_four_seed_ppo_grid_contains_incumbent_and_unique_runs():
    runs = run_matrix((1, 2, 3, 4), (0.0005, 0.001, 0.002), (2, 4), 20_000_000)
    assert len(runs) == 24
    assert len({run.name for run in runs}) == 24
    assert (
        len(
            [run for run in runs if (run.learning_rate, run.update_epochs) == INCUMBENT]
        )
        == 4
    )
    assert all(run.steps == 20_000_000 for run in runs)


def test_completion_requires_successful_status_and_final_checkpoint(tmp_path):
    run = run_matrix((1,), (0.002,), (4,), 20_000_000)[0]
    checkpoint = (
        tmp_path / "checkpoints" / f"{run.name}-wandbid" / "final" / "model.safetensors"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    assert not completed(tmp_path, run)
    status = tmp_path / "status" / f"{run.name}.json"
    status.parent.mkdir()
    status.write_text(json.dumps({"status": "failed"}))
    assert not completed(tmp_path, run)
    status.write_text(json.dumps({"status": "completed"}))
    assert completed(tmp_path, run)


def test_incumbent_baseline_matches_existing_none_command_except_metadata(tmp_path):
    args = SimpleNamespace(
        num_envs=128,
        num_minibatches=4,
        checkpoint_interval=1_000_000,
        wandb_mode="disabled",
        project="test",
        update_epochs=4,
        learning_rate=0.002,
    )
    new_run = run_matrix((1,), (0.002,), (4,), 20_000_000)[0]
    old_run = SweepRun("6s9z_vs_6s10z", 1, 20_000_000, "none")
    new_command = command(Path("/repo"), tmp_path, args, new_run)
    old_command = old_sweep_command(Path("/repo"), tmp_path, args, old_run)
    excluded_prefixes = (
        "MATRIX_PROFILE=",
        "PROTOCOL_VERSION=",
        "METRICS_JSONL=",
        "hydra.run.dir=",
    )
    assert [item for item in new_command if not item.startswith(excluded_prefixes)] == [
        item for item in old_command if not item.startswith(excluded_prefixes)
    ]
    assert "ALIGNMENT_COEF=0" in new_command
    assert "ACTOR_SCORE_RECOVERY=false" in new_command


def test_analysis_reports_mean_and_worst_seed_without_hiding_variance(tmp_path):
    runs = run_matrix((1, 2, 3, 4), (0.001, 0.002), (4,), 100)
    manifest = {
        "protocol": PROTOCOL,
        "map_name": "6s9z_vs_6s10z",
        "seeds": [1, 2, 3, 4],
        "budget": 100,
        "runs": [dict(run.__dict__, run_name=run.name) for run in runs],
    }
    (tmp_path / "experiment_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "status").mkdir()
    (tmp_path / "metrics").mkdir()
    for run in runs:
        (tmp_path / "status" / f"{run.name}.json").write_text(
            json.dumps({"status": "completed"})
        )
        # The incumbent has a higher mean but much worse lowest seed.
        value = (0.1 if run.seed == 1 else 1.0) if run.learning_rate == 0.002 else 0.7
        with (tmp_path / "metrics" / f"{run.name}.jsonl").open("w") as file:
            for step in (50, 100):
                file.write(
                    json.dumps(
                        {
                            "env_step": step,
                            "returns": value,
                            "win_rate": value / 2,
                        }
                    )
                    + "\n"
                )
    result = summarize(tmp_path)
    assert result["winner_by_mean_return_auc"]["learning_rate"] == 0.002
    assert result["winner_by_worst_seed_return_auc"]["learning_rate"] == 0.001
    assert result["incumbent"]["sd_return_auc"] > 0
    assert (
        result["winner_by_worst_seed_return_auc"][
            "return_auc_positive_seeds_vs_incumbent"
        ]
        == 1
    )
    assert (tmp_path / "analysis" / "config_summary.csv").is_file()
    assert (tmp_path / "analysis" / "seed_level.csv").is_file()


def test_analysis_rejects_incomplete_matrix(tmp_path):
    (tmp_path / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "map_name": "6s9z_vs_6s10z",
                "seeds": [1],
                "budget": 100,
                "runs": [{"run_name": "missing"}],
            }
        )
    )
    with pytest.raises(RuntimeError, match="not complete"):
        summarize(tmp_path)
