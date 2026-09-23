"""No-GPU protocol tests for the final SMAX four-method control plane."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.smax_four_method import (
    PROTOCOL, Run, atomic_json, checkpoint_dir, final_five_checkpoints,
    make_grid, map_names, summarize, train_command, validate_artifacts,
    wandb_id, group_for,
)


def args(**changes):
    values = dict(
        map_name="10m_vs_11m", methods=("none", "mse", "cka", "arec"),
        seed_start=1, seed_count=4, total_timesteps=10_000_000,
        num_steps=128, ppo_lrs=(0.002,), ppo_epochs=(4,),
        num_envs_grid=(128,), num_minibatches_grid=(4,),
        mse_coefs=(0.1,), cka_coefs=(0.3, 1.0),
        arec_coefs=(3e-5, 1e-4), arec_q_steps=(4, 8),
        arec_q_lrs=(1e-3,), arec_fisher_ridges=(1e-3,),
        checkpoint_interval=1_000_000, wandb_mode="disabled", project="test",
    )
    values.update(changes)
    return SimpleNamespace(**values)


def overrides(command):
    return dict(part.split("=", 1) for part in command[2:] if "=" in part)


def test_all_methods_share_exact_seed_budget_and_ppo_grid(tmp_path):
    runs = make_grid(args())
    assert len(runs) == 4 * (1 + 1 + 2 + 4)
    assert {run.seed for run in runs} == {1, 2, 3, 4}
    assert {run.timesteps for run in runs} == {9_994_240}
    assert len({run.name for run in runs}) == len(runs)
    assert len([run for run in runs if run.method == "none"]) == 4
    assert Run(**{**runs[0].__dict__}).ident == runs[0].ident
    assert {run.map_name for run in runs} == {"10m_vs_11m"}
    assert {"10m_vs_11m", "3s5z_vs_3s6z", "6s9z_vs_6s10z", "smacv2_10_units"} <= map_names()


@pytest.mark.parametrize("method,alignment,distance,arec", [
    ("none", "none", "ln_mse", "false"),
    ("mse", "c_to_a", "ln_mse", "false"),
    ("cka", "c_to_a", "linear_cka", "false"),
    ("arec", "none", "ln_mse", "true"),
])
def test_exact_method_routing(tmp_path, method, alignment, distance, arec):
    run = next(run for run in make_grid(args()) if run.method == method)
    values = overrides(train_command(tmp_path, args(), run))
    assert values["ACTOR_PARAMETER_SHARING"] == "false"
    assert values["MATCHED_COMPARISON"] == "true"
    assert values["ALIGN_MODE"] == alignment
    assert values["ALIGN_DISTANCE"] == distance
    assert values["ACTOR_SCORE_RECOVERY"] == arec
    assert values["ALIGNMENT_COEF"] == ("0" if method in {"none", "arec"} else str(run.coef))
    assert float(values["ACTOR_SCORE_RECOVERY_COEF"]) == (run.coef if method == "arec" else 0)
    assert values["WANDB_RUN_ID"] == wandb_id(tmp_path, run)
    assert values["TOTAL_TIMESTEPS"] == str(run.timesteps)
    assert "ORACLE_LATENT_DISTORTION" not in values
    assert "SCORE_RECOVERY" not in values


def test_grid_rejects_incompatible_minibatches():
    with pytest.raises(ValueError, match="divisible"):
        make_grid(args(num_envs_grid=(128, 96), num_minibatches_grid=(5,)))


def test_fresh_sweep_root_has_distinct_wandb_identity(tmp_path):
    run = make_grid(args())[0]
    other = tmp_path / "another"
    assert wandb_id(tmp_path, run) != wandb_id(other, run)
    assert group_for(tmp_path, run) != group_for(other, run)
    assert wandb_id(tmp_path, run) == wandb_id(tmp_path, run)


def test_final_checkpoint_wins_when_interval_has_same_step(tmp_path):
    run = make_grid(args())[0]
    parent = checkpoint_dir(tmp_path, "test", run).parent
    for step in range(6_000_000, 10_000_001, 1_000_000):
        directory = parent / f"step_{step:012d}"
        directory.mkdir(parents=True)
        (directory / "model.safetensors").write_bytes(b"checkpoint")
        atomic_json(directory / "metadata.json", {"nominal_env_step": step})
    final = parent / "final"
    final.mkdir()
    (final / "model.safetensors").write_bytes(b"final checkpoint")
    atomic_json(final / "metadata.json", {"nominal_env_step": 10_000_000})
    selected = final_five_checkpoints(tmp_path, "test", run.__dict__)
    assert len(selected) == 5
    assert selected[-1] == final


def test_completion_needs_matching_metadata_and_full_metrics(tmp_path):
    run = next(run for run in make_grid(args()) if run.method == "none")
    directory = checkpoint_dir(tmp_path, "test", run)
    directory.mkdir(parents=True)
    (directory / "model.safetensors").write_bytes(b"checkpoint")
    atomic_json(directory / "metadata.json", {
        "is_final": True, "env_step": run.timesteps,
        "map_name": run.map_name, "seed": run.seed, "git_commit": "abc",
        "wandb_run_id": wandb_id(tmp_path, run), "wandb_run_name": run.name,
    })
    atomic_json(directory / "config.json", {
        "PROTOCOL_VERSION": PROTOCOL,
        "EXPERIMENT_CONDITION": run.condition,
        "WANDB_RUN_ID": wandb_id(tmp_path, run),
        "TOTAL_TIMESTEPS": run.timesteps, "NUM_STEPS": run.num_steps,
        "NUM_ENVS": run.num_envs,
        "NUM_MINIBATCHES": run.num_minibatches,
        "UPDATE_EPOCHS": run.epochs, "LR": run.lr,
        "ALIGNMENT_COEF": 0.0,
        "ACTOR_SCORE_RECOVERY_COEF": 0.0,
    })
    path = tmp_path / "metrics" / f"{run.name}.jsonl"
    path.parent.mkdir()
    path.write_text(json.dumps({"env_step": run.timesteps}) + "\n")
    assert validate_artifacts(tmp_path, "test", run, "abc") is None
    assert "mismatch" in validate_artifacts(tmp_path, "test", run, "def")
    path.write_text(json.dumps({"env_step": 0}) + "\n")
    assert "incomplete" in validate_artifacts(tmp_path, "test", run, "abc")


def test_seed_aggregation_is_deterministic_and_requires_full_cohort(tmp_path):
    runs = make_grid(args(methods=("none",), total_timesteps=5_000_000))
    manifest = {
        "map_name": "10m_vs_11m", "seed_count": 4,
        "checkpoint_interval": 1_000_000,
        "runs": [dict(run.__dict__, name=run.name, group=run.group) for run in runs],
    }
    atomic_json(tmp_path / "experiment_manifest.json", manifest)
    for run in runs:
        atomic_json(tmp_path / "status" / f"{run.name}.json", {"status": "completed"})
        path = tmp_path / "metrics" / f"{run.name}.jsonl"
        path.parent.mkdir(exist_ok=True)
        path.write_text("".join(
            json.dumps({"env_step": step, "returns": run.seed + step / 1e6,
                        "win_rate": step / 1e7}) + "\n"
            for step in range(1_000_000, 6_000_000, 1_000_000)
        ))
    summarize(tmp_path, bootstrap=200, bootstrap_seed=7)
    first = (tmp_path / "summary" / "summary.csv").read_bytes()
    summarize(tmp_path, bootstrap=200, bootstrap_seed=7)
    assert (tmp_path / "summary" / "summary.csv").read_bytes() == first
    with (tmp_path / "summary" / "summary.csv").open() as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 2
    assert {row["metric"] for row in records} == {"returns", "win_rate"}
