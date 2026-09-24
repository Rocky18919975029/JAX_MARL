"""Checks that the paired benchmark launcher cannot drift between conditions."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "experiments/mava_jumanji/run_matched_optimal.py"
spec = importlib.util.spec_from_file_location("mava_matched_optimal", MODULE)
launcher = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(launcher)
CONFIG = json.loads((MODULE.parent / "optimal_rec_mappo.json").read_text())


def test_official_recurrent_mappo_rows_are_transcribed():
    assert CONFIG["mava_commit"] == "9f67e612654ecb7b7d45ff8052ce9ccfc6c68d93"
    assert CONFIG["common"] == {
        "arch.num_envs": 64,
        "system.update_batch_size": 2,
        "system.rollout_length": 128,
        "system.num_updates": 1220,
        "arch.num_evaluation": 122,
        "arch.num_eval_episodes": 32,
        "arch.num_absolute_metric_eval_episodes": 320,
        "arch.absolute_metric": True,
        "system.gamma": 0.99,
        "system.gae_lambda": 0.9,
        "system.vf_coef": 0.5,
        "system.add_agent_id": True,
    }
    assert CONFIG["tasks"]["lbf_15x15-4p-5f"]["optimal"] == {
        "system.num_minibatches": 2,
        "system.max_grad_norm": 10.0,
        "system.ppo_epochs": 4,
        "system.clip_eps": 0.1,
        "system.recurrent_chunk_size": 64,
        "system.ent_coef": 0.01,
        "system.critic_lr": 0.001,
        "system.actor_lr": 0.001,
    }
    assert CONFIG["tasks"]["rware_large-8ag"]["optimal"] == {
        "system.num_minibatches": 8,
        "system.max_grad_norm": 10.0,
        "system.ppo_epochs": 8,
        "system.clip_eps": 0.05,
        "system.recurrent_chunk_size": 8,
        "system.ent_coef": 0.0,
        "system.critic_lr": 0.0005,
        "system.actor_lr": 0.0001,
    }


def test_four_seeds_are_paired_and_seed_first():
    tasks = tuple(CONFIG["tasks"])
    jobs = launcher.make_jobs(CONFIG, tasks, launcher.parse_seeds("1-4"), False)
    assert len(jobs) == 16
    for index in range(0, len(jobs), 2):
        none, arec = jobs[index : index + 2]
        assert (none["seed"], none["task"]) == (arec["seed"], arec["task"])
        assert (none["condition"], arec["condition"]) == ("none", "arec")
        assert none["shared_overrides"] == arec["shared_overrides"]
        assert none["shared_overrides"]["system.seed"] == none["seed"]
        assert none["shared_overrides"]["system.num_updates"] == 1220
    assert [job["seed"] for job in jobs] == sorted(job["seed"] for job in jobs)


def test_arec_only_jobs_do_not_launch_any_baseline():
    jobs = launcher.make_jobs(
        CONFIG, tuple(CONFIG["tasks"]), (1, 2, 3, 4), False, ("arec",)
    )
    assert len(jobs) == 8
    assert {job["condition"] for job in jobs} == {"arec"}
    assert [job["seed"] for job in jobs] == sorted(job["seed"] for job in jobs)


def test_default_manifest_shape_remains_backward_compatible(tmp_path):
    args = argparse.Namespace(
        smoke=False, mava_root=tmp_path, arec_coef=1e-4,
        arec_q_steps=4, arec_q_lr=1e-3, arec_fisher_ridge=1e-3,
    )
    (tmp_path / "mava/systems/ppo/anakin").mkdir(parents=True)
    (tmp_path / "mava/systems/ppo/anakin/rec_mappo.py").write_text("baseline")
    original = launcher.manifest(CONFIG, args, tuple(CONFIG["tasks"]), (1,), launcher.CONDITIONS)
    only_arec = launcher.manifest(CONFIG, args, tuple(CONFIG["tasks"]), (1,), ("arec",))
    assert "conditions" not in original
    assert only_arec["conditions"] == ["arec"]


def test_command_only_adds_arec_knobs(tmp_path):
    none, arec = launcher.make_jobs(CONFIG, ("rware_large-8ag",), (1,), False)
    hyperparams = {"coef": 1e-4, "q_steps": 4, "q_lr": 1e-3, "fisher_ridge": 1e-3}
    baseline_cmd = launcher.command(tmp_path / "mava", tmp_path / "out", none, hyperparams)
    arec_cmd = launcher.command(tmp_path / "mava", tmp_path / "out", arec, hyperparams)
    baseline_overrides = {
        part.split("=", 1)[0]: part.split("=", 1)[1] for part in baseline_cmd[3:]
    }
    arec_overrides = {
        part.split("=", 1)[0]: part.split("=", 1)[1] for part in arec_cmd[3:]
    }
    assert baseline_cmd[2].endswith("/rec_mappo.py")
    assert arec_cmd[2].endswith("/rec_mappo_arec.py")
    for key in ("logger.base_exp_path", "hydra.run.dir"):
        baseline_overrides.pop(key)
        arec_overrides.pop(key)
    assert {key: arec_overrides.pop(key) for key in hyperparams_to_overrides()} == {
        "arec.coef": "0.0001",
        "arec.q_steps": "4",
        "arec.q_lr": "0.001",
        "arec.fisher_ridge": "0.001",
    }
    assert baseline_overrides == arec_overrides


def hyperparams_to_overrides():
    return ("arec.coef", "arec.q_steps", "arec.q_lr", "arec.fisher_ridge")


def test_smoke_preserves_training_hyperparameters_except_length():
    task = "lbf_15x15-4p-5f"
    formal = launcher.task_overrides(CONFIG, task, 3, False)
    smoke = launcher.task_overrides(CONFIG, task, 3, True)
    for key in (
        "system.num_updates", "arch.num_evaluation", "arch.num_eval_episodes",
        "arch.absolute_metric", "arch.num_envs", "system.update_batch_size",
        "system.ppo_epochs", "system.num_minibatches",
    ):
        formal.pop(key)
        smoke.pop(key)
    assert formal == smoke


@pytest.mark.parametrize("bad", ("4-1", "1,1", "-1", ""))
def test_invalid_seeds_rejected(bad):
    with pytest.raises(ValueError):
        launcher.parse_seeds(bad)
