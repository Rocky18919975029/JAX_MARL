"""No-GPU tests for the first four-panel tuned YAML and replay control plane."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_smax_first_four_panel_tuned import (
    TASKS, HISTORICAL_PROTOCOL, launch_args, load_pair, verify_historical,
)
from scripts.smax_four_method import make_grid, train_command


EXPECTED = {
    "10m_vs_11m": (10_000_000, 3e-6, 4),
    "3s5z_vs_3s6z": (20_000_000, 3e-4, 4),
    "6s9z_vs_6s10z": (20_000_000, 1e-4, 8),
    "smacv2_10_units": (10_000_000, 3e-5, 4),
}


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.mark.parametrize("task", TASKS)
def test_yaml_pairs_generate_exact_best_cells(task: str, tmp_path: Path) -> None:
    none, arec, paths = load_pair(task)
    budget, coefficient, q_steps = EXPECTED[task]
    args = SimpleNamespace(
        task=task, run_root=tmp_path / "fresh", gpus=("0", "1"),
        max_runs_per_gpu=2, project="test", wandb_mode="disabled",
        dry_run=True, seed_start=1, seed_count=4,
    )
    launch = launch_args(args, none, arec, paths, None)
    runs = make_grid(launch)
    assert len(runs) == 8
    assert {run.seed for run in runs} == {1, 2, 3, 4}
    assert {run.method for run in runs} == {"none", "arec"}
    assert launch.total_timesteps == budget
    assert launch.arec_coefs == (coefficient,)
    assert launch.arec_q_steps == (q_steps,)
    assert all(run.lr == 0.002 and run.epochs == 4 for run in runs)
    commands = {run.method: train_command(args.run_root, launch, run)
                for run in runs if run.seed == 1}
    assert "ACTOR_SCORE_RECOVERY=false" in commands["none"]
    assert "ACTOR_SCORE_RECOVERY=true" in commands["arec"]
    assert f"ACTOR_SCORE_RECOVERY_Q_STEPS={q_steps}" in commands["arec"]


@pytest.mark.parametrize("task", TASKS)
def test_six_seed_extension_uses_same_frozen_cell(task: str, tmp_path: Path) -> None:
    none, arec, paths = load_pair(task)
    args = SimpleNamespace(
        task=task, run_root=tmp_path / "extension_6seed", gpus=("0", "1"),
        max_runs_per_gpu=2, project="test", wandb_mode="disabled",
        dry_run=True, seed_start=5, seed_count=6,
    )
    runs = make_grid(launch_args(args, none, arec, paths, None))
    assert len(runs) == 12
    assert {run.seed for run in runs} == set(range(5, 11))
    assert all({run.method for run in runs if run.seed == seed} == {"none", "arec"}
               for seed in range(5, 11))


def test_historical_report_and_checkpoint_verification(tmp_path: Path) -> None:
    task = "10m_vs_11m"
    none, arec, _ = load_pair(task)
    source = tmp_path / none["historical_source_root"]
    report = {
        "source_roots": {task: str(source)},
        "selection": {task: {"parameters": {
            "coef": 3e-6, "q_steps": 4,
            "q_learning_rate": 0.001, "fisher_ridge": 0.001,
        }}},
    }
    _write(tmp_path / "actor_score_recovery_best_return_report_v1" /
           "report_manifest.json", report)
    runs = []
    for profile in (none, arec):
        method, config = profile["method"], profile["config"]
        for seed in (1, 2, 3, 4):
            name = f"historical-{method}-seed{seed}"
            run = dict(run_name=name, condition=("none" if method == "none"
                                                 else "actor_score_recovery"),
                       seed=seed, coef=config["ACTOR_SCORE_RECOVERY_COEF"],
                       q_steps=config["ACTOR_SCORE_RECOVERY_Q_STEPS"],
                       q_learning_rate=0.001, fisher_ridge=0.001)
            runs.append(run)
            _write(source / "status" / f"{name}.json", {"status": "completed"})
            old_config = dict(config, SEED=seed, GIT_COMMIT="historical-commit")
            if method == "arec":
                old_config["EXPERIMENT_CONDITION"] = "actor_score_recovery"
            _write(source / "checkpoints" / "project" / f"{name}-hash" /
                   "final" / "config.json", old_config)
    _write(source / "experiment_manifest.json", dict(
        protocol=HISTORICAL_PROTOCOL, maps=[task], seeds=[1, 2, 3, 4],
        budgets={task: 10_000_000}, learning_rate=0.002, update_epochs=4,
        num_envs=128, num_minibatches=4, checkpoint_interval=1_000_000,
        runs=runs,
    ))
    verified = verify_historical(task, (none, arec), tmp_path)
    assert verified["selected_arec"]["coef"] == 3e-6
    assert verified["historical_git_commits"] == ["historical-commit"]
    wrong = dict(report)
    wrong["selection"] = {task: {"parameters": {
        **report["selection"][task]["parameters"], "coef": 3e-5,
    }}}
    _write(tmp_path / "actor_score_recovery_best_return_report_v1" /
           "report_manifest.json", wrong)
    with pytest.raises(RuntimeError, match="selected ARec cell differs"):
        verify_historical(task, (none, arec), tmp_path)
