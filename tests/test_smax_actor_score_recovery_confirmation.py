import json
from pathlib import Path

import pytest

from scripts.run_smax_actor_score_recovery_confirmation import (
    confirmation_plan,
    launcher_command,
)
from scripts.run_smax_actor_score_recovery_sweep import PROTOCOL, run_matrix


def pilot_selection(tmp_path):
    sweep = tmp_path / "pilot"
    sweep.mkdir()
    sweep.joinpath("experiment_manifest.json").write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "maps": ["10m_vs_11m", "3s5z_vs_3s6z"],
                "seeds": [9001, 9002],
                "budgets": {
                    "10m_vs_11m": 10_000_000,
                    "3s5z_vs_3s6z": 20_000_000,
                },
                "update_epochs": 4,
                "learning_rate": 0.002,
                "num_envs": 128,
                "num_minibatches": 4,
                "checkpoint_interval": 1_000_000,
            }
        )
    )
    selection = {}
    for task, budget, coef, q_steps in (
        ("10m_vs_11m", 10_000_000, 3e-5, 4),
        ("3s5z_vs_3s6z", 20_000_000, 3e-4, 8),
    ):
        selection[task] = {
            "selection_metric": "mean seed-paired training win-rate AUC gain",
            "promote_to_confirmatory": True,
            "top_candidate": {
                "condition": "actor_score_recovery",
                "budget": budget,
                "coef": coef,
                "q_steps": q_steps,
                "q_learning_rate": 1e-3,
                "fisher_ridge": 1e-3,
                "paired_delta_win_rate_auc_vs_none": 0.1,
            },
        }
    return sweep, selection


def test_confirmation_is_per_task_full_budget_and_baseline_paired(tmp_path):
    sweep, selection = pilot_selection(tmp_path)
    root = tmp_path / "confirmation"
    plan = confirmation_plan(
        sweep,
        root,
        ("10m_vs_11m", "3s5z_vs_3s6z"),
        (1, 2, 3, 4),
        selection,
    )
    assert plan["tasks"]["10m_vs_11m"]["coef"] == 3e-5
    assert plan["tasks"]["3s5z_vs_3s6z"]["coef"] == 3e-4
    assert plan["tasks"]["10m_vs_11m"]["training_budget"] == 10_000_000
    assert plan["tasks"]["3s5z_vs_3s6z"]["training_budget"] == 20_000_000

    for task, cell in plan["tasks"].items():
        cmd = launcher_command(
            Path("/repo"),
            task,
            cell,
            plan["pilot_ppo"],
            ("0", "1", "2", "3"),
            2,
            "project",
            "disabled",
        )
        assert cmd[cmd.index("--maps") + 1] == task
        assert cmd[cmd.index("--seeds") + 1] == "1,2,3,4"
        assert cmd[cmd.index("--conditions") + 1] == "none,actor_score_recovery"
        assert cmd[cmd.index("--coefs") + 1] == str(cell["coef"])
        assert "--budget-fraction" not in cmd
        assert "--total-timesteps" not in cmd
        assert cmd[cmd.index("--learning-rate") + 1] == "0.002"
        assert cmd[cmd.index("--max-runs-per-gpu") + 1] == "2"
        runs = run_matrix(
            (task,),
            (1, 2, 3, 4),
            {task: cell["training_budget"]},
            (cell["coef"],),
            (cell["q_steps"],),
            (cell["q_learning_rate"],),
            (cell["fisher_ridge"],),
        )
        assert len(runs) == 8
        assert sum(run.condition == "none" for run in runs) == 4


def test_confirmation_rejects_reused_pilot_seed(tmp_path):
    sweep, selection = pilot_selection(tmp_path)
    with pytest.raises(ValueError, match="overlap"):
        confirmation_plan(
            sweep,
            tmp_path / "confirmation",
            ("10m_vs_11m",),
            (9001, 1),
            selection,
        )
