import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.analyze_smax_actor_score_recovery_sweep import metric_summary, summarize
from scripts.plot_smax_actor_score_recovery_sweep import (
    make_figure,
    seed_bootstrap_curves,
)
from scripts.run_smax_actor_score_recovery_sweep import (
    PROTOCOL,
    command,
    hydra_float,
    positive_float_items,
    run_matrix,
)


def test_sweep_contains_one_isolated_run_per_task_seed_and_unique_grid():
    budgets = {"10m_vs_11m": 10_000_000, "3s5z_vs_3s6z": 20_000_000}
    runs = run_matrix(
        tuple(budgets),
        (9001, 9002),
        budgets,
        (3e-5, 1e-4, 3e-4),
        (4, 8),
        (1e-3,),
        (1e-3,),
    )
    assert len(runs) == 28
    assert len({run.name for run in runs}) == 28
    assert sum(run.condition == "none" for run in runs) == 4
    assert all(run.steps == budgets[run.map_name] for run in runs)
    none_only = run_matrix(
        tuple(budgets),
        (1, 2, 3, 4),
        budgets,
        (1e-4,),
        (8,),
        (1e-3,),
        (1e-3,),
        conditions=("none",),
    )
    assert len(none_only) == 8
    assert all(run.condition == "none" for run in none_only)


def test_baseline_disables_all_auxiliary_losses_and_matches_ppo_settings(tmp_path):
    args = SimpleNamespace(
        update_epochs=4,
        learning_rate=0.002,
        num_envs=128,
        num_minibatches=4,
        checkpoint_interval=1_000_000,
        wandb_mode="disabled",
        project="test",
    )
    baseline, recovery = run_matrix(
        ("10m_vs_11m",),
        (9001,),
        {"10m_vs_11m": 10_000_000},
        (1e-4,),
        (8,),
        (1e-3,),
        (1e-3,),
    )
    base_command = command(Path("/repo"), tmp_path, args, baseline)
    recovery_command = command(Path("/repo"), tmp_path, args, recovery)
    assert "ACTOR_SCORE_RECOVERY=false" in base_command
    assert "ACTOR_SCORE_RECOVERY_COEF=0" in base_command
    assert "EXPERIMENT_CONDITION=none" in base_command
    assert "ACTOR_SCORE_RECOVERY=true" in recovery_command
    assert "ACTOR_SCORE_RECOVERY_COEF=0.0001" in recovery_command
    assert "EXPERIMENT_CONDITION=actor_score_recovery" in recovery_command
    for setting in (
        "ACTOR_PARAMETER_SHARING=false",
        "MATCHED_COMPARISON=true",
        "ALIGN_MODE=none",
        "ALIGNMENT_COEF=0",
        "SCORE_RECOVERY=false",
        "ORACLE_LATENT_DISTORTION=false",
        "TOTAL_TIMESTEPS=10000000",
        "UPDATE_EPOCHS=4",
        "LR=0.002",
        "NUM_ENVS=128",
        "NUM_MINIBATCHES=4",
    ):
        assert setting in base_command
        assert setting in recovery_command


def test_float_grid_rejects_nonpositive_and_aliases():
    assert positive_float_items("0.00003,0.0001") == (3e-5, 1e-4)
    assert hydra_float(3e-5) == "0.00003"
    with pytest.raises(Exception):
        positive_float_items("0.0001,1e-4")
    with pytest.raises(Exception):
        positive_float_items("0,0.1")


def test_auc_and_last_five_are_distinct():
    history = [
        {"env_step": step, "win_rate": value}
        for step, value in ((25, 0.2), (50, 0.4), (75, 0.6), (100, 0.8))
    ]
    auc, last5 = metric_summary(history, "win_rate", 100)
    assert auc == pytest.approx(0.425)
    assert last5 == pytest.approx(0.5)


def test_analysis_selects_per_task_against_seed_paired_none(tmp_path):
    budgets = {"10m_vs_11m": 100, "3s5z_vs_3s6z": 200}
    runs = run_matrix(
        tuple(budgets),
        (9001, 9002),
        budgets,
        (1e-4, 3e-4),
        (8,),
        (1e-3,),
        (1e-3,),
    )
    (tmp_path / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "maps": list(budgets),
                "seeds": [9001, 9002],
                "budgets": budgets,
                "runs": [dict(run.__dict__, run_name=run.name) for run in runs],
            }
        )
    )
    (tmp_path / "status").mkdir()
    (tmp_path / "metrics").mkdir()
    for run in runs:
        (tmp_path / "status" / f"{run.name}.json").write_text(
            json.dumps({"status": "completed"})
        )
        if run.condition == "none":
            gain = 0
        elif run.map_name == "10m_vs_11m":
            gain = 0.2 if run.coef == 3e-4 else 0.1
        else:
            gain = -0.1 if run.coef == 3e-4 else 0.05
        with (tmp_path / "metrics" / f"{run.name}.jsonl").open("w") as file:
            for step in (run.steps // 2, run.steps):
                value = step / run.steps * 0.4 + gain + run.seed * 0.00001
                file.write(
                    json.dumps({"env_step": step, "returns": value, "win_rate": value})
                    + "\n"
                )
    selection = summarize(tmp_path)
    assert selection["10m_vs_11m"]["top_candidate"]["coef"] == 3e-4
    assert selection["3s5z_vs_3s6z"]["top_candidate"]["coef"] == 1e-4
    for task in budgets:
        assert (tmp_path / "analysis" / task / "seed_level.csv").is_file()
        assert (tmp_path / "analysis" / task / "task_condition_summary.csv").is_file()


def test_seed_bootstrap_plot_keeps_tasks_and_baseline_separate(tmp_path):
    pytest.importorskip("matplotlib")
    budgets = {"10m_vs_11m": 10_000_000, "3s5z_vs_3s6z": 20_000_000}
    runs = run_matrix(
        tuple(budgets),
        (9001, 9002),
        budgets,
        (3e-5, 1e-4, 3e-4),
        (4, 8),
        (1e-3,),
        (1e-3,),
    )
    (tmp_path / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "maps": list(budgets),
                "seeds": [9001, 9002],
                "budgets": budgets,
                "runs": [dict(run.__dict__, run_name=run.name) for run in runs],
            }
        )
    )
    (tmp_path / "status").mkdir()
    (tmp_path / "metrics").mkdir()
    for run in runs:
        (tmp_path / "status" / f"{run.name}.json").write_text(
            json.dumps({"status": "completed"})
        )
        task_offset = 0.0 if run.map_name == "10m_vs_11m" else 0.4
        seed_offset = 0.0 if run.seed == 9001 else 0.1
        condition_offset = (
            0.0
            if run.condition == "none"
            else {3e-5: 0.02, 1e-4: 0.05, 3e-4: 0.08}[run.coef]
        )
        q_offset = 0.0 if run.condition == "none" else run.q_steps * 0.001
        with (tmp_path / "metrics" / f"{run.name}.jsonl").open("w") as file:
            for step in (run.steps // 4, run.steps // 2, 3 * run.steps // 4, run.steps):
                value = 0.1 + task_offset + seed_offset + condition_offset + q_offset
                file.write(json.dumps({"env_step": step, "win_rate": value}) + "\n")

    manifest, rows = seed_bootstrap_curves(
        tmp_path, bootstrap_samples=1000, bootstrap_seed=7
    )
    assert len(rows) == 2 * 7 * 4  # two tasks, baseline + six settings, four steps
    first = next(
        row
        for row in rows
        if row["task"] == "10m_vs_11m"
        and row["condition"] == "none"
        and row["env_step"] == 2_500_000
    )
    second = next(
        row
        for row in rows
        if row["task"] == "3s5z_vs_3s6z"
        and row["condition"] == "none"
        and row["env_step"] == 5_000_000
    )
    assert manifest["seeds"] == [9001, 9002]
    assert first["mean_win_rate"] == pytest.approx(0.15)
    assert first["ci95_low"] == pytest.approx(0.1)
    assert first["ci95_high"] == pytest.approx(0.2)
    assert second["mean_win_rate"] == pytest.approx(0.55)
    assert second["ci95_low"] == pytest.approx(0.5)
    assert second["ci95_high"] == pytest.approx(0.6)

    image = make_figure(tmp_path, bootstrap_samples=1000, bootstrap_seed=7)
    assert image.is_file()
    assert image.with_suffix(".pdf").is_file()
    assert image.with_suffix(".svg").is_file()
    assert (tmp_path / "analysis" / "10m_vs_11m" / f"{image.stem}-curves.csv").is_file()
    assert (
        tmp_path / "analysis" / "3s5z_vs_3s6z" / f"{image.stem}-curves.csv"
    ).is_file()


def test_seed_bootstrap_plot_rejects_incomplete_runs(tmp_path):
    runs = run_matrix(
        ("10m_vs_11m",),
        (9001, 9002),
        {"10m_vs_11m": 100},
        (1e-4,),
        (8,),
        (1e-3,),
        (1e-3,),
    )
    (tmp_path / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "maps": ["10m_vs_11m"],
                "seeds": [9001, 9002],
                "budgets": {"10m_vs_11m": 100},
                "runs": [dict(run.__dict__, run_name=run.name) for run in runs],
            }
        )
    )
    with pytest.raises(RuntimeError, match="not completed"):
        seed_bootstrap_curves(tmp_path, bootstrap_samples=100)
