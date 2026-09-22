"""Numerical and protocol tests for the offline action-conditioned ridge probe."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.analyze_smax_linear_score_recoverability import analyze
from scripts.run_smax_linear_score_recoverability import (
    build_jobs,
    completed_matches,
    main,
    run_job,
)
from scripts.smax_linear_score_recoverability import (
    fit_action_conditioned_ridge,
    measure_from_collection,
    normalized_error,
    sample_agent_split,
)
from scripts.smax_score_recoverability import episode_fit_mask


def test_action_specific_slopes_and_unpenalized_intercepts():
    x = np.linspace(-2, 2, 100)[:, None]
    action = np.tile([0, 1], 50)
    target = np.where(action == 0, 2 * x[:, 0] + 3, -4 * x[:, 0] - 5)[:, None]
    fit_pred, test_pred, support = fit_action_conditioned_ridge(
        x, action, target, x, action, action_dim=2, ridge=1e-8
    )
    np.testing.assert_allclose(fit_pred, target, atol=1e-6)
    np.testing.assert_allclose(test_pred, target, atol=1e-6)
    assert all(row["fitted"] for row in support)

    # A very large ridge shrinks slopes, but must not shrink the intercept.
    constant_x = np.zeros((20, 1))
    constants = np.full((20, 1), 7.0)
    _, prediction, _ = fit_action_conditioned_ridge(
        constant_x,
        np.zeros(20, dtype=int),
        constants,
        constant_x,
        np.zeros(20, dtype=int),
        action_dim=1,
        ridge=1e6,
    )
    np.testing.assert_allclose(prediction, constants, atol=1e-10)


def test_rare_action_uses_zero_predictor_and_reports_support():
    fit_z = np.ones((8, 2))
    fit_action = np.array([0] * 7 + [1])
    fit_u = np.ones((8, 2))
    test_z = np.ones((4, 2))
    test_action = np.array([1, 1, 0, 2])
    _, prediction, support = fit_action_conditioned_ridge(
        fit_z,
        fit_action,
        fit_u,
        test_z,
        test_action,
        action_dim=3,
        ridge=1e-3,
    )
    np.testing.assert_array_equal(prediction[[0, 1, 3]], 0)
    assert [row["fallback_test_count"] for row in support] == [0, 2, 1]
    assert normalized_error(np.ones((4, 2)), np.zeros((4, 2)))[2] == 1.0


def fixture_collection(tmp_path):
    rng = np.random.default_rng(51)
    episodes, steps, agents, dimensions, actions = 16, 32, 2, 4, 3
    critic = rng.normal(size=(episodes, steps, agents, dimensions)).astype(np.float32)
    action = rng.integers(0, actions, size=(episodes, steps, agents), dtype=np.int32)
    maps = rng.normal(size=(agents, actions, dimensions, dimensions))
    score = np.empty_like(critic)
    for agent in range(agents):
        for which in range(actions):
            mask = action[:, :, agent] == which
            score[:, :, agent][mask] = critic[:, :, agent][mask] @ maps[agent, which]
    arrays = {
        "active": np.ones((episodes, steps), dtype=bool),
        "alive": np.ones((episodes, steps, agents), dtype=bool),
        "available_actions": np.ones(
            (episodes, steps, agents, actions), dtype=np.float32
        ),
        "action": action,
        "critic_latent": critic,
        "actor_score": score,
    }
    collected = tmp_path / "collected"
    collected.mkdir()
    np.savez(collected / "shard.npz", **arrays)
    metadata = {
        "map_name": "10m_vs_11m",
        "condition": "none",
        "align_distance": "ln_mse",
        "training_seed": 1,
        "checkpoint": "/frozen/checkpoint/final",
        "actor_parameter_sharing": False,
        "array_profile": "score_recoverability",
        "episodes": episodes,
        "shards": [{"path": "shard.npz"}],
    }
    (collected / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    summary = {
        "task": "10m_vs_11m",
        "condition": "none",
        "training_seed": 1,
        "checkpoint": metadata["checkpoint"],
        "episodes": episodes,
        "fit_fraction": 0.75,
        "split_seed": 13,
        "sampling_seed": 17,
        "fit_samples_per_agent": 240,
        "test_samples_per_agent": 90,
        "fisher_ridge_absolute": 1e-3,
    }
    return collected, summary, arrays


def test_collection_measurement_uses_disjoint_episodes_and_is_recoverable(tmp_path):
    collected, summary, arrays = fixture_collection(tmp_path)
    mask = episode_fit_mask(16, summary["fit_fraction"], summary["split_seed"])
    fit = sample_agent_split(arrays, 0, mask, 240, summary["sampling_seed"])
    test = sample_agent_split(
        arrays, 0, ~mask, 90, summary["sampling_seed"] + 1_000_000
    )
    assert not set(fit["episode"]) & set(test["episode"])

    output = tmp_path / "result"
    result = measure_from_collection(collected, summary, output)
    assert result["num_agents"] == 2
    assert result["epsilon_rec_lin_normalized"] < 0.05
    assert result["fallback_test_fraction"] == 0
    assert result["fit_episodes"] == 12
    assert result["test_episodes"] == 4
    assert (output / "agent_metrics.csv").is_file()
    assert (output / "action_support.csv").is_file()


def test_runner_reuses_source_and_fails_on_changed_protocol(tmp_path):
    source_root = tmp_path / "source"
    cell = source_root / "runs" / "10m_vs_11m" / "none" / "seed_1"
    cell.mkdir(parents=True)
    collected, source_summary, _ = fixture_collection(cell)
    (cell / "summary.json").write_text(json.dumps(source_summary), encoding="utf-8")
    jobs = build_jobs(source_root, ["10m_vs_11m"], ["none"], [1])
    run_root = tmp_path / "linear"
    result = run_job(jobs[0], str(run_root), 1e-3, None)
    assert result["epsilon_rec_lin_normalized"] < 0.05
    assert completed_matches(jobs[0], run_root, 1e-3, None)
    with pytest.raises(RuntimeError, match="ridge"):
        completed_matches(jobs[0], run_root, 1e-2, None)
    (run_root / "protocol.json").write_text(
        json.dumps(
            {
                "protocol": "smax-linear-score-recoverability-v1.0",
                "tasks": ["10m_vs_11m"],
                "conditions": ["none"],
                "seeds": [1],
                "rare_action_policy": "zero_predictor_and_report_fraction",
            }
        ),
        encoding="utf-8",
    )
    analyze(run_root)
    assert (
        run_root / "analysis" / "10m_vs_11m" / "task_condition_summary.csv"
    ).is_file()
    assert (
        run_root / "analysis" / "10m_vs_11m" / "linear-score-recoverability.png"
    ).is_file()


def test_runner_cli_executes_and_resumes_without_recollection(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    cell = source_root / "runs" / "10m_vs_11m" / "none" / "seed_1"
    cell.mkdir(parents=True)
    _, source_summary, _ = fixture_collection(cell)
    (cell / "summary.json").write_text(json.dumps(source_summary), encoding="utf-8")
    (source_root / "protocol.json").write_text(
        json.dumps(
            {
                "protocol": "smax-score-recoverability-v1.0",
                "tasks": ["10m_vs_11m"],
                "conditions": ["none"],
                "seeds": [1],
                "selected_budgets": {"10m_vs_11m": 10_000_000},
            }
        ),
        encoding="utf-8",
    )
    run_root = tmp_path / "linear"
    argv = [
        "run_smax_linear_score_recoverability.py",
        "--source-root",
        str(source_root),
        "--run-root",
        str(run_root),
        "--workers",
        "1",
        "--skip-analysis",
    ]
    monkeypatch.setattr("sys.argv", argv)
    main()
    summary_path = run_root / "runs" / "10m_vs_11m" / "none" / "seed_1" / "summary.json"
    assert summary_path.is_file()
    assert json.loads(summary_path.read_text())["epsilon_rec_lin_normalized"] < 0.05
    main()
