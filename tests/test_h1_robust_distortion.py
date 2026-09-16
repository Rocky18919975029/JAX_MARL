import json
import sys

import numpy as np
import pytest

from scripts.h1_robust_distortion import (
    aggregate_fisher_metrics,
    grouped_statistics,
    main,
    make_group_statistics,
    uniform_phase_marginalized_gae,
)
from scripts.h1_latent_distortion import reconstruct_training_gae


def test_uniform_phase_marginalized_gae_keeps_boundary_bootstrap():
    rewards = np.ones((1, 3, 1), dtype=np.float64)
    values = np.zeros_like(rewards)
    values[0, :, 0] = (10.0, 20.0, 30.0)
    done = np.asarray([[False, False, True]])
    active = np.ones((1, 3), dtype=bool)
    result = uniform_phase_marginalized_gae(
        rewards, values, done, active, gamma=1.0, gae_lambda=1.0, rollout_steps=2
    )
    # phase 0: [22, 11, -29]; phase 1: [11, -18, -29].  A transition that
    # ends a rollout retains its one-step value bootstrap but does not include
    # the next transition's recursive GAE term.
    np.testing.assert_allclose(result[0, :, 0], [16.5, -3.5, -29.0])


def test_uniform_phase_differs_from_episode_aligned_raw_gae():
    rewards = np.ones((1, 3, 1), dtype=np.float64)
    values = np.zeros_like(rewards)
    done = np.asarray([[False, False, True]])
    active = np.ones((1, 3), dtype=bool)
    raw = reconstruct_training_gae(
        rewards, values, done, active, gamma=1.0, gae_lambda=1.0, rollout_steps=2
    )
    phase = uniform_phase_marginalized_gae(
        rewards, values, done, active, gamma=1.0, gae_lambda=1.0, rollout_steps=2
    )
    np.testing.assert_allclose(raw[0, :, 0], [2.0, 1.0, 1.0])
    np.testing.assert_allclose(phase[0, :, 0], [1.5, 1.5, 1.0])


def test_natural_cosine_and_optimal_scale_remove_positive_scale_only_error():
    scores = np.asarray([[1.0], [-1.0], [2.0], [-2.0]])
    reference = np.asarray([2.0, -2.0, 4.0, -4.0])
    critic = reference / 2.0
    group = make_group_statistics(
        scores,
        reference,
        critic,
        critic,
        key=(0,),
        slot=0,
        unit_type=None,
        weight=1.0,
    )
    metrics, _ = aggregate_fisher_metrics([group], 1e-3)
    assert metrics["epsilon_lat_raw"] > 0
    assert metrics["epsilon_lat_phase_matched"] > 0
    assert metrics["fisher_natural_gradient_cosine"] == pytest.approx(1.0)
    assert metrics["optimal_nonnegative_critic_scale"] == pytest.approx(2.0)
    assert metrics["epsilon_lat_optimal_scale"] == pytest.approx(0.0, abs=1e-12)


def test_cached_fisher_eigensystem_matches_direct_solve():
    rng = np.random.default_rng(7)
    scores = rng.normal(size=(80, 6))
    reference = rng.normal(size=80)
    raw = rng.normal(size=80)
    phase = rng.normal(size=80)
    group = make_group_statistics(
        scores,
        reference,
        raw,
        phase,
        key=(0,),
        slot=0,
        unit_type=None,
        weight=1.0,
    )
    ridge = 0.003
    metrics, _ = aggregate_fisher_metrics([group], ridge)
    fisher = scores.T @ scores / len(scores)
    g_reference = np.mean(scores * reference[:, None], axis=0)
    g_raw = np.mean(scores * raw[:, None], axis=0)
    delta = g_reference - g_raw
    expected = float(delta @ np.linalg.solve(fisher + ridge * np.eye(6), delta))
    assert metrics["epsilon_lat_raw"] == pytest.approx(expected, rel=1e-10)


def test_slot_and_slot_type_are_identical_for_one_fixed_type():
    episodes, timesteps, agents, dimension = 4, 3, 2, 2
    active = np.ones((episodes, timesteps), dtype=bool)
    alive = np.ones((episodes, timesteps, agents), dtype=bool)
    scores = np.arange(
        1, episodes * timesteps * agents * dimension + 1, dtype=np.float64
    ).reshape(episodes, timesteps, agents, dimension)
    arrays = {
        "active": active,
        "alive": alive,
        "diagnostic_episode_id": np.arange(episodes),
        "mc_return": np.ones((episodes, timesteps, agents)),
        "actor_score": scores,
        "state_unit_types": np.zeros((episodes, timesteps, agents * 2)),
    }
    raw = np.full((episodes, timesteps, agents), 0.75)
    phase = np.full((episodes, timesteps, agents), 0.5)
    slot = grouped_statistics(arrays, raw, phase, 4, "slot", agents)
    typed = grouped_statistics(arrays, raw, phase, 4, "slot_x_type", agents)
    slot_metrics, _ = aggregate_fisher_metrics(slot, 1e-3)
    typed_metrics, _ = aggregate_fisher_metrics(typed, 1e-3)
    for name in (
        "epsilon_lat_raw",
        "epsilon_lat_phase_matched",
        "fisher_natural_gradient_cosine",
        "epsilon_lat_optimal_scale",
    ):
        assert typed_metrics[name] == pytest.approx(slot_metrics[name])


def test_checkpoint_recomputation_writes_all_audit_tables(tmp_path, monkeypatch):
    diagnostic = tmp_path / "diagnostic"
    output = tmp_path / "output"
    diagnostic.mkdir()
    episodes, timesteps, agents, dimension = 4, 3, 2, 2
    active = np.ones((episodes, timesteps), dtype=bool)
    done = np.zeros((episodes, timesteps), dtype=bool)
    done[:, -1] = True
    reward = np.ones((episodes, timesteps, agents), dtype=np.float32)
    arrays = {
        "active": active,
        "alive": np.ones((episodes, timesteps, agents), dtype=bool),
        "diagnostic_episode_id": np.arange(episodes, dtype=np.int32),
        "mc_return": np.asarray(
            [[[3.0, 3.0], [2.0, 2.0], [1.0, 1.0]]] * episodes,
            dtype=np.float32,
        ),
        "reward": reward,
        "value": np.zeros_like(reward),
        "global_done": done,
        "actor_score": np.arange(
            1, episodes * timesteps * agents * dimension + 1, dtype=np.float32
        ).reshape(episodes, timesteps, agents, dimension),
        "state_unit_types": np.zeros((episodes, timesteps, agents * 2), dtype=np.int32),
    }
    np.savez_compressed(diagnostic / "episodes_0000.npz", **arrays)
    metadata = {
        "shards": [{"path": "episodes_0000.npz"}],
        "actor_parameter_sharing": False,
        "gamma": 1.0,
        "gae_lambda": 1.0,
        "training_rollout_steps": 2,
        "run_id": "id",
        "run_name": "run",
        "map_name": "map",
        "condition": "none",
        "align_distance": "ln_mse",
        "training_seed": 1,
        "checkpoint_env_step": 0,
        "checkpoint_nominal_env_step": 0,
        "protocol_version": "test",
        "git_commit": "commit",
        "num_agents": agents,
    }
    (diagnostic / "metadata.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "h1_robust_distortion.py",
            "--diagnostics-dir",
            str(diagnostic),
            "--output-dir",
            str(output),
            "--fisher-ridges",
            "0.001,0.01",
        ],
    )
    main()
    summary = json.loads((output / "robust_distortion_summary.json").read_text())
    assert summary["heldout_episodes"] == 4
    assert len(summary["aggregate_metrics"]) == 3 * 2 * 2
    assert summary["single_type_aggregation_consistency"]["status"] == "pass"
    for name in (
        "robust_distortion_metrics.csv",
        "robust_distortion_groups.csv",
        "robust_distortion_convergence.csv",
    ):
        assert (output / name).is_file()

    canonical_raw = next(
        row["epsilon_lat_raw"]
        for row in summary["aggregate_metrics"]
        if row["aggregation"] == "slot"
        and row["episode_budget_label"] == "4M"
        and row["fisher_ridge_absolute"] == 0.001
    )
    (diagnostic / "latent_summary.json").write_text(
        json.dumps({"epsilon_lat": canonical_raw, "fisher_ridge_absolute": 0.001})
    )
    second_output = tmp_path / "second_output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "h1_robust_distortion.py",
            "--diagnostics-dir",
            str(diagnostic),
            "--output-dir",
            str(second_output),
            "--fisher-ridges",
            "0.001,0.01",
        ],
    )
    main()
    second = json.loads((second_output / "robust_distortion_summary.json").read_text())
    assert second["legacy_raw_reproduction"]["available"] is True
    assert second["legacy_raw_reproduction"]["absolute_error"] == pytest.approx(0.0)
