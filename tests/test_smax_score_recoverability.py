import json
from types import SimpleNamespace

import numpy as np

from scripts.run_smax_score_recoverability import common_sample_counts
from scripts.smax_score_recoverability import (
    available_samples_by_agent,
    episode_fit_mask,
    fisher_whiten,
    prepare_probe_data,
)


def test_episode_split_is_deterministic_and_nonempty():
    first = episode_fit_mask(20, 0.75, 17)
    second = episode_fit_mask(20, 0.75, 17)
    assert np.array_equal(first, second)
    assert first.sum() == 15
    assert (~first).sum() == 5


def test_fisher_whitening_uses_ridged_empirical_fisher():
    fit = np.asarray(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0], [0.0, -2.0]],
        dtype=np.float32,
    )
    test = np.asarray([[1.0, 2.0]], dtype=np.float32)
    fit_u, test_u, eigenvalues = fisher_whiten(fit, test, ridge=0.5)
    assert np.allclose(eigenvalues, [0.5, 2.0])
    assert np.allclose(test_u, [[1.0, 2.0 / np.sqrt(2.5)]], atol=1e-6)
    expected_covariance = np.diag(eigenvalues / (eigenvalues + 0.5))
    assert np.allclose(fit_u.T @ fit_u / len(fit_u), expected_covariance)


def test_prepare_probe_data_enforces_equal_per_agent_counts_and_valid_actions():
    rng = np.random.default_rng(5)
    episodes, timesteps, agents, latent_dim, action_dim = 16, 8, 3, 5, 4
    action = rng.integers(0, action_dim, size=(episodes, timesteps, agents))
    available = np.ones((episodes, timesteps, agents, action_dim), dtype=np.float32)
    arrays = {
        "active": np.ones((episodes, timesteps), dtype=bool),
        "alive": np.ones((episodes, timesteps, agents), dtype=bool),
        "available_actions": available,
        "action": action,
        "actor_latent": rng.normal(
            size=(episodes, timesteps, agents, latent_dim)
        ).astype(np.float32),
        "critic_latent": rng.normal(
            size=(episodes, timesteps, agents, latent_dim)
        ).astype(np.float32),
        "actor_score": rng.normal(
            size=(episodes, timesteps, agents, latent_dim)
        ).astype(np.float32),
        "diagnostic_episode_id": np.arange(episodes),
    }
    prepared = prepare_probe_data(
        arrays,
        fit_fraction=0.75,
        split_seed=11,
        sampling_seed=12,
        fit_samples_per_agent=32,
        test_samples_per_agent=16,
        fisher_ridge=1e-3,
    )
    assert prepared["fit_x"].shape == (agents, 32, latent_dim + action_dim)
    assert prepared["fit_y"].shape == (agents, 32, latent_dim)
    assert prepared["test_x"].shape == (agents, 16, latent_dim + action_dim)
    assert prepared["test_y"].shape == (agents, 16, latent_dim)
    assert len(prepared["fisher_audit"]) == agents


def test_invalid_stored_action_is_rejected():
    rng = np.random.default_rng(9)
    arrays = {
        "active": np.ones((4, 4), dtype=bool),
        "alive": np.ones((4, 4, 1), dtype=bool),
        "available_actions": np.ones((4, 4, 1, 2), dtype=np.float32),
        "action": np.zeros((4, 4, 1), dtype=np.int32),
        "actor_latent": rng.normal(size=(4, 4, 1, 3)).astype(np.float32),
        "critic_latent": rng.normal(size=(4, 4, 1, 3)).astype(np.float32),
        "actor_score": rng.normal(size=(4, 4, 1, 3)).astype(np.float32),
        "diagnostic_episode_id": np.arange(4),
    }
    arrays["available_actions"][..., 0] = 0.0
    try:
        prepare_probe_data(
            arrays,
            fit_fraction=0.5,
            split_seed=1,
            sampling_seed=2,
            fit_samples_per_agent=2,
            test_samples_per_agent=2,
            fisher_ridge=1e-3,
        )
    except RuntimeError as error:
        assert "invalid under the stored SMAX mask" in str(error)
    else:
        raise AssertionError("Invalid sampled actions must fail the audit")


def test_common_sample_count_uses_taskwide_minimum(tmp_path):
    jobs = []
    for condition, alive_count in (("none", 4), ("c_to_a_cka", 3)):
        checkpoint = tmp_path / condition / "final"
        output = tmp_path / "runs" / condition
        collected = output / "collected"
        checkpoint.mkdir(parents=True)
        collected.mkdir(parents=True)
        active = np.ones((100, 5), dtype=bool)
        alive = np.ones((100, 5, 2), dtype=bool)
        alive[:, alive_count:, 1] = False
        np.savez_compressed(collected / "episodes_0000.npz", active=active, alive=alive)
        (collected / "metadata.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint),
                    "episodes": 100,
                    "shards": [{"path": "episodes_0000.npz", "episodes": 100}],
                }
            )
        )
        jobs.append(
            (
                condition,
                SimpleNamespace(task="test_map", checkpoint=checkpoint),
                output,
            )
        )
    counts = common_sample_counts(
        jobs,
        fit_fraction=0.5,
        split_seed=7,
        fit_cap=200,
        test_cap=200,
    )
    assert counts["test_map"]["fit_samples_per_agent"] == 150
    assert counts["test_map"]["test_samples_per_agent"] == 150
    assert len(counts["test_map"]["census"]) == 2


def test_availability_counts_only_active_and_alive_steps():
    active = np.ones((4, 3), dtype=bool)
    active[:, -1] = False
    alive = np.ones((4, 3, 2), dtype=bool)
    alive[:, 0, 1] = False
    fit, test = available_samples_by_agent({"active": active, "alive": alive}, 0.5, 1)
    assert fit.tolist() == [4, 2]
    assert test.tolist() == [4, 2]
