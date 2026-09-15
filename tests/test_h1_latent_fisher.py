import numpy as np

from scripts.h1_latent_distortion import (
    convergence_episode_budgets,
    fisher_metrics,
    fisher_metrics_from_statistics,
    fisher_statistics,
    heldout_episode_returns,
    reconstruct_training_gae,
    vector_cosine,
)


def test_cached_fisher_statistics_preserve_metrics():
    rng = np.random.default_rng(17)
    scores = rng.normal(size=(512, 8))
    reference = rng.normal(size=512)
    critic = rng.normal(size=512)
    direct = fisher_metrics(scores, reference, critic, 1e-3)
    cached = fisher_metrics_from_statistics(
        fisher_statistics(scores, reference, critic), 1e-3
    )
    for key in direct:
        assert np.allclose(direct[key], cached[key], equal_nan=True)


def test_baseline_free_reference_is_raw_mc_score_weighting():
    rng = np.random.default_rng(19)
    scores = rng.normal(size=(1024, 6))
    mc_returns = rng.normal(size=1024)
    critic_advantage = rng.normal(size=1024)
    statistics = fisher_statistics(scores, mc_returns, critic_advantage)
    expected = np.mean(scores * mc_returns[:, None], axis=0)
    assert np.allclose(statistics["g_reference"], expected)
    assert np.allclose(statistics["delta"], expected - statistics["g_critic"])


def test_training_gae_bootstraps_at_rollout_boundaries():
    rewards = np.zeros((1, 4, 1))
    values = np.asarray([[[1.0], [2.0], [3.0], [4.0]]])
    done = np.asarray([[False, False, False, True]])
    active = np.ones((1, 4), dtype=bool)
    actual = reconstruct_training_gae(
        rewards, values, done, active, gamma=1.0, gae_lambda=1.0, rollout_steps=2
    )
    assert np.allclose(actual[:, :, 0], [[2.0, 1.0, -3.0, -4.0]])


def test_nested_mc_budgets_and_fisher_ridge_are_fixed():
    assert convergence_episode_budgets(np.arange(512)) == (128, 256, 512)
    metrics = fisher_metrics(np.eye(3), np.ones(3), np.zeros(3), 0.125)
    assert metrics["fisher_ridge_absolute"] == 0.125
    assert np.isclose(vector_cosine(np.ones(3), np.ones(3)), 1.0)


def test_episode_return_ignores_post_termination_padding():
    reward = np.asarray([[[1.0], [2.0], [100.0]]])
    active = np.asarray([[True, True, False]])
    assert np.array_equal(heldout_episode_returns(reward, active), [3.0])
