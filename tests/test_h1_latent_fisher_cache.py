import numpy as np

from scripts.h1_latent_distortion import (
    aggregate_agent_metrics,
    convergence_episode_budgets,
    fisher_metrics,
    fisher_metrics_from_statistics,
    fisher_statistics,
    reweight_reference,
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


def test_zero_baseline_mc_reference_is_used_without_return_transformation():
    rng = np.random.default_rng(19)
    scores = rng.normal(size=(1024, 6))
    mc_returns = rng.normal(size=1024)
    critic_advantage = rng.normal(size=1024)
    statistics = fisher_statistics(scores, mc_returns, critic_advantage)

    expected = np.mean(scores * mc_returns[:, None], axis=0)
    assert np.allclose(statistics["g_reference"], expected)
    assert np.allclose(statistics["delta"], expected - statistics["g_critic"])


def test_cross_fitted_signal_reuses_identical_fisher_sample_pool():
    rng = np.random.default_rng(23)
    scores = rng.normal(size=(400, 5))
    mc_returns = rng.normal(size=400)
    critic_advantage = rng.normal(size=400)
    control_variate_signal = mc_returns - rng.normal(size=400)
    primary = fisher_statistics(scores, mc_returns, critic_advantage)
    sensitivity = reweight_reference(primary, scores, control_variate_signal)

    assert sensitivity["fisher"] is primary["fisher"]
    assert np.allclose(
        sensitivity["g_reference"],
        np.mean(scores * control_variate_signal[:, None], axis=0),
    )


def test_nested_mc_episode_budgets_and_aggregate_ratio():
    assert convergence_episode_budgets(np.arange(512)) == (128, 256, 512)
    metrics = [
        {"epsilon_lat": 2.0, "energy_ref": 4.0},
        {"epsilon_lat": 4.0, "energy_ref": 8.0},
    ]
    aggregate = aggregate_agent_metrics(metrics)
    assert aggregate["epsilon_lat"] == 3.0
    assert aggregate["energy_ref"] == 6.0
    assert np.isclose(aggregate["r_lat"], 3.0 / (6.0 + 1e-8))
    assert np.isclose(vector_cosine(np.ones(3), np.ones(3)), 1.0)
