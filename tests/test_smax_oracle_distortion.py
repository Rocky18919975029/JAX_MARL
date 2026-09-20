import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("distrax")
pytest.importorskip("flax")
pytest.importorskip("optax")
pytest.importorskip("wandb")

from baselines.MAPPO.mappo_rnn_smax import (
    categorical_latent_score,
    complete_mc_return_to_go,
    oracle_latent_distortion,
)


def test_complete_mc_return_masks_unfinished_rollout_suffix():
    reward = jnp.asarray([[1.0], [2.0], [3.0], [4.0], [5.0]])
    done = jnp.asarray([[False], [True], [False], [False], [False]])
    returns, valid = complete_mc_return_to_go(reward, done, 0.5)
    np.testing.assert_allclose(np.asarray(returns[:2, 0]), [2.0, 2.0])
    np.testing.assert_array_equal(np.asarray(valid[:, 0]), [True, True, False, False, False])


def test_latent_score_matches_autodiff():
    latent = jnp.asarray([0.3, -0.7, 1.1])
    hidden = {
        "kernel": jnp.asarray([[0.2, -0.4], [0.7, 0.1], [-0.3, 0.8]]),
        "bias": jnp.asarray([0.4, 0.5]),
    }
    logits = {
        "kernel": jnp.asarray([[0.2, -0.1, 0.4], [0.5, 0.3, -0.2]]),
        "bias": jnp.asarray([0.1, -0.2, 0.3]),
    }
    available = jnp.ones(3)
    action = jnp.asarray(2)

    def log_probability(z):
        h = jax.nn.relu(z @ hidden["kernel"] + hidden["bias"])
        return jax.nn.log_softmax(h @ logits["kernel"] + logits["bias"])[action]

    expected = jax.grad(log_probability)(latent)
    actual = categorical_latent_score(latent, action, available, hidden, logits)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-6, atol=1e-6)


def test_reference_and_gae_are_stop_gradient_but_scores_are_trainable():
    reference_scores = jnp.asarray(
        [[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -0.2]]]]
    )
    critic_scores = jnp.asarray([[[[1.0, 0.0], [0.0, 1.0]]]])
    reference = jnp.asarray([[[1.0, 2.0, 3.0, -1.0]]])
    returns = reference + 0.25
    critic = jnp.asarray([[[0.5, 1.0]]])
    reference_mask = jnp.ones((1, 1, 4), dtype=bool)
    critic_mask = jnp.ones((1, 1, 2), dtype=bool)
    reference_weight = jnp.ones_like(reference)
    critic_weight = jnp.ones_like(critic)

    def loss(rs, cs, r, ret, a):
        return oracle_latent_distortion(
            rs,
            r,
            ret,
            reference_mask,
            reference_weight,
            cs,
            a,
            critic_mask,
            critic_weight,
            1e-3,
        )["epsilon_lat"]

    (
        reference_score_grad,
        critic_score_grad,
        reference_grad,
        return_grad,
        critic_grad,
    ) = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(
        reference_scores, critic_scores, reference, returns, critic
    )
    assert np.linalg.norm(np.asarray(reference_score_grad)) > 0
    assert np.linalg.norm(np.asarray(critic_score_grad)) > 0
    np.testing.assert_array_equal(np.asarray(reference_grad), np.zeros_like(reference))
    np.testing.assert_array_equal(np.asarray(return_grad), np.zeros_like(returns))
    np.testing.assert_array_equal(np.asarray(critic_grad), np.zeros_like(critic))


def test_distortion_is_zero_when_reference_matches_critic():
    scores = jnp.arange(24, dtype=jnp.float32).reshape((2, 2, 3, 2)) / 10
    signal = jnp.arange(12, dtype=jnp.float32).reshape((2, 2, 3))
    mask = jnp.ones((2, 2, 3), dtype=bool)
    weight = jnp.ones_like(signal)
    result = oracle_latent_distortion(
        scores,
        signal,
        signal,
        mask,
        weight,
        scores,
        signal,
        mask,
        weight,
        1e-3,
    )
    np.testing.assert_allclose(np.asarray(result["epsilon_lat"]), 0.0, atol=1e-7)


def test_reference_and_critic_can_use_different_sample_counts():
    reference_scores = jnp.ones((2, 7, 5, 3))
    critic_scores = jnp.ones((2, 2, 5, 3))
    reference_signal = jnp.ones((2, 7, 5))
    critic_signal = jnp.ones((2, 2, 5))
    result = oracle_latent_distortion(
        reference_scores,
        reference_signal,
        reference_signal,
        jnp.ones_like(reference_signal, dtype=bool),
        jnp.ones_like(reference_signal),
        critic_scores,
        critic_signal,
        jnp.ones_like(critic_signal, dtype=bool),
        jnp.ones_like(critic_signal),
        1e-3,
    )
    assert float(result["reference_to_critic_sample_ratio"]) == pytest.approx(3.5)
