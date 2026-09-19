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
    scores = jnp.asarray([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]])
    reference = jnp.asarray([[[1.0, 2.0, 3.0]]])
    critic = jnp.asarray([[[0.5, 1.0, 1.5]]])
    mask = jnp.ones((1, 1, 3), dtype=bool)

    def loss(s, r, a):
        return oracle_latent_distortion(s, r, a, mask, 1e-3)["epsilon_lat"]

    score_grad, reference_grad, critic_grad = jax.grad(loss, argnums=(0, 1, 2))(
        scores, reference, critic
    )
    assert np.isfinite(np.asarray(score_grad)).all()
    assert np.linalg.norm(np.asarray(score_grad)) > 0
    np.testing.assert_array_equal(np.asarray(reference_grad), np.zeros_like(reference))
    np.testing.assert_array_equal(np.asarray(critic_grad), np.zeros_like(critic))


def test_distortion_is_zero_when_reference_matches_critic():
    scores = jnp.arange(24, dtype=jnp.float32).reshape((2, 2, 3, 2)) / 10
    signal = jnp.arange(12, dtype=jnp.float32).reshape((2, 2, 3))
    mask = jnp.ones((2, 2, 3), dtype=bool)
    result = oracle_latent_distortion(scores, signal, signal, mask, 1e-3)
    np.testing.assert_allclose(np.asarray(result["epsilon_lat"]), 0.0, atol=1e-7)
