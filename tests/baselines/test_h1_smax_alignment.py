import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("distrax")
pytest.importorskip("flax")
pytest.importorskip("optax")
pytest.importorskip("wandb")

from baselines.MAPPO.mappo_rnn_smax import (
    latent_distance,
    maybe_shuffle_targets_within_agent,
)


def test_h1_shuffle_is_a_deterministic_alive_derangement():
    steps, agents, envs, width = 5, 3, 7, 8
    target = jax.random.normal(jax.random.PRNGKey(0), (steps, agents * envs, width))
    alive = jnp.ones((steps, agents * envs), dtype=jnp.bool_)
    alive = alive.at[0, 0].set(False).at[-1, -1].set(False)
    first, first_mask, first_audit = maybe_shuffle_targets_within_agent(
        target, alive, True, 101, 9, agents, envs
    )
    second, second_mask, second_audit = maybe_shuffle_targets_within_agent(
        target, alive, True, 101, 9, agents, envs
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_mask, second_mask)
    assert int(first_audit["shuffle_permutation_checksum"]) == int(
        second_audit["shuffle_permutation_checksum"]
    )
    assert float(first_audit["shuffle_fixed_point_proportion"]) == 0.0

    original = np.asarray(target).reshape(steps, agents, envs, width)
    shuffled = np.asarray(first).reshape(steps, agents, envs, width)
    mask = np.asarray(alive).reshape(steps, agents, envs)
    for agent in range(agents):
        original_pool = original[:, agent][mask[:, agent]]
        shuffled_pool = shuffled[:, agent][mask[:, agent]]
        np.testing.assert_allclose(original_pool.mean(0), shuffled_pool.mean(0))
        np.testing.assert_allclose(original_pool.std(0), shuffled_pool.std(0))


def test_h1_shuffle_disabled_is_bitwise_identity():
    target = jax.random.normal(jax.random.PRNGKey(1), (4, 6, 3))
    alive = jnp.asarray([[True, True, False, True, True, True]] * 4, dtype=jnp.bool_)
    output, output_mask, audit = maybe_shuffle_targets_within_agent(
        target, alive, False, 101, 0, 2, 3
    )
    np.testing.assert_array_equal(output, target)
    np.testing.assert_array_equal(output_mask, alive)
    assert int(audit["shuffle_enabled"]) == 0


@pytest.mark.parametrize(
    ("mode", "actor_nonzero", "critic_nonzero"),
    [
        ("none", False, False),
        ("c_to_a", True, False),
        ("a_to_c", False, True),
        ("reciprocal", True, True),
        ("joint", True, True),
    ],
)
def test_h1_alignment_gradient_routing(mode, actor_nonzero, critic_nonzero):
    actor = jax.random.normal(jax.random.PRNGKey(2), (17, 8))
    critic = jax.random.normal(jax.random.PRNGKey(3), (17, 8))
    actor_target = jax.lax.stop_gradient(actor)
    critic_target = jax.lax.stop_gradient(critic)
    mask = jnp.ones((17,), dtype=jnp.bool_)

    def loss(actor_value, critic_value):
        if mode == "none":
            return jnp.zeros(())
        if mode == "c_to_a":
            return latent_distance(actor_value, critic_target, mask)
        if mode == "a_to_c":
            return latent_distance(critic_value, actor_target, mask)
        if mode == "reciprocal":
            return latent_distance(actor_value, critic_target, mask) + latent_distance(
                critic_value, actor_target, mask
            )
        return latent_distance(actor_value, critic_value, mask)

    actor_grad, critic_grad = jax.grad(loss, argnums=(0, 1))(actor, critic)
    assert (float(jnp.linalg.norm(actor_grad)) > 1e-9) is actor_nonzero
    assert (float(jnp.linalg.norm(critic_grad)) > 1e-9) is critic_nonzero
