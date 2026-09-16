import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("distrax")
pytest.importorskip("flax")
pytest.importorskip("optax")
pytest.importorskip("wandb")

from baselines.MAPPO.mappo_rnn_smax import (
    directional_subspace_containment,
    latent_distance,
    linear_cka_distance,
    maybe_shuffle_targets_within_agent,
    representation_distance,
)


def test_linear_cka_ignores_an_orthogonal_sign_flip_but_ln_mse_does_not():
    source = jax.random.normal(jax.random.PRNGKey(20), (3, 7, 8))
    target = -source
    mask = jnp.ones(source.shape[:-1], dtype=jnp.bool_)

    assert abs(float(linear_cka_distance(source, target, mask))) < 1e-5
    assert float(latent_distance(source, target, mask)) > 3.9


def test_linear_cka_mask_excludes_invalid_samples():
    source = jax.random.normal(jax.random.PRNGKey(21), (2, 8, 6))
    target = jax.random.normal(jax.random.PRNGKey(22), (2, 8, 6))
    mask = jnp.ones((2, 8), dtype=jnp.bool_).at[:, -2:].set(False)
    changed_source = source.at[:, -2:].set(1e6)
    changed_target = target.at[:, -2:].set(-1e6)

    np.testing.assert_allclose(
        linear_cka_distance(source, target, mask),
        linear_cka_distance(changed_source, changed_target, mask),
        atol=1e-6,
    )


def test_grouped_linear_cka_is_the_mean_of_per_agent_distances():
    source = jax.random.normal(jax.random.PRNGKey(23), (4, 6, 5))
    target = jax.random.normal(jax.random.PRNGKey(24), (4, 6, 5))
    mask = jnp.ones((4, 6), dtype=jnp.bool_)
    grouped = representation_distance(
        source,
        target,
        mask,
        distance_name="linear_cka",
        num_agent_groups=3,
    )
    expected = jnp.mean(
        jnp.asarray(
            [
                linear_cka_distance(
                    source[:, 2 * agent : 2 * (agent + 1)],
                    target[:, 2 * agent : 2 * (agent + 1)],
                    mask[:, 2 * agent : 2 * (agent + 1)],
                )
                for agent in range(3)
            ]
        )
    )
    np.testing.assert_allclose(grouped, expected, atol=1e-6)


def test_grouped_containment_is_directional_and_type_conditioned():
    samples, agents, envs = 4, 2, 32
    key = jax.random.PRNGKey(31)
    source = jax.random.normal(key, (samples, agents * envs, 4))
    target = source.at[..., 3].set(0.0)
    mask = jnp.ones((samples, agents * envs), dtype=jnp.bool_)
    unit_type = jnp.tile(
        jnp.concatenate(
            (jnp.zeros((envs // 2,)), jnp.ones((envs // 2,))), axis=0
        ),
        (samples, agents),
    ).astype(jnp.int32)
    distance = representation_distance(
        source,
        target,
        mask,
        distance_name="containment",
        num_agent_groups=agents,
        group_labels=unit_type,
        num_label_groups=2,
        min_group_samples=32,
    )
    reverse = representation_distance(
        target,
        source,
        mask,
        distance_name="containment",
        num_agent_groups=agents,
        group_labels=unit_type,
        num_label_groups=2,
        min_group_samples=32,
    )
    assert float(distance) > 0.15
    assert float(reverse) < 5e-3


def test_containment_requires_enough_samples_per_slot_type():
    source = jax.random.normal(jax.random.PRNGKey(32), (2, 6, 4))
    target = jax.random.normal(jax.random.PRNGKey(33), (2, 6, 4))
    mask = jnp.ones((2, 6), dtype=jnp.bool_)
    labels = jnp.zeros((2, 6), dtype=jnp.int32)
    distance = representation_distance(
        source,
        target,
        mask,
        distance_name="containment",
        num_agent_groups=3,
        group_labels=labels,
        num_label_groups=1,
        min_group_samples=5,
    )
    assert float(distance) == 0.0


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


@pytest.mark.parametrize(
    ("mode", "actor_nonzero", "critic_nonzero"),
    [
        ("c_to_a", True, False),
        ("a_to_c", False, True),
        ("joint", True, True),
    ],
)
def test_linear_cka_gradient_routing(mode, actor_nonzero, critic_nonzero):
    actor = jax.random.normal(jax.random.PRNGKey(25), (3, 12, 8))
    critic = jax.random.normal(jax.random.PRNGKey(26), (3, 12, 8))
    mask = jnp.ones((3, 12), dtype=jnp.bool_)

    def distance(source, target):
        return representation_distance(
            source,
            target,
            mask,
            distance_name="linear_cka",
            num_agent_groups=3,
        )

    def loss(actor_value, critic_value):
        if mode == "c_to_a":
            return distance(actor_value, jax.lax.stop_gradient(critic_value))
        if mode == "a_to_c":
            return distance(critic_value, jax.lax.stop_gradient(actor_value))
        return distance(actor_value, critic_value)

    actor_grad, critic_grad = jax.grad(loss, argnums=(0, 1))(actor, critic)
    assert (float(jnp.linalg.norm(actor_grad)) > 1e-9) is actor_nonzero
    assert (float(jnp.linalg.norm(critic_grad)) > 1e-9) is critic_nonzero


@pytest.mark.parametrize(
    ("mode", "actor_nonzero", "critic_nonzero"),
    [
        ("c_to_a", True, False),
        ("a_to_c", False, True),
        ("joint", True, True),
    ],
)
def test_containment_gradient_routing(mode, actor_nonzero, critic_nonzero):
    actor = jax.random.normal(jax.random.PRNGKey(34), (3, 32, 8))
    critic = jax.random.normal(jax.random.PRNGKey(35), (3, 32, 8))
    mask = jnp.ones((3, 32), dtype=jnp.bool_)

    def distance(source, target):
        return representation_distance(
            source,
            target,
            mask,
            distance_name="containment",
            num_agent_groups=3,
        )

    def loss(actor_value, critic_value):
        if mode == "c_to_a":
            return distance(actor_value, jax.lax.stop_gradient(critic_value))
        if mode == "a_to_c":
            return distance(critic_value, jax.lax.stop_gradient(actor_value))
        return distance(actor_value, critic_value)

    actor_grad, critic_grad = jax.grad(loss, argnums=(0, 1))(actor, critic)
    assert (float(jnp.linalg.norm(actor_grad)) > 1e-9) is actor_nonzero
    assert (float(jnp.linalg.norm(critic_grad)) > 1e-9) is critic_nonzero
