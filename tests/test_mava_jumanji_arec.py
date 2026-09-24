"""Unit checks for the Mava LBF/RWARE actor-side score recovery entry point."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("mava")
import jax.numpy as jnp
import numpy as np

from mava.networks import RecurrentActor, RecurrentValueNet, ScannedRNN
from mava.networks.heads import DiscreteActionHead
from mava.networks.torsos import MLPTorso
from mava.types import ObservationGlobalState


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/mava_jumanji/rec_mappo_arec.py"
)
spec = importlib.util.spec_from_file_location("mava_jumanji_arec", MODULE_PATH)
arec = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(arec)


def _torso():
    return MLPTorso(layer_sizes=(128,))


def _actor(cls):
    return cls(
        pre_torso=_torso(),
        post_torso=_torso(),
        action_head=DiscreteActionHead(action_dim=5),
        hidden_state_dim=128,
    )


def _critic(cls):
    return cls(
        pre_torso=_torso(),
        post_torso=_torso(),
        centralised_critic=True,
        hidden_state_dim=128,
    )


def _inputs():
    obs = ObservationGlobalState(
        agents_view=jnp.ones((1, 2, 4, 6), dtype=jnp.float32),
        action_mask=jnp.ones((1, 2, 4, 5), dtype=jnp.bool_),
        global_state=jnp.ones((1, 2, 4, 24), dtype=jnp.float32),
        step_count=None,
    )
    done = jnp.zeros((1, 2, 4), dtype=jnp.bool_)
    carry = ScannedRNN.initialize_carry((2, 4), 128)
    return carry, (obs, done)


def _compare_trees(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_actor_and_critic_preserve_mava_forward_and_parameter_layout():
    carry, inputs = _inputs()
    for baseline, augmented in (
        (_actor(RecurrentActor), _actor(arec.Actor)),
        (_critic(RecurrentValueNet), _critic(arec.Critic)),
    ):
        key = jax.random.PRNGKey(7)
        baseline_params = baseline.init(key, carry, inputs)
        augmented_params = augmented.init(key, carry, inputs)
        _compare_trees(baseline_params, augmented_params)
        baseline_carry, baseline_output = baseline.apply(baseline_params, carry, inputs)
        augmented_carry, augmented_output, latent = augmented.apply(
            augmented_params, carry, inputs, return_latent=True
        )
        _compare_trees(baseline_carry, augmented_carry)
        assert latent.shape == (1, 2, 4, 128)
        if isinstance(baseline, RecurrentActor):
            actions = jnp.zeros((1, 2, 4), dtype=jnp.int32)
            np.testing.assert_allclose(
                baseline_output.log_prob(actions),
                augmented_output.log_prob(actions),
                rtol=0,
                atol=0,
            )
        else:
            np.testing.assert_allclose(baseline_output, augmented_output, rtol=0, atol=0)


def test_score_uses_masked_mava_head_and_has_actor_gradient():
    carry, inputs = _inputs()
    obs, done = inputs
    mask = obs.action_mask.at[..., 4].set(False)
    obs = obs._replace(action_mask=mask)
    actor = _actor(arec.Actor)
    params = actor.init(jax.random.PRNGKey(8), carry, (obs, done))
    _, _, latent = actor.apply(params, carry, (obs, done), return_latent=True)
    action = jnp.zeros((1, 2, 4), dtype=jnp.int32)
    score = arec.score_from_latent(actor, params, latent, action, mask)
    assert score.shape == latent.shape
    assert np.isfinite(np.asarray(score)).all()

    def objective(p):
        _, _, z = actor.apply(p, carry, (obs, done), return_latent=True)
        s = arec.score_from_latent(actor, p, z, action, mask)
        return jnp.square(s).sum()

    gradients = jax.grad(objective)(params)
    assert any(np.any(np.asarray(x) != 0) for x in jax.tree.leaves(gradients))


def test_whitening_and_zero_initial_recovery_head():
    scores = jax.random.normal(jax.random.PRNGKey(9), (8, 2, 4, 128))
    target, inverse_root = arec.whiten_scores(scores, ridge=1e-3)
    assert target.shape == scores.shape
    assert inverse_root.shape == (4, 128, 128)
    assert np.isfinite(np.asarray(target)).all()

    head = arec.RecoveryHead(action_dim=5, latent_dim=128)
    params = head.init(
        jax.random.PRNGKey(10),
        jnp.zeros((2, 128)),
        jnp.zeros((2,), dtype=jnp.int32),
    )
    output = head.apply(params, jnp.ones((2, 128)), jnp.zeros((2,), dtype=jnp.int32))
    np.testing.assert_array_equal(output, jnp.zeros_like(output))


def test_fisher_geometry_and_teacher_targets_are_frozen():
    scores = jax.random.normal(jax.random.PRNGKey(11), (4, 2, 3, 128))

    def detached_target_energy(x):
        target, inverse_root = arec.whiten_scores(x, ridge=1e-3)
        return jnp.square(target).sum() + jnp.square(inverse_root).sum()

    gradient = jax.grad(detached_target_energy)(scores)
    np.testing.assert_array_equal(gradient, jnp.zeros_like(scores))


def test_fisher_uses_all_equal_sized_rollout_shards():
    shards = jax.random.normal(jax.random.PRNGKey(13), (2, 4, 2, 3, 128))
    _, global_inverse_root = arec.whiten_scores(shards.reshape(8, 2, 3, 128), 1e-3)
    _, shard_inverse_roots = jax.vmap(
        lambda shard: arec.whiten_scores(shard, 1e-3, axis_names=("shard",)),
        axis_name="shard",
    )(shards)
    # Float32 eigendecompositions of rank-deficient 128D Fishers differ slightly
    # with reduction order even when the input covariance is the same.
    np.testing.assert_allclose(shard_inverse_roots[0], global_inverse_root, rtol=1e-3, atol=2e-2)
    np.testing.assert_allclose(shard_inverse_roots[1], global_inverse_root, rtol=1e-3, atol=2e-2)
    np.testing.assert_array_equal(shard_inverse_roots[0], shard_inverse_roots[1])


def test_recovery_head_is_independent_for_each_agent():
    head = arec.RecoveryHead(action_dim=5, latent_dim=128)
    keys = jax.random.split(jax.random.PRNGKey(12), 3)
    params = jax.vmap(head.init, in_axes=(0, None, None))(
        keys,
        jnp.zeros((1, 128)),
        jnp.zeros((1,), dtype=jnp.int32),
    )
    assert all(leaf.shape[0] == 3 for leaf in jax.tree.leaves(params))
    assert any(
        not np.array_equal(np.asarray(leaf[0]), np.asarray(leaf[1]))
        for leaf in jax.tree.leaves(params)
        if leaf.size > 3
    )
