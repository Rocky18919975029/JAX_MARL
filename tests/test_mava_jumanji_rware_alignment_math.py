"""JAX checks for RWARE actor-only LN-MSE and per-agent linear CKA."""

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


SOURCE = Path(__file__).resolve().parents[1] / "experiments/mava_jumanji/rec_mappo_alignment.py"
spec = importlib.util.spec_from_file_location("mava_rware_alignment", SOURCE)
alignment = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(alignment)


def test_exposed_gru_latents_preserve_original_mava_parameter_layout_and_output():
    torso = lambda: MLPTorso(layer_sizes=(128,))
    actor_args = dict(pre_torso=torso(), post_torso=torso(),
                      action_head=DiscreteActionHead(action_dim=5), hidden_state_dim=128)
    critic_args = dict(pre_torso=torso(), post_torso=torso(),
                       centralised_critic=True, hidden_state_dim=128)
    observation = ObservationGlobalState(
        agents_view=jnp.ones((1, 2, 4, 6), dtype=jnp.float32),
        action_mask=jnp.ones((1, 2, 4, 5), dtype=jnp.bool_),
        global_state=jnp.ones((1, 2, 4, 24), dtype=jnp.float32),
        step_count=None,
    )
    inputs = (observation, jnp.zeros((1, 2, 4), dtype=jnp.bool_))
    carry = ScannedRNN.initialize_carry((2, 4), 128)
    for baseline, augmented in (
        (RecurrentActor(**actor_args), alignment.Actor(**actor_args)),
        (RecurrentValueNet(**critic_args), alignment.Critic(**critic_args)),
    ):
        key = jax.random.PRNGKey(7)
        original_params = baseline.init(key, carry, inputs)
        aligned_params = augmented.init(key, carry, inputs)
        assert jax.tree.structure(original_params) == jax.tree.structure(aligned_params)
        for left, right in zip(jax.tree.leaves(original_params),
                               jax.tree.leaves(aligned_params), strict=True):
            np.testing.assert_array_equal(left, right)
        _, original_output = baseline.apply(original_params, carry, inputs)
        _, aligned_output, latent = augmented.apply(
            aligned_params, carry, inputs, return_latent=True
        )
        assert latent.shape == (1, 2, 4, 128)
        if isinstance(baseline, RecurrentActor):
            action = jnp.zeros((1, 2, 4), dtype=jnp.int32)
            np.testing.assert_array_equal(
                original_output.log_prob(action), aligned_output.log_prob(action)
            )
        else:
            np.testing.assert_array_equal(original_output, aligned_output)


def test_ln_mse_matches_featurewise_normalized_square_error():
    actor = jax.random.normal(jax.random.PRNGKey(1), (8, 2, 3, 16))
    critic = jax.random.normal(jax.random.PRNGKey(2), (8, 2, 3, 16))
    expected = jnp.square(
        alignment.normalize_latent_samples(actor)
        - alignment.normalize_latent_samples(critic)
    ).mean()
    np.testing.assert_allclose(alignment.alignment_distance(actor, critic, "ln_mse"), expected)


def test_cka_is_agent_specific_and_invariant_to_feature_rotation():
    actor = jax.random.normal(jax.random.PRNGKey(3), (16, 2, 3, 8))
    # A feature permutation preserves samplewise LayerNorm and linear CKA,
    # but generally changes coordinatewise LN-MSE.
    critic = jnp.flip(actor, axis=-1)
    cka = alignment.alignment_distance(actor, critic, "linear_cka")
    mse = alignment.alignment_distance(actor, critic, "ln_mse")
    assert float(cka) == pytest.approx(0, abs=1e-5)
    assert float(mse) > 0.1
    assert jnp.isfinite(jax.jit(alignment.alignment_distance, static_argnums=2)(
        actor, critic, "linear_cka"
    ))


def test_alignment_gradient_reaches_actor_but_not_frozen_critic():
    actor = jax.random.normal(jax.random.PRNGKey(4), (8, 2, 3, 16))
    critic = jax.random.normal(jax.random.PRNGKey(5), (8, 2, 3, 16))
    for mode in ("ln_mse", "linear_cka"):
        grad_actor, grad_critic = jax.grad(
            lambda a, c: alignment.alignment_distance(a, jax.lax.stop_gradient(c), mode),
            argnums=(0, 1),
        )(actor, critic)
        assert np.linalg.norm(np.asarray(grad_actor)) > 0
        np.testing.assert_array_equal(grad_critic, jnp.zeros_like(critic))
