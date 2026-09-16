"""Representation-alignment utilities shared by continuous MAPPO experiments."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax


def vmapped_optimizer(tx: optax.GradientTransformation):
    """Apply one optimizer independently to every leading parameter slice."""

    def init_fn(params):
        return jax.vmap(tx.init)(params)

    def update_fn(updates, state, params=None):
        if params is None:
            return jax.vmap(lambda grad, item: tx.update(grad, item))(updates, state)
        return jax.vmap(tx.update)(updates, state, params)

    return optax.GradientTransformation(init_fn, update_fn)


def tree_l2_norm(tree):
    """Euclidean norm of a parameter pytree."""

    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree.leaves(tree)))


def normalize_latent_samples(latent, epsilon=1e-5):
    """Parameter-free LayerNorm over each sample's feature axis."""

    mean = latent.mean(axis=-1, keepdims=True)
    variance = jnp.square(latent - mean).mean(axis=-1, keepdims=True)
    return (latent - mean) * jax.lax.rsqrt(variance + epsilon)


def layernorm_mse_distance(source, target, mask):
    """Masked coordinate-wise MSE after per-sample LayerNorm."""

    distance = jnp.square(
        normalize_latent_samples(source) - normalize_latent_samples(target)
    ).mean(axis=-1)
    weights = mask.astype(distance.dtype)
    return (distance * weights).sum() / jnp.maximum(weights.sum(), 1.0)


def linear_cka_distance(source, target, mask, epsilon=1e-8):
    """Masked linear-CKA distance after per-sample LayerNorm."""

    source = normalize_latent_samples(source).reshape((-1, source.shape[-1]))
    target = normalize_latent_samples(target).reshape((-1, target.shape[-1]))
    weights = mask.reshape((-1,)).astype(source.dtype)
    count = weights.sum()
    safe_count = jnp.maximum(count, 1.0)
    weights = weights[:, None]

    source_mean = (source * weights).sum(axis=0, keepdims=True) / safe_count
    target_mean = (target * weights).sum(axis=0, keepdims=True) / safe_count
    source_centered = (source - source_mean) * weights
    target_centered = (target - target_mean) * weights

    cross = source_centered.T @ target_centered
    source_self = source_centered.T @ source_centered
    target_self = target_centered.T @ target_centered
    numerator = jnp.square(cross).sum()
    denominator = jnp.sqrt(jnp.square(source_self).sum()) * jnp.sqrt(
        jnp.square(target_self).sum()
    )
    similarity = numerator / (denominator + jnp.asarray(epsilon, source.dtype))
    distance = 1.0 - similarity
    return jnp.where(count > 1, distance, jnp.zeros_like(distance))


def representation_distance(
    source,
    target,
    mask,
    distance_name="ln_mse",
    *,
    agent_axis=1,
    epsilon=1e-8,
):
    """Compute alignment independently inside each agent's sample pool.

    ``source`` and ``target`` must retain an explicit agent axis.  This is
    essential for NPS: latent coordinates emitted by different actor networks
    are never pooled into one CKA estimate.
    """

    if source.shape != target.shape:
        raise ValueError("source and target latents must have identical shapes")
    if mask.shape != source.shape[:-1]:
        raise ValueError("mask must match every non-feature latent axis")
    if distance_name not in {"ln_mse", "linear_cka"}:
        raise ValueError(f"Unknown alignment distance: {distance_name!r}")

    source = jnp.moveaxis(source, agent_axis, 0)
    target = jnp.moveaxis(target, agent_axis, 0)
    mask = jnp.moveaxis(mask, agent_axis, 0)
    distance_fn = (
        layernorm_mse_distance
        if distance_name == "ln_mse"
        else lambda left, right, weights: linear_cka_distance(
            left, right, weights, epsilon
        )
    )
    return jax.vmap(distance_fn)(source, target, mask).mean()
