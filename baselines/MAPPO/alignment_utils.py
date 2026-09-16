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


def directional_subspace_containment(
    source,
    target,
    mask,
    ridge_ratio=1e-3,
    epsilon=1e-6,
):
    """Ridge-whitened directional containment of ``source`` in ``target``.

    Unlike LN-MSE and linear CKA, this distance does not apply per-sample
    LayerNorm.  It asks what fraction of the effective source subspace is
    represented by the target subspace after whitening both feature
    covariances.  ``source`` is the representation receiving the directional
    update and ``target`` is normally a stop-gradient rollout target.

    Returns ``(loss, similarity, source_effective_rank, valid_samples)`` so
    training can audit both the optimized quantity and covariance rank.
    """

    source = source.reshape((-1, source.shape[-1]))
    target = target.reshape((-1, target.shape[-1]))
    weights = mask.reshape((-1,)).astype(source.dtype)
    count = weights.sum()
    safe_count = jnp.maximum(count, 1.0)
    column_weights = weights[:, None]

    source_mean = (source * column_weights).sum(axis=0, keepdims=True) / safe_count
    target_mean = (target * column_weights).sum(axis=0, keepdims=True) / safe_count
    normalization = jnp.sqrt(jnp.maximum(count - 1.0, 1.0))
    source_centered = (source - source_mean) * jnp.sqrt(column_weights) / normalization
    target_centered = (target - target_mean) * jnp.sqrt(column_weights) / normalization

    source_covariance = source_centered.T @ source_centered
    target_covariance = target_centered.T @ target_centered
    cross_covariance = source_centered.T @ target_centered
    source_dim = source_covariance.shape[0]
    target_dim = target_covariance.shape[0]
    ridge_ratio = jnp.asarray(ridge_ratio, source.dtype)
    epsilon = jnp.asarray(epsilon, source.dtype)
    source_ridge = (
        ridge_ratio * jnp.trace(source_covariance) / source_dim + epsilon
    )
    target_ridge = (
        ridge_ratio * jnp.trace(target_covariance) / target_dim + epsilon
    )
    regularized_source = source_covariance + source_ridge * jnp.eye(
        source_dim, dtype=source.dtype
    )
    regularized_target = target_covariance + target_ridge * jnp.eye(
        target_dim, dtype=target.dtype
    )

    # Positive ridge terms make both systems positive definite.  Solves avoid
    # materializing either inverse or inverse square root.
    source_whitened_cross = jnp.linalg.solve(
        regularized_source, cross_covariance
    )
    target_whitened_cross = jnp.linalg.solve(
        regularized_target, cross_covariance.T
    )
    overlap = jnp.trace(source_whitened_cross @ target_whitened_cross)
    source_effective_rank = jnp.trace(
        jnp.linalg.solve(regularized_source, source_covariance)
    )
    similarity = overlap / (source_effective_rank + epsilon)
    loss = 1.0 - similarity
    valid = count >= 2
    zero = jnp.zeros((), dtype=source.dtype)
    return (
        jnp.where(valid, loss, zero),
        jnp.where(valid, similarity, zero),
        jnp.where(valid, source_effective_rank, zero),
        count,
    )


def representation_distance(
    source,
    target,
    mask,
    distance_name="ln_mse",
    *,
    agent_axis=1,
    epsilon=1e-8,
    containment_ridge_ratio=1e-3,
    containment_epsilon=1e-6,
):
    """Compute alignment independently inside each agent's sample pool.

    ``source`` and ``target`` must retain an explicit agent axis.  This is
    essential for NPS: latent coordinates emitted by different actor networks
    are never pooled into one CKA estimate.
    """

    if source.shape[:-1] != target.shape[:-1]:
        raise ValueError("source and target sample axes must have identical shapes")
    if mask.shape != source.shape[:-1]:
        raise ValueError("mask must match every non-feature latent axis")
    if distance_name not in {"ln_mse", "linear_cka", "containment"}:
        raise ValueError(f"Unknown alignment distance: {distance_name!r}")

    source = jnp.moveaxis(source, agent_axis, 0)
    target = jnp.moveaxis(target, agent_axis, 0)
    mask = jnp.moveaxis(mask, agent_axis, 0)
    if distance_name == "ln_mse":
        distance_fn = layernorm_mse_distance
    elif distance_name == "linear_cka":
        distance_fn = lambda left, right, weights: linear_cka_distance(
            left, right, weights, epsilon
        )
    else:
        distance_fn = lambda left, right, weights: directional_subspace_containment(
            left,
            right,
            weights,
            containment_ridge_ratio,
            containment_epsilon,
        )[0]
    return jax.vmap(distance_fn)(source, target, mask).mean()


def representation_containment_statistics(
    source,
    target,
    mask,
    *,
    agent_axis=1,
    ridge_ratio=1e-3,
    epsilon=1e-6,
):
    """Return mean DSC similarity/rank/count without pooling agent spaces."""

    if source.shape[:-1] != target.shape[:-1]:
        raise ValueError("source and target sample axes must have identical shapes")
    if mask.shape != source.shape[:-1]:
        raise ValueError("mask must match every non-feature latent axis")
    source = jnp.moveaxis(source, agent_axis, 0)
    target = jnp.moveaxis(target, agent_axis, 0)
    mask = jnp.moveaxis(mask, agent_axis, 0)
    loss, similarity, effective_rank, count = jax.vmap(
        directional_subspace_containment,
        in_axes=(0, 0, 0, None, None),
    )(source, target, mask, ridge_ratio, epsilon)
    valid = (count >= 2).astype(source.dtype)
    denominator = jnp.maximum(valid.sum(), 1.0)
    return {
        "loss": (loss * valid).sum() / denominator,
        "similarity": (similarity * valid).sum() / denominator,
        "source_effective_rank": (effective_rank * valid).sum() / denominator,
        "valid_samples": (count * valid).sum() / denominator,
        "valid_groups": valid.sum(),
    }
