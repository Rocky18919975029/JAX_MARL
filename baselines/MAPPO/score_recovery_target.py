"""Fit-only per-agent Fisher whitening for frozen rollout policy scores."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def whiten_rollout_scores(scores, valid_mask, ridge):
    """Whiten scores once per rollout, not once per PPO minibatch.

    ``scores`` has shape [agent, time, environment, latent] and ``valid_mask``
    has the corresponding first three axes. Invalid scores are zeroed before
    estimating the empirical Fisher and remain zero in the returned target.
    The target and all statistics are stop-gradient.
    """

    if scores.ndim != 4 or valid_mask.shape != scores.shape[:3]:
        raise ValueError("Expected scores[N,T,E,D] and valid_mask[N,T,E]")
    if ridge <= 0:
        raise ValueError("Fisher ridge must be positive")
    mask = valid_mask.astype(scores.dtype)
    masked_scores = scores * mask[..., None]
    valid_count = mask.sum(axis=(1, 2))
    fisher = jnp.einsum("nted,ntef->ndf", masked_scores, masked_scores)
    fisher = fisher / jnp.maximum(valid_count[:, None, None], 1.0)
    fisher = (fisher + jnp.swapaxes(fisher, -2, -1)) / 2
    eigenvalues, eigenvectors = jnp.linalg.eigh(fisher)
    eigenvalues = jnp.maximum(eigenvalues, 0.0)
    inverse_scale = jax.lax.rsqrt(eigenvalues + ridge)
    inverse_root = jnp.matmul(
        eigenvectors * inverse_scale[:, None, :],
        jnp.swapaxes(eigenvectors, -2, -1),
    )
    target = jnp.einsum("nted,ndf->ntef", masked_scores, inverse_root)
    target = jax.lax.stop_gradient(target)
    energy = jnp.sum(jnp.square(target), axis=-1)
    audit = {
        "valid_count_per_agent": jax.lax.stop_gradient(valid_count),
        "fisher_min_eigenvalue": jax.lax.stop_gradient(eigenvalues[:, 0]),
        "fisher_max_eigenvalue": jax.lax.stop_gradient(eigenvalues[:, -1]),
        "target_energy_per_agent": jax.lax.stop_gradient(
            (energy * mask).sum(axis=(1, 2)) / jnp.maximum(valid_count, 1.0)
        ),
    }
    return target, audit
