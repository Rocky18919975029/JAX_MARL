"""Numerically stable, memory-bounded conditional CS divergence.

The equations follow the official MADPO implementation, while sampling is
performed before constructing pairwise kernels so DexHands rollouts do not
materialize O((num_envs * horizon)^2) matrices.
"""

from __future__ import annotations

import torch


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """Return an RBF Gram matrix without copying squared norms."""

    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("RBF inputs must be [samples, features] with equal features")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    squared = (
        x.square().sum(dim=1, keepdim=True)
        + y.square().sum(dim=1, keepdim=True).transpose(0, 1)
        - 2.0 * (x @ y.transpose(0, 1))
    ).clamp_min(0.0)
    return torch.exp(squared * (-0.5 / (sigma * sigma)))


def _sample_rows(
    first: torch.Tensor,
    second: torch.Tensor,
    maximum: int,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if first.shape[0] != second.shape[0]:
        raise ValueError("Paired tensors must contain the same number of samples")
    count = first.shape[0]
    if count <= maximum:
        return first, second
    indices = torch.randperm(count, generator=generator, device="cpu")[:maximum]
    indices = indices.to(device=first.device)
    return first.index_select(0, indices), second.index_select(0, indices)


def conditional_cs_divergence(
    observations: torch.Tensor,
    reference_observations: torch.Tensor,
    policy_statistics: torch.Tensor,
    reference_policy_statistics: torch.Tensor,
    *,
    sigma: float = 1.0,
    max_samples: int = 1024,
    epsilon: float = 1e-8,
    generator: torch.Generator | None = None,
    paired_samples: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Estimate MADPO's conditional Cauchy--Schwarz divergence.

    Sampling from the two empirical distributions is independent, matching the
    distributional estimator. Within each distribution, observation and policy
    statistic rows remain paired. Reference statistics are expected to be
    stop-gradient inputs; this function does not detach the current policy.
    """

    if max_samples < 2:
        raise ValueError("max_samples must be at least two")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if observations.ndim != 2 or reference_observations.ndim != 2:
        raise ValueError("Observations must be rank-two tensors")
    if policy_statistics.ndim != 2 or reference_policy_statistics.ndim != 2:
        raise ValueError("Policy statistics must be rank-two tensors")
    if observations.shape[1] != reference_observations.shape[1]:
        raise ValueError("Compared agents must have equal observation dimensions")
    if policy_statistics.shape[1] != reference_policy_statistics.shape[1]:
        raise ValueError("Compared agents must have equal policy-statistic dimensions")

    if paired_samples:
        if observations.shape[0] != reference_observations.shape[0]:
            raise ValueError("Paired distribution samples must have equal lengths")
        count = observations.shape[0]
        if count > max_samples:
            indices = torch.randperm(count, generator=generator, device="cpu")[
                :max_samples
            ].to(device=observations.device)
            observations = observations.index_select(0, indices)
            policy_statistics = policy_statistics.index_select(0, indices)
            reference_observations = reference_observations.index_select(0, indices)
            reference_policy_statistics = reference_policy_statistics.index_select(
                0, indices
            )
    else:
        observations, policy_statistics = _sample_rows(
            observations, policy_statistics, max_samples, generator
        )
        reference_observations, reference_policy_statistics = _sample_rows(
            reference_observations,
            reference_policy_statistics,
            max_samples,
            generator,
        )
    if observations.shape[0] < 2 or reference_observations.shape[0] < 2:
        zero = policy_statistics.sum() * 0.0
        return zero, {
            "current_samples": observations.new_tensor(observations.shape[0]),
            "reference_samples": observations.new_tensor(
                reference_observations.shape[0]
            ),
        }

    k_current = gaussian_kernel(observations, observations, sigma)
    k_reference = gaussian_kernel(reference_observations, reference_observations, sigma)
    l_current = gaussian_kernel(policy_statistics, policy_statistics, sigma)
    l_reference = gaussian_kernel(
        reference_policy_statistics, reference_policy_statistics, sigma
    )
    k_cross = gaussian_kernel(observations, reference_observations, sigma)
    l_cross = gaussian_kernel(policy_statistics, reference_policy_statistics, sigma)

    current_mass = k_current.sum(dim=1).clamp_min(epsilon)
    reference_mass = k_reference.sum(dim=1).clamp_min(epsilon)
    cross_current_mass = k_cross.sum(dim=1).clamp_min(epsilon)
    cross_reference_mass = k_cross.sum(dim=0).clamp_min(epsilon)

    self_current = (
        ((k_current * l_current).sum(dim=1) / current_mass.square())
        .sum()
        .clamp_min(epsilon)
    )
    self_reference = (
        ((k_reference * l_reference).sum(dim=1) / reference_mass.square())
        .sum()
        .clamp_min(epsilon)
    )
    cross_forward = (
        ((k_cross * l_cross).sum(dim=1) / (current_mass * cross_current_mass))
        .sum()
        .clamp_min(epsilon)
    )
    cross_reverse = (
        ((k_cross * l_cross).sum(dim=0) / (reference_mass * cross_reference_mass))
        .sum()
        .clamp_min(epsilon)
    )

    log_two = observations.new_tensor(2.0).log()
    shared = self_current.log() + self_reference.log()
    forward = (-2.0 * cross_forward.log() + shared) / log_two
    reverse = (-2.0 * cross_reverse.log() + shared) / log_two
    divergence = 0.5 * (forward + reverse)
    return divergence, {
        "current_samples": observations.new_tensor(observations.shape[0]),
        "reference_samples": observations.new_tensor(reference_observations.shape[0]),
        "forward": forward.detach(),
        "reverse": reverse.detach(),
    }
