"""Representation distances shared by HARL training and calibration."""

from __future__ import annotations

import torch


ALIGN_MODES = ("none", "c_to_a", "a_to_c", "joint")
ALIGN_DISTANCES = ("ln_mse", "linear_cka", "containment")


def normalize_latent_samples(latent: torch.Tensor) -> torch.Tensor:
    """Apply parameter-free LayerNorm independently to every sample."""

    mean = latent.mean(dim=-1, keepdim=True)
    variance = (latent - mean).square().mean(dim=-1, keepdim=True)
    return (latent - mean) * torch.rsqrt(variance + 1e-5)


def ln_mse_distance(
    source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Masked per-sample LayerNorm MSE."""

    distance = (
        (normalize_latent_samples(source) - normalize_latent_samples(target))
        .square()
        .mean(dim=-1)
    )
    weights = mask.to(dtype=distance.dtype)
    return (distance * weights).sum() / weights.sum().clamp_min(1.0)


def linear_cka_distance(
    source: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Masked linear CKA after per-sample LayerNorm and batch centering."""

    source = normalize_latent_samples(source).reshape(-1, source.shape[-1])
    target = normalize_latent_samples(target).reshape(-1, target.shape[-1])
    weights = mask.reshape(-1).to(dtype=source.dtype)
    count = weights.sum()
    safe_count = count.clamp_min(1.0)
    column_weights = weights[:, None]

    source_mean = (source * column_weights).sum(dim=0, keepdim=True) / safe_count
    target_mean = (target * column_weights).sum(dim=0, keepdim=True) / safe_count
    source_centered = (source - source_mean) * column_weights
    target_centered = (target - target_mean) * column_weights

    cross = source_centered.transpose(0, 1) @ target_centered
    source_self = source_centered.transpose(0, 1) @ source_centered
    target_self = target_centered.transpose(0, 1) @ target_centered
    numerator = cross.square().sum()
    denominator = source_self.square().sum().sqrt() * target_self.square().sum().sqrt()
    distance = 1.0 - numerator / (denominator + source.new_tensor(epsilon))
    return torch.where(count > 1, distance, torch.zeros_like(distance))


def directional_subspace_containment(
    source: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    ridge_ratio: float = 1e-3,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Ridge-whitened fraction of the source subspace contained in target."""

    source = source.reshape(-1, source.shape[-1])
    target = target.reshape(-1, target.shape[-1])
    weights = mask.reshape(-1).to(dtype=source.dtype)
    count = weights.sum()
    safe_count = count.clamp_min(1.0)
    column_weights = weights[:, None]
    source_mean = (source * column_weights).sum(dim=0, keepdim=True) / safe_count
    target_mean = (target * column_weights).sum(dim=0, keepdim=True) / safe_count
    normalization = (count - 1.0).clamp_min(1.0).sqrt()
    source_centered = (source - source_mean) * column_weights.sqrt() / normalization
    target_centered = (target - target_mean) * column_weights.sqrt() / normalization

    source_covariance = source_centered.transpose(0, 1) @ source_centered
    target_covariance = target_centered.transpose(0, 1) @ target_centered
    cross_covariance = source_centered.transpose(0, 1) @ target_centered
    source_dim = source_covariance.shape[0]
    target_dim = target_covariance.shape[0]
    source_ridge = (
        ridge_ratio * torch.trace(source_covariance) / source_dim + epsilon
    )
    target_ridge = (
        ridge_ratio * torch.trace(target_covariance) / target_dim + epsilon
    )
    regularized_source = source_covariance + source_ridge * torch.eye(
        source_dim, dtype=source.dtype, device=source.device
    )
    regularized_target = target_covariance + target_ridge * torch.eye(
        target_dim, dtype=target.dtype, device=target.device
    )
    source_whitened_cross = torch.linalg.solve(
        regularized_source, cross_covariance
    )
    target_whitened_cross = torch.linalg.solve(
        regularized_target, cross_covariance.transpose(0, 1)
    )
    overlap = torch.trace(source_whitened_cross @ target_whitened_cross)
    source_effective_rank = torch.trace(
        torch.linalg.solve(regularized_source, source_covariance)
    )
    similarity = overlap / (source_effective_rank + epsilon)
    loss = 1.0 - similarity
    valid = count >= 2
    zero = source.sum() * 0.0 + target.sum() * 0.0
    loss = torch.where(valid, loss, zero)
    statistics = {
        "similarity": torch.where(valid, similarity, zero),
        "source_effective_rank": torch.where(valid, source_effective_rank, zero),
        "valid_samples": count,
    }
    return loss, statistics


def representation_distance(
    source_by_agent: torch.Tensor,
    target_by_agent: torch.Tensor,
    mask_by_agent: torch.Tensor,
    distance_name: str,
    epsilon: float = 1e-8,
    containment_ridge_ratio: float = 1e-3,
    containment_epsilon: float = 1e-6,
) -> torch.Tensor:
    """Average the same distance definition over agent-specific sample pools.

    Inputs have shape ``[agents, samples, features]`` and masks have shape
    ``[agents, samples]``. Keeping CKA pools agent-specific avoids mixing the
    arbitrary coordinate systems of independently parameterized actors.
    """

    if source_by_agent.shape != target_by_agent.shape:
        raise ValueError(
            "Actor and critic latents must have identical [agent, sample, feature] "
            f"shapes, got {source_by_agent.shape} and {target_by_agent.shape}"
        )
    if source_by_agent.ndim != 3 or mask_by_agent.shape != source_by_agent.shape[:2]:
        raise ValueError(
            "Expected latent [agent, sample, feature] and mask [agent, sample]"
        )
    if distance_name not in ALIGN_DISTANCES:
        raise ValueError(f"Unknown alignment distance: {distance_name!r}")

    distances = []
    for agent_id in range(source_by_agent.shape[0]):
        source = source_by_agent[agent_id]
        target = target_by_agent[agent_id]
        mask = mask_by_agent[agent_id]
        if distance_name == "ln_mse":
            distances.append(ln_mse_distance(source, target, mask))
        elif distance_name == "linear_cka" and int(
            (mask > 0).sum().detach().cpu()
        ) > 1:
            distances.append(linear_cka_distance(source, target, mask, epsilon))
        elif distance_name == "containment" and int(
            (mask > 0).sum().detach().cpu()
        ) > 1:
            distances.append(
                directional_subspace_containment(
                    source,
                    target,
                    mask,
                    containment_ridge_ratio,
                    containment_epsilon,
                )[0]
            )
    if not distances:
        return source_by_agent.sum() * 0.0 + target_by_agent.sum() * 0.0
    return torch.stack(distances).mean()


def representation_containment_statistics(
    source_by_agent: torch.Tensor,
    target_by_agent: torch.Tensor,
    mask_by_agent: torch.Tensor,
    ridge_ratio: float = 1e-3,
    epsilon: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Aggregate DSC audit statistics without mixing different actors."""

    statistics = []
    for agent_id in range(source_by_agent.shape[0]):
        mask = mask_by_agent[agent_id]
        if int((mask > 0).sum().detach().cpu()) < 2:
            continue
        loss, item = directional_subspace_containment(
            source_by_agent[agent_id],
            target_by_agent[agent_id],
            mask,
            ridge_ratio,
            epsilon,
        )
        item = {**item, "loss": loss}
        statistics.append(item)
    zero = source_by_agent.sum() * 0.0 + target_by_agent.sum() * 0.0
    if not statistics:
        return {
            "loss": zero,
            "containment_similarity": zero,
            "containment_source_effective_rank": zero,
            "containment_valid_samples_mean": zero,
        }
    return {
        "loss": torch.stack([item["loss"] for item in statistics]).mean(),
        "containment_similarity": torch.stack(
            [item["similarity"] for item in statistics]
        ).mean(),
        "containment_source_effective_rank": torch.stack(
            [item["source_effective_rank"] for item in statistics]
        ).mean(),
        "containment_valid_samples_mean": torch.stack(
            [item["valid_samples"] for item in statistics]
        ).mean(),
    }


def select_alignment_loss(
    align_mode: str,
    distance_name: str,
    actor_current: torch.Tensor,
    critic_current: torch.Tensor,
    actor_old: torch.Tensor,
    critic_old: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float = 1e-8,
    containment_ridge_ratio: float = 1e-3,
    containment_epsilon: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build directional alignment with the SMAX experiment's stop-grad rules."""

    if align_mode not in ALIGN_MODES:
        raise ValueError(f"Unknown alignment mode: {align_mode!r}")
    zero = actor_current.sum() * 0.0 + critic_current.sum() * 0.0
    if align_mode == "none":
        return zero, {
            "c_to_a": zero,
            "a_to_c": zero,
            "joint": zero,
            "containment_similarity": zero,
            "containment_source_effective_rank": zero,
            "containment_valid_samples_mean": zero,
        }

    if distance_name == "containment":
        if align_mode == "a_to_c":
            source, target = critic_current, actor_old.detach()
        elif align_mode == "joint":
            source, target = actor_current, critic_current
        else:
            source, target = actor_current, critic_old.detach()
        statistics = representation_containment_statistics(
            source,
            target,
            mask,
            containment_ridge_ratio,
            containment_epsilon,
        )
        selected = statistics.pop("loss")
        distances = {"c_to_a": zero, "a_to_c": zero, "joint": zero}
        distances[align_mode] = selected
        return selected, {**distances, **statistics}

    kwargs = {
        "epsilon": epsilon,
        "containment_ridge_ratio": containment_ridge_ratio,
        "containment_epsilon": containment_epsilon,
    }
    c_to_a = representation_distance(
        actor_current, critic_old.detach(), mask, distance_name, **kwargs
    )
    a_to_c = representation_distance(
        critic_current, actor_old.detach(), mask, distance_name, **kwargs
    )
    joint = representation_distance(
        actor_current, critic_current, mask, distance_name, **kwargs
    )
    selected = {"c_to_a": c_to_a, "a_to_c": a_to_c, "joint": joint}[align_mode]
    return selected, {
        "c_to_a": c_to_a,
        "a_to_c": a_to_c,
        "joint": joint,
        "containment_similarity": zero,
        "containment_source_effective_rank": zero,
        "containment_valid_samples_mean": zero,
    }
