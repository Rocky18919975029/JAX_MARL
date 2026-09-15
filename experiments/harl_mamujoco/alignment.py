"""Representation distances shared by HARL training and calibration."""

from __future__ import annotations

import torch


ALIGN_MODES = ("none", "c_to_a", "a_to_c", "joint")
ALIGN_DISTANCES = ("ln_mse", "linear_cka")


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


def representation_distance(
    source_by_agent: torch.Tensor,
    target_by_agent: torch.Tensor,
    mask_by_agent: torch.Tensor,
    distance_name: str,
    epsilon: float = 1e-8,
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
        elif int((mask > 0).sum().detach().cpu()) > 1:
            distances.append(linear_cka_distance(source, target, mask, epsilon))
    if not distances:
        return source_by_agent.sum() * 0.0 + target_by_agent.sum() * 0.0
    return torch.stack(distances).mean()


def select_alignment_loss(
    align_mode: str,
    distance_name: str,
    actor_current: torch.Tensor,
    critic_current: torch.Tensor,
    actor_old: torch.Tensor,
    critic_old: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build directional alignment with the SMAX experiment's stop-grad rules."""

    if align_mode not in ALIGN_MODES:
        raise ValueError(f"Unknown alignment mode: {align_mode!r}")
    zero = actor_current.sum() * 0.0 + critic_current.sum() * 0.0
    if align_mode == "none":
        return zero, {"c_to_a": zero, "a_to_c": zero, "joint": zero}

    c_to_a = representation_distance(
        actor_current, critic_old.detach(), mask, distance_name, epsilon
    )
    a_to_c = representation_distance(
        critic_current, actor_old.detach(), mask, distance_name, epsilon
    )
    joint = representation_distance(
        actor_current, critic_current, mask, distance_name, epsilon
    )
    selected = {"c_to_a": c_to_a, "a_to_c": a_to_c, "joint": joint}[align_mode]
    return selected, {"c_to_a": c_to_a, "a_to_c": a_to_c, "joint": joint}
