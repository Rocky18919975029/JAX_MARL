"""Actor-slot-specific representation distances for VMAS NPS training."""

from __future__ import annotations

import torch


def normalize_samples(latent: torch.Tensor) -> torch.Tensor:
    """Parameter-free LayerNorm over each latent vector."""

    mean = latent.mean(dim=-1, keepdim=True)
    variance = (latent - mean).square().mean(dim=-1, keepdim=True)
    return (latent - mean) * torch.rsqrt(variance + 1e-5)


def ln_mse(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (normalize_samples(source) - normalize_samples(target)).square().mean()


def linear_cka(
    source: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8
) -> torch.Tensor:
    source = normalize_samples(source)
    target = normalize_samples(target)
    source = source - source.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    cross = source.transpose(0, 1) @ target
    source_self = source.transpose(0, 1) @ source
    target_self = target.transpose(0, 1) @ target
    numerator = cross.square().sum()
    denominator = source_self.square().sum().sqrt() * target_self.square().sum().sqrt()
    return 1.0 - numerator / (denominator + source.new_tensor(epsilon))


def nps_distance(
    actor_latent: torch.Tensor,
    critic_latent: torch.Tensor,
    distance: str,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Average a distance over actor slots without mixing NPS coordinates.

    Both inputs end in ``[..., agents, features]``. Every leading dimension is a
    sample dimension. The critic target is detached here, enforcing C→A.
    """

    if actor_latent.shape != critic_latent.shape:
        raise ValueError(
            f"actor/critic latent shapes differ: {actor_latent.shape} vs "
            f"{critic_latent.shape}"
        )
    if actor_latent.ndim < 3:
        raise ValueError("expected [..., agents, features] latents")
    actor = actor_latent.movedim(-2, 0).reshape(
        actor_latent.shape[-2], -1, actor_latent.shape[-1]
    )
    critic = (
        critic_latent.detach()
        .movedim(-2, 0)
        .reshape(critic_latent.shape[-2], -1, critic_latent.shape[-1])
    )
    values = []
    for slot in range(actor.shape[0]):
        if distance == "ln_mse":
            values.append(ln_mse(actor[slot], critic[slot]))
        elif distance == "linear_cka":
            if actor.shape[1] < 2:
                values.append(actor[slot].sum() * 0.0)
            else:
                values.append(linear_cka(actor[slot], critic[slot], epsilon))
        else:
            raise ValueError(f"unsupported distance: {distance}")
    return torch.stack(values).mean()
