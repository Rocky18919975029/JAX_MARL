"""MADPO actor implemented as an isolated extension of vanilla HARL."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from harl.algorithms.actors.happo import HAPPO
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm

from experiments.harl_dexhands.ccsd import conditional_cs_divergence


def _frozen_policy(policy: nn.Module) -> nn.Module:
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy


class MADPO(HAPPO):
    """HAPPO update augmented by MADPO mutual-policy divergence."""

    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super().__init__(args, obs_space, act_space, device)
        self.div_coef = float(args["div_coef"])
        self.div_weight = float(args["div_weight"])
        self.div_sigma = float(args["div_sigma"])
        self.div_max_samples = int(args["div_max_samples"])
        self.div_epsilon = float(args.get("div_epsilon", 1e-8))
        if self.div_coef < 0:
            raise ValueError("div_coef must be non-negative")
        if not 0.0 <= self.div_weight <= 1.0:
            raise ValueError("div_weight must lie in [0, 1]")
        if self.div_sigma <= 0 or self.div_max_samples < 2:
            raise ValueError("Invalid MADPO kernel configuration")
        self._divergence_generator = torch.Generator(device="cpu")
        self._divergence_generator.manual_seed(0)

    def set_divergence_seed(self, seed: int) -> None:
        self._divergence_generator.manual_seed(int(seed))

    def _tensor(self, value) -> torch.Tensor:
        return check(value).to(**self.tpdv)

    @staticmethod
    def flatten_peer_buffer(buffer) -> dict[str, np.ndarray | None]:
        return {
            "obs": buffer.obs[:-1].reshape(-1, *buffer.obs.shape[2:]),
            "rnn_states": buffer.rnn_states[:-1].reshape(
                -1, *buffer.rnn_states.shape[2:]
            ),
            "actions": buffer.actions.reshape(-1, buffer.actions.shape[-1]),
            "masks": buffer.masks[:-1].reshape(-1, 1),
            "active_masks": buffer.active_masks[:-1].reshape(-1, 1),
            "available_actions": (
                None
                if buffer.available_actions is None
                else buffer.available_actions[:-1].reshape(
                    -1, buffer.available_actions.shape[-1]
                )
            ),
        }

    def _reference_statistics(
        self,
        policy: nn.Module,
        obs,
        rnn_states,
        actions,
        masks,
        available_actions,
        active_masks,
    ) -> torch.Tensor:
        with torch.no_grad():
            log_probs, _, _ = policy.evaluate_actions(
                obs,
                rnn_states,
                actions,
                masks,
                available_actions,
                active_masks,
            )
        return log_probs.detach()

    def update_with_divergence(
        self,
        sample,
        old_policy: nn.Module,
        peer_policy: nn.Module | None,
        peer_data: dict[str, Any] | None,
    ):
        (
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
            factor_batch,
        ) = sample

        obs = self._tensor(obs_batch)
        rnn_states = self._tensor(rnn_states_batch)
        actions = self._tensor(actions_batch)
        masks = self._tensor(masks_batch)
        active_masks = self._tensor(active_masks_batch)
        old_action_log_probs = self._tensor(old_action_log_probs_batch)
        advantages = self._tensor(adv_targ)
        factor = self._tensor(factor_batch)
        available = (
            None
            if available_actions_batch is None
            else self._tensor(available_actions_batch)
        )

        action_log_probs, entropy, _ = self.evaluate_actions(
            obs,
            rnn_states,
            actions,
            masks,
            available,
            active_masks,
        )
        importance = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs),
            dim=-1,
            keepdim=True,
        )
        surrogate_one = importance * advantages
        surrogate_two = (
            torch.clamp(importance, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * advantages
        )
        surrogate = -torch.sum(
            factor * torch.minimum(surrogate_one, surrogate_two),
            dim=-1,
            keepdim=True,
        )
        if self.use_policy_active_masks:
            policy_loss = (
                surrogate * active_masks
            ).sum() / active_masks.sum().clamp_min(1.0)
        else:
            policy_loss = surrogate.mean()

        valid = active_masks.reshape(-1) > 0
        current_obs = obs[valid]
        current_statistics = action_log_probs[valid]
        old_statistics = self._reference_statistics(
            old_policy,
            obs,
            rnn_states,
            actions,
            masks,
            available,
            active_masks,
        )[valid]
        iteration_divergence, iteration_stats = conditional_cs_divergence(
            current_obs,
            current_obs,
            current_statistics,
            old_statistics,
            sigma=self.div_sigma,
            max_samples=self.div_max_samples,
            epsilon=self.div_epsilon,
            generator=self._divergence_generator,
            paired_samples=True,
        )

        peer_divergence = current_statistics.sum() * 0.0
        peer_samples = current_obs.new_tensor(0.0)
        if peer_policy is not None and peer_data is not None:
            peer_obs = self._tensor(peer_data["obs"])
            peer_rnn = self._tensor(peer_data["rnn_states"])
            peer_actions = self._tensor(peer_data["actions"])
            peer_masks = self._tensor(peer_data["masks"])
            peer_active = self._tensor(peer_data["active_masks"])
            peer_available = (
                None
                if peer_data["available_actions"] is None
                else self._tensor(peer_data["available_actions"])
            )
            peer_valid = peer_active.reshape(-1) > 0
            peer_statistics = self._reference_statistics(
                peer_policy,
                peer_obs,
                peer_rnn,
                peer_actions,
                peer_masks,
                peer_available,
                peer_active,
            )[peer_valid]
            peer_divergence, peer_stats = conditional_cs_divergence(
                current_obs,
                peer_obs[peer_valid],
                current_statistics,
                peer_statistics,
                sigma=self.div_sigma,
                max_samples=self.div_max_samples,
                epsilon=self.div_epsilon,
                generator=self._divergence_generator,
            )
            peer_samples = peer_stats["reference_samples"]

        divergence = (
            1.0 - self.div_weight
        ) * iteration_divergence + self.div_weight * peer_divergence
        objective = (
            policy_loss - self.entropy_coef * entropy - self.div_coef * divergence
        )
        finite_values = {
            "policy loss": policy_loss,
            "entropy": entropy,
            "iteration divergence": iteration_divergence,
            "peer divergence": peer_divergence,
            "combined objective": objective,
        }
        for label, value in finite_values.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Non-finite MADPO {label}: {value.detach()}")
        self.actor_optimizer.zero_grad()
        objective.backward()
        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())
        self.actor_optimizer.step()
        return {
            "policy_loss": policy_loss.detach(),
            "dist_entropy": entropy.detach(),
            "actor_grad_norm": (
                actor_grad_norm.detach()
                if isinstance(actor_grad_norm, torch.Tensor)
                else torch.as_tensor(actor_grad_norm)
            ),
            "ratio": importance.detach().mean(),
            "iteration_divergence": iteration_divergence.detach(),
            "peer_divergence": peer_divergence.detach(),
            "combined_divergence": divergence.detach(),
            "divergence_objective": (self.div_coef * divergence).detach(),
            "divergence_to_policy_loss_abs_ratio": (
                (self.div_coef * divergence).abs()
                / policy_loss.detach().abs().clamp_min(self.div_epsilon)
            ).detach(),
            "divergence_current_samples": iteration_stats["current_samples"].detach(),
            "divergence_peer_samples": peer_samples.detach(),
        }

    def train_with_divergence(
        self,
        actor_buffer,
        advantages,
        state_type: str,
        old_policy: nn.Module,
        peer_policy: nn.Module | None,
        peer_buffer,
    ) -> dict[str, float | torch.Tensor]:
        if state_type != "EP":
            raise ValueError("The DexHands MADPO protocol requires EP state")
        keys = (
            "policy_loss",
            "dist_entropy",
            "actor_grad_norm",
            "ratio",
            "iteration_divergence",
            "peer_divergence",
            "combined_divergence",
            "divergence_objective",
            "divergence_to_policy_loss_abs_ratio",
            "divergence_current_samples",
            "divergence_peer_samples",
        )
        totals = {key: 0.0 for key in keys}
        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return totals

        advantages_copy = advantages.copy()
        advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
        advantages = (advantages - np.nanmean(advantages_copy)) / (
            np.nanstd(advantages_copy) + 1e-5
        )
        peer_data = (
            None if peer_buffer is None else self.flatten_peer_buffer(peer_buffer)
        )
        updates = 0
        old_policy = _frozen_policy(old_policy)
        if peer_policy is not None:
            peer_policy.eval()

        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy or self.use_naive_recurrent_policy:
                raise ValueError(
                    "The first DexHands MADPO protocol is feed-forward only"
                )
            generator = actor_buffer.feed_forward_generator_actor(
                advantages, self.actor_num_mini_batch
            )
            for sample in generator:
                metrics = self.update_with_divergence(
                    sample, old_policy, peer_policy, peer_data
                )
                for key in keys:
                    value = metrics[key]
                    totals[key] += (
                        float(value.detach().cpu())
                        if isinstance(value, torch.Tensor)
                        else float(value)
                    )
                updates += 1
        if updates:
            totals = {key: value / updates for key, value in totals.items()}
        return totals
