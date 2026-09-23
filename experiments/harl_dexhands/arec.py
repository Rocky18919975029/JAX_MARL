"""Actor-side policy-score recovery for feed-forward ShadowHandOver policies.

The critic and recovery head are frozen references during each actor update.
Only the actor receives gradients from the recovery objective.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from harl.algorithms.actors.happo import HAPPO
from harl.algorithms.actors.mappo import MAPPO
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm

from experiments.harl_dexhands.madpo import MADPO


def fisher_inverse_sqrt(scores: torch.Tensor, ridge: float) -> torch.Tensor:
    """Compute a detached, per-agent inverse square root from active scores."""
    if scores.ndim != 2 or scores.shape[0] == 0 or ridge <= 0:
        raise ValueError(
            "Expected nonempty [samples, latent] scores and positive ridge"
        )
    fisher = scores.T @ scores / scores.shape[0]
    fisher = 0.5 * (fisher + fisher.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(fisher)
    inverse_root = (
        eigenvectors * torch.rsqrt(eigenvalues.clamp_min(0) + ridge)
    ) @ eigenvectors.T
    return inverse_root.detach()


class RecoveryBufferView:
    """Attach fixed q-teacher targets to HARL's unchanged feed-forward batches."""

    def __init__(self, buffer, teacher: np.ndarray):
        self.buffer = buffer
        self.teacher = teacher
        if teacher.shape[:2] != buffer.actions.shape[:2]:
            raise ValueError("Recovery teacher and rollout dimensions differ")

    def __getattr__(self, name):
        return getattr(self.buffer, name)

    def feed_forward_generator_actor(
        self, advantages, actor_num_mini_batch=None, mini_batch_size=None
    ):
        buffer = self.buffer
        length, threads = buffer.actions.shape[:2]
        batch_size = length * threads
        if mini_batch_size is None:
            if not actor_num_mini_batch or batch_size < actor_num_mini_batch:
                raise ValueError("Invalid actor minibatch count")
            mini_batch_size = batch_size // actor_num_mini_batch
        if actor_num_mini_batch is None:
            actor_num_mini_batch = batch_size // mini_batch_size
        indices = torch.randperm(batch_size).numpy()
        obs = buffer.obs[:-1].reshape(batch_size, *buffer.obs.shape[2:])
        rnn = buffer.rnn_states[:-1].reshape(batch_size, *buffer.rnn_states.shape[2:])
        actions = buffer.actions.reshape(batch_size, -1)
        masks = buffer.masks[:-1].reshape(batch_size, 1)
        active = buffer.active_masks[:-1].reshape(batch_size, 1)
        old_log_probs = buffer.action_log_probs.reshape(batch_size, -1)
        advantage = None if advantages is None else advantages.reshape(batch_size, 1)
        available = (
            None
            if buffer.available_actions is None
            else buffer.available_actions[:-1].reshape(batch_size, -1)
        )
        factor = (
            None if buffer.factor is None else buffer.factor.reshape(batch_size, -1)
        )
        teacher = self.teacher.reshape(batch_size, -1)
        for start in range(0, mini_batch_size * actor_num_mini_batch, mini_batch_size):
            take = indices[start : start + mini_batch_size]
            sample = (
                obs[take],
                rnn[take],
                actions[take],
                masks[take],
                active[take],
                old_log_probs[take],
                None if advantage is None else advantage[take],
                None if available is None else available[take],
            )
            if factor is not None:
                sample += (factor[take],)
            yield sample + (teacher[take],)


class ActorScoreRecoveryMixin:
    """Shared ARec machinery for the existing HAPPO, MAPPO, and MADPO actors."""

    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super().__init__(args, obs_space, act_space, device)
        if act_space.__class__.__name__ != "Box" or len(act_space.shape) != 1:
            raise ValueError("ShadowHandOver ARec requires vector Box actions")
        if self.use_recurrent_policy or self.use_naive_recurrent_policy:
            raise ValueError(
                "ShadowHandOver ARec currently requires feed-forward policies"
            )
        self.arec_coef = float(args["arec_coef"])
        self.arec_q_steps = int(args["arec_q_steps"])
        self.arec_q_lr = float(args["arec_q_lr"])
        self.arec_fisher_ridge = float(args["arec_fisher_ridge"])
        if (
            self.arec_coef <= 0
            or self.arec_q_steps < 1
            or self.arec_q_lr <= 0
            or self.arec_fisher_ridge <= 0
        ):
            raise ValueError(
                "ARec coefficient, q steps/LR, and Fisher ridge must be positive"
            )
        latent_dim = self.actor.hidden_sizes[-1]
        action_dim = int(act_space.shape[0])
        # Initializing q must not advance the global RNG: otherwise agent 1 and
        # the critic start from different weights than their same-seed baseline.
        with torch.random.fork_rng(devices=[]):
            self.arec_q = nn.Sequential(
                nn.Linear(latent_dim + action_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
            ).to(device)
            nn.init.zeros_(self.arec_q[-1].weight)
            nn.init.zeros_(self.arec_q[-1].bias)
        self.arec_q_optimizer = torch.optim.Adam(
            self.arec_q.parameters(), lr=self.arec_q_lr
        )
        self.arec_teacher: np.ndarray | None = None
        self.arec_whitener: torch.Tensor | None = None
        self.arec_audit: dict[str, float] = {}
        self._arec_loss_sum = 0.0
        self._arec_update_count = 0

    def _tensor(self, value) -> torch.Tensor:
        return check(value).to(**self.tpdv)

    def _joint_log_prob(
        self, latent: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        # HARL's Box policy is a diagonal Gaussian; its log_probs are per action
        # dimension, while the score needs the joint action log probability.
        distribution = self.actor.act.action_out(latent)
        return distribution.log_probs(actions).sum(dim=-1)

    def prepare_recovery(self, buffer, critic_latent: torch.Tensor) -> dict[str, float]:
        batch_size = buffer.actions.shape[0] * buffer.actions.shape[1]
        obs = self._tensor(buffer.obs[:-1].reshape(batch_size, -1))
        actions = self._tensor(buffer.actions.reshape(batch_size, -1))
        active = (
            self._tensor(buffer.active_masks[:-1].reshape(batch_size, 1)).squeeze(-1)
            > 0
        )
        if not bool(active.any()):
            raise ValueError("ARec rollout has no active samples")
        if critic_latent.shape[0] != batch_size:
            raise ValueError("Critic features and actor rollout have different lengths")
        with torch.enable_grad():
            latent = self.actor.base(obs)
            score = torch.autograd.grad(
                self._joint_log_prob(latent, actions).sum(), latent
            )[0].detach()
        self.arec_whitener = fisher_inverse_sqrt(score[active], self.arec_fisher_ridge)
        target = (score @ self.arec_whitener).detach()
        q_input = torch.cat((critic_latent.detach(), actions.detach()), dim=-1)
        # q is fitted to a frozen score target; neither actor nor critic is in
        # this optimizer. The teacher is then frozen for all PPO epochs.
        self.arec_q.train()
        with torch.no_grad():
            before = (
                (self.arec_q(q_input[active]) - target[active]).square().sum(-1).mean()
            )
            zero = target[active].square().sum(-1).mean()
        for _ in range(self.arec_q_steps):
            prediction = self.arec_q(q_input[active])
            loss = (prediction - target[active]).square().sum(-1).mean()
            self.arec_q_optimizer.zero_grad()
            loss.backward()
            self.arec_q_optimizer.step()
        self.arec_q.eval()
        with torch.no_grad():
            after = (
                (self.arec_q(q_input[active]) - target[active]).square().sum(-1).mean()
            )
            teacher = self.arec_q(q_input).detach()
        self.arec_teacher = (
            teacher.cpu().numpy().reshape(*buffer.actions.shape[:2], teacher.shape[-1])
        )
        self.arec_audit = {
            "arec_q_loss_pre": float(before),
            "arec_q_loss_post": float(after),
            "arec_q_to_zero_ratio": float(after / zero.clamp_min(1e-12)),
            "arec_target_energy": float(zero),
            "arec_valid_samples": float(active.sum()),
        }
        self._arec_loss_sum = 0.0
        self._arec_update_count = 0
        return self.arec_audit

    def _recovery_loss(self, obs, actions, active_masks, teacher) -> torch.Tensor:
        if self.arec_whitener is None:
            raise RuntimeError("Call prepare_recovery before updating ARec actors")
        obs = self._tensor(obs)
        actions = self._tensor(actions)
        active_masks = self._tensor(active_masks)
        teacher = self._tensor(teacher).detach()
        latent = self.actor.base(obs)
        score = torch.autograd.grad(
            self._joint_log_prob(latent, actions).sum(), latent, create_graph=True
        )[0]
        normalized = score @ self.arec_whitener
        squared = (normalized - teacher).square().sum(-1, keepdim=True)
        loss = (squared * active_masks).sum() / active_masks.sum().clamp_min(1)
        self._arec_loss_sum += float(loss.detach())
        self._arec_update_count += 1
        return loss

    def _recovery_metrics(self) -> dict[str, float]:
        return {
            **self.arec_audit,
            "arec_loss": self._arec_loss_sum / max(self._arec_update_count, 1),
            "arec_weighted_loss": self.arec_coef
            * self._arec_loss_sum
            / max(self._arec_update_count, 1),
        }

    def clear_recovery(self) -> None:
        self.arec_teacher = None
        self.arec_whitener = None

    def update(self, sample):
        teacher = sample[-1]
        base = sample[:-1]
        if isinstance(self, HAPPO):
            if len(base) != 9:
                raise ValueError("HAPPO ARec minibatch must include sequential factor")
            factor_batch = base[-1]
        else:
            if len(base) != 8:
                raise ValueError("MAPPO ARec minibatch must have eight HARL fields")
            factor_batch = None
        (
            obs,
            rnn,
            actions,
            masks,
            active_masks,
            old_log_probs,
            advantages,
            available,
        ) = base[:8]
        old_log_probs = self._tensor(old_log_probs)
        advantages = self._tensor(advantages)
        active_masks = self._tensor(active_masks)
        log_probs, entropy, _ = self.evaluate_actions(
            obs, rnn, actions, masks, available, active_masks
        )
        importance = getattr(torch, self.action_aggregation)(
            torch.exp(log_probs - old_log_probs), dim=-1, keepdim=True
        )
        surrogate_one = importance * advantages
        surrogate_two = (
            torch.clamp(importance, 1 - self.clip_param, 1 + self.clip_param)
            * advantages
        )
        surrogate = torch.minimum(surrogate_one, surrogate_two)
        if factor_batch is not None:
            surrogate = surrogate * self._tensor(factor_batch)
        policy_sample_loss = -surrogate.sum(dim=-1, keepdim=True)
        if self.use_policy_active_masks:
            policy_loss = (
                policy_sample_loss * active_masks
            ).sum() / active_masks.sum().clamp_min(1)
        else:
            policy_loss = policy_sample_loss.mean()
        recovery_loss = self._recovery_loss(obs, actions, active_masks, teacher)
        objective = (
            policy_loss - self.entropy_coef * entropy + self.arec_coef * recovery_loss
        )
        if not torch.isfinite(objective):
            raise FloatingPointError("Non-finite ARec actor objective")
        self.actor_optimizer.zero_grad()
        objective.backward()
        if self.use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm
            )
        else:
            grad_norm = get_grad_norm(self.actor.parameters())
        self.actor_optimizer.step()
        return policy_loss, entropy, grad_norm, importance

    def train(self, actor_buffer, advantages, state_type):
        if self.arec_teacher is None:
            raise RuntimeError("ARec teacher was not prepared")
        result = super().train(
            RecoveryBufferView(actor_buffer, self.arec_teacher), advantages, state_type
        )
        result.update(self._recovery_metrics())
        return result


class ARecHAPPO(ActorScoreRecoveryMixin, HAPPO):
    pass


class ARecMAPPO(ActorScoreRecoveryMixin, MAPPO):
    pass


class ARecMADPO(ActorScoreRecoveryMixin, MADPO):
    def train_with_divergence(
        self, actor_buffer, advantages, state_type, old_policy, peer_policy, peer_buffer
    ):
        if self.arec_teacher is None:
            raise RuntimeError("ARec teacher was not prepared")
        result = super().train_with_divergence(
            RecoveryBufferView(actor_buffer, self.arec_teacher),
            advantages,
            state_type,
            old_policy,
            peer_policy,
            peer_buffer,
        )
        result.update(self._recovery_metrics())
        return result
