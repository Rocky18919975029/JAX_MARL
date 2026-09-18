"""MAPPO extension that adds a critic-to-actor representation objective."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Type

import torch
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from benchmarl.algorithms.mappo import Mappo, MappoConfig
from benchmarl.algorithms.common import Algorithm

from experiments.benchmarl_vmas.alignment import nps_distance


class AlignmentClipPPOLoss(ClipPPOLoss):
    # LossModule inspects annotations on the concrete subclass when it registers
    # functional parameters.  Re-declaring the inherited modules/parameter
    # containers avoids misleading TorchRL warnings without changing runtime
    # types or ownership.
    actor_network: object
    actor_network_params: object
    target_actor_network_params: object
    critic_network: object
    critic_network_params: object
    target_critic_network_params: object

    def __init__(
        self,
        *args,
        group: str,
        align_mode: str,
        align_distance: str,
        alignment_coef: float,
        alignment_epsilon: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if align_mode not in {"none", "c_to_a"}:
            raise ValueError(f"unsupported alignment mode: {align_mode}")
        if align_distance not in {"ln_mse", "linear_cka"}:
            raise ValueError(f"unsupported alignment distance: {align_distance}")
        self.group = group
        self.align_mode = align_mode
        self.align_distance = align_distance
        self.alignment_coef = float(alignment_coef)
        self.alignment_epsilon = float(alignment_epsilon)
        self._gradient_audit_requested = False
        self._gradient_audit = {
            "alignment_rl_only_gradient_norm": 0.0,
            "alignment_aux_only_gradient_norm": 0.0,
            "alignment_combined_gradient_norm": 0.0,
            "alignment_aux_to_rl_gradient_ratio": 0.0,
        }
        self.out_keys = list(self.out_keys) + [
            "alignment_loss",
            "alignment_weighted_loss",
            *self._gradient_audit,
        ]

    def request_gradient_audit(self) -> None:
        """Measure separate actor-gradient norms on the next training minibatch."""

        self._gradient_audit_requested = True

    def _actor_gradient_norm(self, loss: torch.Tensor) -> torch.Tensor:
        parameters = list(self.actor_network_params.flatten_keys().values())
        gradients = torch.autograd.grad(
            loss, parameters, retain_graph=True, allow_unused=True
        )
        squared = [
            gradient.detach().square().sum()
            for gradient in gradients
            if gradient is not None
        ]
        return torch.stack(squared).sum().sqrt() if squared else loss.new_zeros(())

    def alignment_loss(self, tensordict, distance: str | None = None):
        td = tensordict.clone(False)
        actor_context = (
            self.actor_network_params.to_module(self.actor_network)
            if self.functional
            else contextlib.nullcontext()
        )
        critic_context = (
            self.critic_network_params.to_module(self.critic_network)
            if self.functional
            else contextlib.nullcontext()
        )
        with actor_context:
            self.actor_network.get_dist(td)
        with critic_context:
            self.critic_network(td)
        return nps_distance(
            td.get((self.group, "_alignment_actor_latent")),
            td.get((self.group, "_alignment_critic_latent")),
            distance or self.align_distance,
            self.alignment_epsilon,
        )

    def forward(self, tensordict):
        output = super().forward(tensordict)
        rl_objective = output["loss_objective"]
        if self.align_mode == "none":
            alignment = rl_objective * 0.0
        else:
            alignment = self.alignment_loss(tensordict)
        weighted_alignment = self.alignment_coef * alignment
        combined_objective = rl_objective + weighted_alignment
        output.set("loss_objective", combined_objective)

        if self._gradient_audit_requested:
            rl_norm = self._actor_gradient_norm(rl_objective)
            aux_norm = self._actor_gradient_norm(weighted_alignment)
            combined_norm = self._actor_gradient_norm(combined_objective)
            ratio = aux_norm / torch.clamp(rl_norm, min=1e-12)
            self._gradient_audit = {
                "alignment_rl_only_gradient_norm": float(rl_norm.cpu()),
                "alignment_aux_only_gradient_norm": float(aux_norm.cpu()),
                "alignment_combined_gradient_norm": float(combined_norm.cpu()),
                "alignment_aux_to_rl_gradient_ratio": float(ratio.cpu()),
            }
            self._gradient_audit_requested = False

        output.set("alignment_loss", alignment.detach())
        output.set("alignment_weighted_loss", weighted_alignment.detach())
        for key, value in self._gradient_audit.items():
            output.set(key, rl_objective.new_tensor(value))
        return output


class AlignmentMappo(Mappo):
    def __init__(
        self,
        align_mode: str,
        align_distance: str,
        alignment_coef: float,
        alignment_epsilon: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.align_mode = align_mode
        self.align_distance = align_distance
        self.alignment_coef = alignment_coef
        self.alignment_epsilon = alignment_epsilon

    def _get_loss(self, group, policy_for_loss, continuous):
        loss = AlignmentClipPPOLoss(
            actor=policy_for_loss,
            critic=self.get_critic(group),
            clip_epsilon=self.clip_epsilon,
            entropy_coeff=self.entropy_coef,
            critic_coeff=self.critic_coef,
            loss_critic_type=self.loss_critic_type,
            normalize_advantage=False,
            group=group,
            align_mode=self.align_mode,
            align_distance=self.align_distance,
            alignment_coef=self.alignment_coef,
            alignment_epsilon=self.alignment_epsilon,
        )
        loss.set_keys(
            reward=(group, "reward"),
            action=(group, "action"),
            done=(group, "done"),
            terminated=(group, "terminated"),
            advantage=(group, "advantage"),
            value_target=(group, "value_target"),
            value=(group, "state_value"),
            sample_log_prob=(group, "log_prob"),
        )
        loss.make_value_estimator(
            ValueEstimators.GAE, gamma=self.experiment_config.gamma, lmbda=self.lmbda
        )
        return loss, False


@dataclass
class AlignmentMappoConfig(MappoConfig):
    align_mode: str = "none"
    align_distance: str = "ln_mse"
    alignment_coef: float = 0.0
    alignment_epsilon: float = 1e-8

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return AlignmentMappo
