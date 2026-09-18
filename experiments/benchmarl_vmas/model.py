"""BenchMARL MLP that exposes the pre-head actor and critic latents."""

from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Sequence

import torch
from torch import nn

from benchmarl.models.common import Model, ModelConfig


def _mlp(input_dim: int, hidden_sizes: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_sizes:
        layers.extend((nn.Linear(previous, width), nn.Tanh()))
        previous = width
    return nn.Sequential(*layers)


class AlignmentMlp(Model):
    """The official 256×256 MLP split into encoder and output head."""

    def __init__(self, hidden_sizes: Sequence[int], **kwargs):
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer")
        super().__init__(
            input_spec=kwargs.pop("input_spec"),
            output_spec=kwargs.pop("output_spec"),
            agent_group=kwargs.pop("agent_group"),
            input_has_agent_dim=kwargs.pop("input_has_agent_dim"),
            n_agents=kwargs.pop("n_agents"),
            centralised=kwargs.pop("centralised"),
            share_params=kwargs.pop("share_params"),
            device=kwargs.pop("device"),
            action_spec=kwargs.pop("action_spec"),
            model_index=kwargs.pop("model_index"),
            is_critic=kwargs.pop("is_critic"),
        )
        if kwargs:
            raise TypeError(f"unexpected AlignmentMlp arguments: {sorted(kwargs)}")
        self.hidden_sizes = tuple(hidden_sizes)
        self.latent_dim = self.hidden_sizes[-1]
        per_agent_input = sum(
            int(spec.shape[-1]) for spec in self.input_spec.values(True, True)
        )
        self.output_features = int(self.output_leaf_spec.shape[-1])

        if self.input_has_agent_dim and not self.centralised:
            if self.share_params:
                raise ValueError("This protocol requires non-parameter-sharing actors")
            self.encoders = nn.ModuleList(
                [_mlp(per_agent_input, self.hidden_sizes) for _ in range(self.n_agents)]
            )
            self.heads = nn.ModuleList(
                [
                    nn.Linear(self.latent_dim, self.output_features)
                    for _ in range(self.n_agents)
                ]
            )
            self.central_encoder = None
            self.central_head = None
        else:
            central_input = per_agent_input * (
                self.n_agents if self.input_has_agent_dim else 1
            )
            self.central_encoder = _mlp(central_input, self.hidden_sizes)
            self.central_head = nn.Linear(self.latent_dim, self.output_features)
            self.encoders = None
            self.heads = None
        self.to(self.device)

    @property
    def latent_key(self):
        role = "critic" if self.is_critic else "actor"
        return (self.agent_group, f"_alignment_{role}_latent")

    def _input(self, tensordict):
        return torch.cat(
            [tensordict.get(key).flatten(start_dim=-1) for key in self.in_keys], dim=-1
        )

    def _forward(self, tensordict):
        value = self._input(tensordict)
        if self.encoders is not None:
            latent = torch.stack(
                [
                    encoder(value[..., slot, :])
                    for slot, encoder in enumerate(self.encoders)
                ],
                dim=-2,
            )
            output = torch.stack(
                [head(latent[..., slot, :]) for slot, head in enumerate(self.heads)],
                dim=-2,
            )
        else:
            if self.input_has_agent_dim:
                value = value.flatten(start_dim=-2)
            central_latent = self.central_encoder(value)
            output = self.central_head(central_latent)
            latent = central_latent.unsqueeze(-2).expand(
                *central_latent.shape[:-1], self.n_agents, self.latent_dim
            )
        tensordict.set(self.out_key, output)
        # Collection/evaluation run under no_grad. Avoid storing a 256-D latent
        # for every simulator frame; PPO recomputes it on the sampled minibatch.
        if torch.is_grad_enabled():
            tensordict.set(self.latent_key, latent)
        return tensordict


@dataclass
class AlignmentMlpConfig(ModelConfig):
    hidden_sizes: Sequence[int] = MISSING

    @staticmethod
    def associated_class():
        return AlignmentMlp
