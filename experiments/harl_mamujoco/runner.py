"""Matched MAPPO runner with actor--critic representation alignment for HARL."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from harl.runners.on_policy_base_runner import OnPolicyBaseRunner
from harl.utils.envs_tools import check

from experiments.harl_mamujoco.alignment import (
    ALIGN_DISTANCES,
    ALIGN_MODES,
    representation_distance,
    select_alignment_loss,
)


def _float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _global_norm(grads: Iterable[torch.Tensor | None]) -> float:
    total = 0.0
    for grad in grads:
        if grad is not None:
            total += float(grad.detach().square().sum().cpu())
    return math.sqrt(total)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class AlignedMAMuJoCoRunner(OnPolicyBaseRunner):
    """HARL MAPPO runner whose actor and critic are updated in one matched graph."""

    def __init__(
        self,
        args,
        algo_args,
        env_args,
        experiment: dict,
        wandb_run=None,
    ):
        self.experiment = experiment
        self.align_mode = experiment["align_mode"]
        self.align_distance = experiment["align_distance"]
        self.alignment_coef = float(experiment["alignment_coef"])
        self.alignment_epsilon = float(experiment["alignment_epsilon"])
        self.containment_ridge_ratio = float(
            experiment.get("containment_ridge_ratio", 1e-3)
        )
        self.containment_epsilon = float(
            experiment.get("containment_epsilon", 1e-6)
        )
        self.checkpoint_root = (
            Path(experiment["checkpoint_root"]).expanduser().resolve()
        )
        self.checkpoint_interval = int(experiment["checkpoint_interval_steps"])
        self.upload_checkpoints = bool(experiment["wandb_upload_checkpoints"])
        self.wandb_run = wandb_run
        self.last_train_metrics: dict[str, float] = {}
        self._next_checkpoint_step = self.checkpoint_interval

        if self.align_mode not in ALIGN_MODES:
            raise ValueError(f"align_mode must be one of {ALIGN_MODES}")
        if self.align_distance not in ALIGN_DISTANCES:
            raise ValueError(f"align_distance must be one of {ALIGN_DISTANCES}")
        if self.align_mode == "none" and self.align_distance != "ln_mse":
            raise ValueError(
                "The distance-free none baseline must use align_distance=ln_mse"
            )
        if self.alignment_coef < 0 or self.alignment_epsilon <= 0:
            raise ValueError(
                "Alignment coefficient must be non-negative and epsilon positive"
            )
        if self.containment_ridge_ratio < 0 or self.containment_epsilon <= 0:
            raise ValueError(
                "Containment ridge ratio must be non-negative and epsilon positive"
            )

        model = algo_args["model"]
        algo = algo_args["algo"]
        if model["use_recurrent_policy"] or model["use_naive_recurrent_policy"]:
            raise ValueError(
                "The first HARL alignment protocol is locked to feed-forward MAPPO"
            )
        if algo["actor_num_mini_batch"] != 1 or algo["critic_num_mini_batch"] != 1:
            raise ValueError(
                "Matched alignment currently requires one full-batch minibatch"
            )
        if algo["ppo_epoch"] != algo["critic_epoch"]:
            raise ValueError(
                "Actor and critic epoch counts must match for joint updates"
            )
        if algo_args["train"]["model_dir"] is not None:
            raise ValueError(
                "This confirmatory runner does not resume from HARL model_dir"
            )

        super().__init__(args, algo_args, env_args)
        if self.state_type != "EP":
            raise ValueError("Humanoid-v2-17x1 alignment requires EP centralized state")
        if self.num_agents != 17:
            raise ValueError(
                f"Expected Humanoid-v2-17x1 to expose 17 agents, got {self.num_agents}"
            )
        self._initialize_matched_networks()
        actor_dim = self.actor[0].actor.hidden_sizes[-1]
        critic_dim = self.critic.critic.hidden_sizes[-1]
        if actor_dim != critic_dim:
            raise ValueError(
                f"Actor and critic latent dimensions must match, got {actor_dim} and {critic_dim}"
            )

    def _initialize_matched_networks(self) -> None:
        """Make PS/NPS initialization identical without coupling NPS optimizers."""

        from harl.algorithms.actors import ALGO_REGISTRY
        from harl.algorithms.critics.v_critic import VCritic

        seed = int(self.algo_args["seed"]["seed"])
        actor_args = {**self.algo_args["model"], **self.algo_args["algo"]}
        critic_args = {**self.algo_args["model"], **self.algo_args["algo"]}
        cuda_devices = []
        if self.device.type == "cuda":
            cuda_devices = [
                (
                    self.device.index
                    if self.device.index is not None
                    else torch.cuda.current_device()
                )
            ]

        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            template = ALGO_REGISTRY[self.args["algo"]](
                actor_args,
                self.envs.observation_space[0],
                self.envs.action_space[0],
                device=self.device,
            )
            if self.share_param:
                self.actor = [template for _ in range(self.num_agents)]
            else:
                actors = [template]
                for agent_id in range(1, self.num_agents):
                    actor = ALGO_REGISTRY[self.args["algo"]](
                        actor_args,
                        self.envs.observation_space[agent_id],
                        self.envs.action_space[agent_id],
                        device=self.device,
                    )
                    actor.actor.load_state_dict(template.actor.state_dict())
                    actors.append(actor)
                self.actor = actors

            torch.manual_seed(seed + 1)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed + 1)
            self.critic = VCritic(
                critic_args,
                self.envs.share_observation_space[0],
                device=self.device,
            )

        # Rollout sampling begins from the same RNG stream for every distance,
        # direction, and actor parameterization with the same training seed.
        torch.manual_seed(seed + 100_000)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed + 100_000)

    @property
    def unique_actors(self):
        unique = []
        seen = set()
        for actor in self.actor:
            if id(actor) not in seen:
                unique.append(actor)
                seen.add(id(actor))
        return unique

    def _tensor(self, value) -> torch.Tensor:
        return check(value).to(dtype=torch.float32, device=self.device)

    def _normalized_advantages(self) -> list[np.ndarray]:
        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[
                :-1
            ] - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        else:
            advantages = (
                self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
            )

        raw = np.stack([advantages.copy() for _ in range(self.num_agents)], axis=0)
        masks = np.stack(
            [buffer.active_masks[:-1] for buffer in self.actor_buffer], axis=0
        )
        masked = raw.copy()
        masked[masks == 0.0] = np.nan
        mean = np.nanmean(masked)
        std = np.nanstd(masked)
        normalized = (raw - mean) / (std + 1e-5)
        return [normalized[agent_id] for agent_id in range(self.num_agents)]

    def _actor_arrays(self, agent_id: int, advantage: np.ndarray) -> dict:
        buffer = self.actor_buffer[agent_id]
        available = None
        if buffer.available_actions is not None:
            available = buffer.available_actions[:-1].reshape(
                -1, buffer.available_actions.shape[-1]
            )
        return {
            "obs": buffer.obs[:-1].reshape(-1, *buffer.obs.shape[2:]),
            "actions": buffer.actions.reshape(-1, buffer.actions.shape[-1]),
            "old_log_probs": buffer.action_log_probs.reshape(
                -1, buffer.action_log_probs.shape[-1]
            ),
            "active_masks": buffer.active_masks[:-1].reshape(-1, 1),
            "available_actions": available,
            "advantages": advantage.reshape(-1, 1),
        }

    def _critic_arrays(self) -> dict:
        buffer = self.critic_buffer
        return {
            "share_obs": buffer.share_obs[:-1].reshape(-1, *buffer.share_obs.shape[2:]),
            "value_preds": buffer.value_preds[:-1].reshape(-1, 1),
            "returns": buffer.returns[:-1].reshape(-1, 1),
        }

    def _actor_loss(self, agent_id: int, arrays: dict):
        algorithm = self.actor[agent_id]
        model = algorithm.actor
        obs = self._tensor(arrays["obs"])
        actions = self._tensor(arrays["actions"])
        old_log_probs = self._tensor(arrays["old_log_probs"])
        active_masks = self._tensor(arrays["active_masks"])
        advantages = self._tensor(arrays["advantages"])
        available = (
            self._tensor(arrays["available_actions"])
            if arrays["available_actions"] is not None
            else None
        )

        latent = model.base(obs)
        action_log_probs, entropy, _ = model.act.evaluate_actions(
            latent,
            actions,
            available,
            active_masks=active_masks if model.use_policy_active_masks else None,
        )
        ratio = getattr(torch, algorithm.action_aggregation)(
            torch.exp(action_log_probs - old_log_probs), dim=-1, keepdim=True
        )
        surrogate_one = ratio * advantages
        surrogate_two = (
            torch.clamp(ratio, 1.0 - algorithm.clip_param, 1.0 + algorithm.clip_param)
            * advantages
        )
        surrogate = -torch.sum(
            torch.minimum(surrogate_one, surrogate_two), dim=-1, keepdim=True
        )
        if algorithm.use_policy_active_masks:
            policy_loss = (
                surrogate * active_masks
            ).sum() / active_masks.sum().clamp_min(1.0)
        else:
            policy_loss = surrogate.mean()
        objective = policy_loss - algorithm.entropy_coef * entropy
        return objective, policy_loss, entropy, ratio, latent

    def _critic_loss(self, arrays: dict):
        share_obs = self._tensor(arrays["share_obs"])
        value_preds = self._tensor(arrays["value_preds"])
        returns = self._tensor(arrays["returns"])
        model = self.critic.critic
        latent = model.base(share_obs)
        values = model.v_out(latent)
        value_loss = self.critic.cal_value_loss(
            values, value_preds, returns, value_normalizer=self.value_normalizer
        )
        return value_loss * self.critic.value_loss_coef, value_loss, latent

    def _old_latents(self, actor_arrays: list[dict], critic_arrays: dict):
        with torch.no_grad():
            actor_old = torch.stack(
                [
                    self.actor[agent_id].actor.base(
                        self._tensor(actor_arrays[agent_id]["obs"])
                    )
                    for agent_id in range(self.num_agents)
                ],
                dim=0,
            )
            critic_old_single = self.critic.critic.base(
                self._tensor(critic_arrays["share_obs"])
            )
            critic_old = critic_old_single.unsqueeze(0).expand(self.num_agents, -1, -1)
        return actor_old.detach(), critic_old.detach()

    def _forward_objectives(
        self,
        actor_arrays: list[dict],
        critic_arrays: dict,
        actor_old: torch.Tensor,
        critic_old: torch.Tensor,
        distance_name: str | None = None,
    ) -> dict:
        actor_outputs = [
            self._actor_loss(agent_id, actor_arrays[agent_id])
            for agent_id in range(self.num_agents)
        ]
        actor_objectives = torch.stack([output[0] for output in actor_outputs])
        critic_objective, value_loss, critic_single = self._critic_loss(critic_arrays)
        actor_current = torch.stack([output[4] for output in actor_outputs], dim=0)
        critic_current = critic_single.unsqueeze(0).expand(self.num_agents, -1, -1)
        mask = torch.stack(
            [
                self._tensor(arrays["active_masks"]).squeeze(-1)
                for arrays in actor_arrays
            ],
            dim=0,
        )
        alignment, distances = select_alignment_loss(
            self.align_mode,
            distance_name or self.align_distance,
            actor_current,
            critic_current,
            actor_old,
            critic_old,
            mask,
            self.alignment_epsilon,
            self.containment_ridge_ratio,
            self.containment_epsilon,
        )
        return {
            "actor_objective": actor_objectives.mean(),
            "actor_policy_losses": torch.stack([output[1] for output in actor_outputs]),
            "actor_entropies": torch.stack([output[2] for output in actor_outputs]),
            "actor_ratios": torch.stack([output[3].mean() for output in actor_outputs]),
            "critic_objective": critic_objective,
            "value_loss": value_loss,
            "alignment": alignment,
            "distances": distances,
            "actor_current": actor_current,
            "critic_current": critic_current,
            "mask": mask,
        }

    def _parameter_layout(self):
        actor_groups = [list(actor.actor.parameters()) for actor in self.unique_actors]
        actor_params = [parameter for group in actor_groups for parameter in group]
        critic_params = list(self.critic.critic.parameters())
        return actor_groups, actor_params, critic_params

    def _scale_nps_actor_grads(self, grads, actor_count):
        if self.share_param:
            return grads
        return tuple(
            None if grad is None else grad * self.num_agents
            for grad in grads[:actor_count]
        ) + tuple(grads[actor_count:])

    def train(self):
        """Perform matched full-batch actor, critic, and alignment updates."""

        advantages = self._normalized_advantages()
        actor_arrays = [
            self._actor_arrays(agent_id, advantages[agent_id])
            for agent_id in range(self.num_agents)
        ]
        critic_arrays = self._critic_arrays()
        actor_old, critic_old = self._old_latents(actor_arrays, critic_arrays)
        actor_groups, actor_params, critic_params = self._parameter_layout()
        all_params = actor_params + critic_params
        actor_count = len(actor_params)

        accumulators: dict[str, float] = {}
        last_actor_grad_norms = [0.0 for _ in actor_groups]
        last_critic_grad_norm = 0.0
        epochs = self.algo_args["algo"]["ppo_epoch"]

        for _ in range(epochs):
            for actor in self.unique_actors:
                actor.actor_optimizer.zero_grad(set_to_none=True)
            self.critic.critic_optimizer.zero_grad(set_to_none=True)

            outputs = self._forward_objectives(
                actor_arrays, critic_arrays, actor_old, critic_old
            )
            rl_objective = outputs["actor_objective"] + outputs["critic_objective"]
            cross_objective = self.alignment_coef * outputs["alignment"]
            total_objective = rl_objective + cross_objective

            cross_grads = torch.autograd.grad(
                cross_objective,
                all_params,
                retain_graph=True,
                allow_unused=True,
            )
            total_grads = torch.autograd.grad(
                total_objective, all_params, allow_unused=True
            )
            cross_grads = self._scale_nps_actor_grads(cross_grads, actor_count)
            total_grads = self._scale_nps_actor_grads(total_grads, actor_count)
            rl_grads = tuple(
                (
                    None
                    if total is None and cross is None
                    else (torch.zeros_like(cross) if total is None else total)
                    - (torch.zeros_like(total) if cross is None else cross)
                )
                for total, cross in zip(total_grads, cross_grads)
            )

            for parameter, grad in zip(all_params, total_grads):
                parameter.grad = None if grad is None else grad.detach()

            last_actor_grad_norms = []
            for actor, group in zip(self.unique_actors, actor_groups):
                if actor.use_max_grad_norm:
                    norm = nn.utils.clip_grad_norm_(group, actor.max_grad_norm)
                    last_actor_grad_norms.append(_float(norm))
                else:
                    last_actor_grad_norms.append(_global_norm(p.grad for p in group))
                actor.actor_optimizer.step()
            if self.critic.use_max_grad_norm:
                norm = nn.utils.clip_grad_norm_(
                    critic_params, self.critic.max_grad_norm
                )
                last_critic_grad_norm = _float(norm)
            else:
                last_critic_grad_norm = _global_norm(p.grad for p in critic_params)
            self.critic.critic_optimizer.step()

            values = {
                "policy_loss": outputs["actor_policy_losses"].mean(),
                "dist_entropy": outputs["actor_entropies"].mean(),
                "ratio": outputs["actor_ratios"].mean(),
                "value_loss": outputs["value_loss"],
                "alignment_loss": outputs["alignment"],
                "distance_c_to_a": outputs["distances"]["c_to_a"],
                "distance_a_to_c": outputs["distances"]["a_to_c"],
                "distance_joint": outputs["distances"]["joint"],
                "containment_similarity": outputs["distances"][
                    "containment_similarity"
                ],
                "containment_source_effective_rank": outputs["distances"][
                    "containment_source_effective_rank"
                ],
                "containment_valid_samples_mean": outputs["distances"][
                    "containment_valid_samples_mean"
                ],
                "actor_rl_grad_norm": _global_norm(rl_grads[:actor_count]),
                "critic_rl_grad_norm": _global_norm(rl_grads[actor_count:]),
                "actor_cross_grad_norm": _global_norm(cross_grads[:actor_count]),
                "critic_cross_grad_norm": _global_norm(cross_grads[actor_count:]),
            }
            for key, value in values.items():
                accumulators[key] = accumulators.get(key, 0.0) + _float(value)

        metrics = {key: value / epochs for key, value in accumulators.items()}
        metrics["actor_grad_norm"] = float(np.mean(last_actor_grad_norms))
        metrics["critic_grad_norm"] = last_critic_grad_norm
        self.last_train_metrics = metrics

        actor_info = {
            "policy_loss": metrics["policy_loss"],
            "dist_entropy": metrics["dist_entropy"],
            "actor_grad_norm": metrics["actor_grad_norm"],
            "ratio": metrics["ratio"],
        }
        actor_train_infos = [dict(actor_info) for _ in range(self.num_agents)]
        critic_train_info = {
            "value_loss": metrics["value_loss"],
            "critic_grad_norm": metrics["critic_grad_norm"],
        }
        return actor_train_infos, critic_train_info

    def calibration_metrics(self) -> dict[str, float]:
        """Measure unweighted LN-MSE/CKA cross-gradient scales on one rollout."""

        advantages = self._normalized_advantages()
        actor_arrays = [
            self._actor_arrays(agent_id, advantages[agent_id])
            for agent_id in range(self.num_agents)
        ]
        critic_arrays = self._critic_arrays()
        actor_old, critic_old = self._old_latents(actor_arrays, critic_arrays)
        _, actor_params, critic_params = self._parameter_layout()
        recipient_params = (
            actor_params if self.align_mode == "c_to_a" else critic_params
        )

        outputs = self._forward_objectives(
            actor_arrays, critic_arrays, actor_old, critic_old, "ln_mse"
        )
        rl_loss = (
            outputs["actor_objective"]
            if self.align_mode == "c_to_a"
            else outputs["critic_objective"]
        )
        rl_grads = torch.autograd.grad(rl_loss, recipient_params, retain_graph=True)
        ln_grads = torch.autograd.grad(
            outputs["alignment"],
            recipient_params,
            retain_graph=False,
            allow_unused=True,
        )
        rl_norm = _global_norm(rl_grads)
        ln_norm = _global_norm(ln_grads)

        outputs = self._forward_objectives(
            actor_arrays, critic_arrays, actor_old, critic_old, "linear_cka"
        )
        cka_grads = torch.autograd.grad(
            outputs["alignment"], recipient_params, allow_unused=True
        )
        cka_norm = _global_norm(cka_grads)
        outputs = self._forward_objectives(
            actor_arrays, critic_arrays, actor_old, critic_old, "containment"
        )
        containment_grads = torch.autograd.grad(
            outputs["alignment"], recipient_params, allow_unused=True
        )
        containment_norm = _global_norm(containment_grads)
        if min(rl_norm, ln_norm, cka_norm, containment_norm) <= 0:
            raise RuntimeError(
                f"Calibration gradient norms must be positive: RL={rl_norm}, "
                f"LN-MSE={ln_norm}, CKA={cka_norm}, DSC={containment_norm}"
            )
        return {
            "recipient": "actor" if self.align_mode == "c_to_a" else "critic",
            "rl_grad_norm": rl_norm,
            "ln_mse_cross_grad_norm": ln_norm,
            "linear_cka_cross_grad_norm": cka_norm,
            "containment_cross_grad_norm": containment_norm,
            "ln_mse_cross_to_rl_ratio": ln_norm / rl_norm,
            "linear_cka_cross_to_rl_ratio": cka_norm / rl_norm,
            "containment_cross_to_rl_ratio": containment_norm / rl_norm,
        }

    def save_checkpoint(self, label: str, env_step: int) -> Path:
        destination = self.checkpoint_root / self.experiment["run_name"] / label
        destination.mkdir(parents=True, exist_ok=True)
        if self.share_param:
            torch.save(
                self.actor[0].actor.state_dict(), destination / "actor_shared.pt"
            )
            torch.save(
                self.actor[0].actor_optimizer.state_dict(),
                destination / "actor_optimizer_shared.pt",
            )
        else:
            for agent_id, actor in enumerate(self.actor):
                torch.save(
                    actor.actor.state_dict(),
                    destination / f"actor_agent{agent_id:02d}.pt",
                )
                torch.save(
                    actor.actor_optimizer.state_dict(),
                    destination / f"actor_optimizer_agent{agent_id:02d}.pt",
                )
        torch.save(self.critic.critic.state_dict(), destination / "critic.pt")
        torch.save(
            self.critic.critic_optimizer.state_dict(),
            destination / "critic_optimizer.pt",
        )
        if self.value_normalizer is not None:
            torch.save(
                self.value_normalizer.state_dict(), destination / "value_normalizer.pt"
            )
        metadata = {
            **self.experiment,
            "checkpoint_label": label,
            "environment_steps": int(env_step),
            "actor_files": 1 if self.share_param else self.num_agents,
            "centralized_critic": True,
            "critic_latent_pairing": "one EP critic latent repeated over 17 agents",
            "created_at_unix": time.time(),
        }
        _write_json(destination / "metadata.json", metadata)
        print(f"Checkpoint saved: {destination}", flush=True)

        if self.wandb_run is not None and self.upload_checkpoints:
            import wandb

            artifact = wandb.Artifact(
                f"{self.experiment['run_name']}-{label}", type="model"
            )
            artifact.add_dir(str(destination))
            self.wandb_run.log_artifact(artifact)
        return destination

    def _wandb_log(self, payload: dict, env_step: int) -> None:
        if self.wandb_run is not None:
            self.wandb_run.log({**payload, "env_step": env_step})

    def run(self):
        """Train, evaluate, checkpoint, and log using the tuned HARL schedule."""

        print("start aligned HARL MAPPO training", flush=True)
        self.warmup()
        train = self.algo_args["train"]
        episodes = (
            int(train["num_env_steps"])
            // train["episode_length"]
            // train["n_rollout_threads"]
        )
        steps_per_update = train["episode_length"] * train["n_rollout_threads"]
        self.logger.init(episodes)
        self.save_checkpoint("initial", 0)

        for episode in range(1, episodes + 1):
            if train["use_linear_lr_decay"]:
                for actor in self.unique_actors:
                    actor.lr_decay(episode, episodes)
                self.critic.lr_decay(episode, episodes)

            self.logger.episode_init(episode)
            self.prep_rollout()
            for step in range(train["episode_length"]):
                values, actions, action_log_probs, rnn_states, rnn_states_critic = (
                    self.collect(step)
                )
                obs, share_obs, rewards, dones, infos, available_actions = (
                    self.envs.step(actions)
                )
                data = (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )
                self.logger.per_step(data)
                self.insert(data)

            self.compute()
            self.prep_training()
            actor_infos, critic_info = self.train()
            env_step = episode * steps_per_update

            if episode % train["log_interval"] == 0:
                completed_return = (
                    float(np.mean(self.logger.done_episodes_rewards))
                    if self.logger.done_episodes_rewards
                    else float("nan")
                )
                self._wandb_log(
                    {
                        **{
                            f"train/{key}": value
                            for key, value in self.last_train_metrics.items()
                        },
                        "train/rollout_episode_return": completed_return,
                        "train/average_step_reward": self.critic_buffer.get_mean_rewards(),
                    },
                    env_step,
                )
                self.logger.episode_log(
                    actor_infos, critic_info, self.actor_buffer, self.critic_buffer
                )

            if (
                episode % train["eval_interval"] == 0
                and self.algo_args["eval"]["use_eval"]
            ):
                self.prep_rollout()
                self.eval()
                eval_rewards = np.asarray(
                    self.logger.eval_episode_rewards, dtype=np.float64
                )
                self._wandb_log(
                    {
                        "eval/return": float(np.mean(eval_rewards)),
                        "eval/return_std": float(np.std(eval_rewards)),
                        "eval/episodes": int(eval_rewards.shape[0]),
                    },
                    env_step,
                )

            while (
                self.checkpoint_interval > 0 and env_step >= self._next_checkpoint_step
            ):
                self.save_checkpoint(
                    f"step_{self._next_checkpoint_step:012d}",
                    self._next_checkpoint_step,
                )
                self._next_checkpoint_step += self.checkpoint_interval

            self.after_update()

        final_step = episodes * steps_per_update
        self.save_checkpoint("final", final_step)
        _write_json(
            self.checkpoint_root / self.experiment["run_name"] / "completed.json",
            {
                "status": "completed",
                "environment_steps": final_step,
                "run_name": self.experiment["run_name"],
            },
        )
