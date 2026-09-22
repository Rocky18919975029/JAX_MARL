"""HARL runners for a matched ShadowHandOver algorithm comparison."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.runners.on_policy_ma_runner import OnPolicyMARunner
from harl.utils.trans_tools import _t2n


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _number(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


class MADPORunner(OnPolicyHARunner):
    """Sequential HARL runner with corrected MADPO policy references."""

    def __init__(self, args, algo_args, env_args):
        super().__init__(args, algo_args, env_args)
        if self.share_param:
            raise ValueError("MADPO ShadowHandOver is locked to NPS actors")
        if self.state_type != "EP":
            raise ValueError("MADPO ShadowHandOver requires EP centralized state")
        if self.num_agents != 2:
            raise ValueError(
                f"Expected two ShadowHandOver agents, got {self.num_agents}"
            )
        if (
            self.algo_args["model"]["use_recurrent_policy"]
            or self.algo_args["model"]["use_naive_recurrent_policy"]
        ):
            raise ValueError("The first MADPO implementation is feed-forward only")
        base_seed = int(self.algo_args["seed"]["seed"])
        for agent_id, actor in enumerate(self.actor):
            actor.set_divergence_seed(base_seed * 10_000 + agent_id)

    def train(self):
        actor_train_infos = [None] * self.num_agents
        factor = np.ones(
            (
                self.algo_args["train"]["episode_length"],
                self.algo_args["train"]["n_rollout_threads"],
                1,
            ),
            dtype=np.float32,
        )
        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[
                :-1
            ] - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        else:
            advantages = (
                self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
            )

        agent_order = (
            list(range(self.num_agents))
            if self.fixed_order
            else list(torch.randperm(self.num_agents).cpu().numpy())
        )
        previous_agent_id = None
        for agent_id in agent_order:
            buffer = self.actor_buffer[agent_id]
            buffer.update_factor(factor)
            available = (
                None
                if buffer.available_actions is None
                else buffer.available_actions[:-1].reshape(
                    -1, buffer.available_actions.shape[-1]
                )
            )
            with torch.no_grad():
                old_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                    buffer.obs[:-1].reshape(-1, *buffer.obs.shape[2:]),
                    buffer.rnn_states[:-1].reshape(-1, *buffer.rnn_states.shape[2:]),
                    buffer.actions.reshape(-1, buffer.actions.shape[-1]),
                    buffer.masks[:-1].reshape(-1, 1),
                    available,
                    buffer.active_masks[:-1].reshape(-1, 1),
                )
            old_policy = copy.deepcopy(self.actor[agent_id].actor)
            peer_policy = (
                None
                if previous_agent_id is None
                else self.actor[previous_agent_id].actor
            )
            peer_buffer = (
                None
                if previous_agent_id is None
                else self.actor_buffer[previous_agent_id]
            )
            actor_train_infos[agent_id] = self.actor[agent_id].train_with_divergence(
                buffer,
                advantages.copy(),
                self.state_type,
                old_policy,
                peer_policy,
                peer_buffer,
            )
            with torch.no_grad():
                new_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                    buffer.obs[:-1].reshape(-1, *buffer.obs.shape[2:]),
                    buffer.rnn_states[:-1].reshape(-1, *buffer.rnn_states.shape[2:]),
                    buffer.actions.reshape(-1, buffer.actions.shape[-1]),
                    buffer.masks[:-1].reshape(-1, 1),
                    available,
                    buffer.active_masks[:-1].reshape(-1, 1),
                )
            factor *= _t2n(
                getattr(torch, self.action_aggregation)(
                    torch.exp(new_actions_logprob - old_actions_logprob), dim=-1
                ).reshape(
                    self.algo_args["train"]["episode_length"],
                    self.algo_args["train"]["n_rollout_threads"],
                    1,
                )
            )
            previous_agent_id = agent_id

        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)
        return actor_train_infos, critic_train_info


class TrackingMixin:
    """Add atomic progress, JSONL metrics, final models, and W&B logging."""

    def configure_tracking(
        self,
        *,
        status_path: Path,
        metrics_path: Path,
        run_name: str,
        total_env_steps: int,
        base_status: dict,
        wandb_run=None,
    ) -> None:
        self.tracking_status_path = status_path
        self.tracking_metrics_path = metrics_path
        self.tracking_run_name = run_name
        self.tracking_total_env_steps = int(total_env_steps)
        self.tracking_base_status = dict(base_status)
        self.tracking_wandb_run = wandb_run
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

    def _status(self, state: str, steps: int, **extra) -> None:
        _write_json(
            self.tracking_status_path,
            {
                **self.tracking_base_status,
                "status": state,
                "run_name": self.tracking_run_name,
                "pid": __import__("os").getpid(),
                "env_steps": int(steps),
                "total_env_steps": self.tracking_total_env_steps,
                "harl_output": str(self.run_dir),
                **extra,
            },
        )

    def run(self):
        self.warmup()
        horizon = int(self.algo_args["train"]["episode_length"])
        threads = int(self.algo_args["train"]["n_rollout_threads"])
        episodes = self.tracking_total_env_steps // horizon // threads
        self.logger.init(episodes)
        started = time.time()
        self._status("running", 0)

        for episode in range(1, episodes + 1):
            if self.algo_args["train"]["use_linear_lr_decay"]:
                if self.share_param:
                    self.actor[0].lr_decay(episode, episodes)
                else:
                    for actor in self.actor:
                        actor.lr_decay(episode, episodes)
                self.critic.lr_decay(episode, episodes)
            self.logger.episode_init(episode)
            rollout_returns = np.zeros(threads, dtype=np.float64)
            self.prep_rollout()
            for step in range(horizon):
                values, actions, log_probs, rnn_states, critic_rnn = self.collect(step)
                obs, share_obs, rewards, dones, infos, available = self.envs.step(
                    actions
                )
                rollout_returns += np.mean(rewards, axis=1).reshape(-1)
                data = (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available,
                    values,
                    actions,
                    log_probs,
                    rnn_states,
                    critic_rnn,
                )
                self.logger.per_step(data)
                self.insert(data)

            self.compute()
            self.prep_training()
            actor_info, critic_info = self.train()
            env_steps = episode * horizon * threads
            row = {
                "env_step": env_steps,
                "episode_return_mean": float(rollout_returns.mean()),
                "episode_return_std": float(rollout_returns.std()),
                "fps": float(env_steps / max(time.time() - started, 1e-9)),
            }
            actor_keys = sorted(
                {key for info in actor_info if info is not None for key in info}
            )
            for key in actor_keys:
                values = [_number(info[key]) for info in actor_info if key in info]
                row[f"actor/{key}"] = float(np.mean(values))
            for key, value in critic_info.items():
                row[f"critic/{key}"] = _number(value)
            with self.tracking_metrics_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(row, sort_keys=True) + "\n")
            if self.tracking_wandb_run is not None:
                self.tracking_wandb_run.log(row, step=env_steps)
            self._status("running", env_steps, latest_metrics=row)

            if episode % self.algo_args["train"]["log_interval"] == 0:
                self.logger.episode_log(
                    actor_info, critic_info, self.actor_buffer, self.critic_buffer
                )
            if episode % self.algo_args["train"]["eval_interval"] == 0:
                self.save()
            self.after_update()

        self.save()
        self._status(
            "completed",
            episodes * horizon * threads,
            elapsed_seconds=time.time() - started,
        )


class TrackedHAPPORunner(TrackingMixin, OnPolicyHARunner):
    pass


class TrackedMAPPORunner(TrackingMixin, OnPolicyMARunner):
    pass


class TrackedMADPORunner(TrackingMixin, MADPORunner):
    pass
