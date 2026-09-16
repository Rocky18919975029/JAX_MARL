"""Matched feed-forward MAPPO alignment experiments for JaxMARL MABrax.

The actor receives a padded local observation. The shared centralized critic
receives the underlying Brax global observation. ``ACTOR_PARAMETER_SHARING``
selects either one shared ActorFF or one complete ActorFF parameter set per
agent while leaving the critic shared in both cases.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from functools import partial
from pathlib import Path
from typing import Any, Dict, NamedTuple

import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jax.experimental import io_callback
from omegaconf import OmegaConf
from tqdm.auto import tqdm

import jaxmarl

try:
    from baselines.MAPPO.alignment_utils import (
        representation_distance,
        tree_l2_norm,
        vmapped_optimizer,
    )
except ModuleNotFoundError:  # Direct execution from baselines/MAPPO.
    from alignment_utils import (
        representation_distance,
        tree_l2_norm,
        vmapped_optimizer,
    )
from jaxmarl.wrappers.baselines import JaxMARLWrapper, LogWrapper, save_params


def _safe_path_component(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-_") or "run"


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def make_metrics_jsonl_callback(path):
    """Create an optional machine-readable metrics callback."""

    if not path:
        return None
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    def callback(metrics):
        payload = {}
        for key, value in metrics.items():
            array = np.asarray(value)
            payload[key] = array.item() if array.ndim == 0 else array.tolist()
        with destination.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, sort_keys=True, allow_nan=True) + "\n")
            file.flush()

    return callback


def make_checkpoint_callback(config, run):
    """Create a host callback that writes evaluation-ready checkpoints."""

    root = Path(config["CHECKPOINT_DIR"]).expanduser().resolve()
    run_dir = (
        root
        / _safe_path_component(config.get("PROJECT") or "local")
        / f"{_safe_path_component(run.name)}-{_safe_path_component(run.id)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    def write_json_atomic(path, payload):
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True, default=_json_default)
            file.write("\n")
        os.replace(temporary, path)

    def callback(
        actor_params,
        critic_params,
        env_step,
        nominal_env_step,
        is_final,
        is_initial,
    ):
        env_step = int(np.asarray(env_step).item())
        nominal_env_step = int(np.asarray(nominal_env_step).item())
        is_final = bool(np.asarray(is_final).item())
        is_initial = bool(np.asarray(is_initial).item())
        if is_final:
            nominal_env_step = int(config["TOTAL_TIMESTEPS"])
        name = (
            "initial"
            if is_initial
            else "final" if is_final else f"step_{nominal_env_step:012d}"
        )
        directory = run_dir / name
        directory.mkdir(parents=True, exist_ok=True)
        temporary_model = directory / ".model.tmp.safetensors"
        save_params({"actor": actor_params, "critic": critic_params}, temporary_model)
        os.replace(temporary_model, directory / "model.safetensors")

        checkpoint_config = dict(config)
        checkpoint_config.update(
            {
                "CHECKPOINT_ENV_STEP": env_step,
                "CHECKPOINT_NOMINAL_ENV_STEP": nominal_env_step,
                "CHECKPOINT_IS_FINAL": is_final,
                "CHECKPOINT_IS_INITIAL": is_initial,
            }
        )
        write_json_atomic(directory / "config.json", checkpoint_config)
        metadata = {
            "format_version": 1,
            "environment": config["ENV_NAME"],
            "seed": config["SEED"],
            "env_step": env_step,
            "nominal_env_step": nominal_env_step,
            "is_final": is_final,
            "is_initial": is_initial,
            "actor_parameter_sharing": config["ACTOR_PARAMETER_SHARING"],
            "matched_comparison": config["MATCHED_COMPARISON"],
            "align_mode": config["ALIGN_MODE"],
            "align_distance": config["ALIGN_DISTANCE"],
            "alignment_coef": config["ALIGNMENT_COEF"],
            "condition": config["EXPERIMENT_CONDITION"],
            "matrix_profile": config.get("MATRIX_PROFILE", ""),
            "protocol_version": config.get("PROTOCOL_VERSION", ""),
            "git_commit": config.get("GIT_COMMIT", ""),
            "wandb_project": run.project,
            "wandb_run_id": run.id,
            "wandb_run_name": run.name,
        }
        write_json_atomic(directory / "metadata.json", metadata)
        write_json_atomic(
            run_dir / "latest.json",
            {
                "checkpoint": name,
                "env_step": env_step,
                "nominal_env_step": nominal_env_step,
                "is_final": is_final,
                "is_initial": is_initial,
            },
        )
        print(f"Checkpoint saved: {directory}", flush=True)

        if config["WANDB_UPLOAD_CHECKPOINTS"]:
            artifact = wandb.Artifact(
                f"{_safe_path_component(run.name)}-{run.id}-checkpoint",
                type="model",
                metadata=metadata,
            )
            artifact.add_dir(str(directory))
            aliases = ["latest", f"step-{env_step}"]
            if is_initial:
                aliases.append("initial")
            if is_final:
                aliases.append("final")
            run.log_artifact(artifact, aliases=aliases)
        return np.int32(0)

    return callback, run_dir


class MABraxWorldStateWrapper(JaxMARLWrapper):
    """Adds the underlying Brax global observation for a central critic."""

    def _add_world_state(self, obs, state):
        world_state = jnp.broadcast_to(
            state.obs[None, :],
            (self._env.num_agents, state.obs.shape[-1]),
        )
        return {**obs, "world_state": world_state}

    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        obs, state = self._env.reset(key)
        return self._add_world_state(obs, state), state

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, action):
        obs, state, reward, done, info = self._env.step(key, state, action)
        return self._add_world_state(obs, state), state, reward, done, info

    def world_state_size(self) -> int:
        return int(self._env.env.observation_size)


class ActorFF(nn.Module):
    action_dim: int
    hidden_size: int
    activation: str

    @nn.compact
    def __call__(self, obs):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        latent = obs
        for _ in range(2):
            latent = nn.Dense(
                self.hidden_size,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(latent)
            latent = activation(latent)
        mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(latent)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        return distrax.MultivariateNormalDiag(mean, jnp.exp(log_std)), latent


class CriticFF(nn.Module):
    hidden_size: int
    activation: str

    @nn.compact
    def __call__(self, world_state):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        latent = world_state
        for _ in range(2):
            latent = nn.Dense(
                self.hidden_size,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(latent)
            latent = activation(latent)
        value = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(latent)
        return jnp.squeeze(value, axis=-1), latent


class Transition(NamedTuple):
    global_done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    world_state: jax.Array
    actor_latent_old: jax.Array
    critic_latent_old: jax.Array
    alignment_mask: jax.Array
    info: Dict[str, jax.Array]


def batchify_observations(observations, agent_list, num_actors):
    """Pads heterogeneous local observations and returns agent-major batches."""

    max_dim = max(observations[agent].shape[-1] for agent in agent_list)
    padded = [
        jnp.pad(
            observations[agent],
            ((0, 0), (0, max_dim - observations[agent].shape[-1])),
        )
        for agent in agent_list
    ]
    return jnp.stack(padded).reshape((num_actors, max_dim))


def batchify_scalars(values, agent_list, num_actors):
    return jnp.stack([values[agent] for agent in agent_list]).reshape(num_actors)


def batchify_world_state(world_state, num_actors):
    return world_state.swapaxes(0, 1).reshape((num_actors, world_state.shape[-1]))


def batchify_info(info):
    return jax.tree.map(
        lambda value: value.swapaxes(0, 1).reshape((-1,) + tuple(value.shape[2:])),
        info,
    )


def unbatchify_actions(actions, agent_list, num_envs):
    actions = actions.reshape((len(agent_list), num_envs, actions.shape[-1]))
    return {agent: actions[index] for index, agent in enumerate(agent_list)}


def _validate_and_derive_config(config: Dict[str, Any], env) -> Dict[str, Any]:
    config = dict(config)
    for field in (
        "NUM_ENVS",
        "NUM_STEPS",
        "TOTAL_TIMESTEPS",
        "UPDATE_EPOCHS",
        "NUM_MINIBATCHES",
    ):
        if int(config[field]) <= 0:
            raise ValueError(f"{field} must be positive")

    if not config["MATCHED_COMPARISON"]:
        raise ValueError(
            "The MABrax alignment protocol requires MATCHED_COMPARISON=true"
        )
    if config["ALIGN_MODE"] not in {"none", "c_to_a", "a_to_c", "joint"}:
        raise ValueError("ALIGN_MODE must be none, c_to_a, a_to_c, or joint")
    if config["ALIGN_DISTANCE"] not in {"ln_mse", "linear_cka"}:
        raise ValueError("ALIGN_DISTANCE must be ln_mse or linear_cka")
    if config["ALIGN_MODE"] == "none" and config["ALIGN_DISTANCE"] != "ln_mse":
        raise ValueError("The distance-free baseline must use ALIGN_DISTANCE=ln_mse")
    if float(config["ALIGN_DISTANCE_EPS"]) <= 0:
        raise ValueError("ALIGN_DISTANCE_EPS must be positive")

    action_dims = {env.action_space(agent).shape[0] for agent in env.agents}
    if len(action_dims) != 1:
        raise ValueError("MABrax MAPPO requires equal action dimensions across agents")
    config["NUM_AGENTS"] = env.num_agents
    config["NUM_ACTORS"] = env.num_agents * int(config["NUM_ENVS"])
    config["NUM_UPDATES"] = (
        int(config["TOTAL_TIMESTEPS"])
        // int(config["NUM_STEPS"])
        // int(config["NUM_ENVS"])
    )
    if config["NUM_UPDATES"] <= 0:
        raise ValueError("TOTAL_TIMESTEPS is smaller than one rollout batch")
    if int(config["NUM_ENVS"]) % int(config["NUM_MINIBATCHES"]) != 0:
        raise ValueError("NUM_ENVS must be divisible by NUM_MINIBATCHES")
    config["ENVS_PER_MINIBATCH"] = int(config["NUM_ENVS"]) // int(
        config["NUM_MINIBATCHES"]
    )
    config["MINIBATCH_SIZE"] = (
        int(config["NUM_STEPS"]) * env.num_agents * config["ENVS_PER_MINIBATCH"]
    )
    config["ACTION_DIM"] = next(iter(action_dims))
    config["OBS_DIM"] = max(
        env.observation_space(agent).shape[0] for agent in env.agents
    )
    config["WORLD_STATE_DIM"] = env.world_state_size()

    if config["ALIGN_GRADIENT_CALIBRATION"]:
        if config["ALIGN_MODE"] not in {"c_to_a", "a_to_c"}:
            raise ValueError("Calibration requires c_to_a or a_to_c")
        if float(config["ALIGNMENT_COEF"]) != 0.0:
            raise ValueError("Calibration requires ALIGNMENT_COEF=0")
        if config["NUM_UPDATES"] != 1 or int(config["UPDATE_EPOCHS"]) != 1:
            raise ValueError("Calibration requires one rollout and one update epoch")
        if float(config["LR"]) != 0.0:
            raise ValueError("Calibration requires LR=0")
    return config


def make_train(
    config, host_callback, checkpoint_callback=None, metrics_jsonl_callback=None
):
    raw_env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = MABraxWorldStateWrapper(raw_env)
    config = _validate_and_derive_config(config, env)
    env = LogWrapper(env, replace_info=True)
    num_agents = int(config["NUM_AGENTS"])
    num_envs = int(config["NUM_ENVS"])

    actor_network = ActorFF(
        action_dim=config["ACTION_DIM"],
        hidden_size=int(config["HIDDEN_SIZE"]),
        activation=config["ACTIVATION"],
    )
    critic_network = CriticFF(
        hidden_size=int(config["HIDDEN_SIZE"]),
        activation=config["ACTIVATION"],
    )
    optimizer_steps = int(config["UPDATE_EPOCHS"]) * int(config["NUM_MINIBATCHES"])

    def learning_rate_schedule(count):
        completed_updates = count // optimizer_steps
        fraction = 1.0 - completed_updates / config["NUM_UPDATES"]
        return config["LR"] * jnp.maximum(fraction, 0.0)

    learning_rate = learning_rate_schedule if config["ANNEAL_LR"] else config["LR"]
    base_actor_tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(learning_rate=learning_rate, eps=1e-5),
    )
    actor_tx = (
        base_actor_tx
        if config["ACTOR_PARAMETER_SHARING"]
        else vmapped_optimizer(base_actor_tx)
    )
    critic_tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(learning_rate=learning_rate, eps=1e-5),
    )

    def train(rng):
        rng, actor_key, critic_key, reset_key = jax.random.split(rng, 4)
        actor_params = actor_network.init(actor_key, jnp.zeros((1, config["OBS_DIM"])))
        if not config["ACTOR_PARAMETER_SHARING"]:
            actor_params = jax.tree.map(
                lambda value: jnp.repeat(value[None, ...], num_agents, axis=0),
                actor_params,
            )
        critic_params = critic_network.init(
            critic_key, jnp.zeros((1, config["WORLD_STATE_DIM"]))
        )
        actor_state = TrainState.create(
            apply_fn=actor_network.apply, params=actor_params, tx=actor_tx
        )
        critic_state = TrainState.create(
            apply_fn=critic_network.apply, params=critic_params, tx=critic_tx
        )

        def apply_actor(params, observations):
            parameter_axis = None if config["ACTOR_PARAMETER_SHARING"] else 0
            return jax.vmap(actor_network.apply, in_axes=(parameter_axis, 0))(
                params, observations
            )

        def actor_tree_norms(tree):
            if config["ACTOR_PARAMETER_SHARING"]:
                return jnp.asarray([tree_l2_norm(tree)])
            return jax.vmap(tree_l2_norm)(tree)

        if checkpoint_callback is not None:
            io_callback(
                checkpoint_callback,
                jax.ShapeDtypeStruct((), jnp.int32),
                actor_state.params,
                critic_state.params,
                jnp.asarray(0, jnp.int32),
                jnp.asarray(0, jnp.int32),
                jnp.asarray(False),
                jnp.asarray(True),
                ordered=True,
            )

        reset_keys = jax.random.split(reset_key, num_envs)
        obs, env_state = jax.vmap(env.reset)(reset_keys)

        def update_step(runner_state, _):
            actor_state, critic_state, env_state, last_obs, rng, update_count = (
                runner_state
            )

            def env_step(step_state, _):
                actor_state, critic_state, env_state, last_obs, rng = step_state
                flat_obs = batchify_observations(
                    last_obs, env.agents, config["NUM_ACTORS"]
                )
                actor_obs = flat_obs.reshape((num_agents, num_envs, -1))
                flat_world = batchify_world_state(
                    last_obs["world_state"], config["NUM_ACTORS"]
                )
                actor_policy, actor_latent = apply_actor(actor_state.params, actor_obs)
                value, critic_latent = critic_network.apply(
                    critic_state.params, flat_world
                )
                value = value.reshape((num_agents, num_envs))
                critic_latent = critic_latent.reshape((num_agents, num_envs, -1))

                rng, action_key, step_key = jax.random.split(rng, 3)
                action = actor_policy.sample(seed=action_key)
                log_prob = actor_policy.log_prob(action)
                step_keys = jax.random.split(step_key, num_envs)
                next_obs, next_env_state, reward, done, info = jax.vmap(env.step)(
                    step_keys,
                    env_state,
                    unbatchify_actions(action, env.agents, num_envs),
                )
                flat_info = batchify_info(info)
                transition = Transition(
                    global_done=jnp.broadcast_to(
                        done["__all__"][None, :], (num_agents, num_envs)
                    ),
                    action=action,
                    value=value,
                    reward=batchify_scalars(
                        reward, env.agents, config["NUM_ACTORS"]
                    ).reshape((num_agents, num_envs)),
                    log_prob=log_prob,
                    obs=actor_obs,
                    world_state=flat_world.reshape((num_agents, num_envs, -1)),
                    actor_latent_old=jax.lax.stop_gradient(actor_latent),
                    critic_latent_old=jax.lax.stop_gradient(critic_latent),
                    alignment_mask=jnp.ones((num_agents, num_envs)),
                    info=jax.tree.map(
                        lambda value: value.reshape(
                            (num_agents, num_envs) + tuple(value.shape[1:])
                        ),
                        flat_info,
                    ),
                )
                return (
                    actor_state,
                    critic_state,
                    next_env_state,
                    next_obs,
                    rng,
                ), transition

            step_state, trajectory = jax.lax.scan(
                env_step,
                (actor_state, critic_state, env_state, last_obs, rng),
                None,
                int(config["NUM_STEPS"]),
            )
            actor_state, critic_state, env_state, last_obs, rng = step_state
            last_world = batchify_world_state(
                last_obs["world_state"], config["NUM_ACTORS"]
            )
            last_value, _ = critic_network.apply(critic_state.params, last_world)
            last_value = last_value.reshape((num_agents, num_envs))

            def gae_step(carry, transition):
                gae, next_value = carry
                not_done = 1.0 - transition.global_done
                delta = (
                    transition.reward
                    + config["GAMMA"] * next_value * not_done
                    - transition.value
                )
                gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * not_done * gae
                return (gae, transition.value), gae

            _, advantages = jax.lax.scan(
                gae_step,
                (jnp.zeros_like(last_value), last_value),
                trajectory,
                reverse=True,
                unroll=8,
            )
            targets = advantages + trajectory.value

            def update_epoch(epoch_state, _):
                actor_state, critic_state, rng = epoch_state
                rng, permutation_key = jax.random.split(rng)
                permutation = jax.random.permutation(permutation_key, num_envs)

                def stratify(value):
                    value = jnp.take(value, permutation, axis=2)
                    value = value.reshape(
                        (
                            value.shape[0],
                            num_agents,
                            int(config["NUM_MINIBATCHES"]),
                            config["ENVS_PER_MINIBATCH"],
                            *value.shape[3:],
                        )
                    )
                    return jnp.moveaxis(value, 2, 0)

                minibatches = jax.tree.map(stratify, (trajectory, advantages, targets))

                def update_minibatch(train_states, minibatch):
                    actor_state, critic_state = train_states
                    batch, batch_advantages, batch_targets = minibatch
                    normalized_advantage = (
                        batch_advantages - batch_advantages.mean()
                    ) / (batch_advantages.std() + 1e-8)

                    def rl_loss(actor_params, critic_params):
                        obs_by_agent = jnp.swapaxes(batch.obs, 0, 1)
                        actions_by_agent = jnp.swapaxes(batch.action, 0, 1)
                        policy, actor_latent_by_agent = apply_actor(
                            actor_params, obs_by_agent
                        )
                        new_log_prob = jnp.swapaxes(
                            policy.log_prob(actions_by_agent), 0, 1
                        )
                        actor_latent = jnp.swapaxes(actor_latent_by_agent, 0, 1)
                        log_ratio = new_log_prob - batch.log_prob
                        ratio = jnp.exp(log_ratio)
                        unclipped = ratio * normalized_advantage
                        clipped = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * normalized_advantage
                        )
                        policy_loss = -jnp.minimum(unclipped, clipped).mean()
                        entropy = policy.entropy().mean()
                        actor_loss = policy_loss - config["ENT_COEF"] * entropy

                        flat_world = batch.world_state.reshape(
                            (-1, batch.world_state.shape[-1])
                        )
                        value, critic_latent = critic_network.apply(
                            critic_params, flat_world
                        )
                        value = value.reshape(batch.value.shape)
                        critic_latent = critic_latent.reshape(
                            batch.critic_latent_old.shape
                        )
                        clipped_value = batch.value + jnp.clip(
                            value - batch.value, -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        value_loss = (
                            0.5
                            * jnp.maximum(
                                jnp.square(value - batch_targets),
                                jnp.square(clipped_value - batch_targets),
                            ).mean()
                        )
                        critic_loss = config["VF_COEF"] * value_loss
                        return actor_loss + critic_loss, (
                            policy_loss,
                            value_loss,
                            entropy,
                            ((ratio - 1.0) - log_ratio).mean(),
                            (jnp.abs(ratio - 1.0) > config["CLIP_EPS"]).mean(),
                            actor_latent,
                            critic_latent,
                        )

                    def alignment_losses(
                        actor_params,
                        critic_params,
                        distance_name=config["ALIGN_DISTANCE"],
                    ):
                        _, auxiliary = rl_loss(actor_params, critic_params)
                        actor_latent, critic_latent = auxiliary[-2:]
                        kwargs = {
                            "agent_axis": 1,
                            "epsilon": config["ALIGN_DISTANCE_EPS"],
                        }
                        c_to_a = representation_distance(
                            actor_latent,
                            jax.lax.stop_gradient(batch.critic_latent_old),
                            batch.alignment_mask,
                            distance_name,
                            **kwargs,
                        )
                        a_to_c = representation_distance(
                            critic_latent,
                            jax.lax.stop_gradient(batch.actor_latent_old),
                            batch.alignment_mask,
                            distance_name,
                            **kwargs,
                        )
                        joint = representation_distance(
                            actor_latent,
                            critic_latent,
                            batch.alignment_mask,
                            distance_name,
                            **kwargs,
                        )
                        selected = jnp.zeros((), actor_latent.dtype)
                        if config["ALIGN_MODE"] == "c_to_a":
                            selected = c_to_a
                        elif config["ALIGN_MODE"] == "a_to_c":
                            selected = a_to_c
                        elif config["ALIGN_MODE"] == "joint":
                            selected = joint
                        return selected, (c_to_a, a_to_c, joint)

                    def total_loss(actor_params, critic_params):
                        base, rl_aux = rl_loss(actor_params, critic_params)
                        alignment, alignment_aux = alignment_losses(
                            actor_params, critic_params
                        )
                        return base + config["ALIGNMENT_COEF"] * alignment, (
                            rl_aux,
                            alignment,
                            alignment_aux,
                        )

                    (loss, auxiliary), (actor_grads, critic_grads) = jax.value_and_grad(
                        total_loss, argnums=(0, 1), has_aux=True
                    )(actor_state.params, critic_state.params)
                    rl_actor_grads, rl_critic_grads = jax.grad(
                        lambda actor, critic: rl_loss(actor, critic)[0], argnums=(0, 1)
                    )(actor_state.params, critic_state.params)
                    if config["ALIGN_MODE"] == "none":
                        actor_cross_grads = jax.tree.map(jnp.zeros_like, actor_grads)
                        critic_cross_grads = jax.tree.map(jnp.zeros_like, critic_grads)
                    else:
                        actor_cross_grads, critic_cross_grads = jax.grad(
                            lambda actor, critic: config["ALIGNMENT_COEF"]
                            * alignment_losses(actor, critic)[0],
                            argnums=(0, 1),
                        )(actor_state.params, critic_state.params)

                    calibration = None
                    if config["ALIGN_GRADIENT_CALIBRATION"]:

                        def calibration_grads(distance_name):
                            return jax.grad(
                                lambda actor, critic: alignment_losses(
                                    actor, critic, distance_name
                                )[0],
                                argnums=(0, 1),
                            )(actor_state.params, critic_state.params)

                        calibration = (
                            calibration_grads("ln_mse"),
                            calibration_grads("linear_cka"),
                        )

                    if not config["ACTOR_PARAMETER_SHARING"]:

                        def scale_actor(tree):
                            return jax.tree.map(lambda value: value * num_agents, tree)

                        actor_grads = scale_actor(actor_grads)
                        rl_actor_grads = scale_actor(rl_actor_grads)
                        actor_cross_grads = scale_actor(actor_cross_grads)
                        if calibration is not None:
                            calibration = (
                                (scale_actor(calibration[0][0]), calibration[0][1]),
                                (scale_actor(calibration[1][0]), calibration[1][1]),
                            )

                    actor_grad_norms = actor_tree_norms(actor_grads)
                    actor_rl_norms = actor_tree_norms(rl_actor_grads)
                    actor_cross_norms = actor_tree_norms(actor_cross_grads)
                    critic_grad_norm = tree_l2_norm(critic_grads)
                    critic_rl_norm = tree_l2_norm(rl_critic_grads)
                    critic_cross_norm = tree_l2_norm(critic_cross_grads)
                    actor_state = actor_state.apply_gradients(grads=actor_grads)
                    critic_state = critic_state.apply_gradients(grads=critic_grads)

                    rl_aux, alignment, alignment_aux = auxiliary
                    info = {
                        "total_loss": loss,
                        "actor_loss": rl_aux[0],
                        "value_loss": rl_aux[1],
                        "entropy": rl_aux[2],
                        "approx_kl": rl_aux[3],
                        "clip_fraction": rl_aux[4],
                        "alignment_objective": alignment,
                        "alignment_objective_weighted": config["ALIGNMENT_COEF"]
                        * alignment,
                        "alignment_c_to_a_distance": alignment_aux[0],
                        "alignment_a_to_c_distance": alignment_aux[1],
                        "alignment_current_joint_distance": alignment_aux[2],
                        "actor_grad_norm_mean": actor_grad_norms.mean(),
                        "actor_rl_grad_norm_mean": actor_rl_norms.mean(),
                        "actor_cross_grad_norm_mean": actor_cross_norms.mean(),
                        "actor_cross_to_rl_grad_ratio_mean": (
                            actor_cross_norms / jnp.maximum(actor_rl_norms, 1e-12)
                        ).mean(),
                        "critic_grad_norm": critic_grad_norm,
                        "critic_rl_grad_norm": critic_rl_norm,
                        "critic_cross_grad_norm": critic_cross_norm,
                        "critic_cross_to_rl_grad_ratio": critic_cross_norm
                        / jnp.maximum(critic_rl_norm, 1e-12),
                    }
                    if calibration is not None:
                        ln_actor = actor_tree_norms(calibration[0][0])
                        cka_actor = actor_tree_norms(calibration[1][0])
                        ln_critic = tree_l2_norm(calibration[0][1])
                        cka_critic = tree_l2_norm(calibration[1][1])
                        info.update(
                            {
                                "calibration_ln_mse_actor_cross_to_rl_ratio": (
                                    ln_actor / jnp.maximum(actor_rl_norms, 1e-12)
                                ).mean(),
                                "calibration_linear_cka_actor_cross_to_rl_ratio": (
                                    cka_actor / jnp.maximum(actor_rl_norms, 1e-12)
                                ).mean(),
                                "calibration_ln_mse_critic_cross_to_rl_ratio": ln_critic
                                / jnp.maximum(critic_rl_norm, 1e-12),
                                "calibration_linear_cka_critic_cross_to_rl_ratio": cka_critic
                                / jnp.maximum(critic_rl_norm, 1e-12),
                            }
                        )
                    return (actor_state, critic_state), info

                (actor_state, critic_state), loss_info = jax.lax.scan(
                    update_minibatch, (actor_state, critic_state), minibatches
                )
                return (actor_state, critic_state, rng), loss_info

            (actor_state, critic_state, rng), loss_info = jax.lax.scan(
                update_epoch,
                (actor_state, critic_state, rng),
                None,
                int(config["UPDATE_EPOCHS"]),
            )
            loss_info = jax.tree.map(lambda value: value.mean(), loss_info)
            completed = trajectory.info["returned_episode"][:, 0, :]
            completed_count = completed.sum()
            episode_return = (
                trajectory.info["returned_episode_returns"][:, 0, :] * completed
            ).sum() / jnp.maximum(completed_count, 1.0)
            update_count = update_count + 1
            metrics = {
                **loss_info,
                "episode_return": episode_return,
                "episodes_completed": completed_count,
                "mean_step_reward": trajectory.reward.mean(),
                "update_step": update_count,
                "env_step": update_count * int(config["NUM_STEPS"]) * num_envs,
            }
            jax.debug.callback(host_callback, metrics, ordered=True)
            if metrics_jsonl_callback is not None:
                metrics_output = metrics
                if config["ALIGN_GRADIENT_CALIBRATION"]:
                    # Persist no performance signal in the calibration artifact.
                    # Lambda is selected exclusively from initial gradient ratios.
                    audit_keys = {
                        "env_step",
                        "actor_rl_grad_norm_mean",
                        "critic_rl_grad_norm",
                        "alignment_c_to_a_distance",
                        "alignment_a_to_c_distance",
                    }
                    metrics_output = {
                        key: value
                        for key, value in metrics.items()
                        if key.startswith("calibration_") or key in audit_keys
                    }
                jax.debug.callback(metrics_jsonl_callback, metrics_output, ordered=True)

            if checkpoint_callback is not None:
                rollout_steps = num_envs * int(config["NUM_STEPS"])
                completed_steps = update_count * rollout_steps
                previous_steps = (update_count - 1) * rollout_steps
                interval = int(config["CHECKPOINT_INTERVAL_TIMESTEPS"])
                crossed = completed_steps // interval > previous_steps // interval
                is_final = update_count == config["NUM_UPDATES"]
                should_save = jnp.logical_or(crossed, is_final)
                nominal = (completed_steps // interval) * interval

                def save(_):
                    return io_callback(
                        checkpoint_callback,
                        jax.ShapeDtypeStruct((), jnp.int32),
                        actor_state.params,
                        critic_state.params,
                        completed_steps,
                        nominal,
                        is_final,
                        jnp.asarray(False),
                        ordered=True,
                    )

                jax.lax.cond(should_save, save, lambda _: jnp.int32(0), None)

            return (
                actor_state,
                critic_state,
                env_state,
                last_obs,
                rng,
                update_count,
            ), metrics

        initial_state = (
            actor_state,
            critic_state,
            env_state,
            obs,
            rng,
            jnp.array(0, dtype=jnp.int32),
        )
        final_state, metrics = jax.lax.scan(
            update_step, initial_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": final_state, "metrics": metrics}

    return train, config


@hydra.main(version_base=None, config_path="config", config_name="mappo_ff_mabrax")
def main(hydra_config):
    config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(config, dict):
        raise TypeError("Expected a dictionary-like Hydra configuration")
    if not config.get("GIT_COMMIT"):
        try:
            config["GIT_COMMIT"] = subprocess.run(
                ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            config["GIT_COMMIT"] = "unknown"

    condition = config["ALIGN_MODE"]
    if config["ALIGN_DISTANCE"] == "linear_cka" and condition != "none":
        condition = f"{condition}_cka"
    if config.get("EXPERIMENT_CONDITION") not in ("", condition):
        raise ValueError("EXPERIMENT_CONDITION does not match alignment settings")
    config["EXPERIMENT_CONDITION"] = condition
    actor_label = "ps" if config["ACTOR_PARAMETER_SHARING"] else "nps"
    lambda_label = f"{float(config['ALIGNMENT_COEF']):.10g}".replace(".", "p")
    env_name = str(config["ENV_NAME"])
    default_name = f"MABRAX-{env_name}-{actor_label}-{condition}-lam{lambda_label}-seed{int(config['SEED'])}"
    run_name = os.environ.get("WANDB_NAME") or config.get("WANDB_NAME") or default_name
    run_group = (
        os.environ.get("WANDB_RUN_GROUP")
        or f"MABRAX-{env_name}-{actor_label}-{condition}-lam{lambda_label}"
    )
    tags = [
        "MAPPO",
        "FF",
        "MABrax",
        env_name,
        actor_label,
        condition,
        f"distance-{config['ALIGN_DISTANCE']}",
    ] + [item for item in os.environ.get("WANDB_TAGS", "").split(",") if item]
    run = wandb.init(
        entity=config.get("ENTITY") or None,
        project=config["PROJECT"],
        name=run_name,
        group=run_group,
        tags=tags,
        config=config,
        mode=config["WANDB_MODE"],
    )
    run.define_metric("env_step")
    run.define_metric("*", step_metric="env_step")

    checkpoint_callback = None
    if config["SAVE_CHECKPOINTS"]:
        if int(config["CHECKPOINT_INTERVAL_TIMESTEPS"]) <= 0:
            raise ValueError("CHECKPOINT_INTERVAL_TIMESTEPS must be positive")
        checkpoint_callback, checkpoint_dir = make_checkpoint_callback(config, run)
        print(f"Checkpoints: {checkpoint_dir}", flush=True)
    metrics_jsonl_callback = make_metrics_jsonl_callback(config["METRICS_JSONL"])
    num_updates = (
        int(config["TOTAL_TIMESTEPS"])
        // int(config["NUM_STEPS"])
        // int(config["NUM_ENVS"])
    )
    progress = tqdm(
        total=num_updates,
        desc=f"MAPPO-FF {actor_label} {condition} {env_name} seed={config['SEED']}",
        unit="update",
        dynamic_ncols=True,
    )

    def host_callback(metrics):
        values = {key: float(np.asarray(value)) for key, value in metrics.items()}
        step = int(values["env_step"])
        values["env_step"] = step
        update = int(values["update_step"])
        if values["episodes_completed"] <= 0:
            values.pop("episode_return")
        wandb.log(values)
        progress.update(max(0, update - progress.n))
        postfix = {"steps": f"{step:,}"}
        if "episode_return" in values:
            postfix["return"] = f"{values['episode_return']:.1f}"
        progress.set_postfix(postfix)

    try:
        train, derived = make_train(
            config, host_callback, checkpoint_callback, metrics_jsonl_callback
        )
        # The checkpoint callback closes over ``config``.  Keep its serialized
        # config self-contained by adding the environment-derived dimensions.
        config.update(derived)
        wandb.config.update(
            {
                key: derived[key]
                for key in (
                    "NUM_AGENTS",
                    "NUM_ACTORS",
                    "NUM_UPDATES",
                    "MINIBATCH_SIZE",
                    "ACTION_DIM",
                    "OBS_DIM",
                    "WORLD_STATE_DIM",
                )
            },
            allow_val_change=True,
        )
        rng = jax.random.PRNGKey(int(config["SEED"]))
        device_index = int(config.get("DEVICE", 0))
        devices = jax.devices()
        if not 0 <= device_index < len(devices):
            raise ValueError(f"DEVICE={device_index} is invalid for {devices}")
        output = jax.jit(train, device=devices[device_index])(rng)
        jax.tree.map(
            lambda value: (
                value.block_until_ready()
                if hasattr(value, "block_until_ready")
                else value
            ),
            output,
        )
    finally:
        progress.close()
        run.finish()


if __name__ == "__main__":
    main()
