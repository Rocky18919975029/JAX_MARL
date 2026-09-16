"""Feed-forward MAPPO for JaxMARL's continuous-action MABrax tasks.

The decentralized actor receives a padded local observation.  The centralized
critic receives the underlying Brax global observation (``state.obs``).  The
default Hydra configuration targets ``halfcheetah_6x1`` and uses the tuned
HalfCheetah hyperparameters reported in the JaxMARL paper.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, Mapping, NamedTuple

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
from omegaconf import OmegaConf
from tqdm.auto import tqdm

import jaxmarl
from jaxmarl.wrappers.baselines import JaxMARLWrapper, LogWrapper


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
        log_std = self.param(
            "log_std", nn.initializers.zeros, (self.action_dim,)
        )
        return distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))


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
        return jnp.squeeze(value, axis=-1)


class Transition(NamedTuple):
    global_done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    world_state: jax.Array
    info: Dict[str, jax.Array]


def batchify_observations(
    observations: Mapping[str, jax.Array], agent_list, num_actors: int
) -> jax.Array:
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


def batchify_scalars(
    values: Mapping[str, jax.Array], agent_list, num_actors: int
) -> jax.Array:
    return jnp.stack([values[agent] for agent in agent_list]).reshape(num_actors)


def batchify_world_state(world_state: jax.Array, num_actors: int) -> jax.Array:
    """Converts ``[env, agent, feature]`` to agent-major actor batches."""

    return world_state.swapaxes(0, 1).reshape((num_actors, world_state.shape[-1]))


def batchify_info(info: Mapping[str, jax.Array]) -> Dict[str, jax.Array]:
    """Converts LogWrapper info from env-major to agent-major ordering."""

    return jax.tree.map(
        lambda value: value.swapaxes(0, 1).reshape(
            (-1,) + tuple(value.shape[2:])
        ),
        info,
    )


def unbatchify_actions(
    actions: jax.Array, agent_list, num_envs: int
) -> Dict[str, jax.Array]:
    actions = actions.reshape((len(agent_list), num_envs, actions.shape[-1]))
    return {agent: actions[index] for index, agent in enumerate(agent_list)}


def _validate_and_derive_config(config: Dict[str, Any], env) -> Dict[str, Any]:
    config = dict(config)
    positive = (
        "NUM_ENVS",
        "NUM_STEPS",
        "TOTAL_TIMESTEPS",
        "UPDATE_EPOCHS",
        "NUM_MINIBATCHES",
    )
    for field in positive:
        if int(config[field]) <= 0:
            raise ValueError(f"{field} must be positive")

    action_dims = {env.action_space(agent).shape[0] for agent in env.agents}
    if len(action_dims) != 1:
        raise ValueError(
            "mappo_ff_mabrax currently requires equal action dimensions across "
            "agents. halfcheetah_6x1 satisfies this requirement."
        )

    config["NUM_ACTORS"] = env.num_agents * int(config["NUM_ENVS"])
    config["NUM_UPDATES"] = (
        int(config["TOTAL_TIMESTEPS"])
        // int(config["NUM_STEPS"])
        // int(config["NUM_ENVS"])
    )
    if config["NUM_UPDATES"] <= 0:
        raise ValueError("TOTAL_TIMESTEPS is smaller than one rollout batch")

    batch_size = config["NUM_ACTORS"] * int(config["NUM_STEPS"])
    if batch_size % int(config["NUM_MINIBATCHES"]) != 0:
        raise ValueError(
            "NUM_STEPS * NUM_ENVS * num_agents must be divisible by "
            "NUM_MINIBATCHES"
        )
    config["MINIBATCH_SIZE"] = batch_size // int(config["NUM_MINIBATCHES"])
    config["ACTION_DIM"] = next(iter(action_dims))
    config["OBS_DIM"] = max(
        env.observation_space(agent).shape[0] for agent in env.agents
    )
    config["WORLD_STATE_DIM"] = env.world_state_size()
    return config


def make_train(config: Dict[str, Any], host_callback):
    raw_env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = MABraxWorldStateWrapper(raw_env)
    config = _validate_and_derive_config(config, env)
    env = LogWrapper(env, replace_info=True)

    actor_network = ActorFF(
        action_dim=config["ACTION_DIM"],
        hidden_size=int(config["HIDDEN_SIZE"]),
        activation=config["ACTIVATION"],
    )
    critic_network = CriticFF(
        hidden_size=int(config["HIDDEN_SIZE"]),
        activation=config["ACTIVATION"],
    )

    optimizer_steps_per_update = int(config["UPDATE_EPOCHS"]) * int(
        config["NUM_MINIBATCHES"]
    )

    def learning_rate_schedule(count):
        completed_updates = count // optimizer_steps_per_update
        fraction = 1.0 - completed_updates / config["NUM_UPDATES"]
        return config["LR"] * jnp.maximum(fraction, 0.0)

    learning_rate = (
        learning_rate_schedule if config["ANNEAL_LR"] else config["LR"]
    )
    actor_tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(learning_rate=learning_rate, eps=1e-5),
    )
    critic_tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(learning_rate=learning_rate, eps=1e-5),
    )

    def train(rng):
        rng, actor_key, critic_key, reset_key = jax.random.split(rng, 4)
        actor_params = actor_network.init(
            actor_key, jnp.zeros((1, config["OBS_DIM"]))
        )
        critic_params = critic_network.init(
            critic_key, jnp.zeros((1, config["WORLD_STATE_DIM"]))
        )
        actor_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_params,
            tx=actor_tx,
        )
        critic_state = TrainState.create(
            apply_fn=critic_network.apply,
            params=critic_params,
            tx=critic_tx,
        )

        reset_keys = jax.random.split(reset_key, int(config["NUM_ENVS"]))
        obs, env_state = jax.vmap(env.reset)(reset_keys)

        def update_step(runner_state, _):
            actor_state, critic_state, env_state, last_obs, rng, update_count = (
                runner_state
            )

            def env_step(step_state, _):
                actor_state, critic_state, env_state, last_obs, rng = step_state
                obs_batch = batchify_observations(
                    last_obs, env.agents, config["NUM_ACTORS"]
                )
                world_state = batchify_world_state(
                    last_obs["world_state"], config["NUM_ACTORS"]
                )

                rng, action_key, step_key = jax.random.split(rng, 3)
                policy = actor_network.apply(actor_state.params, obs_batch)
                action = policy.sample(seed=action_key)
                log_prob = policy.log_prob(action)
                value = critic_network.apply(critic_state.params, world_state)
                env_actions = unbatchify_actions(
                    action, env.agents, int(config["NUM_ENVS"])
                )

                step_keys = jax.random.split(step_key, int(config["NUM_ENVS"]))
                next_obs, next_env_state, reward, done, info = jax.vmap(env.step)(
                    step_keys, env_state, env_actions
                )
                transition = Transition(
                    global_done=jnp.tile(done["__all__"], env.num_agents),
                    action=action,
                    value=value,
                    reward=batchify_scalars(
                        reward, env.agents, config["NUM_ACTORS"]
                    ),
                    log_prob=log_prob,
                    obs=obs_batch,
                    world_state=world_state,
                    info=batchify_info(info),
                )
                next_step_state = (
                    actor_state,
                    critic_state,
                    next_env_state,
                    next_obs,
                    rng,
                )
                return next_step_state, transition

            step_state = (actor_state, critic_state, env_state, last_obs, rng)
            step_state, trajectory = jax.lax.scan(
                env_step, step_state, None, int(config["NUM_STEPS"])
            )
            actor_state, critic_state, env_state, last_obs, rng = step_state

            last_world_state = batchify_world_state(
                last_obs["world_state"], config["NUM_ACTORS"]
            )
            last_value = critic_network.apply(
                critic_state.params, last_world_state
            )

            def gae_step(carry, transition):
                gae, next_value = carry
                not_done = 1.0 - transition.global_done
                delta = (
                    transition.reward
                    + config["GAMMA"] * next_value * not_done
                    - transition.value
                )
                gae = (
                    delta
                    + config["GAMMA"]
                    * config["GAE_LAMBDA"]
                    * not_done
                    * gae
                )
                return (gae, transition.value), gae

            _, advantages = jax.lax.scan(
                gae_step,
                (jnp.zeros_like(last_value), last_value),
                trajectory,
                reverse=True,
                unroll=8,
            )
            targets = advantages + trajectory.value

            flat_batch_size = int(config["NUM_STEPS"]) * config["NUM_ACTORS"]
            flat_batch = jax.tree.map(
                lambda value: value.reshape(
                    (flat_batch_size,) + tuple(value.shape[2:])
                ),
                (trajectory, advantages, targets),
            )

            def update_epoch(epoch_state, _):
                actor_state, critic_state, rng = epoch_state
                rng, permutation_key = jax.random.split(rng)
                permutation = jax.random.permutation(
                    permutation_key, flat_batch_size
                )
                shuffled = jax.tree.map(
                    lambda value: jnp.take(value, permutation, axis=0),
                    flat_batch,
                )
                minibatches = jax.tree.map(
                    lambda value: value.reshape(
                        (
                            int(config["NUM_MINIBATCHES"]),
                            config["MINIBATCH_SIZE"],
                        )
                        + tuple(value.shape[1:])
                    ),
                    shuffled,
                )

                def update_minibatch(train_states, minibatch):
                    actor_state, critic_state = train_states
                    minibatch_trajectory, minibatch_advantages, minibatch_targets = (
                        minibatch
                    )

                    def actor_loss_fn(params):
                        policy = actor_network.apply(
                            params, minibatch_trajectory.obs
                        )
                        new_log_prob = policy.log_prob(
                            minibatch_trajectory.action
                        )
                        log_ratio = new_log_prob - minibatch_trajectory.log_prob
                        ratio = jnp.exp(log_ratio)
                        normalized_advantage = (
                            minibatch_advantages
                            - minibatch_advantages.mean()
                        ) / (minibatch_advantages.std() + 1e-8)
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
                        loss = policy_loss - config["ENT_COEF"] * entropy
                        approx_kl = ((ratio - 1.0) - log_ratio).mean()
                        clip_fraction = (
                            jnp.abs(ratio - 1.0) > config["CLIP_EPS"]
                        ).mean()
                        return loss, (
                            policy_loss,
                            entropy,
                            approx_kl,
                            clip_fraction,
                        )

                    def critic_loss_fn(params):
                        value = critic_network.apply(
                            params, minibatch_trajectory.world_state
                        )
                        clipped_value = minibatch_trajectory.value + jnp.clip(
                            value - minibatch_trajectory.value,
                            -config["CLIP_EPS"],
                            config["CLIP_EPS"],
                        )
                        value_loss = jnp.square(
                            value - minibatch_targets
                        )
                        clipped_value_loss = jnp.square(
                            clipped_value - minibatch_targets
                        )
                        value_loss = 0.5 * jnp.maximum(
                            value_loss, clipped_value_loss
                        ).mean()
                        return config["VF_COEF"] * value_loss, value_loss

                    (actor_loss, actor_aux), actor_grads = jax.value_and_grad(
                        actor_loss_fn, has_aux=True
                    )(actor_state.params)
                    (critic_loss, value_loss), critic_grads = jax.value_and_grad(
                        critic_loss_fn, has_aux=True
                    )(critic_state.params)
                    actor_state = actor_state.apply_gradients(grads=actor_grads)
                    critic_state = critic_state.apply_gradients(grads=critic_grads)
                    loss_info = {
                        "total_loss": actor_loss + critic_loss,
                        "actor_loss": actor_aux[0],
                        "value_loss": value_loss,
                        "entropy": actor_aux[1],
                        "approx_kl": actor_aux[2],
                        "clip_fraction": actor_aux[3],
                    }
                    return (actor_state, critic_state), loss_info

                (actor_state, critic_state), loss_info = jax.lax.scan(
                    update_minibatch,
                    (actor_state, critic_state),
                    minibatches,
                )
                return (actor_state, critic_state, rng), loss_info

            (actor_state, critic_state, rng), loss_info = jax.lax.scan(
                update_epoch,
                (actor_state, critic_state, rng),
                None,
                int(config["UPDATE_EPOCHS"]),
            )
            loss_info = jax.tree.map(lambda value: value.mean(), loss_info)

            completed = trajectory.info["returned_episode"]
            completed_count = completed.sum()
            episode_return = (
                trajectory.info["returned_episode_returns"] * completed
            ).sum() / jnp.maximum(completed_count, 1.0)
            update_count = update_count + 1
            metrics = {
                **loss_info,
                "episode_return": episode_return,
                "episodes_completed": completed_count / env.num_agents,
                "mean_step_reward": trajectory.reward.mean(),
                "update_step": update_count,
                "env_step": update_count
                * int(config["NUM_STEPS"])
                * int(config["NUM_ENVS"]),
            }
            jax.debug.callback(host_callback, metrics, ordered=True)
            runner_state = (
                actor_state,
                critic_state,
                env_state,
                last_obs,
                rng,
                update_count,
            )
            return runner_state, metrics

        initial_state = (
            actor_state,
            critic_state,
            env_state,
            obs,
            rng,
            jnp.array(0, dtype=jnp.int32),
        )
        final_state, metrics = jax.lax.scan(
            update_step,
            initial_state,
            None,
            config["NUM_UPDATES"],
        )
        return {"runner_state": final_state, "metrics": metrics}

    return train, config


@hydra.main(version_base=None, config_path="config", config_name="mappo_ff_mabrax")
def main(hydra_config):
    config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(config, dict):
        raise TypeError("Expected a dictionary-like Hydra configuration")

    env_name = str(config["ENV_NAME"])
    run_name = config.get("WANDB_NAME") or (
        f"MAPPO-FF-{env_name}-seed{int(config['SEED'])}"
    )
    run = wandb.init(
        entity=config.get("ENTITY") or None,
        project=config["PROJECT"],
        name=run_name,
        group=config.get("WANDB_GROUP") or f"MAPPO-FF-{env_name}",
        tags=["MAPPO", "FF", "MABrax", env_name],
        config=config,
        mode=config["WANDB_MODE"],
    )

    num_updates = (
        int(config["TOTAL_TIMESTEPS"])
        // int(config["NUM_STEPS"])
        // int(config["NUM_ENVS"])
    )
    progress = tqdm(
        total=num_updates,
        desc=f"MAPPO-FF {env_name} seed={int(config['SEED'])}",
        unit="update",
        dynamic_ncols=True,
    )

    def host_callback(metrics):
        values = {
            key: float(np.asarray(value)) for key, value in metrics.items()
        }
        step = int(values.pop("env_step"))
        update = int(values["update_step"])
        if values["episodes_completed"] <= 0:
            values.pop("episode_return")
        wandb.log(values, step=step)
        progress.update(max(0, update - progress.n))
        postfix = {"steps": f"{step:,}"}
        if "episode_return" in values:
            postfix["return"] = f"{values['episode_return']:.1f}"
        progress.set_postfix(postfix)

    try:
        train, derived_config = make_train(config, host_callback)
        wandb.config.update(
            {
                key: derived_config[key]
                for key in (
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
            raise ValueError(
                f"DEVICE={device_index} is invalid for visible devices {devices}"
            )
        train_jit = jax.jit(train, device=devices[device_index])
        output = train_jit(rng)
        jax.tree.map(
            lambda value: value.block_until_ready()
            if hasattr(value, "block_until_ready")
            else value,
            output,
        )
    finally:
        progress.close()
        run.finish()


if __name__ == "__main__":
    main()
