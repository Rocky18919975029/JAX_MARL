"""
Based on PureJaxRL Implementation of IPPO, with changes to give a centralised critic.
"""

import functools
import json
import os
import re
from functools import partial
from pathlib import Path
from typing import Dict, NamedTuple, Sequence

import jax

# TensorFlow Probability still probes this legacy location when imported by
# Distrax. JAX 0.10 removed it from jax.interpreters.xla but keeps the mapping
# in jax.core. Providing the alias before importing Distrax avoids requiring a
# manual edit inside site-packages.
try:
    jax.interpreters.xla.pytype_aval_mappings
except AttributeError:
    jax.interpreters.xla.pytype_aval_mappings = jax.core.pytype_aval_mappings

import distrax
import flax.linen as nn
import hydra
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jax.experimental import io_callback
from omegaconf import OmegaConf

import wandb
from tqdm.auto import tqdm
from jaxmarl.environments.smax import HeuristicEnemySMAX, map_name_to_scenario
from jaxmarl.wrappers.baselines import JaxMARLWrapper, SMAXLogWrapper, save_params


def _safe_path_component(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-_") or "run"


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def make_checkpoint_callback(config, run):
    """Create a host callback that writes evaluation-ready checkpoints."""

    checkpoint_root = Path(config["CHECKPOINT_DIR"]).expanduser().resolve()
    project_name = _safe_path_component(config.get("PROJECT") or "local")
    run_name = _safe_path_component(run.name)
    run_id = _safe_path_component(run.id)
    run_dir = checkpoint_root / project_name / f"{run_name}-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    def write_json_atomic(path, payload):
        temporary_path = path.with_name(f".{path.name}.tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True, default=_json_default)
            file.write("\n")
        os.replace(temporary_path, path)

    def checkpoint_callback(actor_params, critic_params, env_step, is_final):
        env_step = int(np.asarray(env_step).item())
        is_final = bool(np.asarray(is_final).item())
        checkpoint_name = "final" if is_final else f"step_{env_step:012d}"
        checkpoint_dir = run_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model_path = checkpoint_dir / "model.safetensors"
        temporary_model_path = checkpoint_dir / ".model.tmp.safetensors"
        save_params(
            {"actor": actor_params, "critic": critic_params},
            temporary_model_path,
        )
        os.replace(temporary_model_path, model_path)

        checkpoint_config = dict(config)
        checkpoint_config["CHECKPOINT_ENV_STEP"] = env_step
        checkpoint_config["CHECKPOINT_IS_FINAL"] = is_final
        write_json_atomic(checkpoint_dir / "config.json", checkpoint_config)
        write_json_atomic(
            checkpoint_dir / "metadata.json",
            {
                "format_version": 1,
                "env_step": env_step,
                "is_final": is_final,
                "map_name": config["MAP_NAME"],
                "seed": config["SEED"],
                "actor_parameter_sharing": config["ACTOR_PARAMETER_SHARING"],
                "matched_comparison": config["MATCHED_COMPARISON"],
                "align_mode": config["ALIGN_MODE"],
                "alignment_coef": config["ALIGNMENT_COEF"],
                "wandb_project": run.project,
                "wandb_run_id": run.id,
                "wandb_run_name": run.name,
            },
        )
        write_json_atomic(
            run_dir / "latest.json",
            {
                "checkpoint": checkpoint_name,
                "env_step": env_step,
                "is_final": is_final,
            },
        )
        print(f"Checkpoint saved: {checkpoint_dir}", flush=True)

        if config["WANDB_UPLOAD_CHECKPOINTS"]:
            artifact = wandb.Artifact(
                f"{run_name}-{run_id}-checkpoint",
                type="model",
                metadata={
                    "env_step": env_step,
                    "is_final": is_final,
                    "map_name": config["MAP_NAME"],
                    "seed": config["SEED"],
                },
            )
            artifact.add_dir(str(checkpoint_dir))
            aliases = ["latest", f"step-{env_step}"]
            if is_final:
                aliases.append("final")
            run.log_artifact(artifact, aliases=aliases)

        return np.int32(0)

    return checkpoint_callback, run_dir


class SMAXWorldStateWrapper(JaxMARLWrapper):
    """
    Provides a `"world_state"` observation for the centralised critic.
    world state observation of dimension: (num_agents, world_state_size)
    """

    def __init__(
        self,
        env: HeuristicEnemySMAX,
        obs_with_agent_id=True,
    ):
        super().__init__(env)
        self.obs_with_agent_id = obs_with_agent_id

        if not self.obs_with_agent_id:
            self._world_state_size = self._env.state_size
            self.world_state_fn = self.ws_just_env_state
        else:
            self._world_state_size = self._env.state_size + self._env.num_allies
            self.world_state_fn = self.ws_with_agent_id

    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state_fn(obs, env_state)
        return obs, env_state

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, action):
        obs, env_state, reward, done, info = self._env.step(key, state, action)
        obs["world_state"] = self.world_state_fn(obs, state)
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def ws_just_env_state(self, obs, state):
        # return all_obs
        world_state = obs["world_state"]
        world_state = world_state[None].repeat(self._env.num_allies, axis=0)
        return world_state

    @partial(jax.jit, static_argnums=0)
    def ws_with_agent_id(self, obs, state):
        # all_obs = jnp.array([obs[agent] for agent in self._env.agents])
        world_state = obs["world_state"]
        world_state = world_state[None].repeat(self._env.num_allies, axis=0)
        one_hot = jnp.eye(self._env.num_allies)
        return jnp.concatenate((world_state, one_hot), axis=1)

    def world_state_size(self):

        return self._world_state_size


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        # print('ins', ins)
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, avail_actions = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, actor_latent = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(actor_latent)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)

        pi = distrax.Categorical(logits=action_logits)

        return hidden, pi, actor_latent


class CriticRNN(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(world_state)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, critic_latent = ScannedRNN()(hidden, rnn_in)

        critic = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(critic_latent)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, jnp.squeeze(critic, axis=-1), critic_latent


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    actor_latent_old: jnp.ndarray
    critic_latent_old: jnp.ndarray
    alive_mask: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    # print('batchify', x.shape)
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def vmapped_optimizer(tx):
    """Apply one optimizer independently to every leading parameter slice."""

    def init_fn(params):
        return jax.vmap(tx.init)(params)

    def update_fn(updates, state, params=None):
        if params is None:
            return jax.vmap(lambda g, s: tx.update(g, s))(updates, state)
        return jax.vmap(tx.update)(updates, state, params)

    return optax.GradientTransformation(init_fn, update_fn)


def tree_l2_norm(tree):
    return jnp.sqrt(
        sum(jnp.sum(jnp.square(x)) for x in jax.tree.leaves(tree))
    )


def latent_distance(source, target, mask):
    """Masked MSE after parameter-free, per-sample layer normalization."""

    def normalize(x):
        mean = x.mean(axis=-1, keepdims=True)
        variance = jnp.square(x - mean).mean(axis=-1, keepdims=True)
        return (x - mean) * jax.lax.rsqrt(variance + 1e-5)

    distance = jnp.square(normalize(source) - normalize(target)).mean(axis=-1)
    mask = mask.astype(distance.dtype)
    return (distance * mask).sum() / jnp.maximum(mask.sum(), 1.0)


def make_train(config, progress_bar=None, checkpoint_callback=None):
    scenario = map_name_to_scenario(config["MAP_NAME"])
    env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    valid_align_modes = {"none", "c_to_a", "a_to_c", "reciprocal", "joint"}
    if config["ALIGN_MODE"] not in valid_align_modes:
        raise ValueError(
            f"ALIGN_MODE must be one of {sorted(valid_align_modes)}, got "
            f"{config['ALIGN_MODE']!r}."
        )
    if config["ALIGN_MODE"] != "none" and not config["MATCHED_COMPARISON"]:
        raise ValueError(
            "Representation alignment requires MATCHED_COMPARISON=true."
        )
    if (
        (not config["ACTOR_PARAMETER_SHARING"] or config["MATCHED_COMPARISON"])
        and config["NUM_ENVS"] % config["NUM_MINIBATCHES"] != 0
    ):
        raise ValueError(
            "NUM_ENVS must be divisible by NUM_MINIBATCHES for independent actors "
            "or the matched comparison protocol."
        )
    config["CLIP_EPS"] = (
        config["CLIP_EPS"] / env.num_agents
        if config["SCALE_CLIP_EPS"]
        else config["CLIP_EPS"]
    )

    env = SMAXWorldStateWrapper(env, config["OBS_WITH_AGENT_ID"])
    env = SMAXLogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        actor_network = ActorRNN(env.action_space(env.agents[0]).n, config=config)
        critic_network = CriticRNN(config=config)
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        ac_init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
            jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).n)),
        )
        ac_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )
        if (
            not config["ACTOR_PARAMETER_SHARING"]
            and not config["MATCHED_COMPARISON"]
        ):
            actor_rngs = jax.random.split(_rng_actor, env.num_agents)
            actor_network_params = jax.vmap(
                actor_network.init, in_axes=(0, None, None)
            )(actor_rngs, ac_init_hstate, ac_init_x)
        else:
            actor_network_params = actor_network.init(
                _rng_actor, ac_init_hstate, ac_init_x
            )
            if not config["ACTOR_PARAMETER_SHARING"]:
                actor_network_params = jax.tree.map(
                    lambda x: jnp.repeat(x[None, ...], env.num_agents, axis=0),
                    actor_network_params,
                )
        cr_init_x = (
            jnp.zeros(
                (
                    1,
                    config["NUM_ENVS"],
                    env.world_state_size(),
                )
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        cr_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )
        critic_network_params = critic_network.init(
            _rng_critic, cr_init_hstate, cr_init_x
        )

        if config["ANNEAL_LR"]:
            base_actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            base_actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        actor_tx = (
            base_actor_tx
            if config["ACTOR_PARAMETER_SHARING"]
            else vmapped_optimizer(base_actor_tx)
        )
        actor_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network_params,
            tx=actor_tx,
        )
        critic_train_state = TrainState.create(
            apply_fn=critic_network.apply,
            params=critic_network_params,
            tx=critic_tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        ac_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )
        cr_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, last_done, hstates, rng = (
                    runner_state
                )

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                )
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                    avail_actions,
                )
                if (
                    config["ACTOR_PARAMETER_SHARING"]
                    and not config["MATCHED_COMPARISON"]
                ):
                    ac_hstate, pi, actor_latent = actor_network.apply(
                        train_states[0].params, hstates[0], ac_in
                    )
                    action = pi.sample(seed=_rng)
                    log_prob = pi.log_prob(action)
                else:
                    actor_hstate = hstates[0].reshape(
                        (
                            env.num_agents,
                            config["NUM_ENVS"],
                            config["GRU_HIDDEN_DIM"],
                        )
                    )
                    actor_in = (
                        obs_batch.reshape(
                            (env.num_agents, config["NUM_ENVS"], -1)
                        )[:, None, ...],
                        last_done.reshape(
                            (env.num_agents, config["NUM_ENVS"])
                        )[:, None, ...],
                        avail_actions.reshape(
                            (env.num_agents, config["NUM_ENVS"], -1)
                        ),
                    )
                    actor_rngs = jax.random.split(_rng, env.num_agents)

                    def apply_and_sample(params, hidden, inputs, sample_rng):
                        hidden, pi, latent = actor_network.apply(
                            params, hidden, inputs
                        )
                        action = pi.sample(seed=sample_rng)
                        return hidden, action, pi.log_prob(action), latent

                    parameter_axis = (
                        None if config["ACTOR_PARAMETER_SHARING"] else 0
                    )
                    ac_hstate, action, log_prob, actor_latent = jax.vmap(
                        apply_and_sample,
                        in_axes=(parameter_axis, 0, 0, 0),
                    )(
                        train_states[0].params,
                        actor_hstate,
                        actor_in,
                        actor_rngs,
                    )
                    action = action.reshape((1, config["NUM_ACTORS"]))
                    log_prob = log_prob.reshape((1, config["NUM_ACTORS"]))
                    actor_latent = actor_latent.reshape(
                        (1, config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])
                    )
                    ac_hstate = ac_hstate.reshape(
                        (config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])
                    )
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # VALUE
                # output of wrapper is (num_envs, num_agents, world_state_size)
                # swap axes to (num_agents, num_envs, world_state_size) before reshaping to (num_actors, world_state_size)
                world_state = last_obs["world_state"].swapaxes(0, 1)
                world_state = world_state.reshape((config["NUM_ACTORS"], -1))

                cr_in = (
                    world_state[None, :],
                    last_done[np.newaxis, :],
                )
                cr_hstate, value, critic_latent = critic_network.apply(
                    train_states[1].params, hstates[1], cr_in
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    world_state,
                    jax.lax.stop_gradient(actor_latent.squeeze(axis=0)),
                    jax.lax.stop_gradient(critic_latent.squeeze(axis=0)),
                    jnp.sum(avail_actions, axis=-1) > 1,
                    info,
                    avail_actions,
                )
                runner_state = (
                    train_states,
                    env_state,
                    obsv,
                    done_batch,
                    (ac_hstate, cr_hstate),
                    rng,
                )
                return runner_state, transition

            initial_hstates = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_states, env_state, last_obs, last_done, hstates, rng = runner_state

            last_world_state = last_obs["world_state"].swapaxes(0, 1)
            last_world_state = last_world_state.reshape((config["NUM_ACTORS"], -1))

            cr_in = (
                last_world_state[None, :],
                last_done[np.newaxis, :],
            )
            _, last_val, _ = critic_network.apply(
                train_states[1].params, hstates[1], cr_in
            )
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states
                    ac_init_hstate, cr_init_hstate, traj_batch, advantages, targets = (
                        batch_info
                    )

                    def _actor_loss_fn(actor_params, init_hstate, traj_batch, gae):
                        # RERUN NETWORK
                        _, pi, actor_latent = actor_network.apply(
                            actor_params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done, traj_batch.avail_actions),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        if not config["MATCHED_COMPARISON"]:
                            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        # debug
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                        actor_loss = loss_actor - config["ENT_COEF"] * entropy

                        return actor_loss, (
                            loss_actor,
                            entropy,
                            ratio,
                            approx_kl,
                            clip_frac,
                            actor_latent,
                        )

                    def _critic_loss_fn(
                        critic_params, init_hstate, traj_batch, targets
                    ):
                        # RERUN NETWORK
                        _, value, critic_latent = critic_network.apply(
                            critic_params,
                            init_hstate.squeeze(),
                            (traj_batch.world_state, traj_batch.done),
                        )

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        critic_loss = config["VF_COEF"] * value_loss
                        return critic_loss, (value_loss, critic_latent)

                    def split_agents(x):
                        x = x.reshape(
                            (
                                x.shape[0],
                                env.num_agents,
                                -1,
                                *x.shape[2:],
                            )
                        )
                        return jnp.swapaxes(x, 0, 1)

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    if config["MATCHED_COMPARISON"]:
                        actor_advantages = (
                            advantages - advantages.mean()
                        ) / (advantages.std() + 1e-8)
                    else:
                        actor_advantages = advantages

                    if config["MATCHED_COMPARISON"]:
                        actor_batch = jax.tree.map(split_agents, traj_batch)
                        actor_hstates = split_agents(ac_init_hstate)
                        actor_advantages = split_agents(actor_advantages)
                        parameter_axis = (
                            None if config["ACTOR_PARAMETER_SHARING"] else 0
                        )

                        def matched_total_loss(actor_params, critic_params):
                            actor_losses, actor_aux = jax.vmap(
                                _actor_loss_fn,
                                in_axes=(parameter_axis, 0, 0, 0),
                            )(
                                actor_params,
                                actor_hstates,
                                actor_batch,
                                actor_advantages,
                            )
                            critic_rl_loss, critic_aux = _critic_loss_fn(
                                critic_params,
                                cr_init_hstate,
                                traj_batch,
                                targets,
                            )

                            actor_latent = actor_aux[5].swapaxes(0, 1).reshape(
                                traj_batch.actor_latent_old.shape
                            )
                            critic_latent = critic_aux[1]
                            mask = traj_batch.alive_mask
                            c_to_a_loss = latent_distance(
                                actor_latent,
                                jax.lax.stop_gradient(
                                    traj_batch.critic_latent_old
                                ),
                                mask,
                            )
                            a_to_c_loss = latent_distance(
                                critic_latent,
                                jax.lax.stop_gradient(
                                    traj_batch.actor_latent_old
                                ),
                                mask,
                            )
                            joint_loss = latent_distance(
                                actor_latent, critic_latent, mask
                            )
                            zero = jnp.zeros((), dtype=actor_latent.dtype)
                            actor_alignment = zero
                            critic_alignment = zero
                            joint_alignment = zero
                            if config["ALIGN_MODE"] == "c_to_a":
                                actor_alignment = c_to_a_loss
                            elif config["ALIGN_MODE"] == "a_to_c":
                                critic_alignment = a_to_c_loss
                            elif config["ALIGN_MODE"] == "reciprocal":
                                actor_alignment = c_to_a_loss
                                critic_alignment = a_to_c_loss
                            elif config["ALIGN_MODE"] == "joint":
                                joint_alignment = joint_loss

                            alignment_objective = (
                                actor_alignment
                                + critic_alignment
                                + joint_alignment
                            )
                            total_loss = (
                                actor_losses.mean()
                                + critic_rl_loss
                                + config["ALIGNMENT_COEF"]
                                * alignment_objective
                            )
                            return total_loss, (
                                actor_losses,
                                actor_aux,
                                critic_rl_loss,
                                critic_aux,
                                c_to_a_loss,
                                a_to_c_loss,
                                joint_loss,
                                actor_alignment,
                                critic_alignment,
                                joint_alignment,
                            )

                        (
                            (combined_total_loss, matched_aux),
                            (actor_grads, critic_grads),
                        ) = jax.value_and_grad(
                            matched_total_loss,
                            argnums=(0, 1),
                            has_aux=True,
                        )(
                            actor_train_state.params,
                            critic_train_state.params,
                        )
                        if not config["ACTOR_PARAMETER_SHARING"]:
                            actor_grads = jax.tree.map(
                                lambda x: x * env.num_agents, actor_grads
                            )
                        actor_loss = (matched_aux[0], matched_aux[1])
                        critic_loss = (matched_aux[2], matched_aux[3])
                        c_to_a_loss = matched_aux[4]
                        a_to_c_loss = matched_aux[5]
                        current_joint_loss = matched_aux[6]
                        actor_alignment_loss = matched_aux[7]
                        critic_alignment_loss = matched_aux[8]
                        joint_alignment_loss = matched_aux[9]
                    elif config["ACTOR_PARAMETER_SHARING"]:
                        actor_loss, actor_grads = actor_grad_fn(
                            actor_train_state.params,
                            ac_init_hstate,
                            traj_batch,
                            actor_advantages,
                        )
                        critic_loss, critic_grads = jax.value_and_grad(
                            _critic_loss_fn, has_aux=True
                        )(
                            critic_train_state.params,
                            cr_init_hstate,
                            traj_batch,
                            targets,
                        )
                        combined_total_loss = actor_loss[0] + critic_loss[0]
                        zero = jnp.zeros((), dtype=advantages.dtype)
                        c_to_a_loss = zero
                        a_to_c_loss = zero
                        current_joint_loss = zero
                        actor_alignment_loss = zero
                        critic_alignment_loss = zero
                        joint_alignment_loss = zero
                    else:
                        actor_batch = jax.tree.map(split_agents, traj_batch)
                        actor_hstates = split_agents(ac_init_hstate)
                        actor_advantages = split_agents(actor_advantages)
                        actor_loss, actor_grads = jax.vmap(actor_grad_fn)(
                            actor_train_state.params,
                            actor_hstates,
                            actor_batch,
                            actor_advantages,
                        )
                        critic_loss, critic_grads = jax.value_and_grad(
                            _critic_loss_fn, has_aux=True
                        )(
                            critic_train_state.params,
                            cr_init_hstate,
                            traj_batch,
                            targets,
                        )
                        combined_total_loss = (
                            actor_loss[0].mean() + critic_loss[0]
                        )
                        zero = jnp.zeros((), dtype=advantages.dtype)
                        c_to_a_loss = zero
                        a_to_c_loss = zero
                        current_joint_loss = zero
                        actor_alignment_loss = zero
                        critic_alignment_loss = zero
                        joint_alignment_loss = zero

                    if config["ACTOR_PARAMETER_SHARING"]:
                        actor_grad_norms = jnp.asarray(
                            [tree_l2_norm(actor_grads)]
                        )
                    else:
                        actor_grad_norms = jax.vmap(tree_l2_norm)(actor_grads)
                    critic_grad_norm = tree_l2_norm(critic_grads)

                    old_actor_params = actor_train_state.params
                    old_critic_params = critic_train_state.params
                    actor_train_state = actor_train_state.apply_gradients(
                        grads=actor_grads
                    )
                    critic_train_state = critic_train_state.apply_gradients(
                        grads=critic_grads
                    )
                    actor_param_updates = jax.tree.map(
                        lambda new, old: new - old,
                        actor_train_state.params,
                        old_actor_params,
                    )
                    critic_param_updates = jax.tree.map(
                        lambda new, old: new - old,
                        critic_train_state.params,
                        old_critic_params,
                    )
                    if config["ACTOR_PARAMETER_SHARING"]:
                        actor_update_norms = jnp.asarray(
                            [tree_l2_norm(actor_param_updates)]
                        )
                    else:
                        actor_update_norms = jax.vmap(tree_l2_norm)(
                            actor_param_updates
                        )
                    actor_grad_norms_after_clip = jnp.minimum(
                        actor_grad_norms, config["MAX_GRAD_NORM"]
                    )

                    mean_actor_loss = actor_loss[0].mean()
                    loss_info = {
                        "total_loss": combined_total_loss,
                        "actor_loss": mean_actor_loss,
                        "value_loss": critic_loss[1][0],
                        "entropy": actor_loss[1][1].mean(),
                        "ratio": actor_loss[1][2],
                        "approx_kl": actor_loss[1][3].mean(),
                        "clip_frac": actor_loss[1][4].mean(),
                        "alignment_c_to_a_distance": c_to_a_loss,
                        "alignment_a_to_c_distance": a_to_c_loss,
                        "alignment_current_joint_distance": current_joint_loss,
                        "alignment_actor_objective": actor_alignment_loss,
                        "alignment_critic_objective": critic_alignment_loss,
                        "alignment_joint_objective": joint_alignment_loss,
                        "actor_grad_norm_mean": actor_grad_norms.mean(),
                        "actor_grad_norm_max": actor_grad_norms.max(),
                        "actor_grad_norm_after_clip_mean": (
                            actor_grad_norms_after_clip.mean()
                        ),
                        "actor_grad_norm_after_clip_max": (
                            actor_grad_norms_after_clip.max()
                        ),
                        "actor_grad_clipped_fraction": jnp.mean(
                            actor_grad_norms > config["MAX_GRAD_NORM"]
                        ),
                        "actor_update_norm_mean": actor_update_norms.mean(),
                        "actor_update_norm_max": actor_update_norms.max(),
                        "critic_grad_norm": critic_grad_norm,
                        "critic_grad_norm_after_clip": jnp.minimum(
                            critic_grad_norm, config["MAX_GRAD_NORM"]
                        ),
                        "critic_grad_clipped": (
                            critic_grad_norm > config["MAX_GRAD_NORM"]
                        ),
                        "critic_update_norm": tree_l2_norm(
                            critic_param_updates
                        ),
                    }

                    if config["MATCHED_COMPARISON"]:
                        raw_advantages_by_agent = split_agents(advantages)
                        loss_info["advantage_global_mean"] = advantages.mean()
                        loss_info["advantage_global_std"] = advantages.std()
                        for agent_idx in range(env.num_agents):
                            agent_advantages = raw_advantages_by_agent[agent_idx]
                            loss_info[f"advantage_mean_agent_{agent_idx}"] = (
                                agent_advantages.mean()
                            )
                            loss_info[f"advantage_std_agent_{agent_idx}"] = (
                                agent_advantages.std()
                            )

                    return (actor_train_state, critic_train_state), loss_info

                (
                    train_states,
                    init_hstates,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                init_hstates = jax.tree.map(
                    lambda x: jnp.reshape(x, (1, config["NUM_ACTORS"], -1)),
                    init_hstates,
                )

                batch = (
                    init_hstates[0],
                    init_hstates[1],
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                if (
                    config["ACTOR_PARAMETER_SHARING"]
                    and not config["MATCHED_COMPARISON"]
                ):
                    permutation = jax.random.permutation(
                        _rng, config["NUM_ACTORS"]
                    )
                    shuffled_batch = jax.tree.map(
                        lambda x: jnp.take(x, permutation, axis=1), batch
                    )
                    minibatches = jax.tree.map(
                        lambda x: jnp.swapaxes(
                            jnp.reshape(
                                x,
                                [x.shape[0], config["NUM_MINIBATCHES"], -1]
                                + list(x.shape[2:]),
                            ),
                            1,
                            0,
                        ),
                        shuffled_batch,
                    )
                else:
                    # Keep every agent represented in every minibatch. Matched
                    # shared and independent runs therefore give the critic the
                    # same samples in the same update order.
                    permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                    envs_per_minibatch = (
                        config["NUM_ENVS"] // config["NUM_MINIBATCHES"]
                    )

                    def make_stratified_minibatches(x):
                        x = x.reshape(
                            (
                                x.shape[0],
                                env.num_agents,
                                config["NUM_ENVS"],
                                *x.shape[2:],
                            )
                        )
                        x = jnp.take(x, permutation, axis=2)
                        x = x.reshape(
                            (
                                x.shape[0],
                                env.num_agents,
                                config["NUM_MINIBATCHES"],
                                envs_per_minibatch,
                                *x.shape[3:],
                            )
                        )
                        x = jnp.moveaxis(x, 2, 0)
                        return x.reshape(
                            (
                                config["NUM_MINIBATCHES"],
                                x.shape[1],
                                env.num_agents * envs_per_minibatch,
                                *x.shape[4:],
                            )
                        )

                    minibatches = jax.tree.map(
                        make_stratified_minibatches, batch
                    )

                # train_states = (actor_train_state, critic_train_state)
                train_states, loss_info = jax.lax.scan(
                    _update_minbatch, train_states, minibatches
                )
                update_state = (
                    train_states,
                    jax.tree.map(lambda x: x.squeeze(), init_hstates),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, loss_info

            update_state = (
                train_states,
                initial_hstates,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            loss_info["ratio_0"] = loss_info["ratio"].at[0, 0].get()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

            train_states = update_state[0]
            metric = traj_batch.info
            metric = jax.tree.map(
                lambda x: x.reshape(
                    (config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)
                ),
                traj_batch.info,
            )
            metric["loss"] = loss_info
            rng = update_state[-1]

            def callback(metric):
                # The metrics have an agent dimension, but they are identical
                # for all agents, so index into the 0th agent.
                log_data = {
                    "returns": metric["returned_episode_returns"][:, :, 0][
                        metric["returned_episode"][:, :, 0]
                    ].mean(),
                    "win_rate": metric["returned_won_episode"][:, :, 0][
                        metric["returned_episode"][:, :, 0]
                    ].mean(),
                    "env_step": (metric["update_steps"] + 1)
                    * config["NUM_ENVS"]
                    * config["NUM_STEPS"],
                    **metric["loss"],
                }
                if progress_bar is not None:
                    completed = int(np.asarray(metric["update_steps"]).item()) + 1
                    progress_bar.update(max(0, completed - progress_bar.n))
                    postfix = {"steps": f"{int(log_data['env_step']):,}"}
                    win_rate = float(np.asarray(log_data["win_rate"]).item())
                    if np.isfinite(win_rate):
                        postfix["win"] = f"{win_rate:.3f}"
                    progress_bar.set_postfix(postfix)
                wandb.log(log_data)

            metric["update_steps"] = update_steps
            jax.debug.callback(callback, metric, ordered=True)

            if checkpoint_callback is not None:
                rollout_env_steps = config["NUM_ENVS"] * config["NUM_STEPS"]
                completed_updates = update_steps + 1
                completed_env_steps = completed_updates * rollout_env_steps
                previous_env_steps = update_steps * rollout_env_steps
                checkpoint_interval = config["CHECKPOINT_INTERVAL_TIMESTEPS"]
                crossed_interval = (
                    completed_env_steps // checkpoint_interval
                    > previous_env_steps // checkpoint_interval
                )
                is_final = completed_updates == config["NUM_UPDATES"]
                should_save = jnp.logical_or(crossed_interval, is_final)

                def save_checkpoint(_):
                    return io_callback(
                        checkpoint_callback,
                        jax.ShapeDtypeStruct((), jnp.int32),
                        train_states[0].params,
                        train_states[1].params,
                        completed_env_steps,
                        is_final,
                        ordered=True,
                    )

                jax.lax.cond(
                    should_save,
                    save_checkpoint,
                    lambda _: jnp.int32(0),
                    operand=None,
                )

            update_steps = update_steps + 1
            runner_state = (train_states, env_state, last_obs, last_done, hstates, rng)
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            (actor_train_state, critic_train_state),
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            (ac_init_hstate, cr_init_hstate),
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train


@hydra.main(
    version_base=None, config_path="config", config_name="mappo_homogenous_rnn_smax"
)
def main(config):

    config = OmegaConf.to_container(config)
    sharing_mode = (
        "shared-actor"
        if config["ACTOR_PARAMETER_SHARING"]
        else "independent-actors"
    )
    actor_mode = (
        f"matched-{sharing_mode}"
        if config["MATCHED_COMPARISON"]
        else sharing_mode
    )

    run = wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[t for t in os.environ.get("WANDB_TAGS", "").split(",") if t],
        group=os.environ.get("WANDB_RUN_GROUP") or None,
        name=os.environ.get("WANDB_NAME")
        or (
            f"MAPPO-{actor_mode}-{config['ALIGN_MODE']}-"
            f"{config['MAP_NAME']}-seed{config['SEED']}"
        ),
        config=config,
        mode=config["WANDB_MODE"],
    )
    checkpoint_callback = None
    checkpoint_run_dir = None
    if config["SAVE_CHECKPOINTS"]:
        checkpoint_interval = int(config["CHECKPOINT_INTERVAL_TIMESTEPS"])
        if checkpoint_interval <= 0:
            raise ValueError("CHECKPOINT_INTERVAL_TIMESTEPS must be positive")
        config["CHECKPOINT_INTERVAL_TIMESTEPS"] = checkpoint_interval
        checkpoint_callback, checkpoint_run_dir = make_checkpoint_callback(config, run)
        print(f"Checkpoints: {checkpoint_run_dir}", flush=True)

    rng = jax.random.PRNGKey(config["SEED"])
    num_updates = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    progress_bar = tqdm(
        total=num_updates,
        desc=(
            f"MAPPO {actor_mode} {config['ALIGN_MODE']} "
            f"{config['MAP_NAME']} seed={config['SEED']}"
        ),
        unit="update",
        dynamic_ncols=True,
    )
    try:
        with jax.disable_jit(False):
            train_jit = jax.jit(
                make_train(config, progress_bar, checkpoint_callback)
            )
            result = train_jit(rng)
            jax.block_until_ready(result)
    finally:
        progress_bar.close()
        wandb.finish()


if __name__ == "__main__":
    main()
