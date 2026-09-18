"""
Based on PureJaxRL Implementation of IPPO, with changes to give a centralised critic.
"""

import functools
import json
import os
from functools import partial
from pathlib import Path
from typing import Dict, NamedTuple, Sequence

import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

import jaxmarl
import wandb
from baselines.MAPPO.alignment_utils import (
    representation_distance,
    tree_l2_norm,
    vmapped_optimizer,
)
from jaxmarl.wrappers.baselines import JaxMARLWrapper, MPELogWrapper


class MPEWorldStateWrapper(JaxMARLWrapper):
    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state(obs)
        return obs, env_state

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, action):
        obs, env_state, reward, done, info = self._env.step(key, state, action)
        obs["world_state"] = self.world_state(obs)
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def world_state(self, obs):
        """
        For each agent: [agent obs, all other agent obs]
        """

        @partial(jax.vmap, in_axes=(0, None))
        def _roll_obs(aidx, all_obs):
            robs = jnp.roll(all_obs, -aidx, axis=0)
            robs = robs.flatten()
            return robs

        all_obs = jnp.array([obs[agent] for agent in self._env.agents]).flatten()
        all_obs = jnp.expand_dims(all_obs, axis=0).repeat(self._env.num_agents, axis=0)
        return all_obs

    def world_state_size(self):
        spaces = [self._env.observation_space(agent) for agent in self._env.agents]
        return sum([space.shape[-1] for space in spaces])


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
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(*rnn_state.shape),
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
        obs, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=action_logits)

        return hidden, pi, actor_mean


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
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        critic_latent = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic_latent = nn.relu(critic_latent)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic_latent
        )

        return hidden, jnp.squeeze(value, axis=-1), critic_latent


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    critic_latent: jnp.ndarray
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    config.setdefault("ACTOR_PARAMETER_SHARING", True)
    config.setdefault("MATCHED_COMPARISON", False)
    config.setdefault("ALIGN_MODE", "none")
    config.setdefault("ALIGN_DISTANCE", "ln_mse")
    config.setdefault("ALIGNMENT_COEF", 0.0)
    config.setdefault("ALIGN_DISTANCE_EPS", 1e-8)
    if config["ALIGN_MODE"] not in {"none", "c_to_a"}:
        raise ValueError("MPE alignment supports only none and c_to_a")
    if config["ALIGN_DISTANCE"] not in {"ln_mse", "linear_cka"}:
        raise ValueError("MPE alignment supports only ln_mse and linear_cka")
    if config["ALIGN_MODE"] != "none" and config["ACTOR_PARAMETER_SHARING"]:
        raise ValueError("The MPE alignment protocol requires independent actors")
    if config["ALIGN_MODE"] != "none" and not config["MATCHED_COMPARISON"]:
        raise ValueError("Alignment requires MATCHED_COMPARISON=true")
    if not config["ACTOR_PARAMETER_SHARING"] and (
        config["NUM_ENVS"] % config["NUM_MINIBATCHES"]
    ):
        raise ValueError("NUM_ENVS must be divisible by NUM_MINIBATCHES for NPS")
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = (
        config["CLIP_EPS"] / env.num_agents
        if config["SCALE_CLIP_EPS"]
        else config["CLIP_EPS"]
    )

    env = MPEWorldStateWrapper(env)
    env = MPELogWrapper(env)

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
        )
        ac_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )
        actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)
        if not config["ACTOR_PARAMETER_SHARING"]:
            if config["MATCHED_COMPARISON"]:
                actor_network_params = jax.tree.map(
                    lambda value: jnp.repeat(value[None], env.num_agents, axis=0),
                    actor_network_params,
                )
            else:
                actor_rngs = jax.random.split(_rng_actor, env.num_agents)
                actor_network_params = jax.vmap(
                    actor_network.init, in_axes=(0, None, None)
                )(actor_rngs, ac_init_hstate, ac_init_x)

        cr_init_x = (
            jnp.zeros(
                (
                    1,
                    config["NUM_ENVS"],
                    env.world_state_size(),
                )
            ),  #  + env.observation_space(env.agents[0]).shape[0]
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        cr_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )
        critic_network_params = critic_network.init(
            _rng_critic, cr_init_hstate, cr_init_x
        )

        if config["ANNEAL_LR"]:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        actor_tx = (
            actor_tx
            if config["ACTOR_PARAMETER_SHARING"]
            else vmapped_optimizer(actor_tx)
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
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )
                if config["ACTOR_PARAMETER_SHARING"]:
                    ac_hstate, pi, _ = actor_network.apply(
                        train_states[0].params, hstates[0], ac_in
                    )
                else:
                    actor_obs = obs_batch.reshape(
                        (env.num_agents, config["NUM_ENVS"], -1)
                    )
                    actor_done = last_done.reshape((env.num_agents, config["NUM_ENVS"]))
                    actor_hidden = hstates[0].reshape(
                        (env.num_agents, config["NUM_ENVS"], -1)
                    )

                    def apply_actor(params, hidden, observation, done):
                        return actor_network.apply(
                            params,
                            hidden,
                            (observation[None], done[None]),
                        )

                    ac_hstate, pi, _ = jax.vmap(apply_actor)(
                        train_states[0].params,
                        actor_hidden,
                        actor_obs,
                        actor_done,
                    )
                    ac_hstate = ac_hstate.reshape((config["NUM_ACTORS"], -1))
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                action = action.reshape((1, config["NUM_ACTORS"]))
                log_prob = log_prob.reshape((1, config["NUM_ACTORS"]))
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
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
                    jax.lax.stop_gradient(critic_latent.squeeze(0)),
                    info,
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
                            init_hstate.squeeze(axis=0),
                            (traj_batch.obs, traj_batch.done),
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
                        _, value, _ = critic_network.apply(
                            critic_params,
                            init_hstate.squeeze(axis=0),
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
                        return critic_loss, (value_loss)

                    if config["ACTOR_PARAMETER_SHARING"]:
                        actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                        actor_loss, actor_grads = actor_grad_fn(
                            actor_train_state.params,
                            ac_init_hstate,
                            traj_batch,
                            advantages,
                        )
                        alignment_loss = jnp.asarray(0.0)
                        weighted_alignment_loss = jnp.asarray(0.0)
                        actor_rl_grad_norm = tree_l2_norm(actor_grads)
                        actor_alignment_grad_norm = jnp.asarray(0.0)
                        actor_combined_grad_norm = actor_rl_grad_norm
                    else:
                        minibatch_envs = traj_batch.obs.shape[1] // env.num_agents

                        def split_agents(value):
                            value = value.reshape(
                                (
                                    value.shape[0],
                                    env.num_agents,
                                    minibatch_envs,
                                    *value.shape[2:],
                                )
                            )
                            return jnp.swapaxes(value, 0, 1)

                        actor_hstate = split_agents(ac_init_hstate)
                        actor_traj = jax.tree.map(split_agents, traj_batch)
                        actor_advantages = (advantages - advantages.mean()) / (
                            advantages.std() + 1e-8
                        )
                        actor_advantages = split_agents(actor_advantages)

                        def actor_outputs(params):
                            return jax.vmap(_actor_loss_fn)(
                                params,
                                actor_hstate,
                                actor_traj,
                                actor_advantages,
                            )

                        def distance_from_outputs(actor_aux, distance_name):
                            current_actor_latent = jnp.swapaxes(actor_aux[5], 0, 1)
                            target_critic_latent = jnp.swapaxes(
                                actor_traj.critic_latent, 0, 1
                            )
                            return representation_distance(
                                current_actor_latent,
                                jax.lax.stop_gradient(target_critic_latent),
                                jnp.ones(current_actor_latent.shape[:-1], dtype=bool),
                                distance_name,
                                agent_axis=1,
                                epsilon=config["ALIGN_DISTANCE_EPS"],
                            )

                        def alignment_objective(params, distance_name):
                            _, actor_aux = actor_outputs(params)
                            return distance_from_outputs(actor_aux, distance_name)

                        def actor_objective(params):
                            per_agent_loss, actor_aux = actor_outputs(params)
                            if config["ALIGN_MODE"] == "c_to_a":
                                alignment = distance_from_outputs(
                                    actor_aux, config["ALIGN_DISTANCE"]
                                )
                            else:
                                alignment = jnp.asarray(0.0)
                            weighted = config["ALIGNMENT_COEF"] * alignment
                            return per_agent_loss.mean() + weighted, (
                                actor_aux,
                                alignment,
                                weighted,
                            )

                        (actor_objective_value, actor_objective_aux), actor_grads = (
                            jax.value_and_grad(actor_objective, has_aux=True)(
                                actor_train_state.params
                            )
                        )
                        actor_aux, alignment_loss, weighted_alignment_loss = (
                            actor_objective_aux
                        )
                        actor_loss = (actor_objective_value, actor_aux)
                        if config["ALIGN_MODE"] == "c_to_a":
                            alignment_grads = jax.grad(
                                lambda params: config["ALIGNMENT_COEF"]
                                * alignment_objective(params, config["ALIGN_DISTANCE"])
                            )(actor_train_state.params)
                        else:
                            alignment_grads = jax.tree.map(jnp.zeros_like, actor_grads)
                        rl_grads = jax.tree.map(
                            lambda total, cross: total - cross,
                            actor_grads,
                            alignment_grads,
                        )
                        actor_rl_grad_norm = tree_l2_norm(rl_grads)
                        actor_alignment_grad_norm = tree_l2_norm(alignment_grads)
                        actor_combined_grad_norm = tree_l2_norm(actor_grads)
                        # The objective averages agents.  Restore each independent
                        # actor's PPO/aux scale before its private optimizer step.
                        actor_grads = jax.tree.map(
                            lambda value: value * env.num_agents, actor_grads
                        )
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params, cr_init_hstate, traj_batch, targets
                    )

                    actor_train_state = actor_train_state.apply_gradients(
                        grads=actor_grads
                    )
                    critic_train_state = critic_train_state.apply_gradients(
                        grads=critic_grads
                    )

                    total_loss = actor_loss[0] + critic_loss[0]
                    loss_info = {
                        "total_loss": total_loss,
                        "actor_loss": actor_loss[0],
                        "value_loss": critic_loss[0],
                        "entropy": actor_loss[1][1],
                        "ratio": actor_loss[1][2],
                        "approx_kl": actor_loss[1][3],
                        "clip_frac": actor_loss[1][4],
                        "alignment_loss": alignment_loss,
                        "alignment_weighted_loss": weighted_alignment_loss,
                        "actor_rl_gradient_norm": actor_rl_grad_norm,
                        "actor_alignment_gradient_norm": actor_alignment_grad_norm,
                        "actor_combined_gradient_norm": actor_combined_grad_norm,
                        "actor_alignment_to_rl_gradient_ratio": (
                            actor_alignment_grad_norm
                            / jnp.maximum(actor_rl_grad_norm, 1e-12)
                        ),
                    }

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
                if config["ACTOR_PARAMETER_SHARING"]:
                    permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])
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
                    # Use the same environment permutation for every actor.  This
                    # preserves one trajectory slice per NPS actor in each
                    # minibatch and keeps actor/critic latent samples paired.
                    permutation = jax.random.permutation(_rng, config["NUM_ENVS"])

                    def nps_minibatches(value):
                        value = value.reshape(
                            (
                                value.shape[0],
                                env.num_agents,
                                config["NUM_ENVS"],
                                *value.shape[2:],
                            )
                        )
                        value = jnp.take(value, permutation, axis=2)
                        value = value.reshape(
                            (
                                value.shape[0],
                                env.num_agents,
                                config["NUM_MINIBATCHES"],
                                config["NUM_ENVS"] // config["NUM_MINIBATCHES"],
                                *value.shape[3:],
                            )
                        )
                        value = jnp.moveaxis(value, 2, 0)
                        return value.reshape(
                            (
                                config["NUM_MINIBATCHES"],
                                value.shape[1],
                                env.num_agents
                                * (config["NUM_ENVS"] // config["NUM_MINIBATCHES"]),
                                *value.shape[4:],
                            )
                        )

                    minibatches = jax.tree.map(nps_minibatches, batch)

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
            metric["loss"] = loss_info
            rng = update_state[-1]

            def callback(metric):
                env_step = (
                    (int(metric["update_steps"]) + 1)
                    * config["NUM_ENVS"]
                    * config["NUM_STEPS"]
                )
                payload = {
                    "returns": float(
                        np.asarray(metric["returned_episode_returns"][-1, :]).mean()
                    ),
                    "env_step": env_step,
                    **{
                        key: float(np.asarray(value).mean())
                        for key, value in metric["loss"].items()
                    },
                }
                wandb.log(payload)
                metrics_path = config.get("METRICS_JSONL", "")
                if metrics_path:
                    destination = Path(metrics_path)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(payload, sort_keys=True) + "\n")
                status_path = config.get("STATUS_JSON", "")
                if status_path:
                    destination = Path(status_path)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_suffix(destination.suffix + ".tmp")
                    status = dict(config.get("STATUS_METADATA", {}))
                    status.update(status="running", env_steps=env_step)
                    temporary.write_text(
                        json.dumps(status, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    os.replace(temporary, destination)

            metric["update_steps"] = update_steps
            jax.experimental.io_callback(callback, None, metric)
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
        final_train_states = runner_state[0][0]
        return {
            "runner_state": runner_state,
            "actor_params": final_train_states[0].params,
            "critic_params": final_train_states[1].params,
        }

    return train


@hydra.main(
    version_base=None, config_path="config", config_name="mappo_homogenous_rnn_mpe"
)
def main(config):

    config = OmegaConf.to_container(config)
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[t for t in os.environ.get("WANDB_TAGS", "").split(",") if t],
        group=os.environ.get("WANDB_RUN_GROUP") or None,
        name=os.environ.get("WANDB_NAME") or None,
        config=config,
        mode=config["WANDB_MODE"],
    )
    rng = jax.random.PRNGKey(config["SEED"])
    with jax.disable_jit(False):
        train_jit = jax.jit(make_train(config))
        train_jit(rng)


if __name__ == "__main__":
    main()
