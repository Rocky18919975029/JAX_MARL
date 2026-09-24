# Derived from Mava rec_mappo.py at commit 9f67e612654ecb7b7d45ff8052ce9ccfc6c68d93.
# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import time
from typing import Any, NamedTuple, Tuple

import chex
import flax
import hydra
import jax
import jax.numpy as jnp
import optax
from colorama import Fore, Style
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from flax.linen.initializers import orthogonal
from jax import tree
from omegaconf import DictConfig, OmegaConf

from mava.evaluator import get_eval_fn, get_num_eval_envs, make_rec_eval_act_fn
from mava.networks import RecurrentActor as BaseActor
from mava.networks import RecurrentValueNet as BaseCritic
from mava.networks import ScannedRNN
from mava.systems.ppo.types import (
    HiddenStates,
    OptStates,
    Params,
    RNNPPOTransition,
)
from mava.types import (
    ExperimentOutput,
    LearnerFn,
    MarlEnv,
    Metrics,
)
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import check_total_timesteps
from mava.utils.jax_utils import add_batch_dim, unreplicate_batch_dim, unreplicate_n_dims
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.multistep import calculate_gae
from mava.utils.network_utils import get_action_head
from mava.utils.training import make_learning_rate
from mava.wrappers.episode_metrics import get_final_step_metrics


class Actor(BaseActor):
    """Mava's recurrent actor, exposing only its pre-head GRU representation."""

    @nn.compact
    def __call__(self, policy_hidden_state, observation_done, *, return_latent=False):
        observation, done = observation_done
        policy_embedding = self.pre_torso(observation.agents_view)
        policy_hidden_state, latent = ScannedRNN(self.hidden_state_dim)(
            policy_hidden_state, (policy_embedding, done)
        )
        pi = self.action_head(self.post_torso(latent), observation.action_mask)
        if return_latent:
            return policy_hidden_state, pi, latent
        return policy_hidden_state, pi

    @nn.compact
    def log_prob_from_latent(self, latent, action, action_mask):
        """Use the exact Mava post-torso and masked action head."""
        pi = self.action_head(self.post_torso(latent), action_mask)
        return pi.log_prob(action)


class Critic(BaseCritic):
    """Mava's centralised recurrent value net, exposing its GRU representation."""

    @nn.compact
    def __call__(self, value_hidden_state, observation_done, *, return_latent=False):
        observation, done = observation_done
        value_embedding = self.pre_torso(observation.global_state)
        value_hidden_state, latent = ScannedRNN(self.hidden_state_dim)(
            value_hidden_state, (value_embedding, done)
        )
        value = nn.Dense(1, kernel_init=orthogonal(1.0))(self.post_torso(latent))
        value = jnp.squeeze(value, axis=-1)
        if return_latent:
            return value_hidden_state, value, latent
        return value_hidden_state, value


class RecoveryHead(nn.Module):
    action_dim: int
    latent_dim: int

    @nn.compact
    def __call__(self, critic_latent, action):
        x = jnp.concatenate(
            (critic_latent, jax.nn.one_hot(action, self.action_dim)), axis=-1
        )
        x = nn.relu(nn.Dense(self.latent_dim)(x))
        return nn.Dense(
            self.latent_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(x)


class ARecLearnerState(NamedTuple):
    params: Params
    opt_states: OptStates
    key: chex.PRNGKey
    env_state: Any
    timestep: Any
    dones: jax.Array
    hstates: HiddenStates
    q_params: FrozenDict
    q_opt_state: optax.OptState


def score_from_latent(actor: Actor, params, latent, action, action_mask):
    """One representation-level score per sample; supports higher-order actor grads."""
    def summed_log_prob(z):
        return actor.apply(
            params, z, action, action_mask, method=Actor.log_prob_from_latent
        ).sum()

    return jax.grad(summed_log_prob)(latent)


def whiten_scores(scores, ridge, axis_names=()):
    """Per-agent empirical Fisher from the full update batch [T,E,N,D]."""
    fisher = jnp.einsum("tend,tenf->ndf", scores, scores)
    fisher /= scores.shape[0] * scores.shape[1]
    # Each Mava update has a vmapped learner batch and potentially several devices.
    # Average their equally sized rollout shards before whitening any local scores.
    for axis_name in axis_names:
        fisher = jax.lax.pmean(fisher, axis_name=axis_name)
    fisher = 0.5 * (fisher + jnp.swapaxes(fisher, -1, -2))
    eigenvalues, eigenvectors = jnp.linalg.eigh(fisher)
    inverse_root = (
        eigenvectors
        * jax.lax.rsqrt(jnp.maximum(eigenvalues, 0.0) + ridge)[:, None, :]
    ) @ jnp.swapaxes(eigenvectors, -1, -2)
    target = jnp.einsum("tend,ndf->tenf", scores, inverse_root)
    return jax.lax.stop_gradient(target), jax.lax.stop_gradient(inverse_root)


def get_learner_fn(
    env: MarlEnv,
    actor_network: Actor,
    recovery_head: RecoveryHead,
    apply_fns: Tuple[Any, Any],
    update_fns: Tuple[Any, Any, Any],
    config: DictConfig,
) -> LearnerFn[ARecLearnerState]:
    """Get the learner function."""
    actor_apply_fn, critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn, q_update_fn = update_fns
    q_apply_fn = jax.vmap(recovery_head.apply, in_axes=(0, 2, 2), out_axes=2)

    def _update_step(learner_state: ARecLearnerState, _: Any) -> Tuple[ARecLearnerState, Tuple]:
        """A single update of the network.

        This function steps the environment and records the trajectory batch for
        training. It then calculates advantages and targets based on the recorded
        trajectory and updates the actor and critic networks based on the calculated
        losses.

        Args:
        ----
            learner_state (NamedTuple):
                - params (Params): The current model parameters.
                - opt_states (OptStates): The current optimizer states.
                - key (PRNGKey): The random number generator state.
                - env_state (State): The environment state.
                - prev_timestep (TimeStep): The previous environment timestep.
                - prev_done (bool): Whether the previous timestep was a terminal state.
                - hstates (HiddenStates): The hidden state of the policy and critic RNN.
            _ (Any): The current metrics info.

        """

        def _env_step(
            learner_state: ARecLearnerState, _: Any
        ) -> Tuple[ARecLearnerState, Tuple[RNNPPOTransition, Metrics]]:
            """Step the environment."""
            (
                params,
                opt_states,
                key,
                env_state,
                prev_timestep,
                prev_done,
                prev_hstates,
                q_params,
                q_opt_state,
            ) = learner_state

            key, policy_key = jax.random.split(key)

            # Add a batch dimension to the observation.
            batched_observation = add_batch_dim(prev_timestep.observation)
            ac_in = (batched_observation, prev_done[jnp.newaxis, :])

            # Run the network.
            policy_hidden_state, actor_policy = actor_apply_fn(
                params.actor_params, prev_hstates.policy_hidden_state, ac_in
            )
            critic_hidden_state, value = critic_apply_fn(
                params.critic_params, prev_hstates.critic_hidden_state, ac_in
            )

            # Sample action from the policy and squeeze out the batch dimension.
            action = actor_policy.sample(seed=policy_key)
            log_prob = actor_policy.log_prob(action)

            action, log_prob, value = action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)

            # Step the environment.
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            done = timestep.last().repeat(env.num_agents).reshape(config.arch.num_envs, -1)
            hstates = HiddenStates(policy_hidden_state, critic_hidden_state)
            transition = RNNPPOTransition(
                prev_done,
                action,
                value,
                timestep.reward,
                log_prob,
                prev_timestep.observation,
                prev_hstates,
            )
            learner_state = ARecLearnerState(
                params, opt_states, key, env_state, timestep, done, hstates,
                q_params, q_opt_state,
            )
            metrics = timestep.extras["episode_metrics"] | timestep.extras["env_metrics"]
            return learner_state, (transition, metrics)

        # Step environment for rollout length
        learner_state, (traj_batch, episode_metrics) = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        # Calculate advantage
        (
            params, opt_states, key, env_state, final_timestep, final_done,
            hstates, q_params, q_opt_state,
        ) = learner_state

        # Add a batch dimension to the observation.
        batched_final_observation = add_batch_dim(final_timestep.observation)
        ac_in = (batched_final_observation, final_done[jnp.newaxis, :])

        # Run the network.
        _, final_val = critic_apply_fn(params.critic_params, hstates.critic_hidden_state, ac_in)

        # Squeeze out the batch dimension and mask out the value of terminal states.
        final_val = final_val.squeeze(0)

        advantages, targets = calculate_gae(
            traj_batch, final_val, final_done, config.system.gamma, config.system.gae_lambda
        )

        # Old policy and critic are frozen while constructing this rollout's target.
        obs_and_done = (traj_batch.obs, traj_batch.done)
        _, _, actor_latent = actor_apply_fn(
            params.actor_params,
            traj_batch.hstates.policy_hidden_state[0],
            obs_and_done,
            return_latent=True,
        )
        _, _, critic_latent = critic_apply_fn(
            params.critic_params,
            traj_batch.hstates.critic_hidden_state[0],
            obs_and_done,
            return_latent=True,
        )
        action_mask = traj_batch.obs.action_mask
        old_scores = score_from_latent(
            actor_network, params.actor_params, actor_latent,
            traj_batch.action, action_mask,
        )
        recovery_target, inverse_root = whiten_scores(
            old_scores, config.arec.fisher_ridge, axis_names=("batch", "device")
        )
        critic_latent = jax.lax.stop_gradient(critic_latent)

        def q_fit_loss(p):
            prediction = q_apply_fn(p, critic_latent, traj_batch.action)
            squared = jnp.square(prediction - recovery_target).sum(axis=-1)
            per_agent = squared.mean(axis=(0, 1))
            # Summation keeps each independent agent's optimizer gradient at its own scale.
            return per_agent.sum(), per_agent.mean()

        _, q_loss_pre = q_fit_loss(q_params)

        def fit_q(q_state, _):
            current_params, current_opt_state = q_state
            (_, loss_mean), grads = jax.value_and_grad(q_fit_loss, has_aux=True)(
                current_params
            )
            grads = jax.lax.pmean(grads, axis_name="batch")
            grads = jax.lax.pmean(grads, axis_name="device")
            updates, next_opt_state = q_update_fn(grads, current_opt_state)
            return (optax.apply_updates(current_params, updates), next_opt_state), loss_mean

        (q_params, q_opt_state), _ = jax.lax.scan(
            fit_q, (q_params, q_opt_state), None, config.arec.q_steps
        )
        _, q_loss_post = q_fit_loss(q_params)
        recovery_teacher = jax.lax.stop_gradient(
            q_apply_fn(q_params, critic_latent, traj_batch.action)
        )
        target_energy = jnp.square(recovery_target).sum(axis=-1).mean()

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""
                params, opt_states, key = train_state
                traj_batch, advantages, targets, recovery_teacher = batch_info

                def _actor_loss_fn(
                    actor_params: FrozenDict,
                    traj_batch: RNNPPOTransition,
                    gae: jax.Array,
                    teacher: jax.Array,
                    key: chex.PRNGKey,
                ) -> Tuple:
                    """Calculate the actor loss."""
                    # Rerun network
                    obs_and_done = (traj_batch.obs, traj_batch.done)
                    _, actor_policy, latent = actor_apply_fn(
                        actor_params, traj_batch.hstates.policy_hidden_state[0],
                        obs_and_done, return_latent=True,
                    )
                    log_prob = actor_policy.log_prob(traj_batch.action)

                    # Calculate actor loss
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    # Nomalise advantage at minibatch level
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    actor_loss1 = ratio * gae
                    actor_loss2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config.system.clip_eps,
                            1.0 + config.system.clip_eps,
                        )
                        * gae
                    )
                    actor_loss = -jnp.minimum(actor_loss1, actor_loss2)
                    actor_loss = actor_loss.mean()
                    # The seed will be used in the TanhTransformedDistribution:
                    entropy = actor_policy.entropy(seed=key).mean()

                    scores = score_from_latent(
                        actor_network, actor_params, latent,
                        traj_batch.action, traj_batch.obs.action_mask,
                    )
                    normalized_scores = jnp.einsum(
                        "tend,ndf->tenf", scores, inverse_root
                    )
                    recovery_loss = jnp.square(normalized_scores - teacher).sum(axis=-1).mean()
                    total_loss = (
                        actor_loss - config.system.ent_coef * entropy
                        + config.arec.coef * recovery_loss
                    )
                    return total_loss, (actor_loss, entropy, recovery_loss)

                def _critic_loss_fn(
                    critic_params: FrozenDict,
                    traj_batch: RNNPPOTransition,
                    targets: jax.Array,
                ) -> Tuple:
                    """Calculate the critic loss."""
                    # Rerun network
                    obs_and_done = (traj_batch.obs, traj_batch.done)
                    _, value = critic_apply_fn(
                        critic_params, traj_batch.hstates.critic_hidden_state[0], obs_and_done
                    )

                    # Clipped MSE loss
                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                        -config.system.clip_eps, config.system.clip_eps
                    )
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                    total_loss = config.system.vf_coef * value_loss
                    return total_loss, value_loss

                # Calculate actor loss
                key, entropy_key = jax.random.split(key)
                actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                actor_loss_info, actor_grads = actor_grad_fn(
                    params.actor_params,
                    traj_batch,
                    advantages,
                    recovery_teacher,
                    entropy_key,
                )

                # Calculate critic loss
                critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                value_loss_info, critic_grads = critic_grad_fn(
                    params.critic_params, traj_batch, targets
                )

                # Compute the parallel mean (pmean) over the batch.
                # This pmean could be a regular mean as the batch axis is on the same device.
                actor_grads, actor_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info), axis_name="batch"
                )
                # pmean over devices.
                actor_grads, actor_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info), axis_name="device"
                )

                critic_grads, value_loss_info = jax.lax.pmean(
                    (critic_grads, value_loss_info), axis_name="batch"
                )
                # pmean over devices.
                critic_grads, value_loss_info = jax.lax.pmean(
                    (critic_grads, value_loss_info), axis_name="device"
                )

                # Update params and optimiser state
                actor_updates, actor_new_opt_state = actor_update_fn(
                    actor_grads, opt_states.actor_opt_state
                )
                actor_new_params = optax.apply_updates(params.actor_params, actor_updates)

                critic_updates, critic_new_opt_state = critic_update_fn(
                    critic_grads, opt_states.critic_opt_state
                )
                critic_new_params = optax.apply_updates(params.critic_params, critic_updates)

                new_params = Params(actor_new_params, critic_new_params)
                new_opt_state = OptStates(actor_new_opt_state, critic_new_opt_state)

                actor_loss, (_, entropy, recovery_loss) = actor_loss_info
                value_loss, unscaled_value_loss = value_loss_info

                total_loss = actor_loss + value_loss
                loss_info = {
                    "total_loss": total_loss,
                    "value_loss": unscaled_value_loss,
                    "actor_loss": actor_loss,
                    "entropy": entropy,
                    "arec_loss": recovery_loss,
                    "arec_weighted_loss": config.arec.coef * recovery_loss,
                }

                return (new_params, new_opt_state, entropy_key), loss_info

            params, opt_states, traj_batch, advantages, targets, recovery_teacher, key = (
                update_state
            )
            key, shuffle_key, entropy_key = jax.random.split(key, 3)

            # Shuffle minibatches
            batch = (traj_batch, advantages, targets, recovery_teacher)
            num_recurrent_chunks = (
                config.system.rollout_length // config.system.recurrent_chunk_size
            )
            batch = tree.map(
                lambda x: x.reshape(
                    config.system.recurrent_chunk_size,
                    config.arch.num_envs * num_recurrent_chunks,
                    *x.shape[2:],
                ),
                batch,
            )
            permutation = jax.random.permutation(
                shuffle_key, config.arch.num_envs * num_recurrent_chunks
            )
            shuffled_batch = tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)
            reshaped_batch = tree.map(
                lambda x: jnp.reshape(
                    x, (x.shape[0], config.system.num_minibatches, -1, *x.shape[2:])
                ),
                shuffled_batch,
            )
            minibatches = tree.map(lambda x: jnp.swapaxes(x, 1, 0), reshaped_batch)

            # Update minibatches
            (params, opt_states, entropy_key), loss_info = jax.lax.scan(
                _update_minibatch, (params, opt_states, entropy_key), minibatches
            )

            update_state = (
                params,
                opt_states,
                traj_batch,
                advantages,
                targets,
                recovery_teacher,
                key,
            )
            return update_state, loss_info

        update_state = (
            params,
            opt_states,
            traj_batch,
            advantages,
            targets,
            recovery_teacher,
            key,
        )

        # Update epochs
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.ppo_epochs
        )

        params, opt_states, traj_batch, advantages, targets, recovery_teacher, key = (
            update_state
        )
        learner_state = ARecLearnerState(
            params,
            opt_states,
            key,
            env_state,
            final_timestep,
            final_done,
            hstates,
            q_params,
            q_opt_state,
        )
        loss_info = loss_info | {
            "arec_q_loss_pre": q_loss_pre,
            "arec_q_loss_post": q_loss_post,
            "arec_q_to_zero_ratio": q_loss_post / jnp.maximum(target_energy, 1e-8),
            "arec_target_energy": target_energy,
        }
        return learner_state, (episode_metrics, loss_info)

    def learner_fn(learner_state: ARecLearnerState) -> ExperimentOutput[ARecLearnerState]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        Args:
        ----
            learner_state (NamedTuple):
                - params (Params): The initial model parameters.
                - opt_states (OptStates): The initial optimizer states.
                - key (chex.PRNGKey): The random number generator state.
                - env_state (LogEnvState): The environment state.
                - timesteps (TimeStep): The initial timestep in the initial trajectory.
                - dones (bool): Whether the initial timestep was a terminal state.
                - hstates (HiddenStates): The hidden state of the policy and critic RNN.

        """
        batched_update_step = jax.vmap(_update_step, in_axes=(0, None), axis_name="batch")

        learner_state, (episode_info, loss_info) = jax.lax.scan(
            batched_update_step, learner_state, None, config.system.num_updates_per_eval
        )
        return ExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_info,
            train_metrics=loss_info,
        )

    return learner_fn


def learner_setup(
    env: MarlEnv, keys: jax.Array, config: DictConfig
) -> Tuple[LearnerFn[ARecLearnerState], Actor, ARecLearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number of agents.
    num_agents = env.num_agents
    config.system.num_agents = num_agents

    # PRNG keys.
    key, actor_net_key, critic_net_key = keys

    # Define network and optimiser.
    actor_pre_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_post_torso = hydra.utils.instantiate(config.network.actor_network.post_torso)
    action_head, _ = get_action_head(env.action_spec)
    actor_action_head = hydra.utils.instantiate(action_head, action_dim=env.action_dim)
    critic_pre_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)
    critic_post_torso = hydra.utils.instantiate(config.network.critic_network.post_torso)

    actor_network = Actor(
        pre_torso=actor_pre_torso,
        post_torso=actor_post_torso,
        action_head=actor_action_head,
        hidden_state_dim=config.network.hidden_state_dim,
    )
    critic_network = Critic(
        pre_torso=critic_pre_torso,
        post_torso=critic_post_torso,
        hidden_state_dim=config.network.hidden_state_dim,
        centralised_critic=True,
    )
    recovery_head = RecoveryHead(
        action_dim=env.action_dim,
        latent_dim=config.network.hidden_state_dim,
    )

    actor_lr = make_learning_rate(config.system.actor_lr, config)
    critic_lr = make_learning_rate(config.system.critic_lr, config)

    actor_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )
    critic_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(critic_lr, eps=1e-5),
    )
    q_optim = optax.adam(config.arec.q_lr, eps=1e-5)

    # Initialise observation with obs of all agents.
    init_obs = env.observation_spec.generate_value()
    init_obs = tree.map(
        lambda x: jnp.repeat(x[jnp.newaxis, ...], config.arch.num_envs, axis=0),
        init_obs,
    )
    init_obs = add_batch_dim(init_obs)
    init_done = jnp.zeros((1, config.arch.num_envs, num_agents), dtype=bool)
    init_obs_done = (init_obs, init_done)

    # Initialise hidden state.
    init_policy_hstate = ScannedRNN.initialize_carry(
        (config.arch.num_envs, num_agents), config.network.hidden_state_dim
    )
    init_critic_hstate = ScannedRNN.initialize_carry(
        (config.arch.num_envs, num_agents), config.network.hidden_state_dim
    )

    # initialise params and optimiser state.
    actor_params = actor_network.init(actor_net_key, init_policy_hstate, init_obs_done)
    actor_opt_state = actor_optim.init(actor_params)
    critic_params = critic_network.init(critic_net_key, init_critic_hstate, init_obs_done)
    critic_opt_state = critic_optim.init(critic_params)
    q_keys = jax.random.split(
        jax.random.fold_in(actor_net_key, 71227), num_agents
    )
    q_params = jax.vmap(recovery_head.init, in_axes=(0, None, None))(
        q_keys,
        jnp.zeros((1, config.network.hidden_state_dim)),
        jnp.zeros((1,), dtype=jnp.int32),
    )
    q_opt_state = q_optim.init(q_params)

    # Get network apply functions and optimiser updates.
    apply_fns = (actor_network.apply, critic_network.apply)
    update_fns = (actor_optim.update, critic_optim.update, q_optim.update)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(
        env, actor_network, recovery_head, apply_fns, update_fns, config
    )
    learn = jax.pmap(learn, axis_name="device")

    # Pack params and initial states.
    params = Params(actor_params, critic_params)
    hstates = HiddenStates(init_policy_hstate, init_critic_hstate)

    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.logger.system_name,
            **config.logger.checkpointing.load_args,  # Other checkpoint args
        )
        # Restore the learner state from the checkpoint
        restored_params, restored_hstates = loaded_checkpoint.restore_params(
            input_params=params, restore_hstates=True, THiddenState=HiddenStates
        )
        # Update the params and hstates
        params = restored_params
        hstates = restored_hstates if restored_hstates else hstates

    # Initialise environment states and timesteps: across devices and batches.
    key, *env_keys = jax.random.split(
        key, n_devices * config.system.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(
        jnp.stack(env_keys),
    )
    reshape_states = lambda x: x.reshape(
        (n_devices, config.system.update_batch_size, config.arch.num_envs) + x.shape[1:]
    )
    # (devices, update batch size, num_envs, ...)
    env_states = tree.map(reshape_states, env_states)
    timesteps = tree.map(reshape_states, timesteps)

    # Define params to be replicated across devices and batches.
    dones = jnp.zeros(
        (config.arch.num_envs, num_agents),
        dtype=bool,
    )
    key, step_keys = jax.random.split(key)
    opt_states = OptStates(actor_opt_state, critic_opt_state)
    replicate_learner = (
        params, opt_states, hstates, step_keys, dones, q_params, q_opt_state
    )

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config.system.update_batch_size, *x.shape))
    replicate_learner = tree.map(broadcast, replicate_learner)

    # Duplicate learner across devices.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    # Initialise learner state.
    params, opt_states, hstates, step_keys, dones, q_params, q_opt_state = (
        replicate_learner
    )
    init_learner_state = ARecLearnerState(
        params=params,
        opt_states=opt_states,
        key=step_keys,
        env_state=env_states,
        timestep=timesteps,
        dones=dones,
        hstates=hstates,
        q_params=q_params,
        q_opt_state=q_opt_state,
    )
    return learn, actor_network, init_learner_state


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    if _config.env.env_name not in ("LevelBasedForaging", "RobotWarehouse"):
        raise ValueError("This ARec entry point supports only Jumanji LBF and RWARE")
    if _config.arec.coef <= 0 or _config.arec.fisher_ridge <= 0:
        raise ValueError("ARec coefficient and Fisher ridge must be positive")
    if _config.arec.q_steps < 1 or _config.arec.q_lr <= 0:
        raise ValueError("ARec q_steps and q_lr must be positive")
    if _config.logger.checkpointing.load_model:
        raise ValueError("ARec checkpoint restore is not implemented; refusing partial restore")
    _config.logger.system_name = "rec_mappo_arec"
    config = copy.deepcopy(_config)

    n_devices = len(jax.devices())

    # Set recurrent chunk size.
    if config.system.recurrent_chunk_size is None:
        config.system.recurrent_chunk_size = config.system.rollout_length
    else:
        assert (
            config.system.rollout_length % config.system.recurrent_chunk_size == 0
        ), "Rollout length must be divisible by recurrent chunk size."

        assert (
            config.arch.num_envs % config.system.num_minibatches == 0
        ), "Number of envs must be divisibile by number of minibatches."

    # Create the enviroments for train and eval.
    env, eval_env = environments.make(config=config, add_global_state=True)

    # PRNG keys.
    key, key_e, actor_net_key, critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.system.seed), num=4
    )

    # Setup learner.
    learn, actor_network, learner_state = learner_setup(
        env, (key, actor_net_key, critic_net_key), config
    )

    # Setup evaluator.
    # One key per device for evaluation.
    eval_keys = jax.random.split(key_e, n_devices)
    eval_act_fn = make_rec_eval_act_fn(actor_network.apply, config)
    evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=False)

    # Calculate total timesteps.
    config = check_total_timesteps(config)
    assert (
        config.system.num_updates > config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."

    # Calculate number of updates per evaluation.
    config.system.num_updates_per_eval = config.system.num_updates // config.arch.num_evaluation
    steps_per_rollout = (
        n_devices
        * config.system.num_updates_per_eval
        * config.system.rollout_length
        * config.system.update_batch_size
        * config.arch.num_envs
    )
    # Logger setup
    logger = MavaLogger(config)
    logger.log_config(OmegaConf.to_container(config, resolve=True))

    # Set up checkpointer
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,  # Save all config as metadata in the checkpoint
            model_name=config.logger.system_name,
            **config.logger.checkpointing.save_args,  # Checkpoint args
        )

    # Create an initial hidden state used for resetting memory for evaluation
    eval_batch_size = get_num_eval_envs(config, absolute_metric=False)
    eval_hs = ScannedRNN.initialize_carry(
        (n_devices, eval_batch_size, config.system.num_agents),
        config.network.hidden_state_dim,
    )
    # Run experiment for a total number of evaluations.
    max_episode_return = -jnp.inf
    best_params = None
    for eval_step in range(config.arch.num_evaluation):
        # Train.
        start_time = time.time()
        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        # Log the results of the training.
        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        # Separately log timesteps, actoring metrics and training metrics.
        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if ep_completed:  # only log episode metrics if an episode was completed in the rollout.
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        logger.log(learner_output.train_metrics, t, eval_step, LogEvent.TRAIN)

        # Prepare for evaluation.
        trained_params = unreplicate_batch_dim(learner_state.params.actor_params)
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)
        # Evaluate.
        eval_metrics = evaluator(trained_params, eval_keys, {"hidden_state": eval_hs})
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)
        episode_return = jnp.mean(eval_metrics["episode_return"])

        if save_checkpoint:
            # Save checkpoint of learner state
            checkpointer.save(
                timestep=steps_per_rollout * (eval_step + 1),
                unreplicated_learner_state=unreplicate_n_dims(learner_output.learner_state),
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(trained_params)
            max_episode_return = episode_return

        # Update runner state to continue training.
        learner_state = learner_output.learner_state

    # Record the performance for the final evaluation run.
    eval_performance = float(jnp.mean(eval_metrics[config.env.eval_metric]))

    # Measure absolute metric.
    if config.arch.absolute_metric:
        eval_batch_size = get_num_eval_envs(config, absolute_metric=True)
        eval_hs = ScannedRNN.initialize_carry(
            (n_devices, eval_batch_size, config.system.num_agents),
            config.network.hidden_state_dim,
        )
        abs_metric_evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=True)
        eval_keys = jax.random.split(key, n_devices)

        eval_metrics = abs_metric_evaluator(best_params, eval_keys, {"hidden_state": eval_hs})

        t = int(steps_per_rollout * (eval_step + 1))
        logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)

    # Stop the logger.
    logger.stop()

    return eval_performance


@hydra.main(
    config_path="../../../configs/default",
    config_name="rec_mappo_arec.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}Recurrent MAPPO + ARec experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
