"""A bank of separate recurrent Mava actors with the ordinary actor API.

Only masked logits cross the mapped agent boundary: TFP distributions are
constructed after ``vmap`` because TFP distribution objects are not JAX trees.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from mava.networks.distributions import IdentityTransformation


class IndependentActor:
    """One separately initialised actor parameter tree per agent."""

    def __init__(self, single_actor, num_agents: int):
        self.single_actor = single_actor
        self.num_agents = num_agents

    def _agent_inputs(self, carry, observation_done):
        observation, done = observation_done
        views = jnp.moveaxis(observation.agents_view, -2, 0)
        if views.shape[0] != self.num_agents:
            raise ValueError("Independent actor count differs from observation agents")
        return (
            observation,
            jnp.moveaxis(carry, -2, 0),
            views,
            jnp.moveaxis(observation.action_mask, -2, 0),
            jnp.moveaxis(done, -1, 0),
        )

    def init(self, key, carry, observation_done):
        observation, carries, views, masks, dones = self._agent_inputs(
            carry, observation_done
        )
        keys = jax.random.split(key, self.num_agents)

        def init_one(agent_key, agent_carry, view, mask, done):
            # Mava's ScannedRNN indexes resets as [environment, agent].
            # Keep a singleton agent axis inside each independent actor.
            agent_obs = observation._replace(
                agents_view=jnp.expand_dims(view, -2),
                action_mask=jnp.expand_dims(mask, -2),
            )
            return self.single_actor.init(
                agent_key, jnp.expand_dims(agent_carry, -2),
                (agent_obs, jnp.expand_dims(done, -1)),
            )

        return jax.vmap(init_one)(keys, carries, views, masks, dones)

    def apply(self, params, *args, return_latent=False, method=None):
        if method is not None:
            latent, action, mask = args
            if latent.shape[-2] != self.num_agents:
                raise ValueError("Independent actor count differs from latent agents")

            def apply_head(agent_params, agent_latent, agent_action, agent_mask):
                return self.single_actor.apply(
                    agent_params, agent_latent, agent_action, agent_mask, method=method
                )

            log_prob = jax.vmap(apply_head)(
                params, jnp.moveaxis(latent, -2, 0),
                jnp.moveaxis(action, -1, 0), jnp.moveaxis(mask, -2, 0),
            )
            return jnp.moveaxis(log_prob, 0, -1)

        carry, observation_done = args
        observation, carries, views, masks, dones = self._agent_inputs(
            carry, observation_done
        )

        def apply_one(agent_params, agent_carry, view, mask, done):
            agent_obs = observation._replace(
                agents_view=jnp.expand_dims(view, -2),
                action_mask=jnp.expand_dims(mask, -2),
            )
            next_carry, policy, latent = self.single_actor.apply(
                agent_params, jnp.expand_dims(agent_carry, -2),
                (agent_obs, jnp.expand_dims(done, -1)), return_latent=True,
            )
            return (
                jnp.squeeze(next_carry, axis=-2),
                jnp.squeeze(policy.distribution.logits_parameter(), axis=-2),
                jnp.squeeze(latent, axis=-2),
            )

        next_carries, masked_logits, latents = jax.vmap(apply_one)(
            params, carries, views, masks, dones
        )
        next_carry = jnp.moveaxis(next_carries, 0, -2)
        masked_logits = jnp.moveaxis(masked_logits, 0, -2)
        policy = IdentityTransformation(distribution=tfd.Categorical(logits=masked_logits))
        if return_latent:
            return next_carry, policy, jnp.moveaxis(latents, 0, -2)
        return next_carry, policy
