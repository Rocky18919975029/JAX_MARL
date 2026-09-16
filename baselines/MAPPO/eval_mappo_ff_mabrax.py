#!/usr/bin/env python3
"""Held-out deterministic evaluation for MABrax MAPPO checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import jaxmarl

try:
    from baselines.MAPPO.mappo_ff_mabrax import (
        ActorFF,
        MABraxWorldStateWrapper,
        batchify_observations,
        unbatchify_actions,
    )
except ModuleNotFoundError:  # Direct execution from baselines/MAPPO.
    from mappo_ff_mabrax import (
        ActorFF,
        MABraxWorldStateWrapper,
        batchify_observations,
        unbatchify_actions,
    )
from jaxmarl.wrappers.baselines import load_params


def resolve_checkpoint(path):
    path = path.expanduser().resolve()
    model = path if path.name == "model.safetensors" else path / "model.safetensors"
    config = model.parent / "config.json"
    if not model.is_file() or not config.is_file():
        raise FileNotFoundError(
            f"Expected model.safetensors and config.json under {path}"
        )
    return model, config


def mean_se(values):
    values = np.asarray(values, dtype=np.float64)
    standard_error = (
        values.std(ddof=1) / math.sqrt(values.size) if values.size > 1 else 0.0
    )
    return float(values.mean()), float(standard_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes < 1 or args.num_envs < 1:
        parser.error("episodes and num-envs must be positive")

    model_path, config_path = resolve_checkpoint(args.checkpoint)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    env_kwargs = dict(config.get("ENV_KWARGS", {}))
    env_kwargs["auto_reset"] = False
    raw_env = jaxmarl.make(config["ENV_NAME"], **env_kwargs)
    env = MABraxWorldStateWrapper(raw_env)
    num_agents = env.num_agents
    actor = ActorFF(
        action_dim=env.action_space(env.agents[0]).shape[0],
        hidden_size=int(config["HIDDEN_SIZE"]),
        activation=config["ACTIVATION"],
    )
    actor_params = load_params(model_path)["actor"]
    parameter_axis = None if config["ACTOR_PARAMETER_SHARING"] else 0

    def apply_actor(params, observations):
        return jax.vmap(actor.apply, in_axes=(parameter_axis, 0))(params, observations)

    horizon = int(getattr(raw_env, "episode_length", 1000))

    def evaluate_batch(reset_keys):
        observations, states = jax.vmap(env.reset)(reset_keys)
        batch_size = reset_keys.shape[0]
        active = jnp.ones((batch_size,), dtype=bool)
        returns = jnp.zeros((batch_size,))
        lengths = jnp.zeros((batch_size,), dtype=jnp.int32)

        def step(carry, step_key):
            observations, states, active, returns, lengths = carry
            flat_obs = batchify_observations(
                observations, env.agents, num_agents * batch_size
            )
            actor_obs = flat_obs.reshape((num_agents, batch_size, -1))
            policy, _ = apply_actor(actor_params, actor_obs)
            actions = policy.mean()
            step_keys = jax.random.split(step_key, batch_size)
            next_obs, next_states, rewards, dones, _ = jax.vmap(env.step)(
                step_keys,
                states,
                unbatchify_actions(actions, env.agents, batch_size),
            )
            reward = rewards[env.agents[0]]
            returns = returns + active.astype(reward.dtype) * reward
            lengths = lengths + active.astype(lengths.dtype)
            active = jnp.logical_and(active, jnp.logical_not(dones["__all__"]))
            return (next_obs, next_states, active, returns, lengths), None

        rollout_keys = jax.random.split(jax.random.fold_in(reset_keys[0], 1), horizon)
        (_, _, _, returns, lengths), _ = jax.lax.scan(
            step,
            (observations, states, active, returns, lengths),
            rollout_keys,
        )
        return returns, lengths

    evaluate_batch = jax.jit(evaluate_batch)
    rng = jax.random.PRNGKey(args.seed)
    all_returns = []
    all_lengths = []
    remaining = args.episodes
    while remaining:
        batch_size = min(args.num_envs, remaining)
        rng, reset_key = jax.random.split(rng)
        returns, lengths = evaluate_batch(jax.random.split(reset_key, batch_size))
        all_returns.extend(np.asarray(returns).tolist())
        all_lengths.extend(np.asarray(lengths).tolist())
        remaining -= batch_size

    return_mean, return_se = mean_se(all_returns)
    length_mean, length_se = mean_se(all_lengths)
    result = {
        "schema_version": 1,
        "checkpoint": str(model_path.parent),
        "environment": config["ENV_NAME"],
        "policy": "deterministic_mean",
        "episodes": args.episodes,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "return_mean": return_mean,
        "return_standard_error": return_se,
        "episode_length_mean": length_mean,
        "episode_length_standard_error": length_se,
        "episode_returns": all_returns,
        "episode_lengths": all_lengths,
    }
    output = args.output or model_path.parent / (
        f"eval-deterministic-seed{args.seed}-n{args.episodes}.json"
    )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Checkpoint: {model_path.parent}")
    print(f"Environment: {config['ENV_NAME']}")
    print(f"Episodes: {args.episodes}")
    print(f"Return: {return_mean:.6f} ± {return_se:.6f} SE")
    print(f"Episode length: {length_mean:.3f} ± {length_se:.3f} SE")
    print(f"Saved evaluation: {output}")


if __name__ == "__main__":
    main()
