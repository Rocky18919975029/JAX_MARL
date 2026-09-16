"""Collect complete stochastic SMAX episodes for H1 mechanism diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from baselines.MAPPO.eval_mappo_rnn_smax import resolve_checkpoint
from baselines.MAPPO.mappo_rnn_smax import (
    ActorRNN,
    CriticRNN,
    SMAXWorldStateWrapper,
    ScannedRNN,
    batchify,
    unbatchify,
)
from baselines.MAPPO.smax_rollout import smax_rollout_horizon
from jaxmarl.environments.smax import HeuristicEnemySMAX, map_name_to_scenario
from jaxmarl.wrappers.baselines import load_params


SCHEMA_VERSION = 2


def collector_provenance():
    script_path = Path(__file__).resolve()
    try:
        commit = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "collector_git_commit": commit,
        "collector_script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def dense(params, value):
    return value @ params["kernel"] + params["bias"]


def actor_logits_from_latent(actor_params, latent, available_actions):
    """Apply the existing unnamed Dense_1/Dense_2 policy head to a latent."""

    params = actor_params["params"]
    hidden = jax.nn.relu(dense(params["Dense_1"], latent))
    logits = dense(params["Dense_2"], hidden)
    return logits - (1 - available_actions) * 1e10


def action_score(actor_params, latent, available_actions, action):
    return jax.grad(
        lambda z: jax.nn.log_softmax(
            actor_logits_from_latent(actor_params, z, available_actions)
        )[action]
    )(latent)


def compute_complete_mc_returns(arrays, gamma):
    rewards = arrays["reward"]
    global_done = arrays["global_done"]
    active = arrays["active"]
    episodes, steps, _ = rewards.shape
    mc_returns = np.zeros_like(rewards, dtype=np.float32)
    for episode in range(episodes):
        future_return = 0.0
        for timestep in range(steps - 1, -1, -1):
            if not active[episode, timestep]:
                continue
            terminal = float(global_done[episode, timestep])
            reward = rewards[episode, timestep].astype(np.float64)
            future_return = float(reward[0]) + gamma * (1.0 - terminal) * future_return
            mc_returns[episode, timestep] = future_return
    arrays["mc_return"] = mc_returns


def make_collector(config, checkpoint, batch_size):
    scenario = map_name_to_scenario(config["MAP_NAME"])
    env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])
    env = SMAXWorldStateWrapper(env, config["OBS_WITH_AGENT_ID"])
    num_agents = env.num_agents
    num_actors = num_agents * batch_size
    hidden_dim = config["GRU_HIDDEN_DIM"]
    actor = ActorRNN(env.action_space(env.agents[0]).n, config=config)
    critic = CriticRNN(config=config)
    actor_params = checkpoint["actor"]
    critic_params = checkpoint["critic"]
    sharing = config["ACTOR_PARAMETER_SHARING"]
    rollout_horizon = smax_rollout_horizon(env.max_steps)

    def policy_step(actor_hidden, obs, done, env_state, key):
        available = jax.vmap(env.get_avail_actions)(env_state)
        available_batch = batchify(available, env.agents, num_actors)
        obs_batch = batchify(obs, env.agents, num_actors)
        actor_hidden_by_agent = actor_hidden.reshape(
            (num_agents, batch_size, hidden_dim)
        )
        actor_input = (
            obs_batch.reshape((num_agents, batch_size, -1))[:, None, ...],
            done.reshape((num_agents, batch_size))[:, None, ...],
            available_batch.reshape((num_agents, batch_size, -1)),
        )
        action_keys = jax.random.split(key, num_agents)

        def apply_agent(params, hidden, inputs, agent_key):
            hidden_after, distribution, latent = actor.apply(params, hidden, inputs)
            action = distribution.sample(seed=agent_key)
            return (
                hidden_after,
                action,
                distribution.log_prob(action),
                latent,
            )

        parameter_axis = None if sharing else 0
        hidden_after, action, log_probability, latent = jax.vmap(
            apply_agent, in_axes=(parameter_axis, 0, 0, 0)
        )(actor_params, actor_hidden_by_agent, actor_input, action_keys)
        action_flat = action.reshape((1, num_actors))
        latent_by_agent = latent.reshape((num_agents, batch_size, hidden_dim))
        available_by_agent = available_batch.reshape((num_agents, batch_size, -1))
        action_by_agent = action.reshape((num_agents, batch_size))

        def scores_for_agent(params, agent_latent, agent_available, agent_action):
            return jax.vmap(action_score, in_axes=(None, 0, 0, 0))(
                params, agent_latent, agent_available, agent_action
            )

        score = jax.vmap(
            scores_for_agent,
            in_axes=(parameter_axis, 0, 0, 0),
        )(
            actor_params,
            latent_by_agent,
            available_by_agent,
            action_by_agent,
        )
        return (
            hidden_after.reshape((num_actors, hidden_dim)),
            action_flat,
            log_probability.reshape((num_agents, batch_size)),
            latent_by_agent,
            score,
            available_by_agent,
            obs_batch.reshape((num_agents, batch_size, -1)),
        )

    def collect_batch(reset_keys, rollout_key):
        obs, env_state = jax.vmap(env.reset)(reset_keys)
        actor_hidden = ScannedRNN.initialize_carry(num_actors, hidden_dim)
        critic_hidden = ScannedRNN.initialize_carry(num_actors, hidden_dim)
        done = jnp.zeros((num_actors,), dtype=jnp.bool_)
        finished = jnp.zeros((batch_size,), dtype=jnp.bool_)

        def one_step(carry, _):
            (
                obs,
                env_state,
                actor_hidden,
                critic_hidden,
                done,
                finished,
                key,
            ) = carry
            active = jnp.logical_not(finished)
            key, actor_key, env_key = jax.random.split(key, 3)
            actor_hidden_before = actor_hidden
            critic_hidden_before = critic_hidden
            (
                actor_hidden,
                action,
                log_probability,
                actor_latent,
                actor_score,
                available,
                local_obs,
            ) = policy_step(actor_hidden, obs, done, env_state, actor_key)

            world_state = obs["world_state"].swapaxes(0, 1)
            world_state_flat = world_state.reshape((num_actors, -1))
            critic_hidden, value, critic_latent = critic.apply(
                critic_params,
                critic_hidden,
                (world_state_flat[None, :], done[None, :]),
            )
            action_dict = unbatchify(action, env.agents, batch_size, num_agents)
            action_dict = {
                name: item.squeeze(axis=-1) for name, item in action_dict.items()
            }
            env_keys = jax.random.split(env_key, batch_size)

            base_state = env_state.state
            enemy_state = env_state.enemy_policy_state
            next_obs, next_state, reward, env_done, _ = jax.vmap(env.step)(
                env_keys, env_state, action_dict
            )
            reward_array = jnp.stack([reward[name] for name in env.agents], axis=1)
            done_array = jnp.stack([env_done[name] for name in env.agents], axis=1)
            global_done = env_done["__all__"]
            finished = jnp.logical_or(finished, global_done)
            next_done = batchify(env_done, env.agents, num_actors).reshape(
                (num_actors,)
            )

            record = {
                "active": active,
                "local_observation": local_obs.swapaxes(0, 1),
                "world_state": world_state.swapaxes(0, 1),
                "available_actions": available.swapaxes(0, 1),
                "action": action.reshape((num_agents, batch_size)).swapaxes(0, 1),
                "log_probability": log_probability.swapaxes(0, 1),
                "reward": reward_array,
                "done": done_array,
                "global_done": global_done,
                "alive": (available.sum(axis=-1) > 1).swapaxes(0, 1),
                "actor_hidden_before": actor_hidden_before.reshape(
                    (num_agents, batch_size, hidden_dim)
                ).swapaxes(0, 1),
                "actor_hidden_after": actor_hidden.reshape(
                    (num_agents, batch_size, hidden_dim)
                ).swapaxes(0, 1),
                "critic_hidden_before": critic_hidden_before.reshape(
                    (num_agents, batch_size, hidden_dim)
                ).swapaxes(0, 1),
                "critic_hidden_after": critic_hidden.reshape(
                    (num_agents, batch_size, hidden_dim)
                ).swapaxes(0, 1),
                "actor_latent": actor_latent.swapaxes(0, 1),
                "critic_latent": critic_latent.reshape(
                    (num_agents, batch_size, hidden_dim)
                ).swapaxes(0, 1),
                "actor_score": actor_score.swapaxes(0, 1),
                "value": value.reshape((num_agents, batch_size)).swapaxes(0, 1),
                "state_unit_positions": base_state.unit_positions,
                "state_unit_alive": base_state.unit_alive,
                "state_unit_teams": base_state.unit_teams,
                "state_unit_health": base_state.unit_health,
                "state_unit_types": base_state.unit_types,
                "state_unit_weapon_cooldowns": base_state.unit_weapon_cooldowns,
                "state_prev_movement_actions": base_state.prev_movement_actions,
                "state_prev_attack_actions": base_state.prev_attack_actions,
                "state_step": base_state.step,
                "state_done": base_state.done,
                "enemy_default_target": enemy_state.default_target,
                "enemy_last_attacked_enemy": enemy_state.last_attacked_enemy,
                "environment_step_key": env_keys,
            }
            return (
                next_obs,
                next_state,
                actor_hidden,
                critic_hidden,
                next_done,
                finished,
                key,
            ), record

        carry = (
            obs,
            env_state,
            actor_hidden,
            critic_hidden,
            done,
            finished,
            rollout_key,
        )
        carry, records = jax.lax.scan(one_step, carry, None, rollout_horizon)
        return records, carry[5]

    return jax.jit(collect_batch), env


def main():
    args = parse_args()
    if args.episodes <= 0 or args.batch_size <= 0:
        raise ValueError("--episodes and --batch-size must be positive")
    checkpoint_dir, model_path, config_path = resolve_checkpoint(args.checkpoint)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint = load_params(model_path)
    collector, env = make_collector(config, checkpoint, args.batch_size)

    output = args.output_dir.expanduser().resolve()
    metadata_path = output / "metadata.json"
    if metadata_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Diagnostic output already exists: {metadata_path}; pass --overwrite"
        )
    output.mkdir(parents=True, exist_ok=True)
    master_key = jax.random.PRNGKey(args.seed)
    shard_metadata = []
    episodes_written = 0
    num_batches = math.ceil(args.episodes / args.batch_size)
    for shard_index in tqdm(
        range(num_batches), desc="Diagnostic rollout", unit="shard"
    ):
        master_key, reset_key, rollout_key = jax.random.split(master_key, 3)
        reset_keys = jax.random.split(reset_key, args.batch_size)
        records, finished = collector(reset_keys, rollout_key)
        if not bool(np.asarray(finished).all()):
            raise RuntimeError("Some diagnostic episodes did not terminate")
        keep = min(args.batch_size, args.episodes - episodes_written)
        arrays = {
            name: np.asarray(value).swapaxes(0, 1)[:keep]
            for name, value in records.items()
        }
        arrays["reset_key"] = np.asarray(reset_keys)[:keep]
        initial_unit_types = arrays["state_unit_types"][:, 0]
        num_allies = env.num_agents
        type_ids = np.arange(6, dtype=np.int32)
        arrays["ally_unit_type_histogram"] = (
            initial_unit_types[:, :num_allies, None] == type_ids
        ).sum(axis=1, dtype=np.int32)
        arrays["enemy_unit_type_histogram"] = (
            initial_unit_types[:, num_allies:, None] == type_ids
        ).sum(axis=1, dtype=np.int32)
        arrays["diagnostic_episode_id"] = np.arange(
            episodes_written, episodes_written + keep, dtype=np.int32
        )
        compute_complete_mc_returns(arrays, float(config["GAMMA"]))
        shard_path = output / f"episodes_{shard_index:04d}.npz"
        np.savez_compressed(shard_path, **arrays)
        shard_metadata.append(
            {
                "path": shard_path.name,
                "first_episode_id": episodes_written,
                "episodes": keep,
            }
        )
        episodes_written += keep

    checkpoint_metadata = json.loads(
        (checkpoint_dir / "metadata.json").read_text(encoding="utf-8")
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "collector": "collect_mappo_smax_diagnostics.py",
        **collector_provenance(),
        "checkpoint": str(checkpoint_dir),
        "checkpoint_env_step": config.get("CHECKPOINT_ENV_STEP"),
        "checkpoint_nominal_env_step": config.get("CHECKPOINT_NOMINAL_ENV_STEP"),
        "run_id": checkpoint_metadata.get("wandb_run_id"),
        "run_name": checkpoint_metadata.get("wandb_run_name"),
        "map_name": config["MAP_NAME"],
        "training_seed": int(config["SEED"]),
        "diagnostic_seed": args.seed,
        "episodes": args.episodes,
        "batch_size": args.batch_size,
        "max_steps": env.max_steps,
        "rollout_horizon": smax_rollout_horizon(env.max_steps),
        "num_agents": env.num_agents,
        "actor_parameter_sharing": config["ACTOR_PARAMETER_SHARING"],
        "condition": config.get("EXPERIMENT_CONDITION", config["ALIGN_MODE"]),
        "align_distance": config.get("ALIGN_DISTANCE", "ln_mse"),
        "matrix_profile": config.get("MATRIX_PROFILE", ""),
        "alignment_coef": config["ALIGNMENT_COEF"],
        "gamma": float(config["GAMMA"]),
        "gae_lambda": float(config["GAE_LAMBDA"]),
        "training_rollout_steps": int(config["NUM_STEPS"]),
        "protocol_version": config.get("PROTOCOL_VERSION", ""),
        "git_commit": config.get("GIT_COMMIT", ""),
        "axis_convention": {
            "trajectory_arrays": "episode,timestep,agent,...",
            "environment_state_arrays": "episode,timestep,unit,...",
            "active_and_global_done": "episode,timestep",
            "diagnostic_episode_id": "episode",
            "reset_key_and_unit_histograms": "episode,...",
        },
        "shards": shard_metadata,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(metadata_path)


if __name__ == "__main__":
    main()
