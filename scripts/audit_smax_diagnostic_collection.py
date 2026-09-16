#!/usr/bin/env python3
"""Audit collected SMAX diagnostic trajectories and exactly replay a sample.

The structural audit covers every collected episode.  The replay audit rebuilds
the environment from the checkpoint config and uses the saved reset keys,
environment-step keys, and actions to reproduce selected episodes transition by
transition.  This deliberately does not trust downstream diagnostic metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


STRUCTURAL_ARRAYS = (
    "active",
    "alive",
    "ally_unit_type_histogram",
    "diagnostic_episode_id",
    "done",
    "enemy_unit_type_histogram",
    "global_done",
    "mc_return",
    "reset_key",
    "reward",
    "state_step",
    "state_unit_alive",
    "state_unit_types",
)

REPLAY_ARRAYS = (
    "active",
    "action",
    "available_actions",
    "diagnostic_episode_id",
    "done",
    "enemy_default_target",
    "enemy_last_attacked_enemy",
    "environment_step_key",
    "global_done",
    "local_observation",
    "reset_key",
    "reward",
    "state_done",
    "state_prev_attack_actions",
    "state_prev_movement_actions",
    "state_step",
    "state_unit_alive",
    "state_unit_health",
    "state_unit_positions",
    "state_unit_teams",
    "state_unit_types",
    "state_unit_weapon_cooldowns",
    "world_state",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument(
        "--replay-episodes",
        type=int,
        default=8,
        help="Number of collected episodes to replay exactly; zero disables replay.",
    )
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load_metadata(directory):
    path = directory / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing collected diagnostic metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_shard(path, names):
    with np.load(path) as data:
        missing = set(names) - set(data.files)
        if missing:
            raise RuntimeError(f"Missing arrays {sorted(missing)} from {path}")
        return {name: np.asarray(data[name]) for name in names}


def _episode_prefix_length(active):
    active = np.asarray(active, dtype=bool)
    inactive = np.flatnonzero(~active)
    length = int(inactive[0]) if inactive.size else int(active.size)
    if not active[:length].all() or active[length:].any():
        raise AssertionError("active mask is not a single true prefix")
    return length


def _complete_mc_return(reward, global_done, active, gamma):
    expected = np.zeros_like(reward, dtype=np.float32)
    future = 0.0
    for timestep in range(active.shape[0] - 1, -1, -1):
        if not active[timestep]:
            continue
        future = (
            float(reward[timestep, 0])
            + gamma * (1.0 - float(global_done[timestep])) * future
        )
        expected[timestep] = future
    return expected


def structural_audit(directory, metadata, atol, rtol):
    num_agents = int(metadata["num_agents"])
    if "gamma" in metadata:
        gamma = float(metadata["gamma"])
    else:
        checkpoint = Path(metadata["checkpoint"]).expanduser().resolve()
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        gamma = float(config["GAMMA"])
    expected_episode_id = 0
    reset_keys = set()
    ally_type_counts = Counter()
    enemy_type_counts = Counter()
    episode_lengths = []
    episode_returns = []
    errors = []
    episodes = 0

    for shard in metadata["shards"]:
        path = directory / shard["path"]
        arrays = _read_shard(path, STRUCTURAL_ARRAYS)
        shard_episodes = arrays["active"].shape[0]
        if shard_episodes != int(shard["episodes"]):
            errors.append(
                f"{path.name}: metadata says {shard['episodes']} episodes, "
                f"file contains {shard_episodes}"
            )
        for local_episode in range(shard_episodes):
            label = f"{path.name}:episode={local_episode}"
            try:
                episode_id = int(arrays["diagnostic_episode_id"][local_episode])
                if episode_id != expected_episode_id:
                    raise AssertionError(
                        f"episode id {episode_id}, expected {expected_episode_id}"
                    )
                expected_episode_id += 1

                reset_key = tuple(
                    int(value) for value in arrays["reset_key"][local_episode]
                )
                if reset_key in reset_keys:
                    raise AssertionError(f"duplicate reset key {reset_key}")
                reset_keys.add(reset_key)

                active = arrays["active"][local_episode].astype(bool)
                length = _episode_prefix_length(active)
                if length <= 0:
                    raise AssertionError("empty episode")
                episode_lengths.append(length)

                terminal = np.flatnonzero(
                    active & arrays["global_done"][local_episode].astype(bool)
                )
                if terminal.tolist() != [length - 1]:
                    raise AssertionError(
                        f"terminal transitions {terminal.tolist()}, expected {[length - 1]}"
                    )

                steps = arrays["state_step"][local_episode, :length]
                if int(steps[0]) != 0 or not np.array_equal(
                    np.diff(steps), np.ones(length - 1, dtype=steps.dtype)
                ):
                    raise AssertionError(
                        f"pre-step state counter is not 0..T-1: {steps.tolist()}"
                    )

                unit_types = arrays["state_unit_types"][local_episode, :length]
                if not np.array_equal(
                    unit_types,
                    np.broadcast_to(unit_types[0], unit_types.shape),
                ):
                    raise AssertionError("unit types change within one active episode")

                ally_types = unit_types[0, :num_agents]
                enemy_types = unit_types[0, num_agents:]
                ally_type_counts.update(int(value) for value in ally_types)
                enemy_type_counts.update(int(value) for value in enemy_types)
                type_ids = np.arange(
                    arrays["ally_unit_type_histogram"].shape[-1], dtype=np.int32
                )
                expected_ally_hist = (ally_types[:, None] == type_ids).sum(axis=0)
                expected_enemy_hist = (enemy_types[:, None] == type_ids).sum(axis=0)
                if not np.array_equal(
                    expected_ally_hist,
                    arrays["ally_unit_type_histogram"][local_episode],
                ):
                    raise AssertionError("ally type histogram does not match state")
                if not np.array_equal(
                    expected_enemy_hist,
                    arrays["enemy_unit_type_histogram"][local_episode],
                ):
                    raise AssertionError("enemy type histogram does not match state")

                rewards = arrays["reward"][local_episode]
                if not np.allclose(
                    rewards[:length],
                    rewards[:length, :1],
                    atol=atol,
                    rtol=rtol,
                ):
                    raise AssertionError("team reward differs across ally agents")
                episode_returns.append(float(rewards[:length, 0].sum()))

                state_alive = arrays["state_unit_alive"][
                    local_episode, :length, :num_agents
                ].astype(bool)
                recorded_alive = arrays["alive"][local_episode, :length].astype(bool)
                if not np.array_equal(state_alive, recorded_alive):
                    mismatch = int(np.count_nonzero(state_alive != recorded_alive))
                    raise AssertionError(
                        f"alive mask disagrees with ally state_unit_alive at {mismatch} samples"
                    )

                expected_mc = _complete_mc_return(
                    rewards,
                    arrays["global_done"][local_episode],
                    active,
                    gamma,
                )
                if not np.allclose(
                    expected_mc,
                    arrays["mc_return"][local_episode],
                    atol=atol,
                    rtol=rtol,
                ):
                    difference = float(
                        np.max(np.abs(expected_mc - arrays["mc_return"][local_episode]))
                    )
                    raise AssertionError(
                        f"stored MC return is inconsistent; max_abs={difference}"
                    )
            except AssertionError as error:
                errors.append(f"{label}: {error}")
            episodes += 1

    if episodes != int(metadata["episodes"]):
        errors.append(
            f"metadata says {metadata['episodes']} episodes, shards contain {episodes}"
        )
    return {
        "status": "pass" if not errors else "fail",
        "episodes": episodes,
        "shards": len(metadata["shards"]),
        "episode_length": {
            "minimum": int(min(episode_lengths)) if episode_lengths else None,
            "mean": float(np.mean(episode_lengths)) if episode_lengths else None,
            "maximum": int(max(episode_lengths)) if episode_lengths else None,
        },
        "episode_return": {
            "mean": float(np.mean(episode_returns)) if episode_returns else None,
            "standard_deviation": (
                float(np.std(episode_returns)) if episode_returns else None
            ),
        },
        "ally_unit_type_counts": {
            str(key): value for key, value in sorted(ally_type_counts.items())
        },
        "enemy_unit_type_counts": {
            str(key): value for key, value in sorted(enemy_type_counts.items())
        },
        "errors": errors,
    }


def _compare(label, actual, expected, atol, rtol, maxima):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{label}: shape {actual.shape} does not match stored {expected.shape}"
        )
    if actual.dtype.kind in "biu" and expected.dtype.kind in "biu":
        if not np.array_equal(actual, expected):
            raise AssertionError(f"{label}: exact values differ")
        field = label.rsplit(":", 1)[-1]
        maxima[field] = max(maxima[field], 0.0)
        return
    difference = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    field = label.rsplit(":", 1)[-1]
    maxima[field] = max(maxima[field], difference)
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError(f"{label}: max_abs={difference}")


def _replay_episode(env, arrays, episode, atol, rtol, maxima):
    import jax.numpy as jnp

    reset_key = jnp.asarray(arrays["reset_key"][episode])
    obs, state = env.reset(reset_key)
    num_agents = env.num_agents
    active = arrays["active"][episode].astype(bool)
    length = _episode_prefix_length(active)

    state_names = (
        "unit_positions",
        "unit_alive",
        "unit_teams",
        "unit_health",
        "unit_types",
        "unit_weapon_cooldowns",
        "prev_movement_actions",
        "prev_attack_actions",
        "step",
        "done",
    )
    for timestep in range(length):
        prefix = f"episode={episode},t={timestep}"
        base_state = state.state
        enemy_state = state.enemy_policy_state
        local_obs = np.stack([np.asarray(obs[name]) for name in env.agents])
        available = env.get_avail_actions(state)
        available = np.stack([np.asarray(available[name]) for name in env.agents])
        _compare(
            f"{prefix}:local_observation",
            local_obs,
            arrays["local_observation"][episode, timestep],
            atol,
            rtol,
            maxima,
        )
        _compare(
            f"{prefix}:world_state",
            np.asarray(obs["world_state"]),
            arrays["world_state"][episode, timestep],
            atol,
            rtol,
            maxima,
        )
        _compare(
            f"{prefix}:available_actions",
            available,
            arrays["available_actions"][episode, timestep],
            atol,
            rtol,
            maxima,
        )
        for state_name in state_names:
            _compare(
                f"{prefix}:state_{state_name}",
                np.asarray(getattr(base_state, state_name)),
                arrays[f"state_{state_name}"][episode, timestep],
                atol,
                rtol,
                maxima,
            )
        _compare(
            f"{prefix}:enemy_default_target",
            np.asarray(enemy_state.default_target),
            arrays["enemy_default_target"][episode, timestep],
            atol,
            rtol,
            maxima,
        )
        _compare(
            f"{prefix}:enemy_last_attacked_enemy",
            np.asarray(enemy_state.last_attacked_enemy),
            arrays["enemy_last_attacked_enemy"][episode, timestep],
            atol,
            rtol,
            maxima,
        )

        action = arrays["action"][episode, timestep]
        action_dict = {
            name: jnp.asarray(action[index]) for index, name in enumerate(env.agents)
        }
        step_key = jnp.asarray(arrays["environment_step_key"][episode, timestep])
        obs, state, reward, done, _ = env.step(step_key, state, action_dict)
        reward_array = np.asarray([reward[name] for name in env.agents])
        done_array = np.asarray([done[name] for name in env.agents])
        _compare(
            f"{prefix}:reward",
            reward_array,
            arrays["reward"][episode, timestep],
            atol,
            rtol,
            maxima,
        )
        _compare(
            f"{prefix}:done",
            done_array,
            arrays["done"][episode, timestep],
            atol,
            rtol,
            maxima,
        )
        _compare(
            f"{prefix}:global_done",
            np.asarray(done["__all__"]),
            arrays["global_done"][episode, timestep],
            atol,
            rtol,
            maxima,
        )
    return length


def replay_audit(directory, metadata, requested_episodes, atol, rtol):
    if requested_episodes <= 0:
        return {"status": "disabled", "episodes": 0, "errors": []}

    from baselines.MAPPO.eval_mappo_rnn_smax import resolve_checkpoint
    from baselines.MAPPO.mappo_rnn_smax import SMAXWorldStateWrapper
    from jaxmarl.environments.smax import HeuristicEnemySMAX, map_name_to_scenario

    checkpoint_dir, _, config_path = resolve_checkpoint(metadata["checkpoint"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["MAP_NAME"] != metadata["map_name"]:
        raise RuntimeError(
            f"Checkpoint map {config['MAP_NAME']} does not match collected map "
            f"{metadata['map_name']}"
        )
    scenario = map_name_to_scenario(config["MAP_NAME"])
    env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])
    env = SMAXWorldStateWrapper(env, config["OBS_WITH_AGENT_ID"])

    maxima = defaultdict(float)
    errors = []
    episodes_replayed = 0
    transitions_replayed = 0
    for shard in metadata["shards"]:
        if episodes_replayed >= requested_episodes:
            break
        path = directory / shard["path"]
        arrays = _read_shard(path, REPLAY_ARRAYS)
        for episode in range(arrays["active"].shape[0]):
            if episodes_replayed >= requested_episodes:
                break
            label = int(arrays["diagnostic_episode_id"][episode])
            try:
                transitions_replayed += _replay_episode(
                    env, arrays, episode, atol, rtol, maxima
                )
            except AssertionError as error:
                errors.append(f"diagnostic_episode_id={label}: {error}")
            episodes_replayed += 1

    return {
        "status": "pass" if not errors else "fail",
        "checkpoint": str(checkpoint_dir),
        "episodes": episodes_replayed,
        "transitions": transitions_replayed,
        "maximum_absolute_error": dict(sorted(maxima.items())),
        "errors": errors,
    }


def main():
    args = parse_args()
    if args.replay_episodes < 0:
        raise ValueError("--replay-episodes must be non-negative")
    directory = args.diagnostics_dir.expanduser().resolve()
    metadata = _load_metadata(directory)
    result = {
        "schema_version": 1,
        "diagnostics_dir": str(directory),
        "map_name": metadata["map_name"],
        "run_name": metadata.get("run_name"),
        "training_seed": metadata["training_seed"],
        "diagnostic_seed": metadata["diagnostic_seed"],
        "structural": structural_audit(directory, metadata, args.atol, args.rtol),
    }
    if result["structural"]["status"] == "pass":
        result["replay"] = replay_audit(
            directory,
            metadata,
            args.replay_episodes,
            args.atol,
            args.rtol,
        )
    else:
        result["replay"] = {
            "status": "skipped_due_to_structural_failure",
            "episodes": 0,
            "errors": [],
        }
    result["status"] = (
        "pass"
        if result["structural"]["status"] == "pass"
        and result["replay"]["status"] in {"pass", "disabled"}
        else "fail"
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(output)
    print(rendered, end="")
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
