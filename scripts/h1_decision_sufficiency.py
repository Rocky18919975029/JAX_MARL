#!/usr/bin/env python3
"""Counterfactual SMAX branching and frozen-latent action-ordering probe."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm.auto import tqdm

from baselines.MAPPO.eval_mappo_rnn_smax import resolve_checkpoint
from baselines.MAPPO.mappo_rnn_smax import (
    ActorRNN,
    SMAXWorldStateWrapper,
    batchify,
)
from jaxmarl.environments.smax import HeuristicEnemySMAX, map_name_to_scenario
from jaxmarl.environments.smax.heuristic_enemy import HeuristicPolicyState
from jaxmarl.environments.smax.heuristic_enemy_smax_env import State as EnemyState
from jaxmarl.environments.smax.smax_env import State as SMAXState
from jaxmarl.wrappers.baselines import load_params


def load_diagnostics(directory):
    directory = directory.expanduser().resolve()
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    arrays = {}
    for shard in metadata["shards"]:
        with np.load(directory / shard["path"]) as data:
            for name in data.files:
                arrays.setdefault(name, []).append(np.asarray(data[name]))
    arrays = {name: np.concatenate(parts, axis=0) for name, parts in arrays.items()}
    return metadata, arrays


def reconstruct_state(arrays, episode, timestep):
    base = SMAXState(
        done=jnp.asarray(arrays["state_done"][episode, timestep]),
        step=jnp.asarray(arrays["state_step"][episode, timestep]),
        unit_positions=jnp.asarray(arrays["state_unit_positions"][episode, timestep]),
        unit_alive=jnp.asarray(arrays["state_unit_alive"][episode, timestep]),
        unit_teams=jnp.asarray(arrays["state_unit_teams"][episode, timestep]),
        unit_health=jnp.asarray(arrays["state_unit_health"][episode, timestep]),
        unit_types=jnp.asarray(arrays["state_unit_types"][episode, timestep]),
        unit_weapon_cooldowns=jnp.asarray(
            arrays["state_unit_weapon_cooldowns"][episode, timestep]
        ),
        prev_movement_actions=jnp.asarray(
            arrays["state_prev_movement_actions"][episode, timestep]
        ),
        prev_attack_actions=jnp.asarray(
            arrays["state_prev_attack_actions"][episode, timestep]
        ),
    )
    enemy_policy = HeuristicPolicyState(
        default_target=jnp.asarray(arrays["enemy_default_target"][episode, timestep]),
        last_attacked_enemy=jnp.asarray(
            arrays["enemy_last_attacked_enemy"][episode, timestep]
        ),
    )
    return EnemyState(state=base, enemy_policy_state=enemy_policy)


def select_anchor_states(arrays, count, seed):
    active = arrays["active"].astype(bool)
    alive = arrays["alive"].astype(bool)
    available = arrays["available_actions"].sum(axis=-1) >= 2
    valid = active & np.any(alive & available, axis=-1)
    candidates = np.argwhere(valid)
    if not len(candidates):
        raise RuntimeError("No valid decision anchors")
    episode_lengths = active.sum(axis=1)
    stage = np.asarray(
        [
            min(2, int(3 * timestep / max(episode_lengths[episode], 1)))
            for episode, timestep in candidates
        ]
    )
    rng = np.random.default_rng(seed)
    selected = []
    per_stage = [count // 3, count // 3, count - 2 * (count // 3)]
    for stage_id, stage_count in enumerate(per_stage):
        pool = candidates[stage == stage_id]
        if len(pool) == 0:
            continue
        indices = rng.choice(len(pool), size=min(stage_count, len(pool)), replace=False)
        selected.extend(map(tuple, pool[indices]))
    if len(selected) < count:
        remaining = [
            tuple(item) for item in candidates if tuple(item) not in set(selected)
        ]
        rng.shuffle(remaining)
        selected.extend(remaining[: count - len(selected)])
    return selected[:count]


def make_brancher(config, checkpoint, continuations):
    scenario = map_name_to_scenario(config["MAP_NAME"])
    env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])
    env = SMAXWorldStateWrapper(env, config["OBS_WITH_AGENT_ID"])
    actor = ActorRNN(env.action_space(env.agents[0]).n, config=config)
    actor_params = checkpoint["actor"]
    sharing = config["ACTOR_PARAMETER_SHARING"]
    num_agents = env.num_agents
    hidden_dim = config["GRU_HIDDEN_DIM"]
    action_dim = env.action_space(env.agents[0]).n
    gamma = float(config["GAMMA"])

    def apply_policy(hidden, obs, done, env_state, key):
        available = jax.vmap(env.get_avail_actions)(env_state)
        available_batch = batchify(available, env.agents, num_agents)
        obs_batch = batchify(obs, env.agents, num_agents)
        inputs = (
            obs_batch.reshape((num_agents, 1, -1))[:, None, ...],
            done.reshape((num_agents, 1))[:, None, ...],
            available_batch.reshape((num_agents, 1, -1)),
        )
        keys = jax.random.split(key, num_agents)

        def apply_agent(params, agent_hidden, agent_input, agent_key):
            next_hidden, distribution, _ = actor.apply(
                params, agent_hidden, agent_input
            )
            return next_hidden, distribution.sample(seed=agent_key)

        parameter_axis = None if sharing else 0
        next_hidden, actions = jax.vmap(apply_agent, in_axes=(parameter_axis, 0, 0, 0))(
            actor_params,
            hidden.reshape((num_agents, 1, hidden_dim)),
            inputs,
            keys,
        )
        return next_hidden.reshape((num_agents, hidden_dim)), actions.reshape(
            (1, num_agents)
        )

    def one_continuation(
        candidate_action,
        forced_agent,
        initial_state,
        initial_obs,
        initial_hidden,
        key,
    ):
        done = jnp.zeros((num_agents,), dtype=jnp.bool_)
        key, policy_key, step_key = jax.random.split(key, 3)
        hidden, actions = apply_policy(
            initial_hidden,
            initial_obs,
            done,
            jax.tree.map(lambda value: value[None], initial_state),
            policy_key,
        )
        actions = actions.at[0, forced_agent].set(candidate_action)
        action_dict = {name: actions[0, index] for index, name in enumerate(env.agents)}
        obs, state, reward, env_done, _ = env.step(step_key, initial_state, action_dict)
        total_return = reward[env.agents[0]]
        finished = env_done["__all__"]
        done = batchify(env_done, env.agents, num_agents).reshape((num_agents,))
        discount = jnp.asarray(gamma)

        def future_step(_, carry):
            obs, state, hidden, done, finished, key, total_return, discount = carry
            key, policy_key, step_key = jax.random.split(key, 3)
            hidden, actions = apply_policy(
                hidden,
                jax.tree.map(lambda x: x[None], obs),
                done,
                jax.tree.map(lambda value: value[None], state),
                policy_key,
            )
            action_dict = {
                name: actions[0, index] for index, name in enumerate(env.agents)
            }
            next_obs, next_state, reward, env_done, _ = env.step(
                step_key, state, action_dict
            )
            active = jnp.logical_not(finished)
            total_return = total_return + active * discount * reward[env.agents[0]]
            finished = jnp.logical_or(finished, env_done["__all__"])
            done = batchify(env_done, env.agents, num_agents).reshape((num_agents,))
            return (
                next_obs,
                next_state,
                hidden,
                done,
                finished,
                key,
                total_return,
                discount * gamma,
            )

        carry = (obs, state, hidden, done, finished, key, total_return, discount)
        carry = jax.lax.fori_loop(1, env.max_steps, future_step, carry)
        return carry[6]

    candidate_actions = jnp.arange(action_dim, dtype=jnp.int32)

    def branch_all(forced_agent, state, obs, hidden, key):
        continuation_keys = jax.random.split(key, continuations)

        def evaluate_action(candidate_action):
            returns = jax.vmap(
                lambda continuation_key: one_continuation(
                    candidate_action,
                    forced_agent,
                    state,
                    obs,
                    hidden,
                    continuation_key,
                )
            )(continuation_keys)
            return returns.mean(), returns.std(ddof=1)

        return jax.vmap(evaluate_action)(candidate_actions)

    return jax.jit(branch_all), env, action_dim


def initialize_probe(key, input_dim, hidden_dim):
    key1, key2 = jax.random.split(key)
    return {
        "w1": jax.random.normal(key1, (input_dim, hidden_dim))
        * math.sqrt(2.0 / input_dim),
        "b1": jnp.zeros((hidden_dim,)),
        "w2": jax.random.normal(key2, (hidden_dim, 1)) / math.sqrt(hidden_dim),
        "b2": jnp.zeros((1,)),
    }


def probe_predict(params, inputs):
    hidden = jax.nn.relu(inputs @ params["w1"] + params["b1"])
    return (hidden @ params["w2"] + params["b2"]).squeeze(-1)


def train_probe(inputs, targets, split, hidden_dim, seed, steps, patience, batch_size):
    train = split == 0
    validation = split == 1
    target_mean = float(targets[train].mean())
    target_std = float(targets[train].std() + 1e-8)
    normalized_targets = (targets - target_mean) / target_std
    params = initialize_probe(jax.random.PRNGKey(seed), inputs.shape[1], hidden_dim)
    optimizer = optax.adam(3e-4)
    optimizer_state = optimizer.init(params)

    @jax.jit
    def update(params, optimizer_state, x, y):
        loss, grads = jax.value_and_grad(
            lambda values: jnp.mean(jnp.square(probe_predict(values, x) - y))
        )(params)
        updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
        return optax.apply_updates(params, updates), optimizer_state, loss

    best_params = params
    best_validation = math.inf
    stale = 0
    history = []
    rng = np.random.default_rng(seed)
    train_indices = np.flatnonzero(train)
    for step in range(steps):
        minibatch = rng.choice(
            train_indices,
            size=min(batch_size, len(train_indices)),
            replace=len(train_indices) < batch_size,
        )
        params, optimizer_state, train_loss = update(
            params,
            optimizer_state,
            jnp.asarray(inputs[minibatch]),
            jnp.asarray(normalized_targets[minibatch]),
        )
        if step % 20 == 0 or step == steps - 1:
            prediction = np.asarray(
                probe_predict(params, jnp.asarray(inputs[validation]))
            )
            validation_loss = float(
                np.mean(np.square(prediction - normalized_targets[validation]))
            )
            history.append((step, float(train_loss), validation_loss))
            if validation_loss < best_validation - 1e-7:
                best_validation = validation_loss
                best_params = params
                stale = 0
            else:
                stale += 20
                if stale >= patience:
                    break
    prediction = (
        np.asarray(probe_predict(best_params, jnp.asarray(inputs))) * target_std
        + target_mean
    )
    return prediction, history


def ordering_metrics(rows, predictions, tie_tolerance):
    grouped = {}
    for index, row in enumerate(rows):
        if row["split"] != 2:
            continue
        grouped.setdefault((row["anchor_id"], row["agent_id"]), []).append(index)
    kendalls = []
    pairwise = []
    top1 = []
    errors = []
    by_agent = {}
    for (_, agent), indices in grouped.items():
        truth = np.asarray([rows[index]["q_value"] for index in indices])
        predicted = predictions[indices]
        concordant = 0
        discordant = 0
        correct = 0
        compared = 0
        for first in range(len(indices)):
            for second in range(first + 1, len(indices)):
                difference = truth[first] - truth[second]
                if abs(difference) <= tie_tolerance:
                    continue
                predicted_difference = predicted[first] - predicted[second]
                compared += 1
                if np.sign(difference) == np.sign(predicted_difference):
                    concordant += 1
                    correct += 1
                else:
                    discordant += 1
        tau = (
            (concordant - discordant) / (concordant + discordant)
            if concordant + discordant
            else math.nan
        )
        accuracy = correct / compared if compared else math.nan
        agreement = float(np.argmax(truth) == np.argmax(predicted))
        mse = float(np.mean(np.square(truth - predicted)))
        kendalls.append(tau)
        pairwise.append(accuracy)
        top1.append(agreement)
        errors.append(mse)
        by_agent.setdefault(agent, []).append((tau, accuracy, agreement, mse))

    def nanmean(values):
        return float(np.nanmean(np.asarray(values, dtype=np.float64)))

    aggregate = {
        "kendall_tau": nanmean(kendalls),
        "epsilon_dec": 1.0 - nanmean(kendalls),
        "pairwise_accuracy": nanmean(pairwise),
        "top1_agreement": nanmean(top1),
        "q_mse": nanmean(errors),
        "num_test_anchor_agents": len(grouped),
        "num_test_anchor_states": len(grouped),
    }
    agent_metrics = []
    for agent, values in sorted(by_agent.items()):
        agent_metrics.append(
            {
                "agent_id": agent,
                "kendall_tau": nanmean([value[0] for value in values]),
                "pairwise_accuracy": nanmean([value[1] for value in values]),
                "top1_agreement": nanmean([value[2] for value in values]),
                "q_mse": nanmean([value[3] for value in values]),
                "num_test_anchor_states": len(values),
            }
        )
    return aggregate, agent_metrics


def append_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--anchors", type=int, default=256)
    parser.add_argument("--continuations", type=int, default=32)
    parser.add_argument("--seed", type=int, default=41001)
    parser.add_argument("--probe-steps", type=int, default=2000)
    parser.add_argument("--probe-patience", type=int, default=200)
    parser.add_argument("--probe-batch-size", type=int, default=2048)
    args = parser.parse_args()
    metadata, arrays = load_diagnostics(args.diagnostics_dir)
    checkpoint_dir, model_path, config_path = resolve_checkpoint(args.checkpoint)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint = load_params(model_path)
    brancher, env, action_dim = make_brancher(config, checkpoint, args.continuations)
    anchors = select_anchor_states(arrays, args.anchors, args.seed)
    rng = np.random.default_rng(args.seed)
    unique_episodes = np.unique([episode for episode, _ in anchors])
    rng.shuffle(unique_episodes)
    split_lookup = {}
    for index, episode in enumerate(unique_episodes):
        fraction = index / max(len(unique_episodes), 1)
        split_lookup[int(episode)] = (
            0 if fraction < 0.6 else (1 if fraction < 0.8 else 2)
        )

    rows = []
    for anchor_id, (episode, timestep) in enumerate(
        tqdm(anchors, desc="Counterfactual anchors", unit="anchor")
    ):
        state = reconstruct_state(arrays, episode, timestep)
        obs = {
            name: jnp.asarray(arrays["local_observation"][episode, timestep, agent])[
                None
            ]
            for agent, name in enumerate(env.agents)
        }
        obs["world_state"] = jnp.asarray(arrays["world_state"][episode, timestep])[None]
        hidden = jnp.asarray(arrays["actor_hidden_before"][episode, timestep])
        for agent in range(env.num_agents):
            if not arrays["alive"][episode, timestep, agent]:
                continue
            available = arrays["available_actions"][episode, timestep, agent].astype(
                bool
            )
            if available.sum() < 2:
                continue
            key = jax.random.fold_in(jax.random.PRNGKey(args.seed), anchor_id)
            key = jax.random.fold_in(key, agent)
            q_values, q_std = brancher(agent, state, obs, hidden, key)
            q_values = np.asarray(q_values)
            q_std = np.asarray(q_std)
            latent = arrays["actor_latent"][episode, timestep, agent]
            unit_type = int(arrays["state_unit_types"][episode, timestep, agent])
            for action in np.flatnonzero(available):
                rows.append(
                    {
                        "anchor_id": anchor_id,
                        "episode_id": int(episode),
                        "timestep": int(timestep),
                        "stage": min(
                            2,
                            int(3 * timestep / max(arrays["active"][episode].sum(), 1)),
                        ),
                        "agent_id": agent,
                        "unit_type": unit_type,
                        "action": int(action),
                        "q_value": float(q_values[action]),
                        "q_continuation_std": float(q_std[action]),
                        "split": split_lookup[int(episode)],
                        "latent": latent,
                    }
                )
    if not rows or not all(
        any(row["split"] == split for row in rows) for split in range(3)
    ):
        raise RuntimeError(
            "Anchor dataset does not contain all train/validation/test splits"
        )

    latents = np.stack([row.pop("latent") for row in rows]).astype(np.float32)
    actions = np.asarray([row["action"] for row in rows])
    inputs = np.concatenate(
        (latents, np.eye(action_dim, dtype=np.float32)[actions]), axis=1
    )
    targets = np.asarray([row["q_value"] for row in rows], dtype=np.float32)
    split = np.asarray([row["split"] for row in rows], dtype=np.int32)
    predictions, history = train_probe(
        inputs,
        targets,
        split,
        int(config["GRU_HIDDEN_DIM"]),
        args.seed + 1,
        args.probe_steps,
        args.probe_patience,
        args.probe_batch_size,
    )
    return_scale = max(float(np.std(targets)), 1.0)
    aggregate, agent_metrics = ordering_metrics(rows, predictions, 1e-3 * return_scale)

    train = split == 0
    action_means = np.asarray(
        [
            (
                targets[train & (actions == action)].mean()
                if np.any(train & (actions == action))
                else targets[train].mean()
            )
            for action in range(action_dim)
        ]
    )
    baseline_predictions = action_means[actions]
    baseline_aggregate, _ = ordering_metrics(
        rows, baseline_predictions, 1e-3 * return_scale
    )
    common = {
        "run_id": metadata.get("run_id"),
        "run_name": metadata["run_name"],
        "task": metadata["map_name"],
        "actor_parameterization": (
            "ps" if metadata["actor_parameter_sharing"] else "nps"
        ),
        "align_mode": metadata["condition"],
        "shuffled": metadata["condition"].endswith("_shuffled"),
        "seed": metadata["training_seed"],
        "checkpoint_step": metadata.get("checkpoint_env_step"),
        "num_anchor_states": len(anchors),
        "continuations_per_action": args.continuations,
        "protocol_version": metadata["protocol_version"],
        "git_commit": metadata["git_commit"],
    }
    output_rows = [
        {
            **common,
            "agent_id": "all",
            "unit_type": "all",
            **aggregate,
        }
    ]
    for agent_metric in agent_metrics:
        output_rows.append(
            {
                **common,
                "agent_id": agent_metric.pop("agent_id"),
                "unit_type": "all",
                **agent_metric,
                "epsilon_dec": 1.0 - agent_metric["kendall_tau"],
                "num_test_anchor_agents": 1,
            }
        )
    for unit_type in sorted(set(row["unit_type"] for row in rows)):
        indices = [
            index for index, row in enumerate(rows) if row["unit_type"] == unit_type
        ]
        type_rows = [rows[index] for index in indices]
        type_predictions = predictions[indices]
        type_aggregate, _ = ordering_metrics(
            type_rows, type_predictions, 1e-3 * return_scale
        )
        output_rows.append(
            {
                **common,
                "agent_id": "all",
                "unit_type": unit_type,
                **type_aggregate,
                "num_test_anchor_states": type_aggregate["num_test_anchor_agents"],
            }
        )
    append_csv(args.output_csv.expanduser().resolve(), output_rows)

    output_dir = args.diagnostics_dir.expanduser().resolve()
    np.savez_compressed(
        output_dir / "decision_branching.npz",
        latent=latents,
        action=actions,
        q_value=targets,
        q_prediction=predictions,
        split=split,
        anchor_id=np.asarray([row["anchor_id"] for row in rows]),
        episode_id=np.asarray([row["episode_id"] for row in rows]),
        agent_id=np.asarray([row["agent_id"] for row in rows]),
        unit_type=np.asarray([row["unit_type"] for row in rows]),
    )
    summary = {
        "schema_version": 1,
        **common,
        **aggregate,
        "action_only_baseline": baseline_aggregate,
        "probe_history": history,
        "episode_disjoint_split": True,
    }
    (output_dir / "decision_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
