#!/usr/bin/env python3
"""Empirical Bellman-closure diagnostic for frozen critic representations."""

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
from jaxmarl.wrappers.baselines import load_params


HEAD_OPTIMIZER = optax.adam(3e-4)


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


def initialize_head(key, input_dim, hidden_dim):
    first, second = jax.random.split(key)
    return {
        "w1": jax.random.normal(first, (input_dim, hidden_dim))
        * math.sqrt(2.0 / input_dim),
        "b1": jnp.zeros((hidden_dim,)),
        "w2": jax.random.normal(second, (hidden_dim, 1)) / math.sqrt(hidden_dim),
        "b2": jnp.zeros((1,)),
    }


def predict(params, inputs):
    hidden = jax.nn.relu(inputs @ params["w1"] + params["b1"])
    return (hidden @ params["w2"] + params["b2"]).squeeze(-1)


def checkpoint_head(critic_params):
    params = critic_params["params"]
    return {
        "w1": params["Dense_1"]["kernel"],
        "b1": params["Dense_1"]["bias"],
        "w2": params["Dense_2"]["kernel"],
        "b2": params["Dense_2"]["bias"],
    }


@jax.jit
def update_head_block(params, optimizer_state, inputs, targets, batch_indices):
    """Run a block of head updates in one device dispatch."""

    def update_one(carry, indices):
        current_params, current_optimizer_state = carry
        x = inputs[indices]
        y = targets[indices]
        loss, grads = jax.value_and_grad(
            lambda values: jnp.mean(jnp.square(predict(values, x) - y))
        )(current_params)
        updates, current_optimizer_state = HEAD_OPTIMIZER.update(
            grads, current_optimizer_state, current_params
        )
        current_params = optax.apply_updates(current_params, updates)
        return (current_params, current_optimizer_state), loss

    return jax.lax.scan(update_one, (params, optimizer_state), batch_indices)


def fit_head(
    inputs,
    targets,
    train_indices,
    validation_indices,
    hidden_dim,
    seed,
    steps,
    batch_size,
    patience,
):
    target_mean = float(targets[train_indices].mean())
    target_std = float(targets[train_indices].std() + 1e-8)
    normalized_targets = ((targets - target_mean) / target_std).astype(np.float32)
    params = initialize_head(jax.random.PRNGKey(seed), inputs.shape[1], hidden_dim)
    optimizer_state = HEAD_OPTIMIZER.init(params)
    device_inputs = jnp.asarray(inputs)
    device_targets = jnp.asarray(normalized_targets)
    validation_indices_device = jnp.asarray(validation_indices)

    rng = np.random.default_rng(seed)
    best_params = params
    best_validation = math.inf
    stale = 0
    history = []
    validation_steps = list(range(0, steps, 50))
    if validation_steps[-1] != steps - 1:
        validation_steps.append(steps - 1)
    previous_step = -1
    minibatch_size = min(batch_size, len(train_indices))
    replace = len(train_indices) < batch_size
    for step in validation_steps:
        block_length = step - previous_step
        minibatches = np.stack(
            [
                rng.choice(
                    train_indices,
                    size=minibatch_size,
                    replace=replace,
                )
                for _ in range(block_length)
            ]
        ).astype(np.int32)
        (params, optimizer_state), train_losses = update_head_block(
            params,
            optimizer_state,
            device_inputs,
            device_targets,
            jnp.asarray(minibatches),
        )
        train_loss = train_losses[-1]
        validation_prediction = np.asarray(
            predict(params, device_inputs[validation_indices_device])
        )
        validation_loss = float(
            np.mean(
                np.square(
                    validation_prediction - normalized_targets[validation_indices]
                )
            )
        )
        history.append((step, float(train_loss), validation_loss))
        if validation_loss < best_validation - 1e-7:
            best_validation = validation_loss
            best_params = params
            stale = 0
        else:
            stale += 50
            if stale >= patience:
                break
        previous_step = step
    # Store affine target scaling alongside the network without changing its
    # architecture. Predictions are mapped back to reward units by this pair.
    return best_params, target_mean, target_std, history


def scaled_predict(model, inputs):
    params, mean, scale = model[:3]
    return np.asarray(predict(params, jnp.asarray(inputs))) * scale + mean


def episode_split(episode_count, seed):
    episodes = np.arange(episode_count)
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    train_end = int(0.6 * episode_count)
    validation_end = int(0.8 * episode_count)
    assignment = np.empty(episode_count, dtype=np.int8)
    assignment[episodes[:train_end]] = 0
    assignment[episodes[train_end:validation_end]] = 1
    assignment[episodes[validation_end:]] = 2
    return assignment


def explained_variance(prediction, target):
    variance = np.var(target)
    return (
        float(1.0 - np.var(target - prediction) / variance)
        if variance > 0
        else math.nan
    )


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    # A checkpoint retry can follow a successful CSV write but failed summary
    # write, so keep this per-checkpoint table idempotent.
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--patience", type=int, default=250)
    parser.add_argument("--seed", type=int, default=51001)
    args = parser.parse_args()
    if args.heads < 1 or args.steps < 1:
        raise ValueError("--heads and --steps must be positive")
    metadata, arrays = load_diagnostics(args.diagnostics_dir)
    checkpoint_dir, model_path, config_path = resolve_checkpoint(args.checkpoint)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint = load_params(model_path)

    active = arrays["active"].astype(bool)
    alive = arrays["alive"].astype(bool)
    episode_count, time_count, agent_count = alive.shape
    assignment = episode_split(episode_count, args.seed)
    episode_grid = np.broadcast_to(np.arange(episode_count)[:, None, None], alive.shape)
    valid = np.broadcast_to(active[..., None], alive.shape) & alive
    z_current = arrays["critic_latent"]
    z_next = np.zeros_like(z_current)
    z_next[:, :-1] = z_current[:, 1:]
    state_current = arrays["world_state"]
    reward = arrays["reward"]
    terminal = np.broadcast_to(arrays["global_done"][..., None], alive.shape)
    mc_return = arrays["mc_return"]
    gae = arrays["gae_raw"]
    stored_value = arrays["value"]

    z = z_current[valid].astype(np.float32)
    next_z = z_next[valid].astype(np.float32)
    state = state_current[valid].astype(np.float32)
    rewards = reward[valid].astype(np.float32)
    terminals = terminal[valid].astype(np.float32)
    mc_targets = mc_return[valid].astype(np.float32)
    raw_gae = gae[valid].astype(np.float32)
    value_targets = stored_value[valid].astype(np.float32)
    sample_episode = episode_grid[valid]
    split = assignment[sample_episode]
    train_indices = np.flatnonzero(split == 0)
    validation_indices = np.flatnonzero(split == 1)
    test_indices = np.flatnonzero(split == 2)
    if min(len(train_indices), len(validation_indices), len(test_indices)) == 0:
        raise RuntimeError("Episode-disjoint split produced an empty sample set")

    source_models = [(checkpoint_head(checkpoint["critic"]), 0.0, 1.0, [])]
    source_scales = []
    rng = np.random.default_rng(args.seed)
    train_episodes = np.flatnonzero(assignment == 0)
    for head_id in tqdm(range(1, args.heads), desc="Source value heads", unit="head"):
        bootstrapped_episodes = rng.choice(
            train_episodes, size=len(train_episodes), replace=True
        )
        bootstrap_indices = np.concatenate(
            [
                np.flatnonzero((sample_episode == episode) & (split == 0))
                for episode in bootstrapped_episodes
            ]
        )
        source_models.append(
            fit_head(
                z,
                mc_targets,
                bootstrap_indices,
                validation_indices,
                int(config["GRU_HIDDEN_DIM"]),
                args.seed + 1000 + head_id,
                args.steps,
                args.batch_size,
                args.patience,
            )
        )

    rows = []
    residuals = []
    state_residuals = []
    for head_id, source_model in enumerate(
        tqdm(source_models, desc="Bellman images", unit="head")
    ):
        next_prediction = scaled_predict(source_model, next_z)
        bellman_target = (
            rewards + float(config["GAMMA"]) * (1.0 - terminals) * next_prediction
        )
        source_output = scaled_predict(source_model, z)
        source_scales.append(float(np.std(source_output[test_indices])))
        target_model = fit_head(
            z,
            bellman_target,
            train_indices,
            validation_indices,
            int(config["GRU_HIDDEN_DIM"]),
            args.seed + 2000 + head_id,
            args.steps,
            args.batch_size,
            args.patience,
        )
        state_model = fit_head(
            state,
            bellman_target,
            train_indices,
            validation_indices,
            2 * int(config["GRU_HIDDEN_DIM"]),
            args.seed + 3000 + head_id,
            args.steps,
            args.batch_size,
            args.patience,
        )
        latent_prediction = scaled_predict(target_model, z[test_indices])
        state_prediction = scaled_predict(state_model, state[test_indices])
        target_test = bellman_target[test_indices]
        residual = float(np.mean(np.square(latent_prediction - target_test)))
        state_residual = float(np.mean(np.square(state_prediction - target_test)))
        residuals.append(residual)
        state_residuals.append(state_residual)
        rows.append(
            {
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
                "source_head_id": head_id,
                "source_output_std": source_scales[-1],
                "bellman_residual": residual,
                "full_state_residual": state_residual,
                "excess_residual": max(residual - state_residual, 0.0),
                "value_mc_mse": float(
                    np.mean(
                        np.square(
                            value_targets[test_indices] - mc_targets[test_indices]
                        )
                    )
                ),
                "td_residual": float(
                    np.mean(
                        np.square(
                            value_targets[test_indices] - bellman_target[test_indices]
                        )
                    )
                ),
                "explained_variance": explained_variance(
                    value_targets[test_indices], mc_targets[test_indices]
                ),
                "gae_mc_correlation": float(
                    np.corrcoef(raw_gae[test_indices], mc_targets[test_indices])[0, 1]
                ),
                "num_test_samples": len(test_indices),
                "protocol_version": metadata["protocol_version"],
                "git_commit": metadata["git_commit"],
            }
        )
    write_csv(args.output_csv.expanduser().resolve(), rows)
    residuals_array = np.asarray(residuals)
    state_residuals_array = np.asarray(state_residuals)
    summary = {
        "schema_version": 1,
        "run_id": metadata.get("run_id"),
        "run_name": metadata["run_name"],
        "checkpoint_step": metadata.get("checkpoint_env_step"),
        "epsilon_bell": float(residuals_array.max()),
        "epsilon_bell_median": float(np.median(residuals_array)),
        "epsilon_bell_p90": float(np.quantile(residuals_array, 0.9)),
        "epsilon_bell_excess": float(
            np.maximum(residuals_array - state_residuals_array, 0.0).max()
        ),
        "source_head_output_scales": source_scales,
        "heads": args.heads,
        "episode_disjoint_split": True,
        "train_episodes": int(np.sum(assignment == 0)),
        "validation_episodes": int(np.sum(assignment == 1)),
        "test_episodes": int(np.sum(assignment == 2)),
    }
    output_dir = args.diagnostics_dir.expanduser().resolve()
    (output_dir / "bellman_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
