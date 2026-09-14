#!/usr/bin/env python3
"""Compute H1 latent policy-update distortion from diagnostic rollout shards."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def load_diagnostics(directory):
    directory = directory.expanduser().resolve()
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    arrays = {}
    for shard in metadata["shards"]:
        with np.load(directory / shard["path"]) as data:
            for name in data.files:
                arrays.setdefault(name, []).append(np.asarray(data[name]))
    arrays = {
        name: (
            np.concatenate(parts, axis=0)
            if parts[0].ndim and parts[0].shape[0] == metadata["shards"][0]["episodes"]
            else parts[0]
        )
        for name, parts in arrays.items()
    }
    return metadata, arrays


def make_features(x, mean, scale, random_weight, random_bias):
    normalized = (x - mean) / scale
    nonlinear = np.maximum(normalized @ random_weight + random_bias, 0.0)
    return np.concatenate((np.ones((len(x), 1)), normalized, nonlinear), axis=1).astype(
        np.float64
    )


def cross_fitted_reference(states, returns, episode_ids, active, seed, width, ridge):
    rng = np.random.default_rng(seed)
    input_dim = states.shape[-1]
    random_weight = rng.normal(
        0.0, 1.0 / math.sqrt(input_dim), size=(input_dim, width)
    ).astype(np.float32)
    random_bias = rng.normal(0.0, 0.1, size=(width,)).astype(np.float32)
    predictions = np.full_like(returns, np.nan, dtype=np.float64)
    folds = episode_ids % 5
    fold_metrics = []
    for fold in range(5):
        train_mask = active & (folds[:, None] != fold)
        test_mask = active & (folds[:, None] == fold)
        train_x = states[train_mask].astype(np.float64)
        train_y = returns[train_mask].astype(np.float64)
        test_x = states[test_mask].astype(np.float64)
        mean = train_x.mean(axis=0)
        scale = train_x.std(axis=0)
        scale[scale < 1e-6] = 1.0
        train_features = make_features(train_x, mean, scale, random_weight, random_bias)
        test_features = make_features(test_x, mean, scale, random_weight, random_bias)
        gram = train_features.T @ train_features
        penalty = ridge * np.trace(gram) / max(gram.shape[0], 1)
        coefficients = np.linalg.solve(
            gram + max(penalty, 1e-8) * np.eye(gram.shape[0]),
            train_features.T @ train_y,
        )
        fold_prediction = test_features @ coefficients
        predictions[test_mask] = fold_prediction
        fold_metrics.append(
            {
                "fold": fold,
                "train_episodes": int(np.sum(folds != fold)),
                "test_episodes": int(np.sum(folds == fold)),
                "test_samples": int(test_mask.sum()),
                "mse": float(np.mean(np.square(fold_prediction - returns[test_mask]))),
            }
        )
    if np.isnan(predictions[active]).any():
        raise RuntimeError("Cross-fitted reference baseline left active samples unset")
    return predictions, fold_metrics


def fisher_statistics(scores, reference_advantage, critic_advantage):
    """Compute ridge-independent Fisher quantities once for one sample pool."""

    scores = np.asarray(scores, dtype=np.float64)
    reference_advantage = np.asarray(reference_advantage, dtype=np.float64)
    critic_advantage = np.asarray(critic_advantage, dtype=np.float64)
    count, dimension = scores.shape
    delta = np.mean(scores * (reference_advantage - critic_advantage)[:, None], axis=0)
    g_reference = np.mean(scores * reference_advantage[:, None], axis=0)
    g_critic = np.mean(scores * critic_advantage[:, None], axis=0)
    fisher = (scores.T @ scores) / count
    fisher = (fisher + fisher.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(fisher)
    return {
        "count": count,
        "dimension": dimension,
        "delta": delta,
        "g_reference": g_reference,
        "g_critic": g_critic,
        "fisher": fisher,
        "eigenvalues": eigenvalues,
    }


def fisher_metrics_from_statistics(statistics, ridge_multiplier):
    """Apply a ridge choice without rebuilding the expensive Fisher matrix."""

    count = statistics["count"]
    dimension = statistics["dimension"]
    delta = statistics["delta"]
    g_reference = statistics["g_reference"]
    g_critic = statistics["g_critic"]
    fisher = statistics["fisher"]
    eigenvalues = statistics["eigenvalues"]
    base_ridge = ridge_multiplier * np.trace(fisher) / dimension
    ridge = max(float(base_ridge), 1e-12)
    attempts = 0
    while True:
        try:
            cholesky = np.linalg.cholesky(fisher + ridge * np.eye(dimension))
            break
        except np.linalg.LinAlgError:
            attempts += 1
            ridge *= 10.0
            if attempts >= 8:
                raise RuntimeError(
                    "Fisher Cholesky failed; eigenvalue range "
                    f"[{eigenvalues.min()}, {eigenvalues.max()}]"
                )

    def quadratic(vector):
        intermediate = np.linalg.solve(cholesky, vector)
        solution = np.linalg.solve(cholesky.T, intermediate)
        return float(vector @ solution)

    epsilon = max(quadratic(delta), 0.0)
    reference_energy = max(quadratic(g_reference), 0.0)
    positive = np.maximum(eigenvalues, 0.0)
    effective_rank = float(
        np.square(positive.sum()) / max(np.square(positive).sum(), 1e-20)
    )
    positive_nonzero = positive[positive > max(positive.max() * 1e-12, 1e-15)]
    condition = (
        float((positive.max() + ridge) / (positive_nonzero.min() + ridge))
        if positive_nonzero.size
        else 1.0
    )
    cosine_denominator = np.linalg.norm(g_reference) * np.linalg.norm(g_critic)
    cosine = (
        float(g_reference @ g_critic / cosine_denominator)
        if cosine_denominator > 0
        else math.nan
    )
    return {
        "epsilon_lat": epsilon,
        "energy_ref": reference_energy,
        "r_lat": epsilon / (reference_energy + 1e-8),
        "delta_l2": float(np.linalg.norm(delta)),
        "gradient_cosine": cosine,
        "fisher_effective_rank": effective_rank,
        "fisher_condition_number": condition,
        "fisher_min_eigenvalue": float(eigenvalues.min()),
        "fisher_max_eigenvalue": float(eigenvalues.max()),
        "fisher_ridge": ridge,
        "fisher_ridge_escalations": attempts,
        "num_valid_samples": count,
    }


def fisher_metrics(scores, reference_advantage, critic_advantage, ridge_multiplier):
    return fisher_metrics_from_statistics(
        fisher_statistics(scores, reference_advantage, critic_advantage),
        ridge_multiplier,
    )


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
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--reference-seed", type=int, default=31001)
    parser.add_argument("--reference-width", type=int, default=256)
    parser.add_argument("--reference-ridge", type=float, default=1e-5)
    parser.add_argument("--fisher-ridge", type=float, default=1e-3)
    parser.add_argument("--minibatch-samples-per-agent", type=int, default=4096)
    args = parser.parse_args()
    metadata, arrays = load_diagnostics(args.diagnostics_dir)
    active = arrays["active"].astype(bool)
    episode_ids = arrays["diagnostic_episode_id"].astype(np.int64)
    # Agent 0's view contains the raw global state plus a constant agent-ID
    # suffix. Using one fixed view prevents the tested critic from defining the
    # independent reference baseline.
    states = arrays["world_state"][:, :, 0]
    team_returns = arrays["mc_return"][:, :, 0]
    baseline, fold_metrics = cross_fitted_reference(
        states,
        team_returns,
        episode_ids,
        active,
        args.reference_seed,
        args.reference_width,
        args.reference_ridge,
    )
    reference_advantage = team_returns - baseline
    rows = []
    per_agent_epsilon = []
    per_agent_energy = []
    ridge_sensitivity = []
    minibatch_distribution = []
    distribution_rng = np.random.default_rng(args.reference_seed + 99)
    unit_types = arrays["state_unit_types"][:, :, : metadata["num_agents"]]
    for agent in range(metadata["num_agents"]):
        mask = active & arrays["alive"][:, :, agent].astype(bool)
        agent_statistics = fisher_statistics(
            arrays["actor_score"][:, :, agent][mask],
            reference_advantage[mask],
            arrays["gae_raw"][:, :, agent][mask],
        )
        metrics = fisher_metrics_from_statistics(
            agent_statistics,
            args.fisher_ridge,
        )
        per_agent_epsilon.append(metrics["epsilon_lat"])
        per_agent_energy.append(metrics["energy_ref"])
        for ridge_multiplier in (1e-4, 1e-3, 1e-2):
            sensitivity_metrics = fisher_metrics_from_statistics(
                agent_statistics,
                ridge_multiplier,
            )
            ridge_sensitivity.append(
                {
                    "agent_id": agent,
                    "ridge_multiplier": ridge_multiplier,
                    "epsilon_lat": sensitivity_metrics["epsilon_lat"],
                    "r_lat": sensitivity_metrics["r_lat"],
                }
            )
        score_pool = arrays["actor_score"][:, :, agent][mask]
        reference_pool = reference_advantage[mask]
        critic_pool = arrays["gae_raw"][:, :, agent][mask]
        permutation = distribution_rng.permutation(len(score_pool))
        chunk_size = args.minibatch_samples_per_agent
        for chunk_index, start in enumerate(range(0, len(permutation), chunk_size)):
            indices = permutation[start : start + chunk_size]
            if len(indices) < max(score_pool.shape[-1] + 1, chunk_size // 2):
                continue
            chunk_metrics = fisher_metrics(
                score_pool[indices],
                reference_pool[indices],
                critic_pool[indices],
                args.fisher_ridge,
            )
            minibatch_distribution.append(
                {
                    "agent_id": agent,
                    "chunk_index": chunk_index,
                    "samples": len(indices),
                    "epsilon_lat": chunk_metrics["epsilon_lat"],
                    "r_lat": chunk_metrics["r_lat"],
                }
            )
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
                "agent_id": agent,
                "unit_type": "all",
                **metrics,
                "protocol_version": metadata["protocol_version"],
                "git_commit": metadata["git_commit"],
            }
        )
        for unit_type in np.unique(unit_types[:, :, agent][mask]):
            type_mask = mask & (unit_types[:, :, agent] == unit_type)
            if type_mask.sum() < arrays["actor_score"].shape[-1] + 1:
                continue
            type_metrics = fisher_metrics(
                arrays["actor_score"][:, :, agent][type_mask],
                reference_advantage[type_mask],
                arrays["gae_raw"][:, :, agent][type_mask],
                args.fisher_ridge,
            )
            rows.append(
                {
                    **rows[-1],
                    "unit_type": int(unit_type),
                    **type_metrics,
                }
            )

    aggregate = {
        "schema_version": 1,
        "run_id": metadata.get("run_id"),
        "run_name": metadata["run_name"],
        "checkpoint_step": metadata.get("checkpoint_env_step"),
        "epsilon_lat": float(np.mean(per_agent_epsilon)),
        "energy_ref": float(np.mean(per_agent_energy)),
        "r_lat": float(np.mean(per_agent_epsilon) / (np.mean(per_agent_energy) + 1e-8)),
        "per_agent_mean_matches": bool(
            np.isclose(
                np.mean(per_agent_epsilon),
                np.mean(
                    [row["epsilon_lat"] for row in rows if row["unit_type"] == "all"]
                ),
            )
        ),
        "reference_baseline_folds": fold_metrics,
        "fisher_ridge_multiplier": args.fisher_ridge,
        "ridge_sensitivity": ridge_sensitivity,
        "training_minibatch_sized_distribution": minibatch_distribution,
    }
    output_dir = args.diagnostics_dir.expanduser().resolve()
    (output_dir / "latent_distortion_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "reference_advantages.npz",
        reference_baseline=baseline,
        reference_advantage=reference_advantage,
    )
    append_csv(args.output_csv.expanduser().resolve(), rows)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
