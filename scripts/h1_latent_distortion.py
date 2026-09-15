#!/usr/bin/env python3
"""Compute the canonical NPS H1 latent-update distortion diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

try:
    from h1_diagnostic_data import LATENT_ARRAYS, load_diagnostics
except ModuleNotFoundError:
    from scripts.h1_diagnostic_data import LATENT_ARRAYS, load_diagnostics


REFERENCE_PROTOCOL = "baseline_free_mc_return_train_matched_gae"


def load_training_protocol(metadata):
    """Read GAE settings from new metadata or the frozen checkpoint config."""

    keys = ("gamma", "gae_lambda", "training_rollout_steps")
    if all(key in metadata for key in keys):
        return tuple(metadata[key] for key in keys)
    checkpoint = metadata.get("checkpoint")
    if not checkpoint:
        raise RuntimeError("Diagnostic metadata does not identify its checkpoint")
    config_path = Path(checkpoint).expanduser().resolve() / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint config is unavailable: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return (
        float(config["GAMMA"]),
        float(config["GAE_LAMBDA"]),
        int(config["NUM_STEPS"]),
    )


def reconstruct_training_gae(
    rewards,
    values,
    global_done,
    active,
    gamma,
    gae_lambda,
    rollout_steps,
):
    """Reproduce training GAE, including bootstrap at rollout boundaries."""

    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    global_done = np.asarray(global_done, dtype=bool)
    active = np.asarray(active, dtype=bool)
    if rewards.shape != values.shape or rewards.shape[:2] != active.shape:
        raise ValueError("Reward, value, and active trajectory shapes are inconsistent")
    if global_done.shape != active.shape or rollout_steps <= 0:
        raise ValueError("Invalid termination mask or training rollout length")

    episodes, timesteps, agents = rewards.shape
    advantages = np.zeros_like(values, dtype=np.float64)
    for episode in range(episodes):
        for block_start in range(0, timesteps, rollout_steps):
            block_stop = min(block_start + rollout_steps, timesteps)
            next_value = (
                values[episode, block_stop].copy()
                if block_stop < timesteps and active[episode, block_stop]
                else np.zeros((agents,), dtype=np.float64)
            )
            gae = np.zeros((agents,), dtype=np.float64)
            for timestep in range(block_stop - 1, block_start - 1, -1):
                if not active[episode, timestep]:
                    continue
                continuation = 1.0 - float(global_done[episode, timestep])
                delta = (
                    rewards[episode, timestep]
                    + gamma * next_value * continuation
                    - values[episode, timestep]
                )
                gae = delta + gamma * gae_lambda * continuation * gae
                advantages[episode, timestep] = gae
                next_value = values[episode, timestep]
    return advantages


def fisher_statistics(scores, reference_signal, critic_signal):
    scores = np.asarray(scores, dtype=np.float64)
    reference_signal = np.asarray(reference_signal, dtype=np.float64)
    critic_signal = np.asarray(critic_signal, dtype=np.float64)
    if scores.ndim != 2 or len(scores) == 0:
        raise ValueError("Actor score samples must be a non-empty matrix")
    if len(reference_signal) != len(scores) or len(critic_signal) != len(scores):
        raise ValueError("Reference, critic, and score sample counts must match")
    g_reference = np.mean(scores * reference_signal[:, None], axis=0)
    g_critic = np.mean(scores * critic_signal[:, None], axis=0)
    fisher = (scores.T @ scores) / len(scores)
    fisher = (fisher + fisher.T) / 2.0
    return {
        "count": len(scores),
        "dimension": scores.shape[1],
        "g_reference": g_reference,
        "g_critic": g_critic,
        "delta": g_reference - g_critic,
        "fisher": fisher,
        "eigenvalues": np.linalg.eigvalsh(fisher),
    }


def fisher_metrics_from_statistics(statistics, ridge_absolute):
    ridge = float(ridge_absolute)
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("The absolute Fisher ridge must be finite and positive")
    fisher = statistics["fisher"]
    dimension = statistics["dimension"]
    try:
        cholesky = np.linalg.cholesky(fisher + ridge * np.eye(dimension))
    except np.linalg.LinAlgError as error:
        eigenvalues = statistics["eigenvalues"]
        raise RuntimeError(
            "Fisher Cholesky failed with the fixed ridge; eigenvalue range "
            f"[{eigenvalues.min()}, {eigenvalues.max()}], ridge={ridge}"
        ) from error

    def quadratic(vector):
        intermediate = np.linalg.solve(cholesky, vector)
        solution = np.linalg.solve(cholesky.T, intermediate)
        return max(float(vector @ solution), 0.0)

    eigenvalues = statistics["eigenvalues"]
    positive = np.maximum(eigenvalues, 0.0)
    effective_rank = float(
        np.square(positive.sum()) / max(np.square(positive).sum(), 1e-20)
    )
    cosine_denominator = np.linalg.norm(statistics["g_reference"]) * np.linalg.norm(
        statistics["g_critic"]
    )
    return {
        "epsilon_lat": quadratic(statistics["delta"]),
        "gradient_cosine": (
            float(
                statistics["g_reference"] @ statistics["g_critic"] / cosine_denominator
            )
            if cosine_denominator > 0
            else math.nan
        ),
        "delta_l2": float(np.linalg.norm(statistics["delta"])),
        "reference_gradient_l2": float(np.linalg.norm(statistics["g_reference"])),
        "fisher_effective_rank": effective_rank,
        "fisher_min_eigenvalue": float(eigenvalues.min()),
        "fisher_max_eigenvalue": float(eigenvalues.max()),
        "fisher_ridge_absolute": ridge,
        "num_valid_samples": statistics["count"],
    }


def fisher_metrics(scores, reference_signal, critic_signal, ridge_absolute):
    """Compute one agent's diagnostic directly from sample arrays."""

    return fisher_metrics_from_statistics(
        fisher_statistics(scores, reference_signal, critic_signal), ridge_absolute
    )


def vector_cosine(first, second):
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(first @ second / denominator) if denominator > 0 else math.nan


def convergence_episode_budgets(episode_ids):
    """Use nested M, 2M, and 4M sets of complete held-out episodes."""

    ids = np.asarray(episode_ids, dtype=np.int64)
    episodes = len(ids)
    if episodes < 4 or episodes % 4 or not np.array_equal(ids, np.arange(episodes)):
        raise ValueError("MC convergence requires 4M sequential, complete episodes")
    base = episodes // 4
    return (base, 2 * base, 4 * base)


def heldout_episode_returns(reward, active):
    """Sum only actual transitions, never the post-termination rollout padding."""

    return np.sum(
        np.where(active, np.asarray(reward)[:, :, 0], 0.0),
        axis=1,
        dtype=np.float64,
    )


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--fisher-ridge-absolute", type=float, required=True)
    args = parser.parse_args()

    metadata, arrays = load_diagnostics(args.diagnostics_dir, LATENT_ARRAYS)
    if metadata["actor_parameter_sharing"]:
        raise ValueError("The canonical H1 analysis is restricted to NPS checkpoints")
    gamma, gae_lambda, rollout_steps = load_training_protocol(metadata)
    active = arrays["active"].astype(bool)
    alive = arrays["alive"].astype(bool)
    episode_ids = arrays["diagnostic_episode_id"].astype(np.int64)
    episode_budgets = convergence_episode_budgets(episode_ids)
    budget_labels = dict(zip(episode_budgets, ("M", "2M", "4M")))
    team_returns = arrays["mc_return"][:, :, 0].astype(np.float64)
    training_gae = reconstruct_training_gae(
        arrays["reward"],
        arrays["value"],
        arrays["global_done"],
        active,
        gamma,
        gae_lambda,
        rollout_steps,
    )

    common = {
        "run_id": metadata.get("run_id"),
        "run_name": metadata["run_name"],
        "task": metadata["map_name"],
        "actor_parameterization": "nps",
        "condition": metadata["condition"],
        "align_distance": metadata.get("align_distance", "ln_mse"),
        "seed": metadata["training_seed"],
        "checkpoint_step": metadata.get("checkpoint_env_step"),
        "checkpoint_nominal_step": metadata.get("checkpoint_nominal_env_step"),
        "reference_protocol": REFERENCE_PROTOCOL,
        "fisher_ridge_absolute": args.fisher_ridge_absolute,
        "protocol_version": metadata["protocol_version"],
        "git_commit": metadata["git_commit"],
    }
    per_agent_rows = []
    full_statistics = []
    convergence_statistics = {budget: [] for budget in episode_budgets}
    convergence_rows = []

    for agent in range(metadata["num_agents"]):
        mask = active & alive[:, :, agent]
        statistics = fisher_statistics(
            arrays["actor_score"][:, :, agent][mask],
            team_returns[mask],
            training_gae[:, :, agent][mask],
        )
        full_statistics.append(statistics)
        metrics = fisher_metrics_from_statistics(statistics, args.fisher_ridge_absolute)
        per_agent_rows.append({**common, "agent_id": agent, **metrics})

        for budget in episode_budgets:
            budget_mask = mask & (episode_ids[:, None] < budget)
            budget_statistics = fisher_statistics(
                arrays["actor_score"][:, :, agent][budget_mask],
                team_returns[budget_mask],
                training_gae[:, :, agent][budget_mask],
            )
            convergence_statistics[budget].append(budget_statistics)
            budget_metrics = fisher_metrics_from_statistics(
                budget_statistics, args.fisher_ridge_absolute
            )
            convergence_rows.append(
                {
                    **common,
                    "scope": "agent",
                    "agent_id": agent,
                    "episode_budget": budget,
                    "episode_budget_label": budget_labels[budget],
                    "reference_gradient_cosine_to_4m": vector_cosine(
                        budget_statistics["g_reference"],
                        statistics["g_reference"],
                    ),
                    "reference_gradient_relative_l2_to_4m": float(
                        np.linalg.norm(
                            budget_statistics["g_reference"] - statistics["g_reference"]
                        )
                        / (np.linalg.norm(statistics["g_reference"]) + 1e-12)
                    ),
                    **budget_metrics,
                }
            )

    epsilon_lat = float(
        np.sum(
            [
                fisher_metrics_from_statistics(item, args.fisher_ridge_absolute)[
                    "epsilon_lat"
                ]
                for item in full_statistics
            ]
        )
    )
    if not math.isfinite(epsilon_lat):
        raise RuntimeError("The fixed-ridge latent distortion is non-finite")
    full_gradient = np.concatenate([item["g_reference"] for item in full_statistics])
    convergence_summary = []
    for budget in episode_budgets:
        items = convergence_statistics[budget]
        budget_gradient = np.concatenate([item["g_reference"] for item in items])
        budget_epsilon = float(
            np.sum(
                [
                    fisher_metrics_from_statistics(item, args.fisher_ridge_absolute)[
                        "epsilon_lat"
                    ]
                    for item in items
                ]
            )
        )
        row = {
            **common,
            "scope": "aggregate",
            "agent_id": "all",
            "episode_budget": budget,
            "episode_budget_label": budget_labels[budget],
            "epsilon_lat": budget_epsilon,
            "epsilon_lat_relative_error_to_4m": float(
                abs(budget_epsilon - epsilon_lat) / (abs(epsilon_lat) + 1e-12)
            ),
            "reference_gradient_l2": float(np.linalg.norm(budget_gradient)),
            "reference_gradient_cosine_to_4m": vector_cosine(
                budget_gradient, full_gradient
            ),
            "reference_gradient_relative_l2_to_4m": float(
                np.linalg.norm(budget_gradient - full_gradient)
                / (np.linalg.norm(full_gradient) + 1e-12)
            ),
            "num_valid_samples": int(np.sum([item["count"] for item in items])),
        }
        convergence_summary.append(row)
        convergence_rows.append(row)

    episode_returns = heldout_episode_returns(arrays["reward"], active)
    output_dir = args.diagnostics_dir.expanduser().resolve()
    summary = {
        "schema_version": 1,
        **common,
        "definition": "sum_i delta_g_i^T (F_i + xi I)^-1 delta_g_i",
        "epsilon_lat": epsilon_lat,
        "num_agents": metadata["num_agents"],
        "reference_signal": "complete_discounted_mc_return_to_go",
        "reference_baseline": "none",
        "reference_has_bootstrap": False,
        "critic_signal": "unnormalized_training_matched_gae",
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "training_rollout_steps": rollout_steps,
        "mask": "active_and_agent_alive",
        "heldout_episodes": len(episode_ids),
        "heldout_episode_return_mean": float(episode_returns.mean()),
        "heldout_episode_return_std": float(episode_returns.std(ddof=1)),
        "heldout_episode_return_stderr": float(
            episode_returns.std(ddof=1) / math.sqrt(len(episode_returns))
        ),
        "heldout_discounted_return_mean": float(team_returns[:, 0].mean()),
        "per_agent": per_agent_rows,
        "mc_convergence": convergence_summary,
    }
    (output_dir / "latent_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "reference_signals.npz",
        complete_mc_return=team_returns,
        training_matched_gae=training_gae,
        g_reference=np.stack([item["g_reference"] for item in full_statistics]),
        g_critic=np.stack([item["g_critic"] for item in full_statistics]),
        delta_g=np.stack([item["delta"] for item in full_statistics]),
    )
    write_csv(args.output_csv.expanduser().resolve(), per_agent_rows)
    write_csv(output_dir / "mc_convergence.csv", convergence_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
