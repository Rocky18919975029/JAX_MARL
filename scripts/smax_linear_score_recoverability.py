"""Action-conditioned linear score recoverability from frozen SMAX rollouts.

The source is an existing score-recoverability collection.  This module never
updates an RL network, and the held-out episodes never enter Fisher estimation,
critic-latent standardization, or ridge fitting.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from scripts.h1_diagnostic_data import load_diagnostics
from scripts.smax_score_recoverability import (
    canonical_condition,
    episode_fit_mask,
    fisher_whiten,
)


ARRAYS = (
    "active",
    "alive",
    "available_actions",
    "action",
    "critic_latent",
    "actor_score",
)


def sample_agent_split(
    arrays: dict,
    agent: int,
    episode_mask: np.ndarray,
    count: int,
    seed: int,
) -> dict:
    """Match the v1 per-agent sampling policy without loading actor latents."""

    if count <= 0:
        raise ValueError("Sample count must be positive")
    valid = (
        np.asarray(arrays["active"], dtype=bool)
        & np.asarray(arrays["alive"][:, :, agent], dtype=bool)
        & episode_mask[:, None]
    )
    candidates = np.argwhere(valid)
    if len(candidates) < count:
        raise RuntimeError(
            f"Agent {agent} has {len(candidates):,} eligible transitions; "
            f"{count:,} requested"
        )
    rng = np.random.default_rng(seed + agent)
    chosen = candidates[rng.choice(len(candidates), size=count, replace=False)]
    episode, timestep = chosen[:, 0], chosen[:, 1]
    action = np.asarray(arrays["action"][episode, timestep, agent], dtype=np.int32)
    available = np.asarray(
        arrays["available_actions"][episode, timestep, agent], dtype=np.float32
    )
    action_dim = available.shape[-1]
    if np.any(action < 0) or np.any(action >= action_dim):
        raise RuntimeError(f"Agent {agent} contains an out-of-range action")
    if np.any(available[np.arange(count), action] <= 0.5):
        raise RuntimeError(f"Agent {agent} contains a masked-out sampled action")
    return {
        "critic_latent": np.asarray(
            arrays["critic_latent"][episode, timestep, agent], dtype=np.float64
        ),
        "score": np.asarray(
            arrays["actor_score"][episode, timestep, agent], dtype=np.float64
        ),
        "action": action,
        "action_dim": action_dim,
        "episode": episode,
        "timestep": timestep,
    }


def fit_action_conditioned_ridge(
    fit_z: np.ndarray,
    fit_action: np.ndarray,
    fit_u: np.ndarray,
    test_z: np.ndarray,
    test_action: np.ndarray,
    *,
    action_dim: int,
    ridge: float,
    min_fit_per_action: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Fit W_a and an unpenalized intercept per action, with zero fallback.

    The normal equations are X.T @ X + ridge * diag(1, ..., 1, 0).
    A test action with insufficient fit examples is predicted by zero and
    recorded as unsupported; test outcomes never decide which model is fitted.
    """

    fit_z = np.asarray(fit_z, dtype=np.float64)
    test_z = np.asarray(test_z, dtype=np.float64)
    fit_u = np.asarray(fit_u, dtype=np.float64)
    fit_action = np.asarray(fit_action, dtype=np.int32)
    test_action = np.asarray(test_action, dtype=np.int32)
    if fit_z.ndim != 2 or test_z.ndim != 2 or fit_u.ndim != 2:
        raise ValueError("Critic latents and scores must be rank-two")
    if fit_z.shape[0] != len(fit_action) or fit_z.shape[0] != len(fit_u):
        raise ValueError("Fit latents, actions, and scores are misaligned")
    if test_z.shape[0] != len(test_action) or test_z.shape[1] != fit_z.shape[1]:
        raise ValueError("Test latents/actions are misaligned with fit latents")
    if action_dim <= 0 or fit_u.shape[1] <= 0:
        raise ValueError("Action and score dimensions must be positive")
    if ridge <= 0 or not math.isfinite(ridge):
        raise ValueError("Ridge must be finite and positive")
    if min_fit_per_action is None:
        min_fit_per_action = fit_z.shape[1] + 1
    if min_fit_per_action < 1:
        raise ValueError("Minimum fit examples per action must be positive")
    if (
        np.any(fit_action < 0)
        or np.any(fit_action >= action_dim)
        or np.any(test_action < 0)
        or np.any(test_action >= action_dim)
    ):
        raise ValueError("Action index is outside the declared action dimension")
    if not all(np.isfinite(x).all() for x in (fit_z, fit_u, test_z)):
        raise ValueError("Probe inputs and targets must be finite")

    fit_prediction = np.zeros_like(fit_u, dtype=np.float64)
    test_prediction = np.zeros((len(test_z), fit_u.shape[1]), dtype=np.float64)
    penalty = np.eye(fit_z.shape[1] + 1, dtype=np.float64)
    penalty[-1, -1] = 0.0
    support = []

    for action in range(action_dim):
        fit_mask = fit_action == action
        test_mask = test_action == action
        n_fit = int(fit_mask.sum())
        n_test = int(test_mask.sum())
        supported = n_fit >= min_fit_per_action
        if supported:
            train = np.column_stack((fit_z[fit_mask], np.ones(n_fit, dtype=np.float64)))
            target = fit_u[fit_mask]
            gram = train.T @ train + ridge * penalty
            right = train.T @ target
            try:
                weights = np.linalg.solve(gram, right)
            except np.linalg.LinAlgError as error:
                raise RuntimeError(
                    f"Ridge solve failed for action {action}, n_fit={n_fit}"
                ) from error
            fit_prediction[fit_mask] = train @ weights
            if n_test:
                held_out = np.column_stack(
                    (test_z[test_mask], np.ones(n_test, dtype=np.float64))
                )
                test_prediction[test_mask] = held_out @ weights
        support.append(
            {
                "action": action,
                "fit_count": n_fit,
                "test_count": n_test,
                "fitted": supported,
                "fallback_test_count": 0 if supported else n_test,
            }
        )
    return fit_prediction, test_prediction, support


def normalized_error(
    target: np.ndarray, prediction: np.ndarray
) -> tuple[float, float, float]:
    """Return per-sample vector MSE, target energy, and their ratio."""

    if target.shape != prediction.shape or len(target) == 0:
        raise ValueError("Targets and predictions must have the same nonempty shape")
    raw = float(np.mean(np.sum(np.square(target - prediction), axis=1)))
    energy = float(np.mean(np.sum(np.square(target), axis=1)))
    if energy <= 1e-12 or not math.isfinite(energy):
        raise RuntimeError("Whitened score energy is too small to normalize")
    return raw, energy, raw / energy


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def measure_from_collection(
    collected_dir: Path,
    source_summary: dict,
    output_dir: Path,
    *,
    ridge: float = 1e-3,
    min_fit_per_action: int | None = None,
) -> dict:
    """Evaluate every agent of one frozen checkpoint and write a result marker."""

    metadata, arrays = load_diagnostics(collected_dir, ARRAYS)
    if metadata.get("array_profile") not in (None, "score_recoverability", "full"):
        raise RuntimeError("Unsupported diagnostic array profile")
    task = str(metadata["map_name"])
    condition = canonical_condition(metadata)
    seed = int(metadata["training_seed"])
    for key, expected in (
        ("task", task),
        ("condition", condition),
        ("training_seed", seed),
        ("checkpoint", metadata["checkpoint"]),
    ):
        if source_summary[key] != expected:
            raise RuntimeError(f"Collection and source summary disagree on {key}")
    if bool(metadata.get("actor_parameter_sharing")):
        raise RuntimeError("Score-recoverability protocol requires NPS actors")

    episode_count = int(arrays["active"].shape[0])
    if episode_count != int(source_summary["episodes"]):
        raise RuntimeError("Episode count differs from the source measurement")
    fit_episode = episode_fit_mask(
        episode_count,
        float(source_summary["fit_fraction"]),
        int(source_summary["split_seed"]),
    )
    agent_rows = []
    support_rows = []
    for agent in range(int(arrays["alive"].shape[2])):
        fit = sample_agent_split(
            arrays,
            agent,
            fit_episode,
            int(source_summary["fit_samples_per_agent"]),
            int(source_summary["sampling_seed"]),
        )
        test = sample_agent_split(
            arrays,
            agent,
            ~fit_episode,
            int(source_summary["test_samples_per_agent"]),
            int(source_summary["sampling_seed"]) + 1_000_000,
        )
        if np.intersect1d(fit["episode"], test["episode"]).size:
            raise RuntimeError("Probe fit/test episodes overlap")
        if fit["action_dim"] != test["action_dim"]:
            raise RuntimeError("Fit/test action dimensions differ")
        fit_u, test_u, eigenvalues = fisher_whiten(
            fit["score"], test["score"], float(source_summary["fisher_ridge_absolute"])
        )
        fit_u = np.asarray(fit_u, dtype=np.float64)
        test_u = np.asarray(test_u, dtype=np.float64)
        fit_mean = fit["critic_latent"].mean(axis=0)
        fit_std = fit["critic_latent"].std(axis=0)
        denominator = fit_std + 1e-6
        fit_z = (fit["critic_latent"] - fit_mean) / denominator
        test_z = (test["critic_latent"] - fit_mean) / denominator
        effective_min = (
            fit_z.shape[1] + 1 if min_fit_per_action is None else min_fit_per_action
        )
        fit_prediction, test_prediction, support = fit_action_conditioned_ridge(
            fit_z,
            fit["action"],
            fit_u,
            test_z,
            test["action"],
            action_dim=fit["action_dim"],
            ridge=ridge,
            min_fit_per_action=effective_min,
        )
        fit_raw, fit_energy, fit_normalized = normalized_error(fit_u, fit_prediction)
        test_raw, test_energy, test_normalized = normalized_error(
            test_u, test_prediction
        )
        fallback_count = sum(row["fallback_test_count"] for row in support)
        agent_rows.append(
            {
                "task": task,
                "condition": condition,
                "seed": seed,
                "agent_id": agent,
                "epsilon_rec_lin": test_raw,
                "epsilon_rec_lin_normalized": test_normalized,
                "score_energy": test_energy,
                "fit_epsilon_rec_lin": fit_raw,
                "fit_epsilon_rec_lin_normalized": fit_normalized,
                "fit_score_energy": fit_energy,
                "zero_predictor_normalized": 1.0,
                "fit_samples": len(fit_z),
                "test_samples": len(test_z),
                "ridge": ridge,
                "min_fit_per_action": effective_min,
                "fallback_test_count": fallback_count,
                "fallback_test_fraction": fallback_count / len(test_z),
                "fitted_action_count": sum(row["fitted"] for row in support),
                "observed_test_action_count": sum(
                    row["test_count"] > 0 for row in support
                ),
                "fisher_trace": float(eigenvalues.sum()),
                "fisher_min_eigenvalue": float(eigenvalues.min()),
                "fisher_max_eigenvalue": float(eigenvalues.max()),
                "fisher_positive_rank": int((eigenvalues > 1e-12).sum()),
                "fisher_ridge_absolute": float(source_summary["fisher_ridge_absolute"]),
                "critic_fit_near_constant_dimensions": int((fit_std < 1e-6).sum()),
                "critic_test_standardized_norm_p99": float(
                    np.percentile(np.linalg.norm(test_z, axis=1), 99)
                ),
                "critic_test_standardized_norm_max": float(
                    np.max(np.linalg.norm(test_z, axis=1))
                ),
                "test_prediction_norm_p99": float(
                    np.percentile(np.linalg.norm(test_prediction, axis=1), 99)
                ),
                "test_prediction_norm_max": float(
                    np.max(np.linalg.norm(test_prediction, axis=1))
                ),
                "test_score_norm_p99": float(
                    np.percentile(np.linalg.norm(test_u, axis=1), 99)
                ),
                "test_score_norm_max": float(
                    np.max(np.linalg.norm(test_u, axis=1))
                ),
            }
        )
        for row in support:
            support_rows.append(
                {
                    "task": task,
                    "condition": condition,
                    "seed": seed,
                    "agent_id": agent,
                    **row,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "agent_metrics.csv", agent_rows)
    write_csv(output_dir / "action_support.csv", support_rows)
    summary = {
        "schema_version": 1,
        "protocol": "smax-linear-score-recoverability-v1.0",
        "task": task,
        "condition": condition,
        "training_seed": seed,
        "checkpoint": metadata["checkpoint"],
        "collected_dir": str(collected_dir),
        "fit_fraction": float(source_summary["fit_fraction"]),
        "split_seed": int(source_summary["split_seed"]),
        "sampling_seed": int(source_summary["sampling_seed"]),
        "episodes": episode_count,
        "fit_episodes": int(fit_episode.sum()),
        "test_episodes": int((~fit_episode).sum()),
        "fit_samples_per_agent": int(source_summary["fit_samples_per_agent"]),
        "test_samples_per_agent": int(source_summary["test_samples_per_agent"]),
        "fisher_ridge_absolute": float(source_summary["fisher_ridge_absolute"]),
        "ridge": ridge,
        "min_fit_per_action": effective_min,
        "critic_standardization": "fit_mean_and_fit_std_plus_1e-6",
        "rare_action_policy": "zero_predictor_and_report_fraction",
        "test_used_for_probe_selection": False,
        "zero_predictor_normalized": 1.0,
        "agent_aggregation": "unweighted_mean",
        "num_agents": len(agent_rows),
        "epsilon_rec_lin": float(np.mean([r["epsilon_rec_lin"] for r in agent_rows])),
        "epsilon_rec_lin_normalized": float(
            np.mean([r["epsilon_rec_lin_normalized"] for r in agent_rows])
        ),
        "fallback_test_fraction": float(
            np.mean([r["fallback_test_fraction"] for r in agent_rows])
        ),
        "max_agent_fallback_test_fraction": float(
            max(r["fallback_test_fraction"] for r in agent_rows)
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
