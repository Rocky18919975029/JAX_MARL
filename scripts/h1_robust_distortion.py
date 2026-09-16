#!/usr/bin/env python3
"""Recompute robust H1 latent-update diagnostics from collected trajectories.

The collector stores complete episodes but not the rollout phase at which an
episode began during training.  Consequently, exact historical rollout phase
cannot be recovered.  This diagnostic marginalizes GAE uniformly over every
possible phase of the fixed training rollout horizon.  It reports that signal
alongside the legacy episode-aligned GAE signal, without modifying either the
collected data or the canonical legacy outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from h1_diagnostic_data import ROBUST_DISTORTION_ARRAYS, load_diagnostics
    from h1_latent_distortion import (
        convergence_episode_budgets,
        heldout_episode_returns,
        load_training_protocol,
        reconstruct_training_gae,
    )
except ModuleNotFoundError:
    from scripts.h1_diagnostic_data import (
        ROBUST_DISTORTION_ARRAYS,
        load_diagnostics,
    )
    from scripts.h1_latent_distortion import (
        convergence_episode_budgets,
        heldout_episode_returns,
        load_training_protocol,
        reconstruct_training_gae,
    )


PROTOCOL = "h1-robust-distortion-v1.0"
PHASE_PROTOCOL = "uniform_marginal_over_all_training_rollout_start_phases"
AGGREGATIONS = ("slot", "slot_x_type")


def write_csv(path: Path, rows):
    if not rows:
        raise RuntimeError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def uniform_phase_marginalized_gae(
    rewards,
    values,
    global_done,
    active,
    gamma,
    gae_lambda,
    rollout_steps,
):
    """Average fixed-boundary GAE over every possible episode start phase.

    A boundary after transition ``t`` cuts the recursive GAE term but retains
    the one-step TD bootstrap through ``value[t + 1]``, exactly as training
    does at the end of a rollout update.
    """

    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    global_done = np.asarray(global_done, dtype=bool)
    active = np.asarray(active, dtype=bool)
    if rewards.shape != values.shape or rewards.shape[:2] != active.shape:
        raise ValueError("Reward, value, and active trajectory shapes disagree")
    if global_done.shape != active.shape or rollout_steps <= 0:
        raise ValueError("Invalid termination mask or rollout horizon")

    episodes, timesteps, agents = rewards.shape
    phases = np.arange(rollout_steps, dtype=np.int64)
    output = np.zeros_like(values, dtype=np.float64)
    for episode in range(episodes):
        length = int(active[episode].sum())
        if length <= 0:
            continue
        if not np.all(active[episode, :length]) or np.any(active[episode, length:]):
            raise ValueError("Each collected episode must have one active prefix")
        gae = np.zeros((rollout_steps, agents), dtype=np.float64)
        for timestep in range(length - 1, -1, -1):
            continuation = 1.0 - float(global_done[episode, timestep])
            next_value = (
                values[episode, timestep + 1]
                if timestep + 1 < length
                else np.zeros((agents,), dtype=np.float64)
            )
            delta = (
                rewards[episode, timestep]
                + gamma * continuation * next_value
                - values[episode, timestep]
            )
            boundary_after = (phases + timestep + 1) % rollout_steps == 0
            recursive = np.where(boundary_after[:, None], 0.0, gae)
            gae = delta[None, :] + gamma * gae_lambda * continuation * recursive
            output[episode, timestep] = gae.mean(axis=0)
    return output


@dataclass(frozen=True)
class GroupStatistics:
    key: tuple
    slot: int
    unit_type: int | None
    count: int
    weight: float
    fisher_eigenvalues: np.ndarray
    g_reference_spectral: np.ndarray
    g_raw_spectral: np.ndarray
    g_phase_spectral: np.ndarray


def make_group_statistics(
    scores,
    reference_signal,
    raw_signal,
    phase_signal,
    key,
    slot,
    unit_type,
    weight,
):
    scores = np.asarray(scores, dtype=np.float64)
    reference_signal = np.asarray(reference_signal, dtype=np.float64)
    raw_signal = np.asarray(raw_signal, dtype=np.float64)
    phase_signal = np.asarray(phase_signal, dtype=np.float64)
    if scores.ndim != 2 or not len(scores):
        raise ValueError(f"Fisher group {key} has no valid score samples")
    if not all(
        len(item) == len(scores)
        for item in (reference_signal, raw_signal, phase_signal)
    ):
        raise ValueError(f"Signal sample counts disagree for Fisher group {key}")
    fisher = (scores.T @ scores) / len(scores)
    fisher = (fisher + fisher.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(fisher)
    # Roundoff can leave tiny negative eigenvalues in a PSD score covariance.
    eigenvalues = np.maximum(eigenvalues, 0.0)
    g_reference = np.mean(scores * reference_signal[:, None], axis=0)
    g_raw = np.mean(scores * raw_signal[:, None], axis=0)
    g_phase = np.mean(scores * phase_signal[:, None], axis=0)
    return GroupStatistics(
        key=key,
        slot=slot,
        unit_type=unit_type,
        count=len(scores),
        weight=float(weight),
        fisher_eigenvalues=eigenvalues,
        g_reference_spectral=eigenvectors.T @ g_reference,
        g_raw_spectral=eigenvectors.T @ g_raw,
        g_phase_spectral=eigenvectors.T @ g_phase,
    )


def grouped_statistics(
    arrays,
    raw_gae,
    phase_gae,
    episode_budget,
    aggregation,
    num_agents,
):
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    active = np.asarray(arrays["active"], dtype=bool)
    alive = np.asarray(arrays["alive"], dtype=bool)
    episode_ids = np.asarray(arrays["diagnostic_episode_id"], dtype=np.int64)
    within_budget = episode_ids[:, None] < int(episode_budget)
    reference = np.asarray(arrays["mc_return"][:, :, 0], dtype=np.float64)
    scores = np.asarray(arrays["actor_score"])
    unit_types = np.asarray(arrays["state_unit_types"][:, :, :num_agents])
    integer_types = unit_types.astype(np.int64)
    if not np.array_equal(unit_types, integer_types):
        raise ValueError("Unit type identifiers must be integer-valued")

    groups = []
    for slot in range(num_agents):
        slot_mask = active & alive[:, :, slot] & within_budget
        if not slot_mask.any():
            raise RuntimeError(f"No valid samples for actor slot {slot}")
        if aggregation == "slot":
            groups.append(
                make_group_statistics(
                    scores[:, :, slot][slot_mask],
                    reference[slot_mask],
                    raw_gae[:, :, slot][slot_mask],
                    phase_gae[:, :, slot][slot_mask],
                    key=(slot,),
                    slot=slot,
                    unit_type=None,
                    weight=1.0,
                )
            )
            continue
        slot_types = integer_types[:, :, slot][slot_mask]
        slot_count = int(slot_mask.sum())
        for unit_type in sorted(np.unique(slot_types).tolist()):
            type_mask = slot_mask & (integer_types[:, :, slot] == unit_type)
            count = int(type_mask.sum())
            groups.append(
                make_group_statistics(
                    scores[:, :, slot][type_mask],
                    reference[type_mask],
                    raw_gae[:, :, slot][type_mask],
                    phase_gae[:, :, slot][type_mask],
                    key=(slot, int(unit_type)),
                    slot=slot,
                    unit_type=int(unit_type),
                    weight=count / slot_count,
                )
            )
    return groups


def solved_inner(group, ridge, first_spectral, second_spectral):
    """Compute a regularized Fisher inner product in cached eigen-coordinates."""

    inverse_spectrum = 1.0 / (group.fisher_eigenvalues + ridge)
    return float(np.sum(first_spectral * second_spectral * inverse_spectrum))


def aggregate_fisher_metrics(groups, ridge_absolute):
    ridge = float(ridge_absolute)
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("Fisher ridge must be finite and positive")
    raw_epsilon = 0.0
    phase_epsilon = 0.0
    reference_norm_sq = 0.0
    phase_norm_sq = 0.0
    reference_phase_inner = 0.0
    group_cache = []
    for group in groups:
        raw_delta = group.g_reference_spectral - group.g_raw_spectral
        phase_delta = group.g_reference_spectral - group.g_phase_spectral
        raw_value = solved_inner(group, ridge, raw_delta, raw_delta)
        phase_value = solved_inner(group, ridge, phase_delta, phase_delta)
        reference_value = solved_inner(
            group,
            ridge,
            group.g_reference_spectral,
            group.g_reference_spectral,
        )
        critic_value = solved_inner(
            group,
            ridge,
            group.g_phase_spectral,
            group.g_phase_spectral,
        )
        cross_value = solved_inner(
            group,
            ridge,
            group.g_reference_spectral,
            group.g_phase_spectral,
        )
        raw_epsilon += group.weight * raw_value
        phase_epsilon += group.weight * phase_value
        reference_norm_sq += group.weight * reference_value
        phase_norm_sq += group.weight * critic_value
        reference_phase_inner += group.weight * cross_value
        group_cache.append(
            (
                group,
                raw_value,
                phase_value,
                reference_value,
                critic_value,
                cross_value,
            )
        )
    denominator = math.sqrt(max(reference_norm_sq * phase_norm_sq, 0.0))
    natural_cosine = (
        float(np.clip(reference_phase_inner / denominator, -1.0, 1.0))
        if denominator > 0
        else math.nan
    )
    optimal_scale = (
        max(0.0, reference_phase_inner / phase_norm_sq) if phase_norm_sq > 0 else 0.0
    )
    scale_corrected = 0.0
    group_rows = []
    for (
        group,
        raw_value,
        phase_value,
        reference_value,
        critic_value,
        cross_value,
    ) in group_cache:
        scale_delta = (
            group.g_reference_spectral - optimal_scale * group.g_phase_spectral
        )
        scale_value = solved_inner(group, ridge, scale_delta, scale_delta)
        scale_corrected += group.weight * scale_value
        group_rows.append(
            {
                "slot": group.slot,
                "unit_type": "all" if group.unit_type is None else group.unit_type,
                "group_weight": group.weight,
                "num_valid_samples": group.count,
                "epsilon_lat_raw": raw_value,
                "epsilon_lat_phase_matched": phase_value,
                "epsilon_lat_optimal_scale": scale_value,
                "reference_natural_norm_sq": reference_value,
                "critic_natural_norm_sq": critic_value,
                "reference_critic_natural_inner": cross_value,
            }
        )
    return {
        "epsilon_lat_raw": max(raw_epsilon, 0.0),
        "epsilon_lat_phase_matched": max(phase_epsilon, 0.0),
        "fisher_natural_gradient_cosine": natural_cosine,
        "epsilon_lat_optimal_scale": max(scale_corrected, 0.0),
        "optimal_nonnegative_critic_scale": optimal_scale,
        "reference_natural_norm_sq": max(reference_norm_sq, 0.0),
        "critic_natural_norm_sq": max(phase_norm_sq, 0.0),
        "reference_critic_natural_inner": reference_phase_inner,
        "num_groups": len(groups),
        "num_valid_samples_weighted": float(
            sum(group.weight * group.count for group in groups)
        ),
    }, group_rows


def parse_ridges(value):
    ridges = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not ridges or any(not math.isfinite(item) or item <= 0 for item in ridges):
        raise argparse.ArgumentTypeError("Fisher ridges must be positive finite values")
    if len(set(ridges)) != len(ridges):
        raise argparse.ArgumentTypeError("Fisher ridge sweep contains duplicates")
    return ridges


def checkpoint_step(metadata, diagnostics_dir):
    nominal = metadata.get("checkpoint_nominal_env_step")
    actual = metadata.get("checkpoint_env_step")
    if nominal is not None:
        return int(actual if actual is not None else nominal), int(nominal)
    if diagnostics_dir.name == "initial":
        return 0, 0
    if diagnostics_dir.name.startswith("step_"):
        step = int(diagnostics_dir.name.removeprefix("step_"))
        return step, step
    config = json.loads(
        (Path(metadata["checkpoint"]) / "config.json").read_text(encoding="utf-8")
    )
    total = int(config["TOTAL_TIMESTEPS"])
    return int(actual if actual is not None else total), total


def validate_episode_type_constancy(arrays, num_agents):
    """Require each actor slot's unit type to stay fixed within an episode."""

    active = np.asarray(arrays["active"], dtype=bool)
    unit_types = np.asarray(arrays["state_unit_types"][:, :, :num_agents])
    changes = 0
    for episode in range(len(active)):
        steps = np.flatnonzero(active[episode])
        if not len(steps):
            continue
        first = unit_types[episode, steps[0]]
        changes += int(np.any(unit_types[episode, steps] != first, axis=0).sum())
    if changes:
        raise RuntimeError(
            f"Observed {changes} within-episode actor-slot unit-type changes"
        )


def canonical_raw_comparison(diagnostics_dir, lookup, full_budget):
    """Verify the slot/raw/full-budget cell reproduces the legacy diagnostic."""

    path = diagnostics_dir / "latent_summary.json"
    if not path.is_file():
        return {"available": False}
    legacy = json.loads(path.read_text(encoding="utf-8"))
    ridge = float(legacy["fisher_ridge_absolute"])
    matching = [
        row
        for (aggregation, row_ridge, budget), row in lookup.items()
        if aggregation == "slot"
        and budget == full_budget
        and math.isclose(row_ridge, ridge, rel_tol=1e-12, abs_tol=1e-15)
    ]
    if len(matching) != 1:
        return {
            "available": False,
            "reason": f"legacy ridge {ridge} is absent from requested sweep",
        }
    recomputed = float(matching[0]["epsilon_lat_raw"])
    canonical = float(legacy["epsilon_lat"])
    absolute_error = abs(recomputed - canonical)
    if not math.isclose(recomputed, canonical, rel_tol=1e-7, abs_tol=1e-10):
        raise RuntimeError(
            "Robust raw/slot recomputation does not reproduce latent_summary.json: "
            f"canonical={canonical}, recomputed={recomputed}, ridge={ridge}"
        )
    return {
        "available": True,
        "fisher_ridge_absolute": ridge,
        "canonical_epsilon_lat": canonical,
        "recomputed_epsilon_lat_raw": recomputed,
        "absolute_error": absolute_error,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fisher-ridges",
        type=parse_ridges,
        default=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
    )
    args = parser.parse_args()
    diagnostics_dir = args.diagnostics_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata, arrays = load_diagnostics(diagnostics_dir, ROBUST_DISTORTION_ARRAYS)
    if metadata.get("actor_parameter_sharing"):
        raise ValueError("Robust H1 distortion is currently restricted to NPS")
    num_agents = int(metadata["num_agents"])
    validate_episode_type_constancy(arrays, num_agents)
    gamma, gae_lambda, rollout_steps = load_training_protocol(metadata)
    active = np.asarray(arrays["active"], dtype=bool)
    episode_ids = np.asarray(arrays["diagnostic_episode_id"], dtype=np.int64)
    episode_budgets = convergence_episode_budgets(episode_ids)
    budget_labels = dict(zip(episode_budgets, ("M", "2M", "4M")))
    raw_gae = reconstruct_training_gae(
        arrays["reward"],
        arrays["value"],
        arrays["global_done"],
        active,
        gamma,
        gae_lambda,
        rollout_steps,
    )
    phase_gae = uniform_phase_marginalized_gae(
        arrays["reward"],
        arrays["value"],
        arrays["global_done"],
        active,
        gamma,
        gae_lambda,
        rollout_steps,
    )
    actual_step, nominal_step = checkpoint_step(metadata, diagnostics_dir)
    common = {
        "run_id": metadata.get("run_id"),
        "run_name": metadata["run_name"],
        "task": metadata["map_name"],
        "actor_parameterization": "nps",
        "condition": metadata["condition"],
        "align_distance": metadata.get("align_distance", "ln_mse"),
        "seed": int(metadata["training_seed"]),
        "checkpoint_step": actual_step,
        "checkpoint_nominal_step": nominal_step,
        "protocol_version": metadata.get("protocol_version", ""),
        "training_git_commit": metadata.get("git_commit", ""),
        "robust_distortion_protocol": PROTOCOL,
        "phase_protocol": PHASE_PROTOCOL,
        "training_rollout_steps": rollout_steps,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
    }
    aggregate_rows = []
    group_rows = []
    full_groups = {}
    for budget in episode_budgets:
        for aggregation in AGGREGATIONS:
            groups = grouped_statistics(
                arrays,
                raw_gae,
                phase_gae,
                budget,
                aggregation,
                num_agents,
            )
            if budget == episode_budgets[-1]:
                full_groups[aggregation] = groups
            for ridge in args.fisher_ridges:
                metrics, details = aggregate_fisher_metrics(groups, ridge)
                row_common = {
                    **common,
                    "aggregation": aggregation,
                    "episode_budget": budget,
                    "episode_budget_label": budget_labels[budget],
                    "fisher_ridge_absolute": ridge,
                }
                aggregate_rows.append({**row_common, **metrics})
                for detail in details:
                    group_rows.append({**row_common, **detail})

    full_budget = episode_budgets[-1]
    lookup = {
        (
            row["aggregation"],
            row["fisher_ridge_absolute"],
            row["episode_budget"],
        ): row
        for row in aggregate_rows
    }
    canonical_comparison = canonical_raw_comparison(
        diagnostics_dir, lookup, full_budget
    )

    single_type_consistency = {"applicable": False}
    full_slot_groups = full_groups["slot"]
    full_type_groups = full_groups["slot_x_type"]
    if len(full_type_groups) == len(full_slot_groups) == num_agents:
        max_absolute_error = 0.0
        for ridge in args.fisher_ridges:
            slot_metrics, _ = aggregate_fisher_metrics(full_slot_groups, ridge)
            type_metrics, _ = aggregate_fisher_metrics(full_type_groups, ridge)
            for metric in (
                "epsilon_lat_raw",
                "epsilon_lat_phase_matched",
                "fisher_natural_gradient_cosine",
                "epsilon_lat_optimal_scale",
            ):
                error = abs(float(slot_metrics[metric]) - float(type_metrics[metric]))
                max_absolute_error = max(max_absolute_error, error)
                if not math.isclose(
                    float(slot_metrics[metric]),
                    float(type_metrics[metric]),
                    rel_tol=1e-8,
                    abs_tol=1e-10,
                ):
                    raise RuntimeError(
                        "Single-type slot and slot_x_type metrics disagree: "
                        f"metric={metric}, ridge={ridge}, error={error}"
                    )
        single_type_consistency = {
            "applicable": True,
            "status": "pass",
            "max_absolute_error": max_absolute_error,
        }
    primary_metrics = (
        "epsilon_lat_raw",
        "epsilon_lat_phase_matched",
        "fisher_natural_gradient_cosine",
        "epsilon_lat_optimal_scale",
    )
    convergence_rows = []
    for row in aggregate_rows:
        full = lookup[
            (
                row["aggregation"],
                row["fisher_ridge_absolute"],
                full_budget,
            )
        ]
        convergence = dict(row)
        for metric in primary_metrics:
            convergence[f"{metric}_difference_to_4m"] = row[metric] - full[metric]
            convergence[f"{metric}_relative_error_to_4m"] = abs(
                row[metric] - full[metric]
            ) / (abs(full[metric]) + 1e-12)
        convergence_rows.append(convergence)

    returns = heldout_episode_returns(arrays["reward"], active)
    summary = {
        "schema_version": 1,
        **common,
        "definition_raw": (
            "sum_groups weight * (g_mc-g_episode_gae)^T "
            "(F+xi I)^-1 (g_mc-g_episode_gae)"
        ),
        "definition_phase_matched": (
            "sum_groups weight * (g_mc-g_phase_marginalized_gae)^T "
            "(F+xi I)^-1 (g_mc-g_phase_marginalized_gae)"
        ),
        "definition_optimal_scale": (
            "min_alpha>=0 sum_groups weight * (g_mc-alpha*g_phase)^T "
            "(F+xi I)^-1 (g_mc-alpha*g_phase)"
        ),
        "phase_recovery_limitation": (
            "The collector did not store historical training rollout phase; "
            "the phase-matched signal is uniformly marginalized over all phases."
        ),
        "slot_x_type_weighting": "sum_slot sum_type p(type|slot) metric(slot,type)",
        "reference_signal": "complete discounted Monte Carlo return-to-go",
        "reference_baseline": "none",
        "fisher_ridges": args.fisher_ridges,
        "episode_budgets": {
            budget_labels[budget]: budget for budget in episode_budgets
        },
        "heldout_episodes": len(episode_ids),
        "heldout_episode_return_mean": float(returns.mean()),
        "heldout_episode_return_std": float(returns.std(ddof=1)),
        "heldout_episode_return_stderr": float(
            returns.std(ddof=1) / math.sqrt(len(returns))
        ),
        "legacy_raw_reproduction": canonical_comparison,
        "single_type_aggregation_consistency": single_type_consistency,
        "aggregate_metrics": aggregate_rows,
    }
    write_csv(output_dir / "robust_distortion_metrics.csv", aggregate_rows)
    write_csv(output_dir / "robust_distortion_groups.csv", group_rows)
    write_csv(output_dir / "robust_distortion_convergence.csv", convergence_rows)
    (output_dir / "robust_distortion_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output_dir / "robust_distortion_summary.json")


if __name__ == "__main__":
    main()
