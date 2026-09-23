#!/usr/bin/env python3
"""Rank 6s9z isolated MAPPO PPO settings and audit four-seed variability."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scripts.analyze_smax_actor_score_recovery_sweep import (
        history,
        metric_summary,
        write_csv,
    )
    from scripts.run_smax_nps_baseline_sweep import INCUMBENT, PROTOCOL
except ModuleNotFoundError:  # Direct execution from scripts/.
    from analyze_smax_actor_score_recovery_sweep import (
        history,
        metric_summary,
        write_csv,
    )
    from run_smax_nps_baseline_sweep import INCUMBENT, PROTOCOL


def bootstrap_mean_ci(
    values: list[float], *, draws: int = 10_000
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(20260924)
    sampled = rng.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(sampled, (0.025, 0.975)))


def summarize(root: Path) -> dict:
    manifest = json.loads(
        (root / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("protocol") != PROTOCOL:
        raise RuntimeError("Expected a 6s9z isolated MAPPO PPO-sweep root")
    seed_rows = []
    missing = []
    for run in manifest["runs"]:
        name = run["run_name"]
        status_path = root / "status" / f"{name}.json"
        status = (
            json.loads(status_path.read_text(encoding="utf-8")).get("status")
            if status_path.is_file()
            else "pending"
        )
        if status != "completed":
            missing.append(f"{name}: {status}")
            continue
        rows = history(root / "metrics" / f"{name}.jsonl")
        budget = int(run["steps"])
        return_auc, return_last5 = metric_summary(rows, "returns", budget)
        win_auc, win_last5 = metric_summary(rows, "win_rate", budget)
        seed_rows.append(
            {
                "task": manifest["map_name"],
                "condition": "none",
                "learning_rate": float(run["learning_rate"]),
                "update_epochs": int(run["update_epochs"]),
                "seed": int(run["seed"]),
                "budget": budget,
                "run_name": name,
                "return_auc": return_auc,
                "return_last5_logged_updates": return_last5,
                "win_rate_auc": win_auc,
                "win_rate_last5_logged_updates": win_last5,
            }
        )
    if missing:
        raise RuntimeError(
            f"Sweep is not complete ({len(missing)} runs): {missing[:10]}"
        )

    grouped = defaultdict(list)
    for row in seed_rows:
        grouped[row["learning_rate"], row["update_epochs"]].append(row)
    expected_seeds = set(manifest["seeds"])
    if INCUMBENT not in grouped and manifest["budget"] == 20_000_000:
        raise RuntimeError(
            "Full-budget sweep must include incumbent LR=0.002, epochs=4"
        )
    incumbent_by_seed = {row["seed"]: row for row in grouped.get(INCUMBENT, [])}
    if incumbent_by_seed and set(incumbent_by_seed) != expected_seeds:
        raise RuntimeError("Incomplete incumbent seed group")
    summary_rows = []
    for (lr, epochs), cell in sorted(grouped.items()):
        if {row["seed"] for row in cell} != expected_seeds:
            raise RuntimeError(f"Incomplete seed group for LR={lr}, epochs={epochs}")
        aucs = [row["return_auc"] for row in cell]
        finals = [row["return_last5_logged_updates"] for row in cell]
        win_aucs = [row["win_rate_auc"] for row in cell]
        win_finals = [row["win_rate_last5_logged_updates"] for row in cell]
        paired_deltas = (
            [
                row["return_auc"] - incumbent_by_seed[row["seed"]]["return_auc"]
                for row in cell
            ]
            if incumbent_by_seed
            else []
        )
        auc_lo, auc_hi = bootstrap_mean_ci(aucs)
        final_lo, final_hi = bootstrap_mean_ci(finals)
        summary_rows.append(
            {
                "task": manifest["map_name"],
                "condition": "none",
                "learning_rate": lr,
                "update_epochs": epochs,
                "n_seeds": len(cell),
                "mean_return_auc": statistics.mean(aucs),
                "sd_return_auc": statistics.stdev(aucs) if len(cell) > 1 else 0.0,
                "min_seed_return_auc": min(aucs),
                "mean_paired_delta_return_auc_vs_incumbent": (
                    statistics.mean(paired_deltas) if paired_deltas else None
                ),
                "return_auc_positive_seeds_vs_incumbent": (
                    sum(delta > 0 for delta in paired_deltas) if paired_deltas else None
                ),
                "return_auc_ci95_low": auc_lo,
                "return_auc_ci95_high": auc_hi,
                "mean_return_last5_logged_updates": statistics.mean(finals),
                "sd_return_last5_logged_updates": (
                    statistics.stdev(finals) if len(cell) > 1 else 0.0
                ),
                "return_last5_ci95_low": final_lo,
                "return_last5_ci95_high": final_hi,
                "mean_win_rate_auc": statistics.mean(win_aucs),
                "sd_win_rate_auc": (
                    statistics.stdev(win_aucs) if len(cell) > 1 else 0.0
                ),
                "mean_win_rate_last5_logged_updates": statistics.mean(win_finals),
                "sd_win_rate_last5_logged_updates": (
                    statistics.stdev(win_finals) if len(cell) > 1 else 0.0
                ),
                "is_incumbent": (lr, epochs) == INCUMBENT,
            }
        )
    summary_rows.sort(key=lambda row: row["mean_return_auc"], reverse=True)
    output = root / "analysis"
    write_csv(output / "seed_level.csv", seed_rows)
    write_csv(output / "config_summary.csv", summary_rows)
    result = {
        "selection_metric": "highest mean four-seed training return AUC",
        "winner_by_mean_return_auc": summary_rows[0],
        "winner_by_worst_seed_return_auc": max(
            summary_rows, key=lambda row: row["min_seed_return_auc"]
        ),
        "incumbent": next((row for row in summary_rows if row["is_incumbent"]), None),
        "screening_seeds": list(manifest["seeds"]),
        "warning": (
            "This is exploratory selection on the same four seeds. To make a "
            "fair intervention comparison, rerun score recovery and none with "
            "the selected PPO hyperparameters on unused confirmation seeds."
        ),
    }
    (output / "selection.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run_root.expanduser().resolve())
    winner = result["winner_by_mean_return_auc"]
    robust = result["winner_by_worst_seed_return_auc"]
    incumbent = result["incumbent"]
    print(
        f"Best mean return AUC: LR={winner['learning_rate']:g}, "
        f"epochs={winner['update_epochs']}, "
        f"AUC={winner['mean_return_auc']:.6g} "
        f"(SD={winner['sd_return_auc']:.6g})"
    )
    print(
        f"Best worst-seed return AUC: LR={robust['learning_rate']:g}, "
        f"epochs={robust['update_epochs']}, "
        f"minimum seed AUC={robust['min_seed_return_auc']:.6g}"
    )
    if incumbent:
        print(
            f"Incumbent: LR={incumbent['learning_rate']:g}, "
            f"epochs={incumbent['update_epochs']}, "
            f"AUC={incumbent['mean_return_auc']:.6g} "
            f"(SD={incumbent['sd_return_auc']:.6g})"
        )
    print("See analysis/config_summary.csv, seed_level.csv, selection.json")


if __name__ == "__main__":
    main()
