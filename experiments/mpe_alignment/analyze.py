#!/usr/bin/env python3
"""Create per-task CSV tables and a publication-style MPE learning curve."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.mpe_alignment.protocol import CONDITIONS, DEFAULT_SEEDS, TASKS


LABELS = {
    "none": "Isolated",
    "c_to_a_mse": "C → A (LN-MSE)",
    "c_to_a_cka": "C → A (Linear CKA)",
}
COLORS = {"none": "#343434", "c_to_a_mse": "#D55E00", "c_to_a_cka": "#0072B2"}


def load_run(root: Path, condition: str, seed: int) -> list[dict]:
    matches = sorted((root / "status").glob(f"MPE-*-nps-{condition}-*-seed{seed}.json"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one status file for {condition}/seed{seed}, got {matches}"
        )
    status = json.loads(matches[0].read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        raise RuntimeError(f"run is not complete: {matches[0]}")
    metric = root / "metrics" / f"{status['run_name']}.jsonl"
    return [
        json.loads(line) for line in metric.read_text().splitlines() if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    root = args.run_root.expanduser().resolve()
    output = (args.output_root or root / "analysis").expanduser().resolve()
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    raw_rows = []
    by_cell = defaultdict(dict)
    for condition in CONDITIONS:
        for seed in DEFAULT_SEEDS:
            rows = load_run(root, condition, seed)
            by_cell[condition][seed] = rows
            for row in rows:
                raw_rows.append(
                    {
                        "task": TASKS[0],
                        "condition": condition,
                        "seed": seed,
                        "env_step": int(row["env_step"]),
                        "episode_return": float(row["returns"]),
                    }
                )
    raw_path = tables / "simple_spread_5_seed_curves.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_rows[0].keys())
        writer.writeheader()
        writer.writerows(raw_rows)

    curve_rows = []
    fig, ax = plt.subplots(figsize=(8.1, 4.9), constrained_layout=True)
    for condition in CONDITIONS:
        per_seed = by_cell[condition]
        common_steps = sorted(
            set.intersection(
                *(
                    set(int(row["env_step"]) for row in rows)
                    for rows in per_seed.values()
                )
            )
        )
        values = np.asarray(
            [
                [
                    {
                        int(row["env_step"]): float(row["returns"])
                        for row in per_seed[seed]
                    }[step]
                    for step in common_steps
                ]
                for seed in DEFAULT_SEEDS
            ]
        )
        mean = values.mean(axis=0)
        se = values.std(axis=0, ddof=1) / math.sqrt(len(DEFAULT_SEEDS))
        low, high = mean - 1.96 * se, mean + 1.96 * se
        for index, step in enumerate(common_steps):
            curve_rows.append(
                {
                    "task": TASKS[0],
                    "condition": condition,
                    "env_step": step,
                    "seed_count": len(DEFAULT_SEEDS),
                    "return_mean": mean[index],
                    "return_se": se[index],
                    "return_ci95_low": low[index],
                    "return_ci95_high": high[index],
                }
            )
        x = np.asarray(common_steps)
        ax.plot(
            x, mean, color=COLORS[condition], linewidth=2.4, label=LABELS[condition]
        )
        ax.fill_between(x, low, high, color=COLORS[condition], alpha=0.16, linewidth=0)

    summary_path = tables / "simple_spread_5_curve_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve_rows[0].keys())
        writer.writeheader()
        writer.writerows(curve_rows)

    ax.set_title("MPE — Simple Spread-5", fontsize=17, fontweight="semibold")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode return")
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.8, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=1, loc="best")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6), useMathText=True)
    figure_path = figures / "mpe-simple_spread_5-learning-curve.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    fig.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(raw_path)
    print(summary_path)
    print(figure_path)


if __name__ == "__main__":
    main()
