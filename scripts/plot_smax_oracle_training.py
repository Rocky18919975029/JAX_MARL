#!/usr/bin/env python3
"""Plot seed-aggregated MAPPO versus oracle SMAX learning curves."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = {"none": "MAPPO", "oracle_latent_distortion": r"MAPPO + oracle $\epsilon_{Lat}$"}
COLORS = {"none": "#333333", "oracle_latent_distortion": "#0072B2"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.run_root.expanduser().resolve()
    output = (args.output_dir or root / "analysis").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "experiment_manifest.json").read_text())

    data = {}
    for condition in manifest["conditions"]:
        seed_rows = []
        for seed in manifest["seeds"]:
            name = f"SMAX-ORACLE-{manifest['map_name']}-nps-{condition}-seed{seed}"
            path = root / "metrics" / f"{name}.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            seed_rows.append({int(row["env_step"]): row for row in rows})
        common_steps = sorted(set.intersection(*(set(rows) for rows in seed_rows)))
        data[condition] = (seed_rows, common_steps)

    summary = []
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), constrained_layout=True)
    for metric, axis, title in zip(
        ("returns", "win_rate"), axes, ("Episode return", "Win rate")
    ):
        for condition in manifest["conditions"]:
            seed_rows, steps = data[condition]
            values = np.asarray([[rows[step][metric] for step in steps] for rows in seed_rows], dtype=float)
            mean = np.nanmean(values, axis=0)
            stderr = np.nanstd(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
            color = COLORS[condition]
            axis.plot(steps, mean, color=color, linewidth=2.2, label=LABELS[condition])
            axis.fill_between(steps, mean - 1.96 * stderr, mean + 1.96 * stderr, color=color, alpha=0.16, linewidth=0)
            for step, center, se in zip(steps, mean, stderr):
                summary.append({"task": manifest["map_name"], "condition": condition, "metric": metric, "env_step": step, "mean": center, "stderr": se, "ci95_low": center - 1.96 * se, "ci95_high": center + 1.96 * se, "num_seeds": values.shape[0]})
        axis.set_title(title, fontweight="semibold")
        axis.set_xlabel("Environment steps")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Performance")
    axes[0].legend(frameon=False, loc="best")
    figure.suptitle(f"SMAX — {manifest['map_name']}", fontweight="bold", fontsize=14)
    figure.savefig(output / "oracle-vs-mappo-learning-curves.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / "oracle-vs-mappo-learning-curves.pdf", bbox_inches="tight")
    plt.close(figure)

    with (output / "oracle-vs-mappo-curve-summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    print(output)


if __name__ == "__main__":
    main()
