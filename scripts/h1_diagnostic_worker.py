#!/usr/bin/env python3
"""Run resumable H1 diagnostic stages for one checkpoint."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run(command):
    print("RUN", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--checkpoint-index", type=int, required=True)
    parser.add_argument("--stages", default="collect,latent,decision,bellman")
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--anchors", type=int, default=256)
    parser.add_argument("--continuations", type=int, default=32)
    parser.add_argument("--bellman-heads", type=int, default=32)
    args = parser.parse_args()
    stages = tuple(item.strip() for item in args.stages.split(",") if item.strip())
    unknown = set(stages) - {"collect", "latent", "decision", "bellman"}
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    diagnostic_seed = 200_000 + 100 * args.training_seed + args.checkpoint_index

    if "collect" in stages and not (output / "metadata.json").is_file():
        run(
            [
                sys.executable,
                str(REPO_ROOT / "baselines/MAPPO/collect_mappo_smax_diagnostics.py"),
                "--checkpoint",
                args.checkpoint,
                "--output-dir",
                str(output),
                "--episodes",
                str(args.episodes),
                "--batch-size",
                str(args.batch_size),
                "--seed",
                str(diagnostic_seed),
            ]
        )
    if (
        any(stage in stages for stage in ("latent", "decision", "bellman"))
        and not (output / "metadata.json").is_file()
    ):
        raise FileNotFoundError(
            f"Diagnostic rollout missing: {output / 'metadata.json'}"
        )

    if "latent" in stages and not (output / "latent_distortion_summary.json").is_file():
        run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/h1_latent_distortion.py"),
                "--diagnostics-dir",
                str(output),
                "--output-csv",
                str(output / "compatibility_metrics.csv"),
                "--reference-seed",
                str(30_000 + diagnostic_seed),
            ]
        )
    if "decision" in stages and not (output / "decision_summary.json").is_file():
        run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/h1_decision_sufficiency.py"),
                "--checkpoint",
                args.checkpoint,
                "--diagnostics-dir",
                str(output),
                "--output-csv",
                str(output / "decision_metrics.csv"),
                "--anchors",
                str(args.anchors),
                "--continuations",
                str(args.continuations),
                "--seed",
                str(40_000 + diagnostic_seed),
            ]
        )
    if "bellman" in stages and not (output / "bellman_summary.json").is_file():
        run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/h1_bellman_compatibility.py"),
                "--checkpoint",
                args.checkpoint,
                "--diagnostics-dir",
                str(output),
                "--output-csv",
                str(output / "bellman_metrics.csv"),
                "--heads",
                str(args.bellman_heads),
                "--seed",
                str(50_000 + diagnostic_seed),
            ]
        )
    print(f"PASS {output}")


if __name__ == "__main__":
    main()
