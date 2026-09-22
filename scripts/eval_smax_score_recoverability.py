#!/usr/bin/env python3
"""Collect a frozen SMAX checkpoint and measure score recoverability offline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.MAPPO.eval_mappo_rnn_smax import resolve_checkpoint
from scripts.smax_score_recoverability import measure_score_recoverability


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--collection-batch-size", type=int, default=64)
    parser.add_argument("--collection-seed", type=int, required=True)
    parser.add_argument("--fit-fraction", type=float, default=0.75)
    parser.add_argument("--split-seed", type=int, default=20260923)
    parser.add_argument("--sampling-seed", type=int, default=20260924)
    parser.add_argument("--fit-samples-per-agent", type=int, default=16384)
    parser.add_argument("--test-samples-per-agent", type=int, default=4096)
    parser.add_argument("--fisher-ridge", type=float, default=1e-3)
    parser.add_argument("--probe-hidden-dim", type=int, default=256)
    parser.add_argument("--probe-steps", type=int, default=2000)
    parser.add_argument("--probe-batch-size", type=int, default=512)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-3)
    parser.add_argument("--probe-seed", type=int, default=20260925)
    return parser.parse_args()


def write_progress(path: Path, stage: str, **extra):
    payload = {"stage": stage, **extra}
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def validate_collection(metadata_path, checkpoint_dir, args):
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "checkpoint": str(checkpoint_dir.resolve()),
        "episodes": args.episodes,
        "diagnostic_seed": args.collection_seed,
        "array_profile": "score_recoverability",
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Cached collection does not match the requested protocol: {mismatches}"
        )


def main():
    args = parse_args()
    checkpoint_dir, _, _ = resolve_checkpoint(args.checkpoint)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    progress = output / "progress.json"
    summary = output / "summary.json"
    if summary.is_file():
        print(summary)
        return

    diagnostics = output / "collected"
    metadata_path = diagnostics / "metadata.json"
    if metadata_path.is_file():
        validate_collection(metadata_path, checkpoint_dir, args)
    else:
        write_progress(progress, "collecting", checkpoint=str(checkpoint_dir))
        command = [
            sys.executable,
            str(REPO_ROOT / "baselines/MAPPO/collect_mappo_smax_diagnostics.py"),
            "--checkpoint",
            str(checkpoint_dir),
            "--output-dir",
            str(diagnostics),
            "--episodes",
            str(args.episodes),
            "--batch-size",
            str(args.collection_batch_size),
            "--seed",
            str(args.collection_seed),
            "--array-profile",
            "score_recoverability",
        ]
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        validate_collection(metadata_path, checkpoint_dir, args)

    write_progress(progress, "fitting_probes", checkpoint=str(checkpoint_dir))
    _, protocol = measure_score_recoverability(
        diagnostics,
        output,
        fit_fraction=args.fit_fraction,
        split_seed=args.split_seed,
        sampling_seed=args.sampling_seed,
        fit_samples_per_agent=args.fit_samples_per_agent,
        test_samples_per_agent=args.test_samples_per_agent,
        fisher_ridge=args.fisher_ridge,
        probe_hidden_dim=args.probe_hidden_dim,
        probe_steps=args.probe_steps,
        probe_batch_size=args.probe_batch_size,
        probe_learning_rate=args.probe_learning_rate,
        probe_seed=args.probe_seed,
    )
    write_progress(
        progress,
        "completed",
        checkpoint=str(checkpoint_dir),
        epsilon_rec_normalized=protocol["epsilon_rec_normalized"],
    )
    print(summary)


if __name__ == "__main__":
    main()
