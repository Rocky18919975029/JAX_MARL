#!/usr/bin/env python3
"""Launch the 12-run NPS Humanoid core matrix used by the main experiment."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERAL_LAUNCHER = REPO_ROOT / "experiments" / "harl_mamujoco" / "run_matrix.py"


def build_delegate_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(GENERAL_LAUNCHER),
        "--run-root",
        str(args.run_root),
        "--harl-root",
        str(args.harl_root),
        "--cka-calibration",
        str(args.cka_calibration),
        "--actor-variants",
        "nps",
        "--directions",
        "c_to_a",
        "--distances",
        "ln_mse,linear_cka",
        "--seeds",
        "1-4",
        "--gpus",
        args.gpus,
        "--max-runs-per-gpu",
        str(args.max_runs_per_gpu),
        "--checkpoint-interval-steps",
        str(args.checkpoint_interval_steps),
        "--wandb-project",
        args.wandb_project,
    ]
    if args.upload_checkpoints:
        command.append("--upload-checkpoints")
    if args.dry_run:
        command.append("--dry-run")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cka-calibration", type=Path, required=True)
    parser.add_argument(
        "--harl-root", type=Path, default=REPO_ROOT / "third_party" / "HARL"
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max-runs-per-gpu", type=int, default=3)
    parser.add_argument("--checkpoint-interval-steps", type=int, default=500_000)
    parser.add_argument(
        "--wandb-project", default="harl-mamujoco-humanoid17x1-nps-core"
    )
    parser.add_argument("--upload-checkpoints", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_runs_per_gpu < 1:
        raise ValueError("max-runs-per-gpu must be positive")
    if args.checkpoint_interval_steps < 1:
        raise ValueError("checkpoint-interval-steps must be positive")
    if not args.cka_calibration.expanduser().is_file():
        raise FileNotFoundError(
            f"Missing CKA calibration: {args.cka_calibration.expanduser()}"
        )

    args.run_root = args.run_root.expanduser().resolve()
    args.harl_root = args.harl_root.expanduser().resolve()
    args.cka_calibration = args.cka_calibration.expanduser().resolve()
    args.run_root.mkdir(parents=True, exist_ok=True)
    os.execv(sys.executable, build_delegate_command(args))


if __name__ == "__main__":
    main()
