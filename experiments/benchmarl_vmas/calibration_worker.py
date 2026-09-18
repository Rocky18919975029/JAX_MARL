#!/usr/bin/env python3
"""Single-GPU worker for VMAS CKA gradient calibration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.benchmarl_vmas.calibrate_cka import run_cell
from experiments.benchmarl_vmas.protocol import TASKS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot-seed", type=int, required=True)
    parser.add_argument("--minibatches", type=int, required=True)
    args = parser.parse_args()
    run_cell(args, args.task, args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
