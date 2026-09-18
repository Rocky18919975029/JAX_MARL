#!/usr/bin/env python3
"""Calibrate Linear-CKA to the λ=0.1 LN-MSE actor-gradient scale."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.mpe_alignment.protocol import (
    CALIBRATION_PILOT_SEED,
    CALIBRATION_PROTOCOL_VERSION,
    REFERENCE_ALIGNMENT_COEF,
    TASKS,
    parse_csv,
)


TRAIN_SCRIPT = REPO_ROOT / "experiments" / "mpe_alignment" / "train_alignment.py"


def read_last_metric(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected one calibration update in {path}, got {len(rows)}"
        )
    return rows[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pilot-seed", type=int, default=CALIBRATION_PILOT_SEED)
    parser.add_argument("--gpus", type=parse_csv, default=("0", "1"))
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    if args.pilot_seed != CALIBRATION_PILOT_SEED:
        raise ValueError(f"pilot seed is frozen at {CALIBRATION_PILOT_SEED}")
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    cases = (
        ("c_to_a_mse", "ln_mse"),
        ("c_to_a_cka", "linear_cka"),
    )

    def launch(index: int, case: tuple[str, str]) -> None:
        condition, distance = case
        run_name = f"MPE-CKA-calibration-{TASKS[0]}-{condition}-seed{args.pilot_seed}"
        metric = root / "probes" / "metrics" / f"{run_name}.jsonl"
        if metric.is_file() and not args.rerun:
            print(
                f"GPU {args.gpus[index % len(args.gpus)]} SKIP  {condition}", flush=True
            )
            return
        command = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--run-root",
            str(root / "probes"),
            "--run-name",
            run_name,
            "--task",
            TASKS[0],
            "--seed",
            str(args.pilot_seed),
            "--condition",
            condition,
            "--align-mode",
            "c_to_a",
            "--align-distance",
            distance,
            "--alignment-coef",
            "1.0",
            "--total-timesteps",
            str(16 * 128),
            "--num-envs",
            "16",
            "--num-steps",
            "128",
            "--update-epochs",
            "1",
            "--num-minibatches",
            "4",
            "--learning-rate",
            "0",
            "--wandb-mode",
            "disabled",
        ]
        gpu = args.gpus[index % len(args.gpus)]
        print(f"GPU {gpu} START {condition}", flush=True)
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        log = root / "logs" / f"{condition}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        print(f"GPU {gpu} END   {condition} status={result.returncode}", flush=True)
        if result.returncode:
            raise RuntimeError(f"calibration failed; inspect {log}")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(launch, index, case) for index, case in enumerate(cases)
        ]
        for future in futures:
            future.result()

    rows = {}
    for condition, _ in cases:
        run_name = f"MPE-CKA-calibration-{TASKS[0]}-{condition}-seed{args.pilot_seed}"
        rows[condition] = read_last_metric(
            root / "probes" / "metrics" / f"{run_name}.jsonl"
        )
    mse_norm = float(rows["c_to_a_mse"]["actor_alignment_gradient_norm"])
    cka_norm = float(rows["c_to_a_cka"]["actor_alignment_gradient_norm"])
    if min(mse_norm, cka_norm) <= 0 or not all(
        math.isfinite(value) for value in (mse_norm, cka_norm)
    ):
        raise RuntimeError("calibration produced non-positive or non-finite gradients")
    coefficient = REFERENCE_ALIGNMENT_COEF * mse_norm / cka_norm
    payload = {
        "schema_version": 1,
        "protocol_version": CALIBRATION_PROTOCOL_VERSION,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "task": TASKS[0],
        "pilot_seed": args.pilot_seed,
        "actor_parameterization": "nps",
        "direction": "c_to_a",
        "selection_uses_return": False,
        "reference_distance": "ln_mse",
        "reference_alignment_coefficient": REFERENCE_ALIGNMENT_COEF,
        "reference_gradient_norm": mse_norm,
        "target_distance": "linear_cka",
        "target_gradient_norm": cka_norm,
        "alignment_coefficient": coefficient,
        "matched_reference_gradient_norm": REFERENCE_ALIGNMENT_COEF * mse_norm,
        "matched_target_gradient_norm": coefficient * cka_norm,
        "minibatch_protocol": "same-seed-same-rollout-agent-stratified",
    }
    artifact = root / "cka_gradient_calibration.json"
    artifact.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"lambda_CKA={coefficient:.10g}")
    print(artifact)


if __name__ == "__main__":
    main()
