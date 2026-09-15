#!/usr/bin/env python3
"""Recompute and analyze the canonical NPS H1 MSE/CKA diagnostics."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS = ("10m_vs_11m", "smacv2_10_units")
SEEDS = (1, 2, 3, 4)
CHECKPOINTS_PER_RUN = 8
CHECKPOINT_NAMES = {
    "initial",
    "step_000000500000",
    "step_000001000000",
    "step_000002000000",
    "step_000004000000",
    "step_000006000000",
    "step_000008000000",
    "final",
}
ROOT_SPECS = (
    ("ln_mse", ("none", "a_to_c", "c_to_a")),
    ("linear_cka", ("a_to_c_cka", "c_to_a_cka")),
)


@dataclass(frozen=True)
class Phase:
    name: str
    command: tuple[str, ...]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_collection(root, distance, conditions):
    expected = {
        (task, condition, seed)
        for task in TASKS
        for condition in conditions
        for seed in SEEDS
    }
    cells = set()
    checkpoint_names = {}
    checkpoints = 0
    errors = []
    for metadata_path in sorted(
        (root / "diagnostics_raw").glob("H1-*/*/metadata.json")
    ):
        metadata = read_json(metadata_path)
        condition = metadata["condition"]
        if metadata.get("actor_parameter_sharing"):
            continue
        if metadata.get("align_distance", "ln_mse") != distance:
            continue
        if condition not in conditions:
            continue
        cell = (metadata["map_name"], condition, int(metadata["training_seed"]))
        cells.add(cell)
        checkpoint_names.setdefault(cell, set()).add(metadata_path.parent.name)
        checkpoints += 1
        if metadata.get("protocol_version") != "h1-v1.0":
            errors.append(f"{metadata_path}: wrong training protocol")
        if int(metadata.get("episodes", -1)) != 512:
            errors.append(f"{metadata_path}: expected 512 episodes")
        shards = metadata.get("shards", [])
        if sum(int(item["episodes"]) for item in shards) != 512:
            errors.append(f"{metadata_path}: incomplete episode shards")
        for shard in shards:
            if not (metadata_path.parent / shard["path"]).is_file():
                errors.append(f"{metadata_path.parent / shard['path']}: missing")
    for cell in expected:
        if checkpoint_names.get(cell, set()) != CHECKPOINT_NAMES:
            errors.append(f"{cell}: preregistered checkpoint set is incomplete")
    expected_checkpoints = len(expected) * CHECKPOINTS_PER_RUN
    if cells != expected or checkpoints != expected_checkpoints or errors:
        raise RuntimeError(
            f"{distance} collection is not complete: "
            f"missing_cells={sorted(expected - cells)}, "
            f"unexpected_cells={sorted(cells - expected)}, "
            f"checkpoints={checkpoints}/{expected_checkpoints}, errors={errors[:5]}"
        )
    return checkpoints


def verify_matched_frozen_configs(mse_root, cka_root):
    configs = []
    for root in (mse_root, cka_root):
        source = root / "protocol" / "frozen_training_config.json"
        if not source.is_file():
            raise FileNotFoundError(f"Frozen training protocol missing: {source}")
        configs.append(read_json(source)["training_config"])
    if configs[0] != configs[1] or configs[0].get("MATCHED_COMPARISON") is not True:
        raise RuntimeError("MSE and CKA frozen optimization protocols are not matched")
    digest = hashlib.sha256(
        json.dumps(configs[0], sort_keys=True).encode("utf-8")
    ).hexdigest()
    return digest


def phase_plan(args):
    python = sys.executable
    scripts = REPO_ROOT / "scripts"
    roots = {
        "ln_mse": args.mse_root.expanduser().resolve(),
        "linear_cka": args.cka_root.expanduser().resolve(),
    }
    phases = []
    for stage in ("latent", "decision", "bellman"):
        for distance, conditions in ROOT_SPECS:
            command = [
                python,
                str(scripts / "run_h1_diagnostics.py"),
                "--run-root",
                str(roots[distance]),
                "--gpus",
                args.gpus,
                "--max-runs-per-gpu",
                str(args.max_runs_per_gpu),
                "--output-tree",
                "diagnostics_raw",
                "--stages",
                stage,
                "--conditions",
                ",".join(conditions),
                "--align-distance",
                distance,
            ]
            if stage == "latent":
                command.extend(
                    ["--fisher-ridge-absolute", str(args.fisher_ridge_absolute)]
                )
            elif stage == "decision":
                command.extend(
                    [
                        "--anchors",
                        str(args.anchors),
                        "--continuations",
                        str(args.continuations),
                    ]
                )
            else:
                command.extend(["--bellman-heads", str(args.bellman_heads)])
            phases.append(Phase(f"{distance}-{stage}", tuple(command)))
    phases.extend(
        (
            Phase(
                "analysis",
                (
                    python,
                    str(scripts / "analyze_h1_mechanisms.py"),
                    "--mse-root",
                    str(roots["ln_mse"]),
                    "--cka-root",
                    str(roots["linear_cka"]),
                    "--output-root",
                    str(args.analysis_root.expanduser().resolve()),
                    "--fisher-ridge-absolute",
                    str(args.fisher_ridge_absolute),
                ),
            ),
            Phase(
                "figures",
                (
                    python,
                    str(scripts / "plot_h1_mechanisms.py"),
                    "--analysis-root",
                    str(args.analysis_root.expanduser().resolve()),
                ),
            ),
        )
    )
    return tuple(phases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mse-root", type=Path, required=True)
    parser.add_argument("--cka-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--fisher-ridge-absolute", type=float, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max-runs-per-gpu", type=int, default=1)
    parser.add_argument("--anchors", type=int, default=256)
    parser.add_argument("--continuations", type=int, default=32)
    parser.add_argument("--bellman-heads", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        parser.error("--gpus must contain unique GPU IDs")
    if args.max_runs_per_gpu < 1 or args.fisher_ridge_absolute <= 0:
        parser.error("worker count and Fisher ridge must be positive")

    mse_root = args.mse_root.expanduser().resolve()
    cka_root = args.cka_root.expanduser().resolve()
    mse_count = verify_collection(mse_root, *ROOT_SPECS[0])
    cka_count = verify_collection(cka_root, *ROOT_SPECS[1])
    frozen_sha256 = verify_matched_frozen_configs(mse_root, cka_root)
    plan = phase_plan(args)
    print(
        f"Collections verified: LN-MSE={mse_count}, Linear CKA={cka_count} checkpoints",
        flush=True,
    )
    for phase in plan:
        print(f"{phase.name}: {shlex.join(phase.command)}", flush=True)
    if args.dry_run:
        return

    analysis_root = args.analysis_root.expanduser().resolve()
    analysis_root.mkdir(parents=True, exist_ok=True)
    analysis_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest_path = analysis_root / "recompute_manifest.json"
    manifest = {
        "schema_version": 1,
        "analysis_protocol_version": "h1-nps-two-distance-v2.1",
        "analysis_git_commit": analysis_commit,
        "training_protocol_version": "h1-v1.0",
        "actor_parameterization": "nps",
        "mse_root": str(mse_root),
        "cka_root": str(cka_root),
        "selected_mse_checkpoints": mse_count,
        "selected_cka_checkpoints": cka_count,
        "matched_frozen_training_config_sha256": frozen_sha256,
        "reference_protocol": "baseline_free_mc_return_train_matched_gae",
        "fisher_ridge_absolute": args.fisher_ridge_absolute,
        "interpretation": "descriptive seed-paired trends without hard pass/fail",
        "anchors": args.anchors,
        "continuations": args.continuations,
        "bellman_heads": args.bellman_heads,
        "baseline_reuse": "ln_mse_none_for_linear_cka",
    }
    lock_path = analysis_root / ".h1_nps_diagnostics.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"A canonical H1 pipeline is already active: {lock_path}"
            ) from error
        if manifest_path.is_file():
            if read_json(manifest_path) != manifest:
                raise RuntimeError(
                    f"Recompute settings changed; cached results cannot be mixed: {manifest_path}"
                )
        else:
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        current = None

        def stop(_signum, _frame):
            if current is not None and current.poll() is None:
                os.killpg(current.pid, signal.SIGTERM)
            raise SystemExit(130)

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        for phase in plan:
            print(f"START {phase.name}", flush=True)
            current = subprocess.Popen(
                phase.command, cwd=REPO_ROOT, start_new_session=True
            )
            status = current.wait()
            if status:
                raise RuntimeError(f"Phase {phase.name} failed with exit code {status}")
            print(f"DONE {phase.name}", flush=True)
            current = None
    print("Canonical NPS H1 MSE/CKA diagnostics: COMPLETE", flush=True)


if __name__ == "__main__":
    main()
