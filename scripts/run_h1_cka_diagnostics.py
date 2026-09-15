#!/usr/bin/env python3
"""Resume the reduced NPS CKA checkpoint diagnostics and post-hoc evaluation.

Run this script on the GPU server after pulling the repository. It never trains
new policies; it reuses the sixteen completed CKA runs and their eight
preregistered checkpoints per run.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from eval_h1_checkpoints import CHECKPOINT_SPECS, discover_tasks
except ModuleNotFoundError:  # Imported as scripts.run_h1_cka_diagnostics in tests.
    from scripts.eval_h1_checkpoints import CHECKPOINT_SPECS, discover_tasks


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = Path(
    "/home/data/zeshenghong/JaxMARL/h1_smax_runs/cka_distance_robustness"
)
MAPS = ("10m_vs_11m", "smacv2_10_units")
CONDITIONS = ("c_to_a_cka", "a_to_c_cka")
SEEDS = (1, 2, 3, 4)
EXPECTED_CELLS = {(task, condition, seed) for task in MAPS for condition in CONDITIONS for seed in SEEDS}


@dataclass(frozen=True)
class Phase:
    name: str
    command: tuple[str, ...]


def verify_matrix(run_root: Path) -> int:
    """Reject partial or mixed training matrices before any diagnostic writes."""

    if not run_root.is_dir():
        raise FileNotFoundError(f"CKA run root does not exist: {run_root}")
    tasks, missing = discover_tasks(run_root)
    if missing:
        raise RuntimeError(f"{len(missing)} preregistered CKA checkpoints are missing")
    run_dirs = {task.run_dir for task in tasks}
    cells = set()
    for run_dir in run_dirs:
        config_path = run_dir / "initial" / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        cell = (config["MAP_NAME"], config["EXPERIMENT_CONDITION"], int(config["SEED"]))
        cells.add(cell)
        required = {
            "PROTOCOL_VERSION": "h1-v1.0",
            "MATRIX_PROFILE": "reduced-nps-cka",
            "ACTOR_PARAMETER_SHARING": False,
            "MATCHED_COMPARISON": True,
            "ALIGN_DISTANCE": "linear_cka",
            "ALIGN_TARGET_SHUFFLE": False,
            "ALIGN_MODE": cell[1].removesuffix("_cka"),
        }
        differences = {
            key: (config.get(key), value)
            for key, value in required.items()
            if config.get(key) != value
        }
        if differences:
            raise RuntimeError(f"Unexpected CKA checkpoint config in {run_dir}: {differences}")
    if cells != EXPECTED_CELLS or len(run_dirs) != len(EXPECTED_CELLS):
        raise RuntimeError(
            "CKA matrix mismatch: "
            f"missing={sorted(EXPECTED_CELLS - cells)}, "
            f"unexpected={sorted(cells - EXPECTED_CELLS)}, "
            f"run_dirs={len(run_dirs)}"
        )
    expected_checkpoints = len(EXPECTED_CELLS) * len(CHECKPOINT_SPECS)
    if len(tasks) != expected_checkpoints:
        raise RuntimeError(
            f"Expected {expected_checkpoints} CKA checkpoints, discovered {len(tasks)}"
        )
    return len(tasks)


def phase_plan(run_root: Path, gpus: str, collect_per_gpu: int) -> tuple[Phase, ...]:
    python = sys.executable
    scripts = REPO_ROOT / "scripts"
    base = ("--run-root", str(run_root))
    gpu_args = ("--gpus", gpus)

    def diagnostics(name: str, per_gpu: int, *extra: str) -> Phase:
        return Phase(
            name,
            (
                python,
                str(scripts / "run_h1_diagnostics.py"),
                *base,
                *gpu_args,
                "--max-runs-per-gpu",
                str(per_gpu),
                "--output-tree",
                "diagnostics_raw",
                "--stages",
                name,
                *extra,
            ),
        )

    return (
        diagnostics("collect", collect_per_gpu, "--episodes", "512", "--batch-size", "64"),
        diagnostics("latent", 1),
        diagnostics("decision", 1, "--anchors", "256", "--continuations", "32"),
        diagnostics("bellman", 1, "--bellman-heads", "32"),
        Phase("merge", (python, str(scripts / "merge_h1_diagnostics.py"), *base)),
        Phase(
            "deterministic-eval",
            (
                python,
                str(scripts / "eval_h1_checkpoints.py"),
                *base,
                *gpu_args,
                "--max-runs-per-gpu",
                "1",
                "--episodes",
                "256",
                "--num-envs",
                "128",
            ),
        ),
        Phase("performance", (python, str(scripts / "analyze_h1_performance.py"), *base)),
        Phase("mechanisms", (python, str(scripts / "analyze_h1_mechanisms.py"), *base)),
        Phase("figures", (python, str(scripts / "plot_h1_mechanisms.py"), *base)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--collect-per-gpu", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids) or args.collect_per_gpu < 1:
        parser.error("select unique GPU IDs and a positive collect-per-gpu")
    run_root = args.run_root.expanduser().resolve()
    count = verify_matrix(run_root)
    plan = phase_plan(run_root, ",".join(gpu_ids), args.collect_per_gpu)
    print(f"CKA protocol: 16 runs, {count} preregistered checkpoints, 0 missing", flush=True)
    for phase in plan:
        print(f"{phase.name}: {shlex.join(phase.command)}", flush=True)
    if args.dry_run:
        return

    # Prevent two copies of this long-running pipeline from racing on the same
    # diagnostic shards. The OS releases the lock if the manager exits.
    lock_path = run_root / ".cka_diagnostics_pipeline.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"A CKA diagnostic pipeline is already running: {lock_path}") from error

        current: subprocess.Popen | None = None

        def stop(_signum, _frame):
            if current is not None and current.poll() is None:
                os.killpg(current.pid, signal.SIGTERM)
            raise SystemExit(130)

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        for phase in plan:
            print(f"START {phase.name}", flush=True)
            current = subprocess.Popen(phase.command, cwd=REPO_ROOT, start_new_session=True)
            status = current.wait()
            if status:
                raise RuntimeError(f"CKA phase {phase.name} failed with exit code {status}")
            print(f"DONE {phase.name}", flush=True)
            current = None
    print("CKA diagnostics, evaluation, and figures: COMPLETE", flush=True)


if __name__ == "__main__":
    main()
