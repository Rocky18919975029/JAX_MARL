"""Run the four-seed LBF/RWARE ARec lambda grid without repeating paired runs.

The original formal run supplies eight MAPPO baselines and eight ARec runs at
lambda=1e-4. Four child run roots contain only the 32 missing ARec runs. All
training commands are delegated to the pinned matched-optimal controller.
"""

from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import subprocess
import sys

import run_matched_optimal as paired


TASKS = ("lbf_15x15-4p-5f", "rware_large-8ag")
SEEDS = (1, 2, 3, 4)
COEFFICIENTS = (3e-6, 1e-5, 3e-5, 1e-4, 3e-4)
REUSED_COEFFICIENT = 1e-4
Q_STEPS = 4
Q_LR = 1e-3
FISHER_RIDGE = 1e-3
PROTOCOL = "mava-jumanji-arec-lambda-same4-v1"
HERE = Path(__file__).resolve().parent


def coefficient_tag(coefficient: float) -> str:
    return format(coefficient, ".0e").replace("e-", "em")


def total_steps(job: dict) -> int:
    overrides = job["shared_overrides"]
    return (
        overrides["system.num_updates"]
        * overrides["arch.num_envs"]
        * overrides["system.update_batch_size"]
        * overrides["system.rollout_length"]
    )


def metric_steps(payload: object):
    if isinstance(payload, dict):
        if "step_count" in payload and "mean_episode_return" in payload:
            yield int(payload["step_count"])
        for value in payload.values():
            yield from metric_steps(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from metric_steps(value)


def validate_reused_run(source_root: Path, config: dict) -> list[dict]:
    """Require all 16 original runs, their logs, and their expected final eval."""
    source_manifest = source_root / "experiment_manifest.json"
    if not source_manifest.is_file():
        raise RuntimeError(f"Missing original paired experiment: {source_manifest}")
    saved = paired.read_json(source_manifest)
    expected_jobs = paired.make_jobs(config, TASKS, SEEDS, False)
    expected_arec = {
        "coef": REUSED_COEFFICIENT,
        "q_steps": Q_STEPS,
        "q_lr": Q_LR,
        "fisher_ridge": FISHER_RIDGE,
    }
    expected_fields = {
        "protocol": "mava-jumanji-rec-mappo-arec-paired-optimal-v1",
        "smoke": False,
        "tasks": list(TASKS),
        "seeds": list(SEEDS),
        "jobs": expected_jobs,
        "arec": expected_arec,
        "benchmark_config_sha256": paired.digest(paired.CONFIG_PATH),
        "baseline_script_sha256": paired.digest(
            Path(saved["mava_root"]) / "mava/systems/ppo/anakin/rec_mappo.py"
        ),
        "arec_script_sha256": paired.digest(paired.AREC_SCRIPT),
        "arec_config_sha256": paired.digest(paired.AREC_CONFIG),
        "mava_commit": config["mava_commit"],
    }
    for field, expected in expected_fields.items():
        if saved.get(field) != expected:
            raise RuntimeError(f"Original experiment {field} differs from this sweep protocol")

    reused = []
    for job in expected_jobs:
        status = paired.status_of(source_root, job)
        if status.get("status") != "completed" or status.get("exit_code") != 0:
            raise RuntimeError(f"Original run is not complete: {job['name']}")
        run_dir = source_root / "runs" / job["name"]
        metric_files = sorted((run_dir / "json").glob("**/metrics.json"))
        complete = []
        for path in metric_files:
            try:
                steps = set(metric_steps(paired.read_json(path)))
            except (OSError, ValueError, TypeError):
                continue
            if total_steps(job) in steps:
                complete.append(path)
        if not complete:
            raise RuntimeError(
                f"No complete evaluation log for {job['name']} in {run_dir / 'json'}"
            )
        # A previous retry may have left several timestamped logs. Reuse one
        # complete attempt, never splice evaluation points across attempts.
        selected = max(complete, key=lambda path: (path.stat().st_mtime_ns, str(path)))
        reused.append({
            "task": job["task"],
            "seed": job["seed"],
            "condition": job["condition"],
            "coefficient": None if job["condition"] == "none" else REUSED_COEFFICIENT,
            "run_name": job["name"],
            "run_dir": str(run_dir),
            "metric_file": str(selected),
            "metric_sha256": paired.digest(selected),
        })
    return reused


def build_manifest(
    config: dict, source_root: Path, run_root: Path, mava_root: Path, reused: list[dict]
) -> dict:
    new_groups = []
    for coefficient in COEFFICIENTS:
        if coefficient == REUSED_COEFFICIENT:
            continue
        child_root = run_root / f"lambda_{coefficient_tag(coefficient)}"
        jobs = paired.make_jobs(config, TASKS, SEEDS, False, ("arec",))
        new_groups.append({
            "coefficient": coefficient,
            "run_root": str(child_root),
            "jobs": jobs,
        })
    return {
        "protocol": PROTOCOL,
        "mava_commit": config["mava_commit"],
        "mava_root": str(mava_root),
        "source_root": str(source_root),
        "source_manifest_sha256": paired.digest(source_root / "experiment_manifest.json"),
        "benchmark_config_sha256": paired.digest(paired.CONFIG_PATH),
        "baseline_script_sha256": paired.digest(
            mava_root / "mava/systems/ppo/anakin/rec_mappo.py"
        ),
        "arec_script_sha256": paired.digest(paired.AREC_SCRIPT),
        "arec_config_sha256": paired.digest(paired.AREC_CONFIG),
        "tasks": list(TASKS),
        "seeds": list(SEEDS),
        "coefficients": list(COEFFICIENTS),
        "q_steps": Q_STEPS,
        "q_lr": Q_LR,
        "fisher_ridge": FISHER_RIDGE,
        "reused": reused,
        "new_groups": new_groups,
    }


def validate_manifest(path: Path, expected: dict) -> None:
    if path.is_file() and paired.read_json(path) != expected:
        raise RuntimeError(f"Existing sweep manifest differs: {path}; use a fresh run root")


def child_command(args: argparse.Namespace, group: dict) -> list[str]:
    command = [
        sys.executable, "-u", str(HERE / "run_matched_optimal.py"), "run",
        "--mava-root", str(args.mava_root),
        "--run-root", group["run_root"],
        "--tasks", ",".join(TASKS),
        "--conditions", "arec",
        "--seeds", "1-4",
        "--arec-coef", str(group["coefficient"]),
        "--arec-q-steps", str(Q_STEPS),
        "--arec-q-lr", str(Q_LR),
        "--arec-fisher-ridge", str(FISHER_RIDGE),
        "--gpus", args.gpus,
        "--max-runs-per-gpu", str(args.max_runs_per_gpu),
    ]
    if args.retry_failed:
        command.append("--retry-failed")
    if args.dry_run:
        command.append("--dry-run")
    return command


def run(args: argparse.Namespace) -> int:
    if (
        args.run_root == args.source_root
        or args.run_root in args.source_root.parents
        or args.source_root in args.run_root.parents
    ):
        raise ValueError("Sweep and original paired run roots must not contain one another")
    if args.max_runs_per_gpu < 1:
        raise ValueError("max-runs-per-gpu must be positive")
    config = paired.read_json(paired.CONFIG_PATH)
    paired.validate_mava(args.mava_root, config)
    reused = validate_reused_run(args.source_root, config)
    planned = build_manifest(config, args.source_root, args.run_root, args.mava_root, reused)
    manifest_path = args.run_root / "experiment_manifest.json"
    validate_manifest(manifest_path, planned)

    if args.dry_run:
        print("REUSED=16 NEW=32 TOTAL=48", flush=True)
        for group in planned["new_groups"]:
            print(f"lambda={group['coefficient']} child={group['run_root']}", flush=True)
            result = subprocess.run(child_command(args, group), check=False)
            if result.returncode:
                return result.returncode
        return 0

    args.run_root.mkdir(parents=True, exist_ok=True)
    with (args.run_root / ".launcher.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another lambda-sweep launcher is using this run root") from error
        validate_manifest(manifest_path, planned)
        if not manifest_path.exists():
            paired.write_json(manifest_path, planned)
        print("REUSED=16 NEW=32 TOTAL=48", flush=True)
        failed = False
        for group in planned["new_groups"]:
            print(f"LAMBDA {group['coefficient']} START", flush=True)
            result = subprocess.run(child_command(args, group), check=False)
            print(f"LAMBDA {group['coefficient']} END status={result.returncode}", flush=True)
            failed |= result.returncode != 0
        return 1 if failed else 0


def status(run_root: Path) -> None:
    manifest_path = run_root / "experiment_manifest.json"
    if not manifest_path.is_file():
        print(f"No sweep manifest at {manifest_path}")
        return
    manifest = paired.read_json(manifest_path)
    if manifest.get("protocol") != PROTOCOL:
        raise RuntimeError(f"Unexpected sweep protocol in {manifest_path}")
    counts = Counter({"completed": len(manifest["reused"])})
    rows = []
    for group in manifest["new_groups"]:
        child_root = Path(group["run_root"])
        for job in group["jobs"]:
            state = paired.status_of(child_root, job)["status"]
            counts[state] += 1
            steps = total_steps(job) if state == "completed" else paired.logged_steps(
                child_root / "logs" / f"{job['name']}.log"
            )
            target = total_steps(job)
            fraction = min(1.0, steps / target)
            filled = int(fraction * 24)
            bar = "█" * filled + "░" * (24 - filled)
            rows.append(
                f"{state.upper():9} [{bar}] {fraction:6.1%} "
                f"{steps:>10,}/{target:,}  lambda={group['coefficient']:g} {job['name']}"
            )
    print(
        f"REUSED=16 NEW_COMPLETED={counts['completed'] - 16} "
        f"RUNNING={counts['running']} FAILED={counts['failed']} "
        f"PENDING={counts['pending']} TOTAL=48"
    )
    print("Existing none baseline: 8 completed; existing lambda=1e-4 ARec: 8 completed")
    for row in rows:
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "status"))
    parser.add_argument("--mava-root", type=Path, default=Path("/home/data/zeshenghong/Mava"))
    parser.add_argument(
        "--source-root", type=Path,
        default=Path("/home/data/zeshenghong/JaxMARL/mava_high_agent_paired/formal_4seed_v1"),
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max-runs-per-gpu", type=int, default=1)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.mava_root = args.mava_root.expanduser().resolve()
    args.source_root = args.source_root.expanduser().resolve()
    args.run_root = args.run_root.expanduser().resolve()
    if args.mode == "status":
        status(args.run_root)
    else:
        raise SystemExit(run(args))


if __name__ == "__main__":
    main()
