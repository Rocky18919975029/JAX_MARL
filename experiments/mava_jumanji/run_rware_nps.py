"""Run seed-matched independent-actor RWARE MAPPO, alignment, and ARec.

The parameter-sharing MAPPO runs cannot serve as this protocol's baseline:
all four conditions initialise and train independent recurrent actor banks.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import fcntl
import math
import os
from pathlib import Path
import re
import subprocess
import time

import run_matched_optimal as paired


TASK = "rware_large-8ag"
SEEDS = (1, 2, 3, 4)
METHODS = ("none", "mse", "cka", "arec")
DISTANCES = {"none": "none", "mse": "ln_mse", "cka": "linear_cka"}
PROTOCOL = "mava-jumanji-rware-rec-mappo-nps-four-method-v1"
HERE = Path(__file__).resolve().parent
NPS_SCRIPT = HERE / "rec_mappo_nps.py"
NPS_CONFIG = HERE / "rec_mappo_nps.yaml"
AREC_SCRIPT = HERE / "rec_mappo_arec_nps.py"
AREC_CONFIG = HERE / "rec_mappo_arec_nps.yaml"
ACTOR_HELPER = HERE / "independent_recurrent_actor.py"
ALGORITHM_NAMES = {
    "none": "rec_mappo_nps",
    "mse": "rec_mappo_nps_ln_mse",
    "cka": "rec_mappo_nps_linear_cka",
    "arec": "rec_mappo_nps_arec",
}


def parse_evaluations(payload: dict, condition: str, seed: int) -> dict[int, float]:
    """Read only complete evaluation records from one Mava JSON run."""
    run = payload["RobotWarehouse"]["large-8ag"][ALGORITHM_NAMES[condition]][f"seed_{seed}"]
    values = {}
    for key, row in run.items():
        if not key.startswith("step_") or not isinstance(row, dict):
            continue
        if "step_count" not in row or "mean_episode_return" not in row:
            continue
        step = int(row["step_count"])
        value = row["mean_episode_return"]
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        value = float(value)
        if step <= 0 or step in values or not math.isfinite(value):
            raise ValueError(f"Invalid RWARE evaluation at step {step}")
        values[step] = value
    return values


def total_steps(job: dict) -> int:
    shared = job["shared_overrides"]
    return (
        shared["system.num_updates"] * shared["arch.num_envs"]
        * shared["system.update_batch_size"] * shared["system.rollout_length"]
    )


def complete_metric(run_dir: Path, job: dict, *, after_time: float = 0) -> Path | None:
    evaluations = job["shared_overrides"]["arch.num_evaluation"]
    stride = total_steps(job) // evaluations
    expected_steps = tuple(stride * index for index in range(1, evaluations + 1))
    complete = []
    for path in (run_dir / "json").glob("**/metrics.json"):
        if path.stat().st_mtime < after_time:
            continue
        try:
            values = parse_evaluations(paired.read_json(path), job["condition"], job["seed"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if tuple(sorted(values)) == expected_steps:
            complete.append(path)
    return max(complete, key=lambda path: (path.stat().st_mtime_ns, str(path))) if complete else None


def validate_mava(mava_root: Path, config: dict) -> None:
    revision = subprocess.check_output(
        ["git", "-C", str(mava_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != config["mava_commit"]:
        raise RuntimeError(f"Mava revision {revision} differs from benchmark pin")
    for source, destination in (
        (NPS_SCRIPT, mava_root / "mava/systems/ppo/anakin/rec_mappo_nps.py"),
        (NPS_CONFIG, mava_root / "mava/configs/default/rec_mappo_nps.yaml"),
        (AREC_SCRIPT, mava_root / "mava/systems/ppo/anakin/rec_mappo_arec_nps.py"),
        (AREC_CONFIG, mava_root / "mava/configs/default/rec_mappo_arec_nps.yaml"),
        (ACTOR_HELPER, mava_root / "mava/systems/ppo/anakin/independent_recurrent_actor.py"),
    ):
        if not destination.is_file() or paired.digest(source) != paired.digest(destination):
            raise RuntimeError(f"Install the current {source.name} at {destination}")
    if not (mava_root / ".venv/bin/python").is_file():
        raise RuntimeError(f"Mava virtual-environment Python is missing at {mava_root}")
    changed = subprocess.check_output(
        ["git", "-C", str(mava_root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).splitlines()
    if changed:
        raise RuntimeError("Mava checkout has tracked changes: " + "; ".join(changed[:8]))


def build_manifest(args: argparse.Namespace, config: dict) -> dict:
    seeds = (1,) if args.smoke else SEEDS
    jobs = paired.make_jobs(config, (TASK,), seeds, args.smoke, METHODS)
    return {
        "protocol": PROTOCOL,
        "smoke": args.smoke,
        "mava_root": str(args.mava_root),
        "mava_commit": config["mava_commit"],
        "benchmark_config_sha256": paired.digest(paired.CONFIG_PATH),
        "nps_script_sha256": paired.digest(NPS_SCRIPT),
        "nps_config_sha256": paired.digest(NPS_CONFIG),
        "arec_script_sha256": paired.digest(AREC_SCRIPT),
        "arec_config_sha256": paired.digest(AREC_CONFIG),
        "actor_helper_sha256": paired.digest(ACTOR_HELPER),
        "baseline_script_sha256": paired.digest(
            args.mava_root / "mava/systems/ppo/anakin/rec_mappo.py"
        ),
        "task": TASK,
        "seeds": list(seeds),
        "actor_parameter_sharing": False,
        "coefficients": {
            "none": 0.0, "mse": args.mse_coef,
            "cka": args.cka_coef, "arec": args.arec_coef,
        },
        "epsilon": args.epsilon,
        "arec": {
            "q_steps": args.q_steps,
            "q_lr": args.q_lr,
            "fisher_ridge": args.fisher_ridge,
        },
        "jobs": jobs,
    }


def worker_command(mava_root: Path, run_root: Path, job: dict, manifest: dict) -> list[str]:
    run_dir = run_root / "runs" / job["name"]
    shared = {
        **job["shared_overrides"],
        "logger.base_exp_path": str(run_dir),
        "logger.loggers.json.enabled": True,
        "hydra.run.dir": str(run_dir / "hydra"),
    }
    if job["condition"] == "arec":
        script = "rec_mappo_arec_nps.py"
        shared.update({
            "arec.coef": manifest["coefficients"]["arec"],
            "arec.q_steps": manifest["arec"]["q_steps"],
            "arec.q_lr": manifest["arec"]["q_lr"],
            "arec.fisher_ridge": manifest["arec"]["fisher_ridge"],
        })
    else:
        script = "rec_mappo_nps.py"
        shared.update({
            "align.distance": DISTANCES[job["condition"]],
            "align.coef": manifest["coefficients"][job["condition"]],
            "align.epsilon": manifest["epsilon"],
        })
    return [
        str(mava_root / ".venv/bin/python"), "-u",
        str(mava_root / "mava/systems/ppo/anakin" / script),
        *(f"{key}={paired.hydra_value(value)}" for key, value in shared.items()),
    ]


def status(run_root: Path) -> None:
    path = run_root / "experiment_manifest.json"
    if not path.is_file():
        print(f"No independent-actor manifest at {path}")
        return
    manifest = paired.read_json(path)
    if manifest.get("protocol") != PROTOCOL:
        raise RuntimeError(f"Unexpected independent-actor protocol in {path}")
    counts = Counter(paired.status_of(run_root, job)["status"] for job in manifest["jobs"])
    print(
        " ".join(f"{name.upper()}={counts[name]}" for name in
                   ("completed", "running", "failed", "pending"))
        + f" TOTAL={len(manifest['jobs'])}"
    )
    for job in manifest["jobs"]:
        state = paired.status_of(run_root, job)["status"]
        target = total_steps(job)
        steps = target if state == "completed" else paired.logged_steps(
            run_root / "logs" / f"{job['name']}.log"
        )
        fraction = min(1.0, steps / target)
        bar = "█" * int(24 * fraction) + "░" * (24 - int(24 * fraction))
        print(f"{state.upper():9} [{bar}] {fraction:6.1%} "
              f"{steps:>10,}/{target:,}  {job['name']}")


def run(args: argparse.Namespace) -> int:
    config = paired.read_json(paired.CONFIG_PATH)
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.mse_coef, args.cka_coef, args.arec_coef, args.epsilon,
            args.q_lr, args.fisher_ridge)):
        raise ValueError("Auxiliary coefficients, learning rate, ridges and epsilon must be finite positive")
    if args.q_steps < 1:
        raise ValueError("ARec q_steps must be positive")
    gpus = paired.parse_names(args.gpus, set(args.gpus.split(",")), "GPUs")
    if any(not re.fullmatch(r"\d+", gpu) for gpu in gpus):
        raise ValueError("GPUs must be comma-separated non-negative device IDs")
    if args.max_runs_per_gpu != 1:
        raise ValueError("Pinned Mava formal runs use one worker per GPU")
    validate_mava(args.mava_root, config)
    manifest = build_manifest(args, config)
    if args.dry_run:
        print(f"NEW={len(manifest['jobs'])}")
        for job in manifest["jobs"]:
            print(job["name"], " ".join(worker_command(args.mava_root, args.run_root, job, manifest)))
        return 0

    args.run_root.mkdir(parents=True, exist_ok=True)
    with (args.run_root / ".launcher.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another independent-actor launcher is using this run root") from error
        manifest_path = args.run_root / "experiment_manifest.json"
        if manifest_path.exists():
            if paired.read_json(manifest_path) != manifest:
                raise RuntimeError("Existing independent-actor manifest differs; use a new run root")
        else:
            paired.write_json(manifest_path, manifest)
        pending = deque()
        for job in manifest["jobs"]:
            old = paired.status_of(args.run_root, job)
            if old["status"] == "completed":
                metric = Path(old["metric_file"]) if old.get("metric_file") else None
                if metric is None or not metric.is_file() or paired.digest(metric) != old.get("metric_sha256"):
                    raise RuntimeError(f"Completed independent-actor metric changed: {job['name']}")
                continue
            if old["status"] == "running" and paired.pid_alive(old.get("pid")):
                raise RuntimeError(f"Worker still running: {job['name']}")
            if old["status"] == "failed" and not args.retry_failed:
                continue
            pending.append(job)
        print(f"selected={len(manifest['jobs'])} "
              f"pending={len(pending)}", flush=True)
        active = {}
        load = Counter()
        while pending or active:
            while pending:
                available = [gpu for gpu in gpus if load[gpu] == 0]
                if not available:
                    break
                gpu = available[0]
                job = pending.popleft()
                command = worker_command(args.mava_root, args.run_root, job, manifest)
                run_dir = args.run_root / "runs" / job["name"]
                run_dir.mkdir(parents=True, exist_ok=True)
                log_path = args.run_root / "logs" / f"{job['name']}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                env = os.environ.copy()
                env.pop("LD_LIBRARY_PATH", None)
                env["CUDA_VISIBLE_DEVICES"] = gpu
                with log_path.open("a") as log:
                    log.write(f"\n===== START gpu={gpu} command={command!r} =====\n")
                    log.flush()
                    process = subprocess.Popen(
                        command, cwd=args.mava_root, env=env, stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                paired.write_json(paired.status_path(args.run_root, job), {
                    "status": "running", "pid": process.pid, "gpu": gpu,
                    "run_name": job["name"], "started_at": time.time(),
                    "command": command,
                })
                active[job["name"]] = (process, gpu, job)
                load[gpu] += 1
                print(f"GPU {gpu} START {job['name']} pid={process.pid}", flush=True)
            if not active:
                continue
            time.sleep(2)
            for name, (process, gpu, job) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                old = paired.status_of(args.run_root, job)
                metric = complete_metric(
                    args.run_root / "runs" / job["name"], job,
                    after_time=old["started_at"],
                ) if code == 0 else None
                effective_code = code if code != 0 or metric is not None else 2
                paired.write_json(paired.status_path(args.run_root, job), {
                    **old, "status": "completed" if effective_code == 0 else "failed",
                    "exit_code": effective_code, "finished_at": time.time(),
                    "metric_file": str(metric) if metric is not None else None,
                    "metric_sha256": paired.digest(metric) if metric is not None else None,
                    "error": "missing full evaluation log" if code == 0 and metric is None else None,
                })
                del active[name]
                load[gpu] -= 1
                print(f"GPU {gpu} END   {name} status={effective_code}", flush=True)
        counts = Counter(paired.status_of(args.run_root, job)["status"] for job in manifest["jobs"])
        print(f"finished completed={counts['completed']} failed={counts['failed']}", flush=True)
        return 1 if counts["failed"] else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "status"))
    parser.add_argument("--mava-root", type=Path, default=Path("/home/data/zeshenghong/Mava"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max-runs-per-gpu", type=int, default=1)
    parser.add_argument("--mse-coef", type=float, default=0.01)
    parser.add_argument("--cka-coef", type=float, default=0.03)
    parser.add_argument("--arec-coef", type=float, default=1e-4)
    parser.add_argument("--q-steps", type=int, default=4)
    parser.add_argument("--q-lr", type=float, default=1e-3)
    parser.add_argument("--fisher-ridge", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    args.mava_root = args.mava_root.expanduser().resolve()
    args.run_root = args.run_root.expanduser().resolve()
    if args.mode == "status":
        status(args.run_root)
    else:
        raise SystemExit(run(args))


if __name__ == "__main__":
    main()
