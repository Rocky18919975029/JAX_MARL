#!/usr/bin/env python3
"""Launch the matched ShadowHandOver HAPPO/MAPPO/MADPO matrix."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "experiments" / "harl_dexhands" / "train.py"
OFFICIAL_CONFIG = Path("tuned_configs/dexhands/ShadowHandOver/happo/config.json")


def training_environment(gpu: str) -> dict[str, str]:
    """Build an Isaac Gym child environment from the active Python runtime."""

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    conda_prefix = Path(environment.get("CONDA_PREFIX", sys.prefix)).expanduser()
    runtime_lib = str(conda_prefix / "lib")
    current = environment.get("LD_LIBRARY_PATH", "")
    entries = [entry for entry in current.split(os.pathsep) if entry]
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        [runtime_lib, *(entry for entry in entries if entry != runtime_lib)]
    )
    return environment


def verify_wandb() -> None:
    """Reject an incomplete or shadowed W&B import before launching workers."""
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B online mode requires the wandb SDK in the active Python "
            "environment; install it or use --wandb-mode disabled"
        ) from error
    if not callable(getattr(wandb, "init", None)):
        raise RuntimeError(
            "The imported wandb module has no callable init; "
            f"loaded from {getattr(wandb, '__file__', None)!r}. "
            "Check for a shadowing local module or an incomplete SDK installation."
        )


def freeze_manifest(root: Path, study_spec: dict, tasks: list) -> dict:
    """Record the exact grid and reject accidental changes on resume."""
    path = root / "experiment_manifest.json"
    payload = {
        "schema_version": 1,
        "study_spec": study_spec,
        "runs": [
            {
                "run_name": task.name,
                "algorithm": task.algorithm,
                "condition": task.condition,
                "seed": task.seed,
                "arec_coef": task.arec_coef if task.condition == "arec" else 0.0,
            }
            for task in tasks
        ],
    }
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(
                f"Run-root manifest differs from this launch: {path}. "
                "Use the original grid/settings or a new run root."
            )
        return existing
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return payload


def read_status(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def process_is_running(pid: int, proc_root: Path = Path("/proc")) -> bool:
    """Ignore exited and zombie workers without relying on stale status JSON."""
    try:
        fields = (proc_root / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return False
    return bool(fields) and fields[0] not in ("Z", "X")


def running_gpu(
    run_name: str, status: dict, gpus: tuple[str, ...],
    proc_root: Path = Path("/proc"),
    expected_log: Path | None = None,
) -> str | None:
    """Identify an already-running Isaac Gym worker before occupying its GPU."""
    if status.get("status") != "running":
        return None
    pid = status.get("pid")
    if not isinstance(pid, int) or pid <= 0 or not process_is_running(pid, proc_root):
        return None
    proc = proc_root / str(pid)
    try:
        command = proc.joinpath("cmdline").read_bytes()
        environment = proc.joinpath("environ").read_bytes().split(b"\0")
    except OSError as error:
        raise RuntimeError(f"Cannot inspect live worker {run_name} pid={pid}") from error
    # HARL may replace argv with a process title; stdout still points to this
    # run's launcher log, which provides a second exact identity check.
    log_matches = False
    if expected_log is not None:
        try:
            log_matches = proc.joinpath("fd", "1").resolve(strict=True) == expected_log.resolve()
        except OSError:
            pass
    if run_name.encode() not in command and not log_matches:
        raise RuntimeError(
            f"Status says {run_name} is running at pid={pid}, but the live process "
            "has a different title and stdout log; refusing to risk a duplicate launch"
        )
    devices = [item.split(b"=", 1)[1].decode() for item in environment
               if item.startswith(b"CUDA_VISIBLE_DEVICES=")]
    if len(devices) != 1 or devices[0] not in gpus:
        raise RuntimeError(
            f"Live worker {run_name} pid={pid} uses GPU {devices!r}; "
            f"select its GPU among {gpus} before resuming"
        )
    return devices[0]


def mark_failed(path: Path, run_name: str, reason: str) -> None:
    payload = read_status(path)
    if payload.get("status") in ("completed", "failed"):
        return
    payload.update(status="failed", run_name=run_name, error=reason)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def archive_failed_attempt(root: Path, run_name: str) -> None:
    """Keep earlier failed metrics/logs separate from a fresh training attempt."""
    status = root / "status" / f"{run_name}.json"
    if read_status(status).get("status") != "failed":
        return
    parent = root / "failed_attempts" / run_name
    attempt = 1
    while (parent / f"attempt_{attempt:02d}").exists():
        attempt += 1
    archive = parent / f"attempt_{attempt:02d}"
    archive.mkdir(parents=True)
    for source, label in (
        (status, "status.json"),
        (root / "metrics" / f"{run_name}.jsonl", "metrics.jsonl"),
        (root / "logs" / f"{run_name}.log", "train.log"),
    ):
        if source.exists():
            shutil.move(str(source), str(archive / label))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--harl-root", type=Path, default=REPO_ROOT / "third_party" / "HARL"
    )
    parser.add_argument("--algorithms", default="happo,mappo,madpo")
    parser.add_argument("--conditions", default="none")
    parser.add_argument("--seeds", default="1-4")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max-runs-per-gpu", type=int, default=1)
    parser.add_argument("--num-env-steps", type=int)
    parser.add_argument("--n-rollout-threads", type=int)
    parser.add_argument("--div-coef", type=float, default=1000.0)
    parser.add_argument("--div-weight", type=float, default=0.05)
    parser.add_argument("--div-sigma", type=float, default=1.0)
    parser.add_argument("--div-max-samples", type=int, default=1024)
    coefficient_group = parser.add_mutually_exclusive_group()
    coefficient_group.add_argument("--arec-coef", type=float, default=0.0001)
    coefficient_group.add_argument("--arec-coefs")
    parser.add_argument("--arec-q-steps", type=int, default=4)
    parser.add_argument("--arec-q-lr", type=float, default=0.001)
    parser.add_argument("--arec-fisher-ridge", type=float, default=0.001)
    parser.add_argument("--wandb-project", default="harl-dexhands-shadowhandover")
    parser.add_argument(
        "--wandb-mode", choices=("online", "disabled"), default="online"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    from experiments.harl_dexhands.protocol import (
        ALGORITHMS,
        AREC_PROTOCOL_VERSION,
        CONDITIONS,
        PROTOCOL_VERSION,
        parse_csv,
        parse_positive_floats,
        parse_seeds,
        task_matrix,
    )

    if args.max_runs_per_gpu < 1:
        raise ValueError("max-runs-per-gpu must be positive")
    if args.max_runs_per_gpu > 2:
        raise ValueError("ShadowHandOver is limited to two concurrent runs per GPU")
    if args.num_env_steps is not None and args.num_env_steps <= 0:
        raise ValueError("num-env-steps must be positive")
    if args.n_rollout_threads is not None and args.n_rollout_threads <= 0:
        raise ValueError("n-rollout-threads must be positive")
    if args.arec_q_steps < 1 or args.arec_q_lr <= 0 or args.arec_fisher_ridge <= 0:
        raise ValueError("ARec q steps/LR and Fisher ridge must be positive")
    algorithms = parse_csv(args.algorithms, ALGORITHMS)
    conditions = parse_csv(args.conditions, CONDITIONS)
    seeds = parse_seeds(args.seeds)
    arec_coefs = (
        (args.arec_coef,)
        if args.arec_coefs is None
        else parse_positive_floats(args.arec_coefs)
    )
    gpus = tuple(piece.strip() for piece in args.gpus.split(",") if piece.strip())
    if not gpus:
        raise ValueError("Select at least one GPU")
    if args.wandb_mode == "online" and not args.dry_run:
        verify_wandb()
    root = args.run_root.expanduser().resolve()
    harl_root = args.harl_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    tasks = task_matrix(
        algorithms,
        seeds,
        div_coef=args.div_coef,
        div_weight=args.div_weight,
        div_sigma=args.div_sigma,
        div_max_samples=args.div_max_samples,
        conditions=conditions,
        arec_coef=args.arec_coef,
        arec_coefs=arec_coefs,
        arec_q_steps=args.arec_q_steps,
        arec_q_lr=args.arec_q_lr,
        arec_fisher_ridge=args.arec_fisher_ridge,
    )
    config_path = harl_root / OFFICIAL_CONFIG
    config_hash = (
        hashlib.sha256(config_path.read_bytes()).hexdigest()
        if config_path.is_file()
        else None
    )
    if not args.dry_run and config_hash is None:
        raise FileNotFoundError(f"Missing official HARL config: {config_path}")
    official_budget = (
        int(json.loads(config_path.read_text())["algo_args"]["train"]["num_env_steps"])
        if config_hash is not None
        else 0
    )
    protocol_version = (
        AREC_PROTOCOL_VERSION if "arec" in conditions else PROTOCOL_VERSION
    )
    freeze_manifest(
        root,
        {
            "protocol_version": protocol_version,
            "algorithms": list(algorithms),
            "conditions": list(conditions),
            "seeds": list(seeds),
            "arec_coefs": list(arec_coefs),
            "arec_q_steps": args.arec_q_steps,
            "arec_q_lr": args.arec_q_lr,
            "arec_fisher_ridge": args.arec_fisher_ridge,
            "div_coef": args.div_coef,
            "div_weight": args.div_weight,
            "div_sigma": args.div_sigma,
            "div_max_samples": args.div_max_samples,
            "num_env_steps": args.num_env_steps or official_budget,
            "n_rollout_threads": args.n_rollout_threads,
            "harl_root": str(harl_root),
            "official_config_sha256": config_hash,
        },
        tasks,
    )

    lock_path = root / ".launcher.lock"
    launcher_lock = lock_path.open("a+")
    try:
        fcntl.flock(launcher_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        launcher_lock.close()
        raise RuntimeError(
            f"Another launcher is already managing {root}; do not start a duplicate"
        ) from error

    def status_path(task) -> Path:
        return root / "status" / f"{task.name}.json"
    external: dict[str, tuple[int, str, int]] = {}
    if not args.dry_run:
        for task in tasks:
            status = read_status(status_path(task))
            if status.get("status") != "running":
                continue
            gpu = running_gpu(
                task.name, status, gpus,
                expected_log=root / "logs" / f"{task.name}.log",
            )
            if gpu is None:
                mark_failed(
                    status_path(task), task.name,
                    f"Stale running status: pid={status.get('pid')} is no longer alive",
                )
            else:
                external[task.name] = (status["pid"], gpu, task.seed)

    pending = [
        task for task in tasks
        if read_status(status_path(task)).get("status") != "completed"
    ]
    print(
        f"Protocol={protocol_version} "
        f"total={len(tasks)} pending={len(pending)} "
        f"algorithms={','.join(algorithms)} conditions={','.join(conditions)} "
        f"seeds={','.join(map(str, seeds))} "
        f"arec_coefs={','.join(map(str, arec_coefs))}",
        flush=True,
    )
    lock = threading.Lock()
    launcher_log = root / "launcher.log"

    def event(message: str) -> None:
        line = f"[{dt.datetime.now().astimezone():%Y-%m-%d %H:%M:%S}] {message}"
        with lock:
            print(line, flush=True)
            with launcher_log.open("a", encoding="utf-8") as file:
                file.write(line + "\n")

    for name, (pid, gpu, seed) in external.items():
        event(f"GPU {gpu} KEEP  {name} pid={pid} seed={seed}")

    def refresh_external() -> None:
        for name, (pid, gpu, _seed) in list(external.items()):
            if process_is_running(pid):
                continue
            del external[name]
            path = root / "status" / f"{name}.json"
            if read_status(path).get("status") == "running":
                mark_failed(path, name, f"Existing worker pid={pid} exited without final status")
            event(f"GPU {gpu} RELEASE {name} pid={pid}")

    def run_one(task, gpu: str, retry: bool) -> int:
        command = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--harl-root", str(harl_root),
            "--run-root", str(root),
            "--run-name", task.name,
            "--algorithm", task.algorithm,
            "--condition", task.condition,
            "--seed", str(task.seed),
            "--div-coef", str(args.div_coef),
            "--div-weight", str(args.div_weight),
            "--div-sigma", str(args.div_sigma),
            "--div-max-samples", str(args.div_max_samples),
            "--arec-coef", str(task.arec_coef),
            "--arec-q-steps", str(args.arec_q_steps),
            "--arec-q-lr", str(args.arec_q_lr),
            "--arec-fisher-ridge", str(args.arec_fisher_ridge),
            "--wandb-project", args.wandb_project,
            "--wandb-mode", args.wandb_mode,
        ]
        if args.num_env_steps is not None:
            command.extend(["--num-env-steps", str(args.num_env_steps)])
        if args.n_rollout_threads is not None:
            command.extend(["--n-rollout-threads", str(args.n_rollout_threads)])
        if args.dry_run:
            event(f"GPU {gpu} START {task.name}")
            event("COMMAND " + " ".join(command))
            return 0
        try:
            if retry:
                archive_failed_attempt(root, task.name)
            event(f"GPU {gpu} {'RETRY' if retry else 'START'} {task.name}")
            output = root / "logs" / f"{task.name}.log"
            with output.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=training_environment(gpu),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            code = result.returncode
            if read_status(status_path(task)).get("status") != "completed":
                mark_failed(
                    status_path(task), task.name,
                    f"Worker exited with code {code} without completed status",
                )
        except Exception as error:
            code = 1
            mark_failed(status_path(task), task.name, f"Launcher error: {error!r}")
            event(f"GPU {gpu} ERROR {task.name}: {error!r}")
        event(f"GPU {gpu} END   {task.name} status={code}")
        return code

    def run_wave(seed: int, jobs: list, retry: bool, executor: ThreadPoolExecutor) -> None:
        waiting = collections.deque(jobs)
        active = {}
        while waiting or active or any(item[2] == seed for item in external.values()):
            refresh_external()
            for future in list(active):
                if not future.done():
                    continue
                future.result()
                del active[future]
            occupied = collections.Counter(item[1] for item in external.values())
            occupied.update(gpu for _task, gpu in active.values())
            for gpu in gpus:
                while waiting and occupied[gpu] < args.max_runs_per_gpu:
                    task = waiting.popleft()
                    if read_status(status_path(task)).get("status") == "completed":
                        continue
                    future = executor.submit(run_one, task, gpu, retry)
                    active[future] = (task, gpu)
                    occupied[gpu] += 1
            if waiting or active or any(item[2] == seed for item in external.values()):
                if active:
                    wait(active, timeout=2, return_when=FIRST_COMPLETED)
                else:
                    time.sleep(2)

    with ThreadPoolExecutor(max_workers=len(gpus) * args.max_runs_per_gpu) as executor:
        for seed in seeds:
            cohort = [task for task in tasks if task.seed == seed]
            primary = [
                task for task in cohort
                if read_status(status_path(task)).get("status") != "completed"
                and (
                    args.dry_run
                    or (
                        read_status(status_path(task)).get("status") != "failed"
                        and task.name not in external
                    )
                )
            ]
            event(f"SEED {seed} primary={len(primary)} existing={sum(item[2] == seed for item in external.values())}")
            run_wave(seed, primary, False, executor)
            if args.dry_run:
                continue
            failed = [
                task for task in cohort
                if read_status(status_path(task)).get("status") == "failed"
            ]
            event(f"SEED {seed} retry={len(failed)}")
            run_wave(seed, failed, True, executor)
            incomplete = [
                task.name for task in cohort
                if read_status(status_path(task)).get("status") != "completed"
            ]
            event(f"SEED {seed} finished completed={len(cohort)-len(incomplete)}/{len(cohort)}")

    if args.dry_run:
        event("dry-run finished")
        launcher_lock.close()
        return
    states = collections.Counter(
        read_status(status_path(task)).get("status", "pending") for task in tasks
    )
    pending_count = len(tasks) - states["completed"] - states["failed"] - states["running"]
    event(
        f"matrix finished; completed={states['completed']} "
        f"failed={states['failed']} running={states['running']} "
        f"pending={pending_count}"
    )
    launcher_lock.close()
    if states["completed"] != len(tasks):
        raise RuntimeError(
            f"{len(tasks)-states['completed']} runs are incomplete; "
            f"inspect {root / 'failed_attempts'} and {root / 'logs'}"
        )


if __name__ == "__main__":
    main()
