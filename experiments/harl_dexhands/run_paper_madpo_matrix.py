#!/usr/bin/env python3
"""Finish a 48-run ShadowHandOver study with paper-profile MADPO.

Completed HAPPO/MAPPO runs are referenced read-only from the legacy study.
Missing HAPPO/MAPPO and all MADPO paper-profile runs are trained in a new root.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = REPO_ROOT / "experiments" / "harl_dexhands" / "train.py"
sys.path.insert(0, str(REPO_ROOT))

from experiments.harl_dexhands.protocol import (  # noqa: E402
    MADPO_PAPER_CONFIG,
    madpo_paper_settings,
    task_matrix,
)
from experiments.harl_dexhands.run_matrix import (  # noqa: E402
    OFFICIAL_CONFIG,
    archive_failed_attempt,
    mark_failed,
    process_is_running,
    read_status,
    running_gpu,
    training_environment,
    verify_wandb,
)


def legacy_launcher_pids(
    legacy_root: Path, proc_root: Path = Path("/proc")
) -> list[int]:
    """Find only a legacy matrix launcher with the exact frozen run root."""
    matches = []
    for proc in proc_root.iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            argv = [
                part.decode()
                for part in (proc / "cmdline").read_bytes().split(b"\0")
                if part
            ]
        except (OSError, UnicodeDecodeError):
            continue
        if not any(
            arg.endswith("experiments/harl_dexhands/run_matrix.py") for arg in argv
        ):
            continue
        for index, arg in enumerate(argv[:-1]):
            if (
                arg == "--run-root"
                and Path(argv[index + 1]).expanduser().resolve() == legacy_root
            ):
                matches.append(int(proc.name))
                break
    return matches


def stop_legacy_launcher(legacy_root: Path, lock_path: Path, timeout: float = 20.0):
    """Stop the old scheduler only; workers are inspected separately below."""
    candidates = legacy_launcher_pids(legacy_root)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Legacy launcher lock is held, but found {len(candidates)} exact launcher "
            f"processes ({candidates}); stop it manually before retrying"
        )
    pid = candidates[0]
    # A second exact process check narrows the PID-reuse window before signaling.
    if pid not in legacy_launcher_pids(legacy_root):
        raise RuntimeError("Legacy launcher changed during preflight; retry")
    print(f"Stopping legacy scheduler pid={pid} for {legacy_root}", flush=True)
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        stream = lock_path.open("a+")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return stream
        except BlockingIOError:
            stream.close()
            time.sleep(0.25)
    raise RuntimeError(f"Legacy launcher did not release {lock_path} after SIGTERM")


def legacy_worker_is_live(
    name: str, status: dict, legacy_root: Path, proc_root: Path = Path("/proc")
) -> bool:
    """Verify a legacy worker without relying on mutable /proc environment data."""
    if status.get("status") != "running":
        return False
    if status.get("run_name") != name:
        raise RuntimeError(f"Legacy status has a different run name: {name}")
    pid = status.get("pid")
    if not isinstance(pid, int) or pid <= 0 or not process_is_running(pid, proc_root):
        return False
    proc = proc_root / str(pid)
    expected_log = legacy_root / "logs" / f"{name}.log"
    try:
        log_matches = (proc / "fd" / "1").resolve(strict=True) == expected_log.resolve(
            strict=True
        )
    except OSError:
        log_matches = False
    if log_matches:
        return True
    try:
        argv = (proc / "cmdline").read_bytes().split(b"\0")
    except OSError as error:
        raise RuntimeError(
            f"Cannot inspect live legacy worker {name} pid={pid}"
        ) from error
    if (
        name.encode() in argv
        and any(arg.endswith(b"experiments/harl_dexhands/train.py") for arg in argv)
        and str(legacy_root).encode() in argv
    ):
        return True
    raise RuntimeError(
        f"Legacy status points to live pid={pid}, but its log and command do not "
        f"identify {name}; refusing to signal that PID"
    )


def stop_legacy_worker(
    name: str, status: dict, legacy_root: Path, *, timeout: float = 20.0
) -> bool:
    """Terminate only a live worker whose run-specific identity was verified."""
    if not legacy_worker_is_live(name, status, legacy_root):
        return False
    pid = int(status["pid"])
    if not legacy_worker_is_live(name, status, legacy_root):
        return True
    print(f"Stopping legacy worker pid={pid} {name}", flush=True)
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not legacy_worker_is_live(name, status, legacy_root):
            return True
        time.sleep(0.25)
    # Never send SIGKILL to an unverified or recycled PID.
    if not legacy_worker_is_live(name, status, legacy_root):
        return True
    os.kill(pid, signal.SIGKILL)
    for _ in range(40):
        if not legacy_worker_is_live(name, status, legacy_root):
            return True
        time.sleep(0.25)
    raise RuntimeError(f"Worker still live after SIGKILL: {name} pid={pid}")


def read_legacy_study(legacy_root: Path, config_hash: str) -> tuple[dict, list]:
    path = legacy_root / "experiment_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen legacy manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    spec = manifest["study_spec"]
    if (
        spec.get("algorithms") != ["happo", "mappo", "madpo"]
        or spec.get("conditions") != ["none", "arec"]
        or spec.get("seeds") != [1, 2, 3, 4]
        or spec.get("arec_coefs") != [3e-5, 1e-4, 3e-4]
        or spec.get("madpo_profile", "legacy") != "legacy"
        or spec.get("official_config_sha256") != config_hash
    ):
        raise RuntimeError("Legacy root is not the expected frozen 48-run study")
    paper = madpo_paper_settings()
    div = paper["algo"]
    tasks = task_matrix(
        ("happo", "mappo", "madpo"),
        (1, 2, 3, 4),
        conditions=("none", "arec"),
        arec_coefs=tuple(spec["arec_coefs"]),
        arec_q_steps=spec["arec_q_steps"],
        arec_q_lr=spec["arec_q_lr"],
        arec_fisher_ridge=spec["arec_fisher_ridge"],
        div_coef=div["div_coef"],
        div_weight=div["div_weight"],
        div_sigma=div["div_sigma"],
        div_max_samples=div["div_max_samples"],
    )
    if len(tasks) != 48:
        raise AssertionError("Expected exactly 48 logical runs")
    old_names = {row["run_name"] for row in manifest["runs"]}
    if any(task.name not in old_names for task in tasks if task.algorithm != "madpo"):
        raise RuntimeError("Legacy manifest is missing a HAPPO/MAPPO run")
    return spec, tasks


def freeze_new_manifest(
    root: Path,
    legacy_root: Path,
    spec: dict,
    tasks: list,
    config_hash: str,
    paper_hash: str,
) -> dict:
    paper_budget = int(madpo_paper_settings()["train"]["num_env_steps"])
    legacy_budget = int(spec["num_env_steps"])
    payload = {
        "schema_version": 2,
        "study_spec": {
            "protocol_version": "harl-dexhands-shadowhandover-paper-madpo-v2.0",
            "legacy_run_root": str(legacy_root),
            "legacy_manifest_sha256": hashlib.sha256(
                (legacy_root / "experiment_manifest.json").read_bytes()
            ).hexdigest(),
            "official_config_sha256": config_hash,
            "madpo_profile": "paper2024",
            "madpo_paper_config_sha256": paper_hash,
            "algorithms": ["happo", "mappo", "madpo"],
            "conditions": ["none", "arec"],
            "seeds": [1, 2, 3, 4],
            "arec_coefs": spec["arec_coefs"],
            "arec_q_steps": spec["arec_q_steps"],
            "arec_q_lr": spec["arec_q_lr"],
            "arec_fisher_ridge": spec["arec_fisher_ridge"],
            "num_env_steps": legacy_budget,
            "num_env_steps_by_algorithm": {
                "happo": legacy_budget,
                "mappo": legacy_budget,
                "madpo": paper_budget,
            },
        },
        "runs": [
            {
                "run_name": task.name,
                "algorithm": task.algorithm,
                "condition": task.condition,
                "seed": task.seed,
                "arec_coef": task.arec_coef if task.condition == "arec" else 0.0,
                "num_env_steps": (
                    paper_budget if task.algorithm == "madpo" else legacy_budget
                ),
                **(
                    {"reuse_completed_from": str(legacy_root)}
                    if task.algorithm in ("happo", "mappo")
                    else {}
                ),
            }
            for task in tasks
        ],
    }
    path = root / "experiment_manifest.json"
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise RuntimeError(
                f"Frozen study differs; use its original settings or a new root: {path}"
            )
    else:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    return payload


def completed_source(task, legacy_root: Path, budget: int) -> bool:
    if task.algorithm == "madpo":
        return False
    status = read_status(legacy_root / "status" / f"{task.name}.json")
    if status.get("status") != "completed":
        return False
    if (
        status.get("run_name") != task.name
        or int(status.get("total_env_steps", -1)) != budget
    ):
        raise RuntimeError(
            f"Completed legacy run has incompatible identity/budget: {task.name}"
        )
    if not (legacy_root / "metrics" / f"{task.name}.jsonl").is_file():
        raise RuntimeError(f"Completed legacy run is missing metrics: {task.name}")
    return True


def _lock(path: Path, description: str):
    stream = path.open("a+")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        stream.close()
        raise RuntimeError(
            f"{description} launcher is still active ({path})"
        ) from error
    return stream


def least_loaded_gpu(
    gpus: tuple[str, ...], occupied: collections.Counter, limit: int
) -> str | None:
    """Spread a seed's runs across devices before filling a second slot."""
    available = [gpu for gpu in gpus if occupied[gpu] < limit]
    return (
        min(available, key=lambda gpu: (occupied[gpu], gpus.index(gpu)))
        if available
        else None
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-run-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--harl-root", type=Path, default=REPO_ROOT / "third_party" / "HARL"
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max-runs-per-gpu", type=int, default=2)
    parser.add_argument(
        "--wandb-project", default="harl-dexhands-shadowhandover-arec-grid4seed"
    )
    parser.add_argument(
        "--wandb-mode", choices=("online", "disabled"), default="online"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.max_runs_per_gpu < 1 or args.max_runs_per_gpu > 2:
        raise ValueError("ShadowHandOver permits one or two concurrent runs per GPU")
    gpus = tuple(piece.strip() for piece in args.gpus.split(",") if piece.strip())
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("Select distinct GPUs")
    legacy_root = args.legacy_run_root.expanduser().resolve()
    root = args.run_root.expanduser().resolve()
    harl_root = args.harl_root.expanduser().resolve()
    if root == legacy_root or not legacy_root.is_dir():
        raise ValueError("Use a new run root and an existing legacy study root")
    config_path = harl_root / OFFICIAL_CONFIG
    if not config_path.is_file() or not (harl_root / "harl").is_dir():
        raise FileNotFoundError("Initialize third_party/HARL before launching")
    if args.wandb_mode == "online" and not args.dry_run:
        verify_wandb()
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    paper_hash = hashlib.sha256(MADPO_PAPER_CONFIG.read_bytes()).hexdigest()
    spec, tasks = read_legacy_study(legacy_root, config_hash)
    legacy_budget = int(spec["num_env_steps"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    launcher_lock = _lock(root / ".launcher.lock", "Replacement")
    try:
        try:
            legacy_lock = _lock(legacy_root / ".launcher.lock", "Legacy")
        except RuntimeError:
            if args.dry_run:
                raise RuntimeError(
                    "Dry-run will not stop the active legacy launcher; run without --dry-run "
                    "to perform the requested replacement"
                )
            legacy_lock = stop_legacy_launcher(
                legacy_root, legacy_root / ".launcher.lock"
            )
    except BaseException:
        launcher_lock.close()
        raise
    try:
        freeze_new_manifest(root, legacy_root, spec, tasks, config_hash, paper_hash)
        status_path = lambda task: root / "status" / f"{task.name}.json"
        completed = lambda task: (
            completed_source(task, legacy_root, legacy_budget)
            or read_status(status_path(task)).get("status") == "completed"
        )
        external: dict[str, tuple[int, str, int]] = {}
        stopped_workers = 0
        for row in json.loads((legacy_root / "experiment_manifest.json").read_text())[
            "runs"
        ]:
            status = read_status(legacy_root / "status" / f"{row['run_name']}.json")
            if status.get("status") != "running":
                continue
            if not legacy_worker_is_live(row["run_name"], status, legacy_root):
                continue
            if args.dry_run:
                print(
                    f"WOULD_STOP legacy pid={status['pid']} {row['run_name']}",
                    flush=True,
                )
                continue
            if stop_legacy_worker(row["run_name"], status, legacy_root):
                stopped_workers += 1
        for task in tasks:
            status = read_status(status_path(task))
            if status.get("status") != "running":
                continue
            gpu = running_gpu(
                task.name,
                status,
                gpus,
                expected_log=root / "logs" / f"{task.name}.log",
            )
            if gpu is not None:
                external[task.name] = (status["pid"], gpu, task.seed)
            elif not args.dry_run:
                mark_failed(status_path(task), task.name, "Stale running worker")

        initial_reused = sum(
            completed_source(t, legacy_root, legacy_budget) for t in tasks
        )
        print(
            f"TOTAL=48 REUSED={initial_reused} STOPPED_OLD={stopped_workers} "
            f"NEW_OR_INCOMPLETE={sum(not completed(t) for t in tasks)}",
            flush=True,
        )
        events_lock = threading.Lock()
        launcher_log = root / "launcher.log"

        def event(message: str) -> None:
            line = f"[{dt.datetime.now().astimezone():%Y-%m-%d %H:%M:%S}] {message}"
            with events_lock:
                print(line, flush=True)
                with launcher_log.open("a", encoding="utf-8") as file:
                    file.write(line + "\n")

        for name, (pid, gpu, _seed) in external.items():
            event(f"GPU {gpu} KEEP  {name} pid={pid}")

        def refresh_external() -> None:
            for name, (pid, gpu, _seed) in list(external.items()):
                if process_is_running(pid):
                    continue
                del external[name]
                event(f"GPU {gpu} RELEASE {name} pid={pid}")

        def run_one(task, gpu: str, retry: bool) -> None:
            command = [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--harl-root",
                str(harl_root),
                "--run-root",
                str(root),
                "--run-name",
                task.name,
                "--algorithm",
                task.algorithm,
                "--condition",
                task.condition,
                "--seed",
                str(task.seed),
                "--arec-coef",
                str(task.arec_coef),
                "--arec-q-steps",
                str(task.arec_q_steps),
                "--arec-q-lr",
                str(task.arec_q_lr),
                "--arec-fisher-ridge",
                str(task.arec_fisher_ridge),
                "--wandb-project",
                args.wandb_project,
                "--wandb-mode",
                args.wandb_mode,
            ]
            if task.algorithm == "madpo":
                command += [
                    "--madpo-profile",
                    "paper2024",
                    "--madpo-paper-config-sha256",
                    paper_hash,
                    "--div-coef",
                    str(task.div_coef),
                    "--div-weight",
                    str(task.div_weight),
                    "--div-sigma",
                    str(task.div_sigma),
                    "--div-max-samples",
                    str(task.div_max_samples),
                ]
            else:
                if spec.get("num_env_steps") != int(
                    json.loads(config_path.read_text())["algo_args"]["train"][
                        "num_env_steps"
                    ]
                ):
                    command += ["--num-env-steps", str(spec["num_env_steps"])]
                if spec.get("n_rollout_threads") is not None:
                    command += ["--n-rollout-threads", str(spec["n_rollout_threads"])]
            if args.dry_run:
                event(f"GPU {gpu} PLAN  {task.name}")
                event("COMMAND " + " ".join(command))
                return
            try:
                if retry:
                    archive_failed_attempt(root, task.name)
                event(f"GPU {gpu} {'RETRY' if retry else 'START'} {task.name}")
                path = root / "logs" / f"{task.name}.log"
                with path.open("w", encoding="utf-8") as log:
                    result = subprocess.run(
                        command,
                        cwd=REPO_ROOT,
                        env=training_environment(gpu),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                if read_status(status_path(task)).get("status") != "completed":
                    mark_failed(
                        status_path(task),
                        task.name,
                        f"Worker exit code {result.returncode}",
                    )
                event(f"GPU {gpu} END   {task.name} status={result.returncode}")
            except Exception as error:
                mark_failed(status_path(task), task.name, f"Launcher error: {error!r}")
                event(f"GPU {gpu} ERROR {task.name}: {error!r}")

        def run_wave(
            seed: int, jobs: list, retry: bool, executor: ThreadPoolExecutor
        ) -> None:
            waiting = collections.deque(jobs)
            active = {}
            while (
                waiting or active or any(item[2] == seed for item in external.values())
            ):
                refresh_external()
                for future in list(active):
                    if future.done():
                        future.result()
                        del active[future]
                occupied = collections.Counter(item[1] for item in external.values())
                occupied.update(gpu for _, gpu in active.values())
                while waiting:
                    gpu = least_loaded_gpu(gpus, occupied, args.max_runs_per_gpu)
                    if gpu is None:
                        break
                    task = waiting.popleft()
                    if completed(task):
                        continue
                    future = executor.submit(run_one, task, gpu, retry)
                    active[future] = (task, gpu)
                    occupied[gpu] += 1
                if (
                    waiting
                    or active
                    or any(item[2] == seed for item in external.values())
                ):
                    if active:
                        wait(active, timeout=2, return_when=FIRST_COMPLETED)
                    else:
                        time.sleep(2)

        with ThreadPoolExecutor(
            max_workers=len(gpus) * args.max_runs_per_gpu
        ) as executor:
            for seed in (1, 2, 3, 4):
                cohort = [task for task in tasks if task.seed == seed]
                primary = [
                    task
                    for task in cohort
                    if not completed(task)
                    and task.name not in external
                    and (
                        args.dry_run
                        or read_status(status_path(task)).get("status") != "failed"
                    )
                ]
                event(f"SEED {seed} primary={len(primary)}")
                run_wave(seed, primary, False, executor)
                if args.dry_run:
                    continue
                deferred = [
                    task
                    for task in cohort
                    if not completed(task) and task.name not in external
                ]
                event(f"SEED {seed} deferred={len(deferred)}")
                run_wave(seed, deferred, True, executor)
                event(
                    f"SEED {seed} finished completed={sum(completed(t) for t in cohort)}/12"
                )

        if args.dry_run:
            event("dry-run finished")
            return
        done = sum(completed(task) for task in tasks)
        event(f"matrix finished; completed={done}/48")
        if done != 48:
            raise RuntimeError(f"{48-done} runs incomplete; inspect {root / 'logs'}")
    finally:
        launcher_lock.close()
        legacy_lock.close()


if __name__ == "__main__":
    main()
