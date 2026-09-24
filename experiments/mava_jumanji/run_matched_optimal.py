"""Run paired Mava recurrent MAPPO/ARec experiments with Sable optimal settings.

This controller uses only the Python standard library. Each worker runs in the
pinned Mava checkout and sees exactly one GPU via CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "optimal_rec_mappo.json"
AREC_SCRIPT = HERE / "rec_mappo_arec.py"
AREC_CONFIG = HERE / "rec_mappo_arec.yaml"
CONDITIONS = ("none", "arec")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_seeds(value: str) -> tuple[int, ...]:
    if re.fullmatch(r"\d+-\d+", value):
        first, last = map(int, value.split("-"))
        seeds = tuple(range(first, last + 1))
    else:
        seeds = tuple(int(part) for part in value.split(","))
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique non-negative integers")
    return seeds


def parse_names(value: str, allowed: set[str], description: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in value.split(","))
    if not names or len(set(names)) != len(names) or any(name not in allowed for name in names):
        raise ValueError(f"Invalid {description}: {value}; allowed: {sorted(allowed)}")
    return names


def hydra_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def task_overrides(config: dict, task: str, seed: int, smoke: bool) -> dict[str, object]:
    task_config = config["tasks"][task]
    overrides = {
        "env": task_config["env"],
        "env/scenario": task_config["scenario"],
        "system.seed": seed,
        **config["common"],
        **task_config["optimal"],
    }
    if smoke:
        # This is a low-cost wiring check, not a benchmark run. Preserve the
        # task/architecture and recurrent chunk size; the formal mode below
        # always uses the exact published optimisation and budget settings.
        overrides.update({
            "arch.num_envs": 2,
            "system.update_batch_size": 1,
            "system.num_updates": 2,
            "system.ppo_epochs": 1,
            "system.num_minibatches": 1,
            "arch.num_evaluation": 1,
            "arch.num_eval_episodes": 2,
            "arch.absolute_metric": False,
        })
    return overrides


def make_jobs(config: dict, tasks: tuple[str, ...], seeds: tuple[int, ...], smoke: bool) -> list[dict]:
    jobs = []
    for seed in seeds:
        for task in tasks:
            shared = task_overrides(config, task, seed, smoke)
            for condition in CONDITIONS:
                name = f"{task}--{condition}--seed{seed}"
                jobs.append({
                    "name": name,
                    "task": task,
                    "seed": seed,
                    "condition": condition,
                    "shared_overrides": shared,
                })
    return jobs


def validate_mava(mava_root: Path, config: dict) -> None:
    if not (mava_root / "mava/systems/ppo/anakin/rec_mappo.py").is_file():
        raise RuntimeError(f"Mava recurrent MAPPO not found under {mava_root}")
    actual_commit = subprocess.check_output(
        ["git", "-C", str(mava_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != config["mava_commit"]:
        raise RuntimeError(f"Mava commit {actual_commit} != required {config['mava_commit']}")
    installed = (
        (AREC_SCRIPT, mava_root / "mava/systems/ppo/anakin/rec_mappo_arec.py"),
        (AREC_CONFIG, mava_root / "mava/configs/default/rec_mappo_arec.yaml"),
    )
    for source, target in installed:
        if not target.is_file() or digest(source) != digest(target):
            raise RuntimeError(f"Install current {source.name} into {target} before launching")
    # Tracked upstream changes invalidate the pin. Untracked outputs from
    # previous Mava experiments do not alter either learner implementation.
    changed = subprocess.check_output(
        ["git", "-C", str(mava_root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).splitlines()
    if changed:
        raise RuntimeError("Mava checkout has tracked changes: " + "; ".join(changed[:8]))
    if not (mava_root / ".venv/bin/python").is_file():
        raise RuntimeError(f"Run uv sync --extra cuda12 in {mava_root} first")


def manifest(config: dict, args: argparse.Namespace, tasks: tuple[str, ...], seeds: tuple[int, ...]) -> dict:
    jobs = make_jobs(config, tasks, seeds, args.smoke)
    return {
        "protocol": "mava-jumanji-rec-mappo-arec-paired-optimal-v1",
        "benchmark_config_sha256": digest(CONFIG_PATH),
        "baseline_script_sha256": digest(
            args.mava_root / "mava/systems/ppo/anakin/rec_mappo.py"
        ),
        "arec_script_sha256": digest(AREC_SCRIPT),
        "arec_config_sha256": digest(AREC_CONFIG),
        "mava_commit": config["mava_commit"],
        "mava_root": str(args.mava_root.resolve()),
        "tasks": list(tasks),
        "seeds": list(seeds),
        "smoke": args.smoke,
        "arec": {
            "coef": args.arec_coef,
            "q_steps": args.arec_q_steps,
            "q_lr": args.arec_q_lr,
            "fisher_ridge": args.arec_fisher_ridge,
        },
        "jobs": jobs,
    }


def command(mava_root: Path, run_root: Path, job: dict, arec: dict) -> list[str]:
    run_dir = run_root / "runs" / job["name"]
    entrypoint = "rec_mappo.py" if job["condition"] == "none" else "rec_mappo_arec.py"
    overrides = dict(job["shared_overrides"])
    overrides.update({
        "logger.base_exp_path": str(run_dir),
        "logger.loggers.json.enabled": True,
        "hydra.run.dir": str(run_dir / "hydra"),
    })
    if job["condition"] == "arec":
        overrides.update({
            "arec.coef": arec["coef"],
            "arec.q_steps": arec["q_steps"],
            "arec.q_lr": arec["q_lr"],
            "arec.fisher_ridge": arec["fisher_ridge"],
        })
    return [
        str(mava_root / ".venv/bin/python"),
        "-u",
        str(mava_root / "mava/systems/ppo/anakin" / entrypoint),
        *(f"{key}={hydra_value(value)}" for key, value in overrides.items()),
    ]


def status_path(run_root: Path, job: dict) -> Path:
    return run_root / "status" / f"{job['name']}.json"


def status_of(run_root: Path, job: dict) -> dict:
    path = status_path(run_root, job)
    return read_json(path) if path.exists() else {"status": "pending"}


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def logged_steps(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    with log_path.open("rb") as stream:
        stream.seek(max(0, log_path.stat().st_size - 262144))
        tail = stream.read().decode("utf8", errors="replace")
    matches = re.findall(r"Timestep:\s*([\d,]+)", tail)
    return int(matches[-1].replace(",", "")) if matches else 0


def display_status(run_root: Path) -> None:
    path = run_root / "experiment_manifest.json"
    if not path.exists():
        print(f"No experiment manifest at {path}")
        return
    saved = read_json(path)
    states = [status_of(run_root, job)["status"] for job in saved["jobs"]]
    counts = Counter(states)
    print(
        " ".join(f"{key.upper()}={counts[key]}" for key in ("completed", "running", "failed", "pending"))
        + f" TOTAL={len(states)}"
    )
    for job, state in zip(saved["jobs"], states, strict=True):
        target = job["shared_overrides"]["system.num_updates"]
        target *= job["shared_overrides"]["arch.num_envs"]
        target *= job["shared_overrides"]["system.update_batch_size"]
        target *= job["shared_overrides"]["system.rollout_length"]
        if state == "completed":
            steps = target
        else:
            steps = logged_steps(run_root / "logs" / f"{job['name']}.log")
        fraction = min(1.0, steps / target)
        bar = "█" * int(fraction * 24) + "░" * (24 - int(fraction * 24))
        print(f"{state.upper():9} [{bar}] {fraction:6.1%} {steps:>10,}/{target:,}  {job['name']}")


def run(args: argparse.Namespace) -> int:
    config = read_json(CONFIG_PATH)
    tasks = parse_names(args.tasks, set(config["tasks"]), "tasks")
    seeds = parse_seeds(args.seeds)
    gpus = parse_names(args.gpus, set(args.gpus.split(",")), "GPUs")
    if any(not re.fullmatch(r"\d+", gpu) for gpu in gpus):
        raise ValueError("GPUs must be comma-separated non-negative device IDs")
    if args.max_runs_per_gpu < 1 or args.arec_coef <= 0 or args.arec_q_steps < 1:
        raise ValueError("GPU capacity, ARec coefficient and q_steps must be positive")
    if args.arec_q_lr <= 0 or args.arec_fisher_ridge <= 0:
        raise ValueError("ARec q_lr and Fisher ridge must be positive")
    validate_mava(args.mava_root, config)
    planned = manifest(config, args, tasks, seeds)
    if args.dry_run:
        for job in planned["jobs"]:
            print(job["name"], " ".join(command(args.mava_root, args.run_root, job, planned["arec"])))
        return 0

    args.run_root.mkdir(parents=True, exist_ok=True)
    with (args.run_root / ".launcher.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another launcher is already using this run root") from error
        manifest_path = args.run_root / "experiment_manifest.json"
        if manifest_path.exists():
            if read_json(manifest_path) != planned:
                raise RuntimeError("Existing manifest differs; use a new run root")
        else:
            write_json(manifest_path, planned)
        pending = deque()
        for job in planned["jobs"]:
            old = status_of(args.run_root, job)
            if old["status"] == "completed":
                continue
            if old["status"] == "running" and pid_alive(old.get("pid")):
                raise RuntimeError(f"Worker still running for {job['name']} pid={old['pid']}")
            if old["status"] == "failed" and not args.retry_failed:
                continue
            pending.append(job)
        print(f"selected={len(planned['jobs'])} pending={len(pending)} skipped={len(planned['jobs'])-len(pending)}", flush=True)
        running: dict[str, tuple[subprocess.Popen, str, dict]] = {}
        gpu_load = Counter()
        while pending or running:
            while pending:
                available = [gpu for gpu in gpus if gpu_load[gpu] < args.max_runs_per_gpu]
                if not available:
                    break
                gpu = min(available, key=lambda item: (gpu_load[item], gpus.index(item)))
                job = pending.popleft()
                cmd = command(args.mava_root, args.run_root, job, planned["arec"])
                log_path = args.run_root / "logs" / f"{job['name']}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                (args.run_root / "runs" / job["name"]).mkdir(parents=True, exist_ok=True)
                environment = os.environ.copy()
                environment.pop("LD_LIBRARY_PATH", None)
                environment["CUDA_VISIBLE_DEVICES"] = gpu
                with log_path.open("a") as log:
                    log.write(f"\n===== START gpu={gpu} command={cmd!r} =====\n")
                    log.flush()
                    process = subprocess.Popen(cmd, cwd=args.mava_root, env=environment, stdout=log, stderr=subprocess.STDOUT)
                write_json(status_path(args.run_root, job), {
                    "status": "running", "pid": process.pid, "gpu": gpu,
                    "run_name": job["name"], "started_at": time.time(), "command": cmd,
                })
                running[job["name"]] = (process, gpu, job)
                gpu_load[gpu] += 1
                print(f"GPU {gpu} START {job['name']} pid={process.pid}", flush=True)
            if not running:
                continue
            time.sleep(2)
            for name, (process, gpu, job) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                current = status_of(args.run_root, job)
                write_json(status_path(args.run_root, job), {
                    **current, "status": "completed" if code == 0 else "failed",
                    "exit_code": code, "finished_at": time.time(),
                })
                gpu_load[gpu] -= 1
                del running[name]
                print(f"GPU {gpu} END   {name} status={code}", flush=True)
        final = Counter(status_of(args.run_root, job)["status"] for job in planned["jobs"])
        print(f"finished completed={final['completed']} failed={final['failed']}", flush=True)
        return 1 if final["failed"] else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "status"))
    parser.add_argument("--mava-root", type=Path, default=Path("/home/data/zeshenghong/Mava"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--tasks", default="lbf_15x15-4p-5f,rware_large-8ag")
    parser.add_argument("--seeds", default="1-4")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max-runs-per-gpu", type=int, default=1)
    parser.add_argument("--arec-coef", type=float, default=1e-4)
    parser.add_argument("--arec-q-steps", type=int, default=4)
    parser.add_argument("--arec-q-lr", type=float, default=1e-3)
    parser.add_argument("--arec-fisher-ridge", type=float, default=1e-3)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.run_root = args.run_root.expanduser().resolve()
    args.mava_root = args.mava_root.expanduser().resolve()
    if args.mode == "status":
        display_status(args.run_root)
    else:
        raise SystemExit(run(args))


if __name__ == "__main__":
    main()
