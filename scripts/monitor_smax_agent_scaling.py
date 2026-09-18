#!/usr/bin/env python3
"""Monitor every run in the NPS SMAX agent-count scaling matrix."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

try:
    from run_smax_agent_scaling import (
        AGENT_COUNTS,
        FAMILY_MAPS,
        parse_counts,
        parse_csv,
        task_matrix,
    )
    from run_h1_smax_confirmatory import parse_seeds
except ModuleNotFoundError:
    from scripts.run_smax_agent_scaling import (
        AGENT_COUNTS,
        FAMILY_MAPS,
        parse_counts,
        parse_csv,
        task_matrix,
    )
    from scripts.run_h1_smax_confirmatory import parse_seeds


STEP_PATTERN = re.compile(
    r"steps=([0-9][0-9,]*(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
)


def parse_latest_step(text):
    matches = STEP_PATTERN.findall(text.replace("\r", "\n"))
    return int(float(matches[-1].replace(",", ""))) if matches else 0


def process_listing():
    return subprocess.run(
        ("pgrep", "-af", "baselines/MAPPO/mappo_rnn_smax.py"),
        text=True,
        capture_output=True,
    ).stdout


def reused_run_names(root, tasks):
    path = root / "reused_runs.json"
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    reused_keys = {
        (parts[0], parts[1], parts[2], int(parts[3]))
        for serialized in payload.get("runs", {})
        if len(parts := serialized.split("|")) == 4
    }
    return {task.run_name for task in tasks if task.key in reused_keys}


def final_exists(root, run_name):
    return any(
        (root / "checkpoints").glob(f"**/{run_name}-*/final/model.safetensors")
    ) or any((root / "checkpoints").glob(f"**/{run_name}/final/model.safetensors"))


def snapshot(root, tasks):
    processes = process_listing()
    reused = reused_run_names(root, tasks)
    rows = []
    counts = {key: 0 for key in ("DONE", "REUSED", "RUNNING", "FAILED", "PENDING")}
    for task in tasks:
        total_timesteps = task.total_timesteps
        log_path = root / "logs" / f"{task.run_name}.log"
        text = log_path.read_text(errors="replace") if log_path.is_file() else ""
        steps = parse_latest_step(text)
        checkpoint_steps = [
            int(match.group(1))
            for path in (root / "checkpoints").glob(f"**/{task.run_name}-*/step_*")
            if (match := re.fullmatch(r"step_([0-9]+)", path.name))
        ]
        steps = max([steps, *checkpoint_steps])
        marker = root / "status" / f"{task.run_name}.json"
        marker_status = ""
        if marker.is_file():
            marker_status = json.loads(marker.read_text(encoding="utf-8")).get(
                "status", ""
            )
        if task.run_name in reused:
            status = "REUSED"
            steps = total_timesteps
        elif final_exists(root, task.run_name) or marker_status == "completed":
            status = "DONE"
            steps = total_timesteps
        elif task.run_name in processes:
            status = "RUNNING"
        elif (
            marker_status == "failed"
            or "Traceback" in text
            or "Error executing job" in text
        ):
            status = "FAILED"
        else:
            status = "PENDING"
        counts[status] += 1
        percent = min(100.0, 100.0 * steps / total_timesteps)
        filled = round(24 * percent / 100.0)
        bar = "█" * filled + "░" * (24 - filled)
        rows.append(
            (
                task.agent_count,
                task.family,
                status,
                bar,
                percent,
                steps,
                total_timesteps,
                task.run_name,
            )
        )
    return counts, rows


def render(root, tasks, clear=False):
    if clear:
        print("\033[2J\033[H", end="")
    counts, rows = snapshot(root, tasks)
    summary = "  ".join(f"{key}={counts[key]}" for key in counts)
    print(f"{summary}  TOTAL={len(tasks)}\n")
    prior_count = None
    for (
        agent_count,
        family,
        status,
        bar,
        percent,
        steps,
        total_timesteps,
        run_name,
    ) in rows:
        if prior_count is not None and agent_count != prior_count:
            print()
        prior_count = agent_count
        print(
            f"n={agent_count:2d} {family[:3]:3s} {status:7s} [{bar}] "
            f"{percent:6.2f}% {steps:>10,}/{total_timesteps:,}  {run_name}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--families",
        type=lambda value: parse_csv(value, FAMILY_MAPS),
        default=tuple(FAMILY_MAPS),
    )
    parser.add_argument("--agent-counts", type=parse_counts, default=AGENT_COUNTS)
    parser.add_argument("--seeds", type=parse_seeds, default=(1, 2, 3, 4))
    parser.add_argument("--cka-coef", type=float, default=0.3515769798)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    root = args.run_root.expanduser().resolve()
    tasks = task_matrix(args.families, args.agent_counts, args.seeds, args.cka_coef)
    if args.once:
        render(root, tasks)
        return
    while True:
        render(root, tasks, clear=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
