#!/usr/bin/env python3
"""Monitor all 12 focused 3s5z_vs_3s6z NPS 20M-step runs."""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

try:
    from run_smax_3s5z_nps_20m import TOTAL_TIMESTEPS, task_matrix
except ModuleNotFoundError:
    from scripts.run_smax_3s5z_nps_20m import TOTAL_TIMESTEPS, task_matrix


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


def final_exists(root, run_name):
    return any(
        (root / "checkpoints").glob(f"**/{run_name}-*/final/model.safetensors")
    ) or any((root / "checkpoints").glob(f"**/{run_name}/final/model.safetensors"))


def snapshot(root):
    processes = process_listing()
    rows = []
    counts = {"DONE": 0, "RUNNING": 0, "FAILED": 0, "PENDING": 0}
    for task in task_matrix():
        log_path = root / "logs" / f"{task.run_name}.log"
        text = log_path.read_text(errors="replace") if log_path.is_file() else ""
        steps = parse_latest_step(text)
        checkpoint_steps = [
            int(match.group(1))
            for path in (root / "checkpoints").glob(f"**/{task.run_name}-*/step_*")
            if (match := re.fullmatch(r"step_([0-9]+)", path.name))
        ]
        steps = max([steps, *checkpoint_steps])
        if final_exists(root, task.run_name):
            status = "DONE"
            steps = TOTAL_TIMESTEPS
        elif task.run_name in processes:
            status = "RUNNING"
        elif "Traceback" in text or "Error executing job" in text:
            status = "FAILED"
        else:
            status = "PENDING"
        counts[status] += 1
        percent = min(100.0, 100.0 * steps / TOTAL_TIMESTEPS)
        filled = round(28 * percent / 100.0)
        bar = "█" * filled + "░" * (28 - filled)
        rows.append((status, bar, percent, steps, task.run_name))
    return counts, rows


def render(root, clear=False):
    if clear:
        print("\033[2J\033[H", end="")
    counts, rows = snapshot(root)
    print(
        f"DONE={counts['DONE']}  RUNNING={counts['RUNNING']}  "
        f"FAILED={counts['FAILED']}  PENDING={counts['PENDING']}  TOTAL=12\n"
    )
    for status, bar, percent, steps, run_name in rows:
        print(
            f"{status:7s} [{bar}] {percent:6.2f}%  "
            f"{steps:>10,}/{TOTAL_TIMESTEPS:,}  {run_name}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    root = args.run_root.expanduser().resolve()
    if args.once:
        render(root)
        return
    while True:
        render(root, clear=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
