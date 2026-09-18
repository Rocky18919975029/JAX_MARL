#!/usr/bin/env python3
"""Display progress bars for the 12-run NPS Humanoid core matrix."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


EXPECTED_RUNS = 12
RUN_PATTERN = re.compile(
    r"^HARL-Humanoid-v2-17x1-nps-(none|c_to_a_mse|c_to_a_cka)-" r"lam[^-]+-seed([1-4])$"
)


def progress_bar(progress: float, width: int = 30) -> str:
    filled = min(width, max(0, round(width * progress)))
    return "█" * filled + "░" * (width - filled)


def checkpoint_progress(root: Path, run_name: str) -> int:
    run_root = root / "checkpoints" / run_name
    completed = run_root / "completed.json"
    if completed.is_file():
        try:
            return int(json.loads(completed.read_text())["environment_steps"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
    maximum = 0
    for metadata in run_root.glob("*/metadata.json"):
        try:
            maximum = max(
                maximum,
                int(json.loads(metadata.read_text())["environment_steps"]),
            )
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return maximum


def load_rows(root: Path, default_total: int) -> list[dict]:
    rows = []
    for status_path in sorted((root / "status").glob("*.json")):
        if RUN_PATTERN.fullmatch(status_path.stem) is None:
            continue
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        run_name = str(payload.get("run_name", status_path.stem))
        state = str(payload.get("status", "unknown")).upper()
        total = int(payload.get("total_env_steps", default_total))
        steps = max(
            int(payload.get("env_steps", 0)), checkpoint_progress(root, run_name)
        )
        if state == "COMPLETED":
            steps = total
        rows.append(
            {
                "run_name": run_name,
                "state": state,
                "steps": min(steps, total),
                "total": total,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--total-env-steps", type=int, default=10_000_000)
    args = parser.parse_args()
    if args.total_env_steps < 1:
        raise ValueError("total-env-steps must be positive")
    root = args.run_root.expanduser().resolve()
    rows = load_rows(root, args.total_env_steps)
    counts = {
        state: sum(row["state"] == state for row in rows)
        for state in ("COMPLETED", "RUNNING", "FAILED")
    }
    pending = max(0, EXPECTED_RUNS - len(rows))
    print(
        f"DONE={counts['COMPLETED']} RUNNING={counts['RUNNING']} "
        f"FAILED={counts['FAILED']} PENDING={pending} "
        f"TOTAL={len(rows)} / {EXPECTED_RUNS}\n"
    )
    for row in rows:
        progress = row["steps"] / row["total"]
        print(
            f"{row['state']:9s} [{progress_bar(progress)}] {progress:7.2%} "
            f"{row['steps']:>10,}/{row['total']:,}  {row['run_name']}"
        )
    launcher_log = root / "launcher.log"
    if launcher_log.is_file():
        print("\nLatest launcher events:")
        print("\n".join(launcher_log.read_text(encoding="utf-8").splitlines()[-12:]))


if __name__ == "__main__":
    main()
