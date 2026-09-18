#!/usr/bin/env python3
"""Print one compact progress snapshot for the VMAS phase-one matrix."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=36)
    args = parser.parse_args()
    root = args.run_root.expanduser().resolve()

    rows = []
    for path in sorted((root / "status").glob("*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rows.append(item)

    counts = Counter(item.get("status", "unknown") for item in rows)
    pending = max(args.expected_runs - len(rows), 0)
    print(
        f"DONE={counts['completed']} RUNNING={counts['running']} "
        f"FAILED={counts['failed']} PENDING={pending} "
        f"TOTAL={len(rows)} / {args.expected_runs}\n"
    )
    order = {"running": 0, "failed": 1, "initializing": 2, "completed": 3}
    for item in sorted(
        rows,
        key=lambda row: (
            order.get(row.get("status", "unknown"), 4),
            row.get("task", ""),
            row.get("seed", -1),
            row.get("condition", ""),
        ),
    ):
        maximum = int(item.get("max_frames", 10_000_000))
        steps = int(item.get("env_steps", 0))
        percent = 100.0 * min(steps, maximum) / maximum if maximum else 0.0
        print(
            f"{item.get('status', 'unknown').upper():12s} "
            f"{percent:6.2f}%  {steps:>10,}/{maximum:<10,}  "
            f"{item.get('run_name', '<unknown>')}"
        )

    launcher = root / "launcher.log"
    if launcher.is_file():
        print("\nLatest launcher events:")
        lines = launcher.read_text(encoding="utf-8", errors="replace").splitlines()
        print("\n".join(lines[-12:]))


if __name__ == "__main__":
    main()
