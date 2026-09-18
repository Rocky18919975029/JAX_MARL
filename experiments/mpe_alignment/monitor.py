#!/usr/bin/env python3
"""Print restart-safe progress for the MPE alignment matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def bar(progress: float, width: int = 28) -> str:
    filled = min(width, max(0, round(width * progress)))
    return "█" * filled + "░" * (width - filled)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--expected-runs", type=int, default=12)
    args = parser.parse_args()
    if args.expected_runs < 1 or args.total_timesteps < 1:
        raise ValueError("expected-runs and total-timesteps must be positive")
    root = args.run_root.expanduser().resolve()
    rows = []
    for path in sorted((root / "status").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        steps = int(payload.get("env_steps", 0))
        state = str(payload.get("status", "unknown")).upper()
        rows.append((payload.get("run_name", path.stem), state, steps))
    counts = {
        state: sum(row[1] == state for row in rows)
        for state in ("COMPLETED", "RUNNING", "FAILED", "INITIALIZING")
    }
    pending = max(0, args.expected_runs - len(rows))
    print(
        f"DONE={counts['COMPLETED']} RUNNING={counts['RUNNING']} "
        f"INITIALIZING={counts['INITIALIZING']} FAILED={counts['FAILED']} "
        f"PENDING={pending} TOTAL={len(rows)} / {args.expected_runs}\n"
    )
    for name, state, steps in rows:
        progress = min(1.0, steps / args.total_timesteps)
        print(
            f"{state:12s} [{bar(progress)}] {progress:7.2%} "
            f"{steps:>10,}/{args.total_timesteps:,}  {name}"
        )
    launcher = root / "launcher.log"
    if launcher.is_file():
        print("\nLatest launcher events:")
        print("\n".join(launcher.read_text(encoding="utf-8").splitlines()[-12:]))


if __name__ == "__main__":
    main()
