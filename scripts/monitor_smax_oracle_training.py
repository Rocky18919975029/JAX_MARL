#!/usr/bin/env python3
"""Print progress for every run in an SMAX oracle comparison root."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.expanduser().resolve()
    manifest = json.loads((root / "experiment_manifest.json").read_text())
    total = int(manifest["total_timesteps"])
    rows = []
    for seed in manifest["seeds"]:
        for condition in manifest["conditions"]:
            name = (
                f"SMAX-ORACLE-{manifest['map_name']}-nps-{condition}-seed{seed}"
            )
            status_path = root / "status" / f"{name}.json"
            status = "pending"
            if status_path.exists():
                status = json.loads(status_path.read_text()).get("status", "unknown")
            step = 0
            metrics = root / "metrics" / f"{name}.jsonl"
            if metrics.exists():
                for line in metrics.read_text().splitlines():
                    if line.strip():
                        step = int(json.loads(line).get("env_step", step))
            rows.append((status, step, name))
    counts = {key: sum(status == key for status, _, _ in rows) for key in ("completed", "running", "failed", "pending")}
    print(" ".join(f"{key.upper()}={value}" for key, value in counts.items()) + f" TOTAL={len(rows)}")
    for status, step, name in rows:
        fraction = min(1.0, step / max(total, 1))
        width = 28
        bar = "█" * round(width * fraction) + "░" * (width - round(width * fraction))
        print(f"{status.upper():9s} [{bar}] {100*fraction:6.2f}% {step:>11,}/{total:,}  {name}")
    launcher = root / "launcher.log"
    if launcher.exists():
        print("\nLatest launcher events:")
        print("\n".join(launcher.read_text().splitlines()[-12:]))


if __name__ == "__main__":
    main()
