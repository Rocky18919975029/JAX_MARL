#!/usr/bin/env python3
"""Merge per-checkpoint H1 diagnostic CSV files without changing raw data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FILES = (
    "compatibility_metrics.csv",
    "decision_metrics.csv",
    "bellman_metrics.csv",
)


def merge(root, filename, output):
    sources = sorted((root / "diagnostics_raw").glob(f"H1-*/*/{filename}"))
    rows = []
    fieldnames = None
    for source in sources:
        with source.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise RuntimeError(f"Schema mismatch in {source}")
            rows.extend(reader)
    if not rows:
        print(f"SKIP {filename}: no inputs")
        return
    destination = output / filename
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{destination}: {len(rows)} rows from {len(sources)} files")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.expanduser().resolve()
    output = root / "diagnostics_summary"
    output.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        merge(root, filename, output)


if __name__ == "__main__":
    main()
