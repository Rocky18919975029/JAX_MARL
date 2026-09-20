#!/usr/bin/env python3
"""Select an oracle coefficient from a reward-free, LR=0 pilot run."""

import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-gradient-ratio", type=float, default=0.1)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.metrics.read_text().splitlines() if line.strip()]
    metric_key = "oracle_actor_to_rl_grad_ratio_mean"
    ratios = [float(row[metric_key]) for row in rows if metric_key in row]
    ratios = [value for value in ratios if math.isfinite(value) and value > 0]
    if not ratios:
        raise RuntimeError("Pilot metrics contain no positive finite gradient ratio")
    unit_ratio = sum(ratios) / len(ratios)
    coefficient = args.target_gradient_ratio / unit_ratio
    payload = {
        "schema_version": 1,
        "selection_signal": "actor_oracle_to_ppo_gradient_norm_ratio",
        "uses_return": False,
        "pilot_oracle_coefficient": 1.0,
        "pilot_ratio_metric": metric_key,
        "pilot_ratio_mean": unit_ratio,
        "target_gradient_ratio": args.target_gradient_ratio,
        "oracle_distortion_coef": coefficient,
        "metrics": str(args.metrics.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
