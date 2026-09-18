"""Frozen matrix and calibration schema for the first MPE experiment."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path


PROTOCOL_VERSION = "mpe-simple-spread5-nps-alignment-v1.0"
CALIBRATION_PROTOCOL_VERSION = "mpe-simple-spread5-nps-cka-calibration-v1.0"
TASKS = ("simple_spread_5",)
CONDITIONS = ("none", "c_to_a_mse", "c_to_a_cka")
DEFAULT_SEEDS = (1, 2, 3, 4)
REFERENCE_ALIGNMENT_COEF = 0.1
CALIBRATION_PILOT_SEED = 9001


def parse_csv(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("expected a non-empty list without duplicates")
    return result


def parse_seeds(value: str) -> tuple[int, ...]:
    result: list[int] = []
    for piece in value.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            start, end = map(int, piece.split("-", 1))
            result.extend(range(start, end + 1))
        else:
            result.append(int(piece))
    if not result or len(result) != len(set(result)) or min(result) < 0:
        raise argparse.ArgumentTypeError("seeds must be unique non-negative integers")
    return tuple(result)


def float_label(value: float) -> str:
    return f"{value:.10g}".replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class Run:
    task: str
    condition: str
    seed: int
    coefficient: float

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"unsupported MPE task: {self.task}")
        if self.condition not in CONDITIONS:
            raise ValueError(f"unsupported condition: {self.condition}")
        if self.seed < 0 or not math.isfinite(self.coefficient):
            raise ValueError("invalid seed or coefficient")

    @property
    def align_mode(self) -> str:
        return "none" if self.condition == "none" else "c_to_a"

    @property
    def align_distance(self) -> str:
        return "linear_cka" if self.condition == "c_to_a_cka" else "ln_mse"

    @property
    def name(self) -> str:
        return (
            f"MPE-{self.task}-nps-{self.condition}-"
            f"lam{float_label(self.coefficient)}-seed{self.seed}"
        )


def matrix(
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    tasks: tuple[str, ...] = TASKS,
    cka_coefficient: float | None = None,
) -> list[Run]:
    if cka_coefficient is None or not math.isfinite(cka_coefficient):
        raise ValueError("a finite CKA coefficient is required")
    if cka_coefficient <= 0:
        raise ValueError("CKA coefficient must be positive")
    runs: list[Run] = []
    for task in tasks:
        for seed in seeds:
            runs.extend(
                (
                    Run(task, "none", seed, 0.0),
                    Run(task, "c_to_a_mse", seed, REFERENCE_ALIGNMENT_COEF),
                    Run(task, "c_to_a_cka", seed, cka_coefficient),
                )
            )
    return runs


def load_cka_coefficient(path: Path) -> float:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    expected = {
        "protocol_version": CALIBRATION_PROTOCOL_VERSION,
        "task": TASKS[0],
        "pilot_seed": CALIBRATION_PILOT_SEED,
        "actor_parameterization": "nps",
        "selection_uses_return": False,
        "reference_distance": "ln_mse",
        "target_distance": "linear_cka",
    }
    mismatch = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatch:
        raise ValueError(f"incompatible CKA calibration artifact: {mismatch}")
    coefficient = float(payload["alignment_coefficient"])
    if not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("invalid calibrated CKA coefficient")
    return coefficient
