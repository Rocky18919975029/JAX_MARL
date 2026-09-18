"""Dependency-free protocol definitions for the first VMAS experiment phase."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path


PROTOCOL_VERSION = "benchmarl-vmas-nps-alignment-v1.0"
CALIBRATION_PROTOCOL_VERSION = "benchmarl-vmas-nps-cka-calibration-v1.0"
TASKS = ("discovery_5", "passage_5", "football_5v5_heuristic")
CONDITIONS = ("none", "c_to_a_mse", "c_to_a_cka")
DEFAULT_SEEDS = (1, 2, 3, 4)
REFERENCE_ALIGNMENT_COEF = 0.1


def parse_csv(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("expected a non-empty list without duplicates")
    return values


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds: list[int] = []
    for piece in value.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            start, end = map(int, piece.split("-", 1))
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(piece))
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be unique non-negative integers")
    return tuple(seeds)


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
            raise ValueError(f"unsupported task: {self.task}")
        if self.condition not in CONDITIONS:
            raise ValueError(f"unsupported condition: {self.condition}")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    @property
    def align_mode(self) -> str:
        return "none" if self.condition == "none" else "c_to_a"

    @property
    def align_distance(self) -> str:
        return "linear_cka" if self.condition == "c_to_a_cka" else "ln_mse"

    @property
    def name(self) -> str:
        return (
            f"VMAS-{self.task}-nps-{self.condition}-"
            f"lam{float_label(self.coefficient)}-seed{self.seed}"
        )


def matrix(
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    tasks: tuple[str, ...] = TASKS,
    cka_coefficient: float | None = None,
) -> list[Run]:
    if (
        cka_coefficient is None
        or not math.isfinite(cka_coefficient)
        or cka_coefficient <= 0
    ):
        raise ValueError("a finite positive CKA coefficient is required")
    runs = []
    for task in tasks:
        if task not in TASKS:
            raise ValueError(f"unsupported task: {task}")
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
        "selection_uses_return": False,
        "performance_fields_persisted": False,
        "reference_distance": "ln_mse",
        "target_distance": "linear_cka",
        "tasks": list(TASKS),
        "actor_parameterization": "nps",
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"incompatible CKA calibration artifact: {mismatches}")
    coefficient = float(payload["global_alignment_coef"])
    if not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError("calibrated CKA coefficient must be finite and positive")
    return coefficient
