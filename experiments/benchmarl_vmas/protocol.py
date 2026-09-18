"""Dependency-free protocol definitions for the first VMAS experiment phase."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path


PROTOCOL_VERSION = "benchmarl-vmas-nps-alignment-v2.0"
CALIBRATION_PROTOCOL_VERSION = "benchmarl-vmas-nps-gradient-calibration-v2.0"
# These names map directly to BenchMARL's official VMAS task YAMLs.  Agent
# counts and every other environment option are owned by those YAMLs and must
# not be overridden by this protocol.
TASKS = ("discovery", "passage", "football")
CONDITIONS = ("none", "c_to_a_mse", "c_to_a_cka")
DEFAULT_SEEDS = (1, 2, 3, 4)
REFERENCE_MSE_ALIGNMENT_COEF = 0.1
CALIBRATION_PILOT_SEED = 9001
CALIBRATION_MINIBATCHES = 8


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
    alignment_coefficients: dict[str, dict[str, float]] | None = None,
) -> list[Run]:
    if alignment_coefficients is None:
        raise ValueError("task-specific alignment coefficients are required")
    runs = []
    for task in tasks:
        if task not in TASKS:
            raise ValueError(f"unsupported task: {task}")
        coefficients = alignment_coefficients.get(task)
        if not isinstance(coefficients, dict):
            raise ValueError(
                f"alignment coefficients are required for {task}"
            )
        expected = {"c_to_a_mse", "c_to_a_cka"}
        if set(coefficients) != expected or any(
            not math.isfinite(value) or value <= 0
            for value in coefficients.values()
        ):
            raise ValueError(f"invalid alignment coefficients for {task}")
        if not math.isclose(
            coefficients["c_to_a_mse"],
            REFERENCE_MSE_ALIGNMENT_COEF,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"LN-MSE coefficient for {task} must remain "
                f"{REFERENCE_MSE_ALIGNMENT_COEF}"
            )
        for seed in seeds:
            runs.extend(
                (
                    Run(task, "none", seed, 0.0),
                    Run(task, "c_to_a_mse", seed, coefficients["c_to_a_mse"]),
                    Run(task, "c_to_a_cka", seed, coefficients["c_to_a_cka"]),
                )
            )
    return runs


def load_alignment_coefficients(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    expected = {
        "protocol_version": CALIBRATION_PROTOCOL_VERSION,
        "selection_uses_return": False,
        "performance_fields_persisted": False,
        "reference_distance": "ln_mse",
        "reference_alignment_coef": REFERENCE_MSE_ALIGNMENT_COEF,
        "target_distance": "linear_cka",
        "tasks": list(TASKS),
        "actor_parameterization": "nps",
        "minibatch_sampling": "training_replay_buffer_random",
        "pilot_seed": CALIBRATION_PILOT_SEED,
        "calibration_minibatches": CALIBRATION_MINIBATCHES,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"incompatible alignment calibration artifact: {mismatches}")
    raw = payload.get("task_alignment_coefs")
    if not isinstance(raw, dict) or set(raw) != set(TASKS):
        raise ValueError("calibration artifact must contain every task")
    if any(not isinstance(raw[task], dict) for task in TASKS):
        raise ValueError("each task must map conditions to coefficients")
    coefficients = {
        task: {
            condition: float(value)
            for condition, value in raw[task].items()
        }
        for task in TASKS
    }
    expected_conditions = {"c_to_a_mse", "c_to_a_cka"}
    if any(set(values) != expected_conditions for values in coefficients.values()):
        raise ValueError("each task needs MSE and CKA coefficients")
    invalid = {
        task: values
        for task, values in coefficients.items()
        if any(not math.isfinite(value) or value <= 0 for value in values.values())
    }
    if invalid:
        raise ValueError(f"invalid calibrated alignment coefficients: {invalid}")
    invalid_mse = {
        task: values["c_to_a_mse"]
        for task, values in coefficients.items()
        if not math.isclose(
            values["c_to_a_mse"],
            REFERENCE_MSE_ALIGNMENT_COEF,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    }
    if invalid_mse:
        raise ValueError(f"LN-MSE coefficient is not frozen at 0.1: {invalid_mse}")
    return coefficients


def load_cka_coefficients(path: Path) -> dict[str, float]:
    """Compatibility helper used by the post-calibration CKA tuning launcher."""

    coefficients = load_alignment_coefficients(path)
    return {task: values["c_to_a_cka"] for task, values in coefficients.items()}
