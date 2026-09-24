"""Frozen configuration helpers for the ShadowHandOver comparison."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path


ALGORITHMS = ("happo", "mappo", "madpo")
CONDITIONS = ("none", "arec")
PROTOCOL_VERSION = "harl-dexhands-shadowhandover-v1.0"
AREC_PROTOCOL_VERSION = "harl-dexhands-shadowhandover-arec-v1.0"
MADPO_PAPER_PROFILE = "paper2024"
MADPO_PAPER_CONFIG = (
    Path(__file__).resolve().parent / "configs" / "madpo_shadowhandover_paper2024.json"
)


def madpo_paper_settings() -> dict:
    """Paper task settings overlaid on the pinned HARL environment config."""
    payload = json.loads(MADPO_PAPER_CONFIG.read_text(encoding="utf-8"))
    if payload["profile"] != MADPO_PAPER_PROFILE:
        raise ValueError("Unexpected MADPO paper profile")
    return payload


def number_label(value: float) -> str:
    return f"{value:.10g}".replace("-", "m").replace(".", "p").replace("+", "")


def parse_csv(value: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    selected = tuple(piece.strip() for piece in value.split(",") if piece.strip())
    if (
        not selected
        or len(selected) != len(set(selected))
        or not set(selected).issubset(allowed)
    ):
        raise ValueError(f"Expected a unique selection from {allowed}, got {value!r}")
    return selected


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
        raise ValueError("Seeds must be a unique non-negative list or range")
    return tuple(seeds)


def parse_positive_floats(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(piece.strip()) for piece in value.split(","))
    except ValueError as error:
        raise ValueError(
            "Expected comma-separated positive finite coefficients"
        ) from error
    if (
        not values
        or len(values) != len(set(values))
        or any(not math.isfinite(number) or number <= 0 for number in values)
    ):
        raise ValueError("Coefficients must be unique, positive, and finite")
    return values


def load_matched_config(
    config_path: Path,
    algorithm: str,
    seed: int,
    log_root: Path,
    *,
    num_env_steps: int | None = None,
    n_rollout_threads: int | None = None,
    div_coef: float | None = None,
    div_weight: float | None = None,
    div_sigma: float | None = None,
    div_max_samples: int | None = None,
    madpo_profile: str = "legacy",
    condition: str = "none",
    arec_coef: float = 0.0001,
    arec_q_steps: int = 4,
    arec_q_lr: float = 0.001,
    arec_fisher_ridge: float = 0.001,
) -> tuple[dict, dict, dict]:
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {algorithm!r}")
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    main_args = copy.deepcopy(payload["main_args"])
    algo_args = copy.deepcopy(payload["algo_args"])
    env_args = copy.deepcopy(payload["env_args"])
    if main_args.get("algo") != "happo" or main_args.get("env") != "dexhands":
        raise ValueError("Source config must be HARL HAPPO on DexHands")
    if env_args.get("task") != "ShadowHandOver":
        raise ValueError("Source config must be ShadowHandOver")
    if madpo_profile not in ("legacy", MADPO_PAPER_PROFILE):
        raise ValueError(f"Unknown MADPO profile {madpo_profile!r}")
    if algorithm == "madpo" and madpo_profile == MADPO_PAPER_PROFILE:
        paper = madpo_paper_settings()
        for section in ("train", "model", "algo"):
            algo_args[section].update(paper[section])

    main_args["algo"] = algorithm
    main_args["env"] = "dexhands"
    algo_args["seed"]["seed_specify"] = True
    algo_args["seed"]["seed"] = int(seed)
    algo_args["algo"]["share_param"] = False
    algo_args["eval"]["use_eval"] = False
    algo_args["logger"]["log_dir"] = str(log_root)
    algo_args["train"]["model_dir"] = None
    if num_env_steps is not None:
        algo_args["train"]["num_env_steps"] = int(num_env_steps)
    if n_rollout_threads is not None:
        algo_args["train"]["n_rollout_threads"] = int(n_rollout_threads)
    if algorithm == "madpo":
        defaults = algo_args["algo"]
        algo_args["algo"].update(
            {
                "div_coef": float(
                    defaults.get("div_coef", 1000.0) if div_coef is None else div_coef
                ),
                "div_weight": float(
                    defaults.get("div_weight", 0.05)
                    if div_weight is None
                    else div_weight
                ),
                "div_sigma": float(
                    defaults.get("div_sigma", 1.0) if div_sigma is None else div_sigma
                ),
                "div_max_samples": int(
                    defaults.get("div_max_samples", 1024)
                    if div_max_samples is None
                    else div_max_samples
                ),
                "div_epsilon": 1e-8,
            }
        )
    if condition == "arec":
        if (
            arec_coef <= 0
            or arec_q_steps < 1
            or arec_q_lr <= 0
            or arec_fisher_ridge <= 0
        ):
            raise ValueError(
                "ARec coefficient, q steps/LR, and Fisher ridge must be positive"
            )
        algo_args["algo"].update(
            {
                "arec_coef": float(arec_coef),
                "arec_q_steps": int(arec_q_steps),
                "arec_q_lr": float(arec_q_lr),
                "arec_fisher_ridge": float(arec_fisher_ridge),
            }
        )
    return main_args, algo_args, env_args


@dataclass(frozen=True)
class Task:
    algorithm: str
    seed: int
    div_coef: float = 1000.0
    div_weight: float = 0.05
    div_sigma: float = 1.0
    div_max_samples: int = 1024
    condition: str = "none"
    arec_coef: float = 0.0001
    arec_q_steps: int = 4
    arec_q_lr: float = 0.001
    arec_fisher_ridge: float = 0.001

    @property
    def name(self) -> str:
        prefix = f"HARL-ShadowHandOver-nps-{self.algorithm}"
        if self.algorithm == "madpo":
            prefix += (
                f"-div{number_label(self.div_coef)}-w{number_label(self.div_weight)}"
                f"-sig{number_label(self.div_sigma)}-k{self.div_max_samples}"
            )
        if self.condition == "arec":
            prefix += (
                f"-arec-lam{number_label(self.arec_coef)}"
                f"-qs{self.arec_q_steps}-qlr{number_label(self.arec_q_lr)}"
                f"-ridge{number_label(self.arec_fisher_ridge)}"
            )
        return f"{prefix}-seed{self.seed}"


def task_matrix(
    algorithms: tuple[str, ...],
    seeds: tuple[int, ...],
    *,
    div_coef: float = 1000.0,
    div_weight: float = 0.05,
    div_sigma: float = 1.0,
    div_max_samples: int = 1024,
    conditions: tuple[str, ...] = ("none",),
    arec_coef: float = 0.0001,
    arec_coefs: tuple[float, ...] | None = None,
    arec_q_steps: int = 4,
    arec_q_lr: float = 0.001,
    arec_fisher_ridge: float = 0.001,
) -> list[Task]:
    grid = (arec_coef,) if arec_coefs is None else arec_coefs
    if (
        not grid
        or len(grid) != len(set(grid))
        or any(not math.isfinite(coef) or coef <= 0 for coef in grid)
    ):
        raise ValueError("ARec coefficient grid must be unique, positive, and finite")
    return [
        Task(
            algorithm,
            seed,
            div_coef,
            div_weight,
            div_sigma,
            div_max_samples,
            condition,
            coef,
            arec_q_steps,
            arec_q_lr,
            arec_fisher_ridge,
        )
        for seed in seeds
        for algorithm in algorithms
        for condition in conditions
        for coef in (grid if condition == "arec" else (arec_coef,))
    ]
