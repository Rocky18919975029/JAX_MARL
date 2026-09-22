"""Frozen configuration helpers for the ShadowHandOver comparison."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path


ALGORITHMS = ("happo", "mappo", "madpo")
PROTOCOL_VERSION = "harl-dexhands-shadowhandover-v1.0"


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


def load_matched_config(
    config_path: Path,
    algorithm: str,
    seed: int,
    log_root: Path,
    *,
    num_env_steps: int | None = None,
    n_rollout_threads: int | None = None,
    div_coef: float = 1000.0,
    div_weight: float = 0.05,
    div_sigma: float = 1.0,
    div_max_samples: int = 1024,
) -> tuple[dict, dict, dict]:
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {algorithm!r}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    main_args = copy.deepcopy(payload["main_args"])
    algo_args = copy.deepcopy(payload["algo_args"])
    env_args = copy.deepcopy(payload["env_args"])
    if main_args.get("algo") != "happo" or main_args.get("env") != "dexhands":
        raise ValueError("Source config must be HARL HAPPO on DexHands")
    if env_args.get("task") != "ShadowHandOver":
        raise ValueError("Source config must be ShadowHandOver")

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
        algo_args["algo"].update(
            {
                "div_coef": float(div_coef),
                "div_weight": float(div_weight),
                "div_sigma": float(div_sigma),
                "div_max_samples": int(div_max_samples),
                "div_epsilon": 1e-8,
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

    @property
    def name(self) -> str:
        prefix = f"HARL-ShadowHandOver-nps-{self.algorithm}"
        if self.algorithm == "madpo":
            label = lambda value: f"{value:.10g}".replace("-", "m").replace(".", "p")
            prefix += (
                f"-div{label(self.div_coef)}-w{label(self.div_weight)}"
                f"-sig{label(self.div_sigma)}-k{self.div_max_samples}"
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
) -> list[Task]:
    return [
        Task(
            algorithm,
            seed,
            div_coef,
            div_weight,
            div_sigma,
            div_max_samples,
        )
        for seed in seeds
        for algorithm in algorithms
    ]
