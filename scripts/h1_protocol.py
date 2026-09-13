#!/usr/bin/env python3
"""Freeze and audit the H1 SMAX confirmatory training protocol."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


PROTOCOL_VERSION = "h1-v1.0"
FROZEN_KEYS = (
    "ENV_NAME",
    "NUM_ENVS",
    "NUM_STEPS",
    "TOTAL_TIMESTEPS",
    "FC_DIM_SIZE",
    "GRU_HIDDEN_DIM",
    "UPDATE_EPOCHS",
    "NUM_MINIBATCHES",
    "LR",
    "GAMMA",
    "GAE_LAMBDA",
    "CLIP_EPS",
    "SCALE_CLIP_EPS",
    "ENT_COEF",
    "VF_COEF",
    "MAX_GRAD_NORM",
    "ACTIVATION",
    "OBS_WITH_AGENT_ID",
    "MATCHED_COMPARISON",
    "ALIGNMENT_COEF",
    "ANNEAL_LR",
    "ENV_KWARGS",
)

MANUAL_REFERENCE = {
    "ENV_NAME": "HeuristicEnemySMAX",
    "NUM_ENVS": 128,
    "NUM_STEPS": 128,
    "TOTAL_TIMESTEPS": 10_000_000,
    "FC_DIM_SIZE": 128,
    "GRU_HIDDEN_DIM": 128,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 4,
    "LR": 0.002,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "SCALE_CLIP_EPS": False,
    "ENT_COEF": 0.0,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.25,
    "ACTIVATION": "relu",
    "OBS_WITH_AGENT_ID": True,
    "MATCHED_COMPARISON": True,
    "ALIGNMENT_COEF": 0.1,
    "ANNEAL_LR": True,
    "ENV_KWARGS": {
        "see_enemy_actions": True,
        "walls_cause_death": True,
        "attack_mode": "closest",
    },
}


def run(command, cwd=None):
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def repository_root():
    return Path(__file__).resolve().parents[1]


def load_mapping(path):
    path = Path(path).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError(
                "Reading YAML requires PyYAML (installed with hydra-core). "
                "Alternatively export the pilot config as JSON."
            ) from error
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    if isinstance(payload.get("training_config"), dict):
        payload = payload["training_config"]
    if isinstance(payload.get("config"), dict):
        payload = payload["config"]

    # W&B's local config.yaml represents each field as {value: ..., desc: ...}.
    unwrapped = {}
    for key, value in payload.items():
        if (
            isinstance(value, dict)
            and "value" in value
            and set(value)
            <= {
                "value",
                "desc",
            }
        ):
            unwrapped[key] = value["value"]
        else:
            unwrapped[key] = value
    return unwrapped, path


def normalize_value(key, value):
    reference = MANUAL_REFERENCE[key]
    if isinstance(reference, bool):
        if isinstance(value, str):
            return value.lower() == "true"
        return bool(value)
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(float(value))
    if isinstance(reference, float):
        return float(value)
    return value


def current_git_state(repo):
    try:
        commit = run(["git", "rev-parse", "HEAD"], cwd=repo)
        status = run(["git", "status", "--porcelain=v1"], cwd=repo)
    except (OSError, subprocess.CalledProcessError):
        return "unknown", "unavailable"
    return commit, status


def freeze(args):
    source, source_path = load_mapping(args.source)
    applied_overrides = []
    for override in args.override:
        if "=" not in override:
            raise ValueError(
                f"Invalid --override {override!r}; expected KEY=JSON_VALUE"
            )
        key, raw_value = override.split("=", 1)
        if key not in FROZEN_KEYS:
            raise ValueError(f"{key!r} is not a frozen training field")
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        previous = source.get(key, "<missing>")
        source[key] = value
        applied_overrides.append(
            {"field": key, "source_value": previous, "frozen_value": value}
        )
    missing = [key for key in FROZEN_KEYS if key not in source]
    if missing:
        raise ValueError("Pilot config is missing frozen fields: " + ", ".join(missing))

    frozen = {key: normalize_value(key, source[key]) for key in FROZEN_KEYS}
    if frozen["MATCHED_COMPARISON"] is not True:
        raise ValueError("H1 requires MATCHED_COMPARISON=true")

    repo = repository_root()
    commit, status = current_git_state(repo)
    run_root = Path(args.run_root).expanduser().resolve()
    protocol_dir = run_root / "protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    destination = protocol_dir / "frozen_training_config.json"
    payload = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "source_path": str(source_path),
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit_at_freeze": commit,
        "worktree_clean_at_freeze": status == "",
        "explicit_overrides": applied_overrides,
        "training_config": frozen,
    }
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    differences = []
    for key in FROZEN_KEYS:
        if frozen[key] != MANUAL_REFERENCE[key]:
            differences.append(
                f"- `{key}`: frozen `{frozen[key]!r}`; manual reference "
                f"`{MANUAL_REFERENCE[key]!r}`"
            )
    deviation_path = protocol_dir / "deviations.md"
    header = (
        "# H1 protocol deviations\n\n"
        f"Protocol: `{PROTOCOL_VERSION}`  \n"
        f"Frozen source: `{source_path}`  \n"
        f"Git commit: `{commit}`\n\n"
    )
    body = (
        "No frozen-training differences from the execution manual.\n"
        if not differences
        else "Differences from the manual reference:\n\n"
        + "\n".join(differences)
        + "\n"
    )
    if applied_overrides:
        body += "\nExplicit overrides applied while freezing:\n\n"
        body += "\n".join(
            f"- `{item['field']}`: source `{item['source_value']!r}` -> "
            f"frozen `{item['frozen_value']!r}`"
            for item in applied_overrides
        )
        body += "\n"
    deviation_path.write_text(header + body, encoding="utf-8")
    print(destination)
    print(deviation_path)


def write_command_output(path, command, cwd=None, allow_failure=False):
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=not allow_failure,
            capture_output=True,
            text=True,
        )
        text = completed.stdout
        if completed.stderr:
            text += "\n[stderr]\n" + completed.stderr
        text += f"\n[exit_code] {completed.returncode}\n"
    except OSError as error:
        text = f"Command unavailable: {error}\n"
    path.write_text(text, encoding="utf-8")


def manifest(args):
    repo = repository_root()
    root = Path(args.run_root).expanduser().resolve()
    output = root / "protocol" / "software_manifest"
    output.mkdir(parents=True, exist_ok=True)

    write_command_output(output / "git_commit.txt", ["git", "rev-parse", "HEAD"], repo)
    write_command_output(
        output / "git_status.txt", ["git", "status", "--porcelain=v1"], repo
    )
    write_command_output(output / "git_diff.patch", ["git", "diff", "--binary"], repo)
    write_command_output(
        output / "pip_freeze.txt", [sys.executable, "-m", "pip", "freeze"]
    )
    write_command_output(output / "nvidia_smi.txt", ["nvidia-smi"], allow_failure=True)

    packages = {}
    for name in (
        "jax",
        "jaxlib",
        "jax-cuda13-plugin",
        "jax-cuda13-pjrt",
        "flax",
        "optax",
        "distrax",
        "tfp-nightly",
        "wandb",
        "hydra-core",
        "numpy",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    system = {
        "schema_version": 1,
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get(
                "XLA_PYTHON_CLIENT_PREALLOCATE"
            ),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        },
        "packages": packages,
    }
    (output / "system.json").write_text(
        json.dumps(system, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)


def validate(args):
    payload, path = load_mapping(args.frozen_config)
    missing = [key for key in FROZEN_KEYS if key not in payload]
    if missing:
        raise ValueError(f"{path} is missing: {', '.join(missing)}")
    print(f"PASS {path} ({len(FROZEN_KEYS)} frozen fields)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--source", required=True)
    freeze_parser.add_argument("--run-root", required=True)
    freeze_parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="explicit frozen override as KEY=JSON_VALUE; repeat as needed",
    )
    freeze_parser.set_defaults(func=freeze)

    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--run-root", required=True)
    manifest_parser.set_defaults(func=manifest)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--frozen-config", required=True)
    validate_parser.set_defaults(func=validate)
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    parsed_args.func(parsed_args)
