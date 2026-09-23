#!/usr/bin/env python3
"""Re-run the first four-panel figure's selected none/ARec configurations.

The eight committed YAML files are effective hyperparameter snapshots. Before
launching, this wrapper can verify them against the old four-panel report,
completed sweep manifests, and final checkpoint configs on the server. The
current, immutable four-method launcher then handles seeds, GPUs and outputs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import yaml

try:
    from scripts import smax_four_method as control
except ModuleNotFoundError:  # Direct execution from scripts/.
    import smax_four_method as control


REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "configs/smax_first_four_panel"
TASKS = ("10m_vs_11m", "3s5z_vs_3s6z", "6s9z_vs_6s10z", "smacv2_10_units")
HISTORICAL_PROTOCOL = "smax-nps-actor-score-recovery-sweep-v1.0"
FIGURE_ID = "smax-arec-selected-return-curves-v1"
CONFIG_KEYS = frozenset({
    "LR", "NUM_ENVS", "NUM_STEPS", "TOTAL_TIMESTEPS", "FC_DIM_SIZE",
    "GRU_HIDDEN_DIM", "UPDATE_EPOCHS", "NUM_MINIBATCHES", "GAMMA",
    "GAE_LAMBDA", "CLIP_EPS", "SCALE_CLIP_EPS", "ENT_COEF", "VF_COEF",
    "MAX_GRAD_NORM", "ACTIVATION", "OBS_WITH_AGENT_ID",
    "ACTOR_PARAMETER_SHARING", "MATCHED_COMPARISON", "ALIGN_MODE",
    "ALIGN_DISTANCE", "ALIGN_DISTANCE_EPS", "ALIGNMENT_COEF",
    "ACTOR_SCORE_RECOVERY", "ACTOR_SCORE_RECOVERY_COEF",
    "ACTOR_SCORE_RECOVERY_FISHER_RIDGE", "ACTOR_SCORE_RECOVERY_Q_LR",
    "ACTOR_SCORE_RECOVERY_Q_STEPS", "EXPERIMENT_CONDITION", "ENV_NAME",
    "MAP_NAME", "ENV_KWARGS", "ANNEAL_LR", "SAVE_CHECKPOINTS",
    "CHECKPOINT_INTERVAL_TIMESTEPS", "WANDB_UPLOAD_CHECKPOINTS",
})
METHOD_KEYS = frozenset({
    "ACTOR_SCORE_RECOVERY", "ACTOR_SCORE_RECOVERY_COEF",
    "ACTOR_SCORE_RECOVERY_Q_STEPS", "EXPERIMENT_CONDITION",
})
RUNTIME_KEYS = frozenset({
    "LR", "NUM_ENVS", "NUM_STEPS", "TOTAL_TIMESTEPS", "UPDATE_EPOCHS",
    "NUM_MINIBATCHES", "MAP_NAME", "ACTOR_SCORE_RECOVERY",
    "ACTOR_SCORE_RECOVERY_COEF", "ACTOR_SCORE_RECOVERY_Q_STEPS",
    "ACTOR_SCORE_RECOVERY_Q_LR", "ACTOR_SCORE_RECOVERY_FISHER_RIDGE",
    "EXPERIMENT_CONDITION", "CHECKPOINT_INTERVAL_TIMESTEPS",
    "SAVE_CHECKPOINTS", "WANDB_UPLOAD_CHECKPOINTS",
})


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def same_value(key: str, observed, expected) -> bool:
    # PyYAML parses the upstream base config's `1e-8` as a string, while
    # OmegaConf/checkpoint JSON may represent the same scalar numerically.
    if key == "ALIGN_DISTANCE_EPS":
        return math.isclose(float(observed), float(expected), rel_tol=0, abs_tol=1e-14)
    return observed == expected


def load_pair(task: str, config_dir: Path = CONFIG_DIR) -> tuple[dict, dict, tuple[Path, Path]]:
    if task not in TASKS:
        raise ValueError(f"Unsupported task: {task}")
    config_dir = config_dir.resolve()
    if config_dir != REPO and REPO not in config_dir.parents:
        raise ValueError("Tuned YAML files must be inside the versioned repository")
    paths = tuple(config_dir / f"{task}_{method}.yaml" for method in ("none", "arec"))
    profiles = []
    base = yaml.safe_load((REPO / "baselines/MAPPO/config/mappo_homogenous_rnn_smax.yaml").read_text())
    for method, path in zip(("none", "arec"), paths):
        profile = yaml.safe_load(path.read_text(encoding="utf-8"))
        if set(profile) != {"schema_version", "figure_id", "historical_source_root",
                            "map_name", "method", "seeds", "config"}:
            raise ValueError(f"Unexpected YAML schema: {path}")
        if (profile["schema_version"] != 1 or profile["figure_id"] != FIGURE_ID
                or profile["map_name"] != task or profile["method"] != method
                or profile["seeds"] != [1, 2, 3, 4]):
            raise ValueError(f"YAML identity/seed mismatch: {path}")
        config = profile["config"]
        if set(config) != CONFIG_KEYS:
            raise ValueError(f"Missing/unexpected effective config keys in {path}: "
                             f"{sorted(CONFIG_KEYS ^ set(config))}")
        if (config["MAP_NAME"] != task or config["EXPERIMENT_CONDITION"] != method
                or config["ACTOR_SCORE_RECOVERY"] != (method == "arec")
                or config["ACTOR_PARAMETER_SHARING"] is not False
                or config["MATCHED_COMPARISON"] is not True
                or config["ALIGN_MODE"] != "none" or config["ALIGNMENT_COEF"] != 0):
            raise ValueError(f"Method routing mismatch: {path}")
        if (method == "none" and config["ACTOR_SCORE_RECOVERY_COEF"] != 0
                or method == "arec" and config["ACTOR_SCORE_RECOVERY_COEF"] <= 0):
            raise ValueError(f"Invalid auxiliary coefficient: {path}")
        if (config["NUM_ENVS"] % config["NUM_MINIBATCHES"]
                or min(config["TOTAL_TIMESTEPS"], config["NUM_STEPS"],
                       config["CHECKPOINT_INTERVAL_TIMESTEPS"]) <= 0):
            raise ValueError(f"Invalid rollout or checkpoint sizing: {path}")
        for key in CONFIG_KEYS - RUNTIME_KEYS:
            if not same_value(key, base.get(key), config[key]):
                raise ValueError(f"Current base config {key} differs from {path}; "
                                 "do not silently change the historical protocol")
        profiles.append(profile)
    none, arec = profiles
    if none["historical_source_root"] != arec["historical_source_root"]:
        raise ValueError("Paired methods must come from the same first-figure sweep")
    for key in CONFIG_KEYS - METHOD_KEYS:
        if none["config"][key] != arec["config"][key]:
            raise ValueError(f"Unpaired common setting {key} in {task}")
    return none, arec, paths


def _selected_historical_runs(manifest: dict, method: str, config: dict) -> dict[int, dict]:
    selected = {}
    for run in manifest["runs"]:
        if run["condition"] != ("none" if method == "none" else "actor_score_recovery"):
            continue
        if method == "arec" and not (
            math.isclose(float(run["coef"]), config["ACTOR_SCORE_RECOVERY_COEF"], rel_tol=0, abs_tol=1e-12)
            and int(run["q_steps"]) == config["ACTOR_SCORE_RECOVERY_Q_STEPS"]
            and float(run["q_learning_rate"]) == config["ACTOR_SCORE_RECOVERY_Q_LR"]
            and float(run["fisher_ridge"]) == config["ACTOR_SCORE_RECOVERY_FISHER_RIDGE"]
        ):
            continue
        seed = int(run["seed"])
        if seed in selected:
            raise RuntimeError(f"Duplicate historical {method} run, seed {seed}")
        selected[seed] = run
    if set(selected) != {1, 2, 3, 4}:
        raise RuntimeError(f"Historical {method} cohort is incomplete")
    return selected


def verify_historical(task: str, profiles: tuple[dict, dict], matrix_root: Path) -> dict:
    none, arec = profiles
    source = (matrix_root / none["historical_source_root"]).resolve()
    report_path = matrix_root / "actor_score_recovery_best_return_report_v1/report_manifest.json"
    report = read_json(report_path)
    if Path(report["source_roots"][task]).resolve() != source:
        raise RuntimeError(f"First-figure source root differs from YAML: {task}")
    selected = report["selection"][task]["parameters"]
    c = arec["config"]
    expected = {
        "coef": c["ACTOR_SCORE_RECOVERY_COEF"],
        "q_steps": c["ACTOR_SCORE_RECOVERY_Q_STEPS"],
        "q_learning_rate": c["ACTOR_SCORE_RECOVERY_Q_LR"],
        "fisher_ridge": c["ACTOR_SCORE_RECOVERY_FISHER_RIDGE"],
    }
    if selected != expected:
        raise RuntimeError(f"First-figure selected ARec cell differs: {task}: {selected}")
    manifest = read_json(source / "experiment_manifest.json")
    common = none["config"]
    if (manifest.get("protocol") != HISTORICAL_PROTOCOL
            or manifest.get("maps") != [task]
            or manifest.get("seeds") != [1, 2, 3, 4]
            or int(manifest["budgets"][task]) != common["TOTAL_TIMESTEPS"]):
        raise RuntimeError(f"Historical sweep identity/budget mismatch: {source}")
    for field, key in (("learning_rate", "LR"), ("update_epochs", "UPDATE_EPOCHS"),
                       ("num_envs", "NUM_ENVS"), ("num_minibatches", "NUM_MINIBATCHES"),
                       ("checkpoint_interval", "CHECKPOINT_INTERVAL_TIMESTEPS")):
        if manifest[field] != common[key]:
            raise RuntimeError(f"Historical sweep {field} differs from YAML: {task}")
    historical_commits = set()
    for profile in profiles:
        method, config = profile["method"], profile["config"]
        for seed, run in _selected_historical_runs(manifest, method, config).items():
            if read_json(source / "status" / f"{run['run_name']}.json").get("status") != "completed":
                raise RuntimeError(f"Historical run did not complete: {run['run_name']}")
            matches = list((source / "checkpoints").glob(
                f"**/{run['run_name']}-*/final/config.json"
            ))
            if len(matches) != 1:
                raise RuntimeError(f"Expected one historical final config: {run['run_name']}")
            observed = read_json(matches[0])
            for key in CONFIG_KEYS:
                expected_value = (
                    "actor_score_recovery"
                    if key == "EXPERIMENT_CONDITION" and method == "arec"
                    else config[key]
                )
                if not same_value(key, observed.get(key), expected_value):
                    raise RuntimeError(f"Historical checkpoint {key} differs from YAML: "
                                       f"{task}/{method}/seed{seed}; "
                                       f"observed={observed.get(key)!r}, "
                                       f"expected={expected_value!r}")
            historical_commits.add(observed.get("GIT_COMMIT", "unknown"))
    return {
        "first_figure_report_sha256": control.sha256(report_path),
        "historical_sweep_manifest_sha256": control.sha256(source / "experiment_manifest.json"),
        "historical_source_root": str(source),
        "historical_git_commits": sorted(historical_commits),
        "selected_arec": expected,
    }


def launch_args(args: argparse.Namespace, none: dict, arec: dict,
                paths: tuple[Path, Path], provenance: dict | None) -> SimpleNamespace:
    c = none["config"]
    a = arec["config"]
    return SimpleNamespace(
        map_name=args.task, methods=("none", "arec"),
        seed_start=1, seed_count=4, total_timesteps=int(c["TOTAL_TIMESTEPS"]),
        num_steps=int(c["NUM_STEPS"]), ppo_lrs=(float(c["LR"]),),
        ppo_epochs=(int(c["UPDATE_EPOCHS"]),),
        num_envs_grid=(int(c["NUM_ENVS"]),),
        num_minibatches_grid=(int(c["NUM_MINIBATCHES"]),),
        mse_coefs=(0.1,), cka_coefs=(0.3,),
        arec_coefs=(float(a["ACTOR_SCORE_RECOVERY_COEF"]),),
        arec_q_steps=(int(a["ACTOR_SCORE_RECOVERY_Q_STEPS"]),),
        arec_q_lrs=(float(a["ACTOR_SCORE_RECOVERY_Q_LR"]),),
        arec_fisher_ridges=(float(a["ACTOR_SCORE_RECOVERY_FISHER_RIDGE"]),),
        checkpoint_interval=int(c["CHECKPOINT_INTERVAL_TIMESTEPS"]),
        run_root=args.run_root, gpus=args.gpus,
        max_runs_per_gpu=args.max_runs_per_gpu,
        project=args.project, wandb_mode=args.wandb_mode, dry_run=args.dry_run,
        tuned_config_paths=paths,
        tuned_selection={"figure_id": FIGURE_ID, "task": args.task,
                         "historical_verification": provenance,
                         "source_yaml": [str(path.relative_to(REPO)) for path in paths]},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--historical-matrix-root", type=Path,
                        help="Root containing first-four-panel report and original sweeps")
    parser.add_argument("--skip-historical-validation", action="store_true",
                        help="Only for environments without original sweep artifacts")
    parser.add_argument("--gpus", type=control.comma_strings, default=("0", "1", "2", "3"))
    parser.add_argument("--max-runs-per-gpu", type=int, default=2)
    parser.add_argument("--project", default="jaxmarl-smax-first-four-panel-tuned")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                        default="online")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    none, arec, paths = load_pair(args.task, args.config_dir)
    if args.skip_historical_validation:
        provenance = None
        print("WARNING: original figure/checkpoints not validated", flush=True)
    else:
        if args.historical_matrix_root is None:
            parser.error("Give --historical-matrix-root or explicitly skip validation")
        provenance = verify_historical(args.task, (none, arec),
                                       args.historical_matrix_root.expanduser().resolve())
    launch = launch_args(args, none, arec, paths, provenance)
    runs = control.make_grid(launch)
    if len(runs) != 8 or {run.method for run in runs} != {"none", "arec"}:
        raise RuntimeError("Expected exactly four paired seeds per method")
    print(f"task={args.task} methods=none,arec seeds=1-4 "
          f"nominal_budget={none['config']['TOTAL_TIMESTEPS']:,} "
          f"effective_budget={runs[0].timesteps:,} "
          f"lambda={arec['config']['ACTOR_SCORE_RECOVERY_COEF']} "
          f"q_steps={arec['config']['ACTOR_SCORE_RECOVERY_Q_STEPS']}", flush=True)
    control.launch(launch)


if __name__ == "__main__":
    main()
