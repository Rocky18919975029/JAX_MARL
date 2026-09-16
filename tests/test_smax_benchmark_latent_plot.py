import json
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from scripts.plot_smax_benchmark_latent import (
    discover_rows,
    paired_tables,
    plot_distance,
    validate_matrix,
)


def make_checkpoint(root, condition, distance, seed, step, return_value, epsilon):
    raw_condition = f"{condition}_cka" if distance == "linear_cka" else condition
    run_name = f"SMAXB4-3s5z_vs_3s6z-nps-{raw_condition}-seed{seed}"
    name = "initial" if step == 0 else f"step_{step:012d}"
    directory = root / "diagnostics_raw" / run_name / name
    directory.mkdir(parents=True)
    metadata = {
        "run_name": run_name,
        "map_name": "3s5z_vs_3s6z",
        "actor_parameter_sharing": False,
        "condition": raw_condition,
        "align_distance": distance,
        "training_seed": seed,
        "checkpoint_nominal_env_step": step,
    }
    latent = {
        "heldout_episode_return_mean": return_value,
        "epsilon_lat": epsilon,
        "fisher_ridge_absolute": 0.001,
        "reference_protocol": "baseline_free_mc_return_train_matched_gae",
    }
    (directory / "metadata.json").write_text(json.dumps(metadata))
    (directory / "latent_summary.json").write_text(json.dumps(latent))


def synthetic_matrix(tmp_path):
    for seed in (1, 2):
        for step in (0, 10):
            make_checkpoint(tmp_path, "none", "ln_mse", seed, step, 1.0, 0.5)
            for distance in ("ln_mse", "linear_cka"):
                for index, condition in enumerate(("c_to_a", "a_to_c", "joint"), 1):
                    make_checkpoint(
                        tmp_path,
                        condition,
                        distance,
                        seed,
                        step,
                        1.0 + index * 0.1 + seed * 0.01,
                        0.5 - index * 0.02,
                    )


def test_seed_paired_tables_and_figures(tmp_path):
    synthetic_matrix(tmp_path)
    rows = discover_rows(tmp_path, "3s5z_vs_3s6z", "nps")
    seeds, steps = validate_matrix(rows, (1, 2))
    seed_rows, summaries = paired_tables(rows, seeds, steps)
    assert len(seed_rows) == 2 * 3 * 2 * 2
    selected = next(
        row
        for row in seed_rows
        if row["align_distance"] == "linear_cka"
        and row["condition"] == "c_to_a"
        and row["seed"] == 1
        and row["nominal_step"] == 10
    )
    assert selected["delta_heldout_return"] == pytest.approx(0.11)
    assert selected["delta_epsilon_lat"] == pytest.approx(-0.02)
    summary = next(
        row
        for row in summaries
        if row["align_distance"] == "linear_cka"
        and row["condition"] == "c_to_a"
        and row["nominal_step"] == 10
        and row["metric"] == "heldout_return"
    )
    assert summary["paired_mean_difference"] == pytest.approx(0.115)
    output = tmp_path / "figures"
    output.mkdir()
    stem = plot_distance(
        output,
        "3s5z_vs_3s6z",
        "nps",
        "linear_cka",
        seed_rows,
        summaries,
    )
    assert stem.with_suffix(".png").is_file()
    assert stem.with_suffix(".pdf").is_file()


def test_missing_condition_is_rejected(tmp_path):
    synthetic_matrix(tmp_path)
    for path in tmp_path.glob("diagnostics_raw/*joint_cka*/*/latent_summary.json"):
        path.unlink()
    with pytest.raises(RuntimeError, match="have no latent_summary"):
        discover_rows(tmp_path, "3s5z_vs_3s6z", "nps")
