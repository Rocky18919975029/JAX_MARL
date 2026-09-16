import json

import matplotlib
import pytest

matplotlib.use("Agg")

from scripts.analyze_h1_robust_distortion import (
    load_checkpoint_rows,
    number_token,
    paired_rows,
    plot_temporal,
    summarize_paired,
)


def write_summary(root, condition, distance, seed, step, aligned):
    raw_condition = f"{condition}_cka" if distance == "linear_cka" else condition
    run = f"run-{raw_condition}-seed{seed}"
    checkpoint = "initial" if step == 0 else "final"
    directory = root / "checkpoints" / run / checkpoint
    directory.mkdir(parents=True)
    rows = []
    for aggregation in ("slot", "slot_x_type"):
        for budget, label in ((1, "M"), (2, "2M"), (4, "4M")):
            for ridge in (0.001, 0.01):
                rows.append(
                    {
                        "aggregation": aggregation,
                        "episode_budget": budget,
                        "episode_budget_label": label,
                        "fisher_ridge_absolute": ridge,
                        "epsilon_lat_raw": 0.4 - 0.1 * aligned,
                        "epsilon_lat_phase_matched": 0.3 - 0.1 * aligned,
                        "fisher_natural_gradient_cosine": 0.5 + 0.1 * aligned,
                        "epsilon_lat_optimal_scale": 0.2 - 0.05 * aligned,
                        "optimal_nonnegative_critic_scale": 1.0,
                        "reference_natural_norm_sq": 0.8,
                        "critic_natural_norm_sq": 0.7,
                        "reference_critic_natural_inner": 0.6,
                    }
                )
    payload = {
        "run_name": run,
        "task": "task",
        "condition": raw_condition,
        "align_distance": distance,
        "seed": seed,
        "checkpoint_step": step,
        "checkpoint_nominal_step": step,
        "heldout_episode_return_mean": 1.0 + 0.2 * aligned,
        "robust_distortion_protocol": "h1-robust-distortion-v1.0",
        "phase_protocol": "uniform",
        "fisher_ridges": [0.001, 0.01],
        "aggregate_metrics": rows,
    }
    (directory / "robust_distortion_summary.json").write_text(json.dumps(payload))


def synthetic_results(tmp_path):
    for seed in (1, 2):
        for step in (0, 10):
            write_summary(tmp_path, "none", "ln_mse", seed, step, 0)
            for distance in ("ln_mse", "linear_cka"):
                for condition in ("c_to_a", "a_to_c"):
                    write_summary(tmp_path, condition, distance, seed, step, 1)


def test_robust_analysis_pairs_baseline_and_plots(tmp_path):
    synthetic_results(tmp_path)
    checkpoints = load_checkpoint_rows(tmp_path)
    paired = paired_rows(checkpoints)
    summaries = summarize_paired(paired)
    selected = next(
        row
        for row in paired
        if row["align_distance"] == "linear_cka"
        and row["condition"] == "c_to_a"
        and row["aggregation"] == "slot_x_type"
        and row["episode_budget_label"] == "4M"
        and row["fisher_ridge_absolute"] == 0.001
    )
    assert selected["delta_heldout_return"] == pytest.approx(0.2)
    assert selected["delta_epsilon_lat_raw"] == pytest.approx(-0.1)
    assert selected["delta_fisher_natural_gradient_cosine"] == pytest.approx(0.1)
    figures = tmp_path / "figures"
    figures.mkdir()
    stem = plot_temporal(
        figures,
        "task",
        "linear_cka",
        "slot_x_type",
        0.001,
        "4M",
        paired,
        summaries,
    )
    assert stem.with_suffix(".png").is_file()
    assert stem.with_suffix(".pdf").is_file()
    assert "ridge0p001" in stem.name


def test_number_token_preserves_decimal_ridge():
    assert number_token(0.001) == "0p001"
    assert number_token(0.0001) == "0p0001"
    assert number_token(1.0) == "1"
