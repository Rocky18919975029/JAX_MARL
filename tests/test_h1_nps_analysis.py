import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.analyze_h1_mechanisms import (
    curve_summary,
    paired_effects,
    validate_and_reuse_baseline,
)
from scripts.plot_h1_mechanisms import (
    draw_paired_panel,
    draw_raw_panel,
    finish_figure,
)


def fixture_rows():
    rows = []
    for task in ("10m_vs_11m", "smacv2_10_units"):
        for distance in ("ln_mse", "linear_cka"):
            for seed in (1, 2, 3, 4):
                for condition in ("none", "a_to_c", "c_to_a"):
                    for step in (
                        0,
                        500_000,
                        1_000_000,
                        2_000_000,
                        4_000_000,
                        6_000_000,
                        8_000_000,
                        10_000_000,
                    ):
                        aligned = condition != "none"
                        rows.append(
                            {
                                "task": task,
                                "align_distance": distance,
                                "condition": condition,
                                "seed": seed,
                                "nominal_step": step,
                                "heldout_return": float(aligned),
                                "epsilon_lat": -float(aligned),
                                "epsilon_dec": 0.005 * float(aligned),
                                "epsilon_bell": 0.01 * float(aligned),
                                "decision_kendall_tau": 0.25,
                                "decision_pairwise_accuracy": 0.625,
                                "decision_top1_agreement": 0.4,
                            }
                        )
    return rows


def test_seed_paired_effects_are_descriptive_without_pass_fail():
    effects = paired_effects(fixture_rows(), np.random.default_rng(7), 100)
    assert len(effects) == 2 * 2 * 2 * 8 * 4
    assert {row["n_paired_seeds"] for row in effects} == {4}
    assert {row["seeds"] for row in effects} == {"1;2;3;4"}
    assert all("noninferior" not in row for row in effects)
    assert all("signature" not in row for row in effects)
    summary = curve_summary(fixture_rows(), np.random.default_rng(8), 100)
    assert len(summary) == 2 * 2 * 3 * 8 * 7
    assert {row["metric"] for row in summary} >= {
        "decision_kendall_tau",
        "decision_pairwise_accuracy",
    }


def test_cka_reuses_baseline_without_new_seed_replicates():
    rows = fixture_rows()
    mse = [
        {**row, "alignment_coef": 0.1}
        for row in rows
        if row["align_distance"] == "ln_mse"
    ]
    cka = [
        {**row, "alignment_coef": 0.037}
        for row in rows
        if row["align_distance"] == "linear_cka" and row["condition"] != "none"
    ]
    combined = validate_and_reuse_baseline(mse, cka)
    reused = [
        row for row in combined if row.get("baseline_source") == "reused_ln_mse_none"
    ]
    assert len(combined) == 384
    assert len(reused) == 64
    assert all(row["align_distance"] == "linear_cka" for row in reused)


def test_figures_overlay_four_seeds_and_show_decision_chance_reference():
    rows = [
        row
        for row in fixture_rows()
        if row["task"] == "10m_vs_11m" and row["align_distance"] == "ln_mse"
    ]
    summaries = curve_summary(rows, np.random.default_rng(9), 100)
    effects = paired_effects(fixture_rows(), np.random.default_rng(10), 100)
    selected_effects = [
        row
        for row in effects
        if row["task"] == "10m_vs_11m" and row["align_distance"] == "ln_mse"
    ]
    figure, axis = plt.subplots()
    draw_raw_panel(axis, summaries, rows, "decision_pairwise_accuracy", chance=0.5)
    assert len(axis.lines) == 1 + 3 * (4 + 1)
    assert float(axis.lines[0].get_ydata()[0]) == 0.5
    plt.close(figure)

    figure, axis = plt.subplots()
    draw_paired_panel(axis, selected_effects, rows, "epsilon_lat")
    assert len(axis.lines) == 1 + 2 * (4 + 1)
    assert float(axis.lines[0].get_ydata()[0]) == 0.0
    plt.close(figure)


def test_shared_legend_has_a_dedicated_right_column():
    rows = [
        row
        for row in fixture_rows()
        if row["task"] == "10m_vs_11m" and row["align_distance"] == "ln_mse"
    ]
    summaries = curve_summary(rows, np.random.default_rng(11), 100)
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8))
    for axis, metric in zip(
        axes.flat,
        ("heldout_return", "epsilon_lat", "epsilon_dec", "epsilon_bell"),
    ):
        draw_raw_panel(axis, summaries, rows, metric)
    legend = finish_figure(figure, axes, "Layout test")

    figure.canvas.draw()
    assert max(axis.get_position().x1 for axis in axes.flat) <= 0.82
    assert legend.get_title().get_text() == "Alignment condition"
    assert [text.get_text() for text in legend.get_texts()] == [
        "none",
        "A → C",
        "C → A",
    ]
    plt.close(figure)
