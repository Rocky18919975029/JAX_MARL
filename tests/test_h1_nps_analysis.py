import numpy as np

from scripts.analyze_h1_mechanisms import (
    h1_signatures,
    paired_effects,
    validate_and_reuse_baseline,
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
                            }
                        )
    return rows


def test_seed_paired_effects_and_signature_use_absolute_errors():
    effects = paired_effects(fixture_rows(), np.random.default_rng(7), 100)
    assert len(effects) == 2 * 2 * 2 * 8 * 4
    assert {row["n_paired_seeds"] for row in effects} == {4}
    assert {row["seeds"] for row in effects} == {"1;2;3;4"}
    signatures = h1_signatures(effects, delta_dec=0.01, delta_bell=0.02)
    assert len(signatures) == 2 * 2 * 2 * 8
    assert all(row["epsilon_dec_noninferior"] for row in signatures)
    assert all(row["epsilon_bell_noninferior"] for row in signatures)
    assert all(row["mean_signature_if_return_improved"] is True for row in signatures)


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
