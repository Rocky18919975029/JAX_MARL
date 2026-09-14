import numpy as np

from scripts.analyze_h1_mechanisms import paired_checkpoint_effects


def test_reduced_seed_mechanism_effects_pair_against_none():
    rows = []
    for seed in (1, 2, 3, 4):
        for condition, offset in (("none", 0.0), ("c_to_a", 0.1)):
            rows.append(
                {
                    "task": "10m_vs_11m",
                    "actor_parameterization": "nps",
                    "condition": condition,
                    "seed": seed,
                    "nominal_step": 500_000,
                    "r_lat": 1.0 + offset,
                    "epsilon_dec": 0.01 + offset,
                    "epsilon_bell_excess": 0.5 + offset,
                }
            )

    effects = paired_checkpoint_effects(rows, np.random.default_rng(7))

    assert len(effects) == 3
    assert {row["n_paired_seeds"] for row in effects} == {4}
    assert {row["seeds"] for row in effects} == {"1;2;3;4"}
