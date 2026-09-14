from scripts.plot_h1_mechanisms import conditions_for


def test_reduced_plot_includes_all_available_conditions():
    rows = [
        {
            "task": "10m_vs_11m",
            "actor_parameterization": "nps",
            "condition": condition,
        }
        for condition in ("joint", "a_to_c", "none", "c_to_a")
    ]

    assert conditions_for(rows, "10m_vs_11m", "nps") == (
        "none",
        "a_to_c",
        "c_to_a",
        "joint",
    )
