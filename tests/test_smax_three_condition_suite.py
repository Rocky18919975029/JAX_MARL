import json

import numpy as np

from scripts.analyze_smax_three_condition_suite import (
    BOOTSTRAP_CI_METHOD,
    CONDITIONS_BY_DIRECTION,
    CONDITIONS,
    Source,
    aggregate_table,
    bootstrap_resample_count,
    checkpoint_steps,
    choose_complete_cohorts,
    discover_sources,
    normalized_auc,
    seed_metrics,
)


def make_source(tmp_path, task, budget, condition, seed, actor_parameterization="nps"):
    run = (
        tmp_path
        / "checkpoints"
        / (f"{task}-{actor_parameterization}-{budget}-{condition}-seed{seed}")
    )
    final = run / "final"
    final.mkdir(parents=True)
    mode = "none" if condition == "none" else condition.rsplit("_", 1)[0]
    distance = "linear_cka" if condition.endswith("_cka") else "ln_mse"
    metadata = {
        "map_name": task,
        "seed": seed,
        "actor_parameter_sharing": actor_parameterization == "ps",
        "align_mode": mode,
        "align_distance": distance,
        "alignment_coef": 0.3 if condition == "c_to_a_cka" else 0.1,
        "wandb_project": "project",
        "wandb_run_id": f"{task}-{budget}-{condition}-{seed}",
        "wandb_run_name": f"run-{condition}-{seed}",
        "protocol_version": "test",
        "nominal_env_step": budget,
        "is_final": True,
    }
    (final / "metadata.json").write_text(json.dumps(metadata))
    (final / "config.json").write_text(json.dumps({"TOTAL_TIMESTEPS": budget}))
    (final / "model.safetensors").write_bytes(b"model")
    for index, step in enumerate(np.linspace(budget / 10, budget, 10), 1):
        checkpoint = run / f"step_{index:02d}"
        checkpoint.mkdir()
        (checkpoint / "metadata.json").write_text(
            json.dumps({"nominal_env_step": int(step), "is_initial": False})
        )
    return Source(
        task=task,
        actor_parameterization=actor_parameterization,
        budget=budget,
        condition=condition,
        seed=seed,
        checkpoint=final.resolve(),
        project="project",
        run_id=metadata["wandb_run_id"],
        run_name=metadata["wandb_run_name"],
        alignment_coef=metadata["alignment_coef"],
        protocol_version="test",
    )


def complete_matrix(
    tmp_path,
    task,
    budget,
    actor_parameterization="nps",
    conditions=CONDITIONS,
):
    return {
        source.key: source
        for condition in conditions
        for seed in (1, 2, 3, 4)
        for source in (
            make_source(
                tmp_path,
                task,
                budget,
                condition,
                seed,
                actor_parameterization,
            ),
        )
    }


def history(offset=0.0, budget=100):
    return [
        {"env_step": step, "returns": step / budget + offset, "win_rate": step / budget}
        for step in range(0, budget + 1, 10)
    ]


def test_discovers_arbitrary_tasks_for_ps_and_nps(tmp_path):
    expected = complete_matrix(tmp_path, "custom_7_agents", 100)
    expected.update(complete_matrix(tmp_path, "custom_7_agents", 100, "ps"))
    discovered = discover_sources(tmp_path)
    assert set(discovered) == set(expected)


def test_discovers_a_to_c_mse_and_cka(tmp_path):
    conditions = CONDITIONS_BY_DIRECTION["a_to_c"]
    expected = complete_matrix(tmp_path, "custom_map", 100, conditions=conditions)
    discovered = discover_sources(tmp_path)
    assert set(discovered) == set(expected)


def test_rejects_non_isolated_interventions_with_align_mode_none(tmp_path):
    isolated = make_source(tmp_path / "isolated", "map", 100, "none", 1)
    for name, flag, declared in (
        ("critic-recovery", "SCORE_RECOVERY", "score_recovery"),
        ("actor-recovery", "ACTOR_SCORE_RECOVERY", "actor_score_recovery"),
        ("oracle", "ORACLE_LATENT_DISTORTION", "oracle_latent_distortion"),
        ("shuffled", "ALIGN_TARGET_SHUFFLE", "none_shuffled"),
    ):
        run = tmp_path / name / "checkpoints" / name / "final"
        run.mkdir(parents=True)
        metadata = json.loads((isolated.checkpoint / "metadata.json").read_text())
        metadata["condition"] = declared
        metadata[flag.lower()] = True
        (run / "metadata.json").write_text(json.dumps(metadata))
        config = json.loads((isolated.checkpoint / "config.json").read_text())
        config[flag] = True
        config["EXPERIMENT_CONDITION"] = declared
        (run / "config.json").write_text(json.dumps(config))
        (run / "model.safetensors").write_bytes(b"model")
    discovered = discover_sources(tmp_path)
    assert set(discovered) == {isolated.key}
    assert discovered[isolated.key].checkpoint == isolated.checkpoint


def test_rejects_condition_label_mismatch_without_auxiliary_flags(tmp_path):
    source = make_source(tmp_path, "map", 100, "none", 1)
    metadata_path = source.checkpoint / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["condition"] = "score_recovery"
    metadata_path.write_text(json.dumps(metadata))
    assert discover_sources(tmp_path) == {}


def test_selects_largest_complete_budget_per_task(tmp_path):
    sources = complete_matrix(tmp_path / "short", "map", 100)
    sources.update(complete_matrix(tmp_path / "long", "map", 200))
    sources.pop(("map", "nps", 200, "c_to_a_cka", 4))
    selected, audit = choose_complete_cohorts(sources, (1, 2, 3, 4))
    assert selected == {("map", "nps"): 100}
    assert any(
        row["training_budget_env_steps"] == 200 and not row["complete"] for row in audit
    )


def test_selects_a_to_c_cohort_without_using_c_to_a_runs(tmp_path):
    a_to_c = CONDITIONS_BY_DIRECTION["a_to_c"]
    sources = complete_matrix(tmp_path / "a-to-c", "map", 100, conditions=a_to_c)
    sources.update(complete_matrix(tmp_path / "c-to-a", "map", 200))
    selected, audit = choose_complete_cohorts(sources, (1, 2, 3, 4), conditions=a_to_c)
    assert selected == {("map", "nps"): 100}
    long_budget = next(row for row in audit if row["training_budget_env_steps"] == 200)
    assert not long_budget["complete"]
    assert "a_to_c_cka:seed1" in long_budget["missing_runs"]


def test_final_performance_is_mean_of_last_five_saved_checkpoints(tmp_path):
    source = make_source(tmp_path, "map", 100, "none", 1)
    row = seed_metrics(source, history(budget=100), 5)
    assert checkpoint_steps(source)[-5:] == [60, 70, 80, 90, 100]
    assert np.isclose(row["final_return_last5_ckpt"], 0.8)
    assert row["final_checkpoint_steps"] == "60;70;80;90;100"


def test_auc_is_time_normalised_trapezoidal_integral():
    assert np.isclose(normalized_auc(history(budget=100), "returns", 100), 0.5)


def test_auc_supports_numpy_without_legacy_trapz(monkeypatch):
    monkeypatch.delattr(np, "trapz", raising=False)
    monkeypatch.setattr(np, "trapezoid", lambda y, x: 50.0, raising=False)
    assert np.isclose(normalized_auc(history(budget=100), "returns", 100), 0.5)


def test_summary_keeps_tasks_separate_and_reports_paired_deltas():
    rows = []
    for condition in CONDITIONS:
        for seed in (1, 2, 3, 4):
            gain = (
                0.2
                if condition == "c_to_a_cka"
                else 0.1 if condition == "c_to_a_mse" else 0.0
            )
            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "alignment_coef": 0.1,
                    "return_auc": seed + gain,
                    "final_return_last5_ckpt": seed + gain,
                    "win_rate_auc": seed / 10 + gain,
                    "final_win_rate_last5_ckpt": seed / 10 + gain,
                }
            )
    table = aggregate_table("map", "nps", 100, rows, (1, 2, 3, 4))
    cka = next(row for row in table if row["condition"] == "c_to_a_cka")
    assert len(table) == 3
    assert np.isclose(cka["delta_return_auc_vs_isolated_mean"], 0.2)
    assert np.isclose(cka["delta_final_return_last5_ckpt_vs_isolated_mean"], 0.2)
    assert cka["ci_method"] == BOOTSTRAP_CI_METHOD
    assert cka["uncertainty_unit"] == "training_seed"
    assert cka["confidence_level"] == 0.95
    assert cka["bootstrap_resamples"] == 4**4
    assert "return_auc_ci95_low" in cka
    assert "return_auc_ci95_high" in cka
    assert "delta_return_auc_vs_isolated_ci95_low" in cka
    assert "delta_return_auc_vs_isolated_ci95_high" in cka


def test_exact_bootstrap_enumerates_all_ordered_four_seed_resamples():
    assert bootstrap_resample_count(4) == 256
