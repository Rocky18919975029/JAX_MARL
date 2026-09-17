import matplotlib

matplotlib.use("Agg")

from scripts.analyze_rq1_core_smax import (
    Cell,
    RunSource,
    build_task_results,
    expected_cells,
    interquartile_mean,
    plot_metric,
)


def history(offset=0.0):
    return [
        {"env_step": 10, "returns": 0.1 + offset, "win_rate": 0.0 + offset},
        {"env_step": 20, "returns": 0.5 + offset, "win_rate": 0.4 + offset},
        {"env_step": 30, "returns": 0.9 + offset, "win_rate": 0.8 + offset},
    ]


def synthetic_complete_task(tmp_path, task):
    sources = {}
    histories = {}
    for cell in expected_cells((task,), (1, 2, 3, 4)):
        checkpoint = tmp_path / cell.actor_variant / cell.distance / cell.mode / str(cell.seed)
        source = RunSource(
            checkpoint=checkpoint,
            project="project",
            run_id=f"{cell.actor_variant}-{cell.distance}-{cell.mode}-{cell.seed}",
            run_name="run",
            alignment_coef=0.1,
            source="test",
        )
        sources[cell.key] = source
        histories[cell.key] = history(0.01 * cell.seed + (0.1 if cell.mode != "none" else 0.0))
    return sources, histories


def test_each_task_has_an_independent_nps_only_28_seed_cell_matrix():
    cells = expected_cells(("10m_vs_11m", "3s5z_vs_3s6z"), (1, 2, 3, 4))
    assert len(cells) == 56
    assert sum(cell.task == "10m_vs_11m" for cell in cells) == 28
    assert sum(cell.task == "3s5z_vs_3s6z" for cell in cells) == 28
    assert {cell.actor_variant for cell in cells} == {"nps"}


def test_four_seed_iqm_uses_the_middle_two_seeds():
    assert interquartile_mean([1.0, 2.0, 100.0, 3.0]) == 2.5


def test_incomplete_four_seed_method_is_left_blank(tmp_path):
    task = "10m_vs_11m"
    sources, histories = synthetic_complete_task(tmp_path, task)
    missing = Cell(task, "nps", "linear_cka", "joint", 4)
    sources.pop(missing.key)
    histories.pop(missing.key)
    seed_rows, _, table, curves, _ = build_task_results(
        task, (1, 2, 3, 4), sources, histories
    )
    selected = next(
        row
        for row in table
        if row["actor_parameterization"] == "nps"
        and row["align_distance"] == "linear_cka"
        and row["align_mode"] == "joint"
    )
    assert selected["data_status"] == "incomplete"
    assert selected["n_complete_seeds"] == 3
    assert selected["missing_seeds"] == "4"
    assert selected["final_return_mean"] == ""
    assert not any(
        row["actor_parameterization"] == "nps"
        and row["align_distance"] == "linear_cka"
        and row["align_mode"] == "joint"
        for row in curves
    )
    seed = next(
        row
        for row in seed_rows
        if row["actor_parameterization"] == "nps"
        and row["align_distance"] == "linear_cka"
        and row["align_mode"] == "joint"
        and row["seed"] == 4
    )
    assert seed["status"] == "missing"
    assert seed["final_return"] == ""


def test_complete_task_writes_separate_return_and_win_figures(tmp_path):
    task = "3s5z_vs_3s6z"
    sources, histories = synthetic_complete_task(tmp_path, task)
    _, _, table, curves, _ = build_task_results(
        task, (1, 2, 3, 4), sources, histories
    )
    assert len(table) == 7
    assert all(row["data_status"] == "complete" for row in table)
    return_stem = plot_metric(task, "returns", table, curves, tmp_path)
    win_stem = plot_metric(task, "win_rate", table, curves, tmp_path)
    for stem in (return_stem, win_stem):
        assert stem.with_suffix(".png").is_file()
        assert stem.with_suffix(".pdf").is_file()


def test_fully_missing_task_still_has_all_blank_method_rows():
    _, history_rows, table, curves, thresholds = build_task_results(
        "10m_vs_11m", (1, 2, 3, 4), {}, {}
    )
    assert not history_rows
    assert not curves
    assert len(table) == 7
    assert all(row["data_status"] == "incomplete" for row in table)
    assert all(row["final_return_mean"] == "" for row in table)
    assert thresholds == {"nps": None}


def test_missing_baseline_leaves_only_paired_fields_blank(tmp_path):
    task = "10m_vs_11m"
    sources, histories = synthetic_complete_task(tmp_path, task)
    for seed in (1, 2, 3, 4):
        baseline = Cell(task, "nps", "distance_free", "none", seed)
        sources.pop(baseline.key)
        histories.pop(baseline.key)
    _, _, table, _, _ = build_task_results(
        task, (1, 2, 3, 4), sources, histories
    )
    selected = next(
        row
        for row in table
        if row["actor_parameterization"] == "nps"
        and row["align_distance"] == "ln_mse"
        and row["align_mode"] == "c_to_a"
    )
    assert selected["data_status"] == "complete"
    assert selected["final_return_mean"] != ""
    assert selected["delta_final_return_mean"] == ""
