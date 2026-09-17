from scripts.monitor_smax_3s5z_nps_20m import parse_latest_step
from scripts.run_smax_3s5z_nps_20m import TOTAL_TIMESTEPS, task_matrix


def test_focused_matrix_has_exactly_twelve_matched_runs():
    tasks = task_matrix()
    assert len(tasks) == 12
    assert {task.seed for task in tasks} == {1, 2, 3, 4}
    assert {task.display_condition for task in tasks} == {
        "none",
        "c_to_a_mse",
        "c_to_a_cka",
    }
    assert {task.experiment_condition for task in tasks} == {
        "none",
        "c_to_a",
        "c_to_a_cka",
    }
    assert TOTAL_TIMESTEPS == 20_000_000


def test_mse_display_name_is_not_used_as_training_condition():
    task = next(task for task in task_matrix() if task.display_condition == "c_to_a_mse")
    assert task.run_name.endswith(f"c_to_a_mse-seed{task.seed}")
    assert task.align_mode == "c_to_a"
    assert task.align_distance == "ln_mse"
    assert task.experiment_condition == "c_to_a"


def test_progress_parser_supports_scientific_and_integer_steps():
    assert parse_latest_step("steps=3.28e+04") == 32_800
    assert parse_latest_step("steps=32,768\rsteps=1.25e+06") == 1_250_000
