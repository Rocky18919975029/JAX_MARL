import json

import numpy as np
import pytest

from scripts.h1_protocol import MANUAL_REFERENCE
from scripts.monitor_smax_agent_scaling import parse_latest_step, reused_run_names
from scripts.run_smax_agent_scaling import (
    AGENT_COUNTS,
    FAMILY_MAPS,
    TOTAL_TIMESTEPS,
    reusable_tasks,
    task_matrix,
)


def test_scaling_maps_match_agent_counts_and_enemy_advantage():
    pytest.importorskip("jax")
    from jaxmarl.environments.smax import map_name_to_scenario

    for family, maps in FAMILY_MAPS.items():
        for agent_count, map_name in maps.items():
            scenario = map_name_to_scenario(map_name)
            unit_types = np.asarray(scenario.unit_types)
            allies = unit_types[:agent_count]
            enemies = unit_types[agent_count:]
            assert scenario.num_allies == agent_count
            assert scenario.num_enemies == agent_count + 1
            assert unit_types.size == 2 * agent_count + 1
            if family == "homogeneous":
                assert np.all(unit_types == 0)
            else:
                assert set(allies.tolist()) == {2, 3}
                assert np.count_nonzero(enemies == 2) == np.count_nonzero(allies == 2)
                assert np.count_nonzero(enemies == 3) == (
                    np.count_nonzero(allies == 3) + 1
                )


def test_full_scaling_matrix_has_120_unique_matched_runs():
    tasks = task_matrix()
    assert AGENT_COUNTS == (3, 5, 8, 10, 15)
    assert len(tasks) == 2 * 5 * 4 * 3 == 120
    assert len({task.run_name for task in tasks}) == len(tasks)
    assert {task.seed for task in tasks} == {1, 2, 3, 4}
    assert {task.family for task in tasks} == {"homogeneous", "heterogeneous"}
    assert {task.display_condition for task in tasks} == {
        "none",
        "c_to_a_mse",
        "c_to_a_cka",
    }
    assert TOTAL_TIMESTEPS == 20_000_000


def test_task_order_advances_from_small_to_large_agent_counts():
    tasks = task_matrix(seeds=(1,))
    seen = []
    for task in tasks:
        if task.agent_count not in seen:
            seen.append(task.agent_count)
    assert tuple(seen) == AGENT_COUNTS


def test_exact_20m_anchor_is_reused_but_10m_anchor_is_not(tmp_path):
    tasks = task_matrix(
        families=("heterogeneous",),
        agent_counts=(8,),
        seeds=(1,),
    )
    target = next(task for task in tasks if task.display_condition == "c_to_a_cka")
    final = tmp_path / "checkpoints" / "anchor" / "final"
    final.mkdir(parents=True)
    metadata = {
        "map_name": target.map_name,
        "seed": target.seed,
        "actor_parameter_sharing": False,
        "align_mode": target.align_mode,
        "align_distance": target.align_distance,
    }
    frozen = dict(MANUAL_REFERENCE)
    frozen["TOTAL_TIMESTEPS"] = TOTAL_TIMESTEPS
    config = dict(frozen)
    config["ALIGNMENT_COEF"] = target.alignment_coef
    (final / "metadata.json").write_text(json.dumps(metadata))
    (final / "config.json").write_text(json.dumps(config))
    (final / "model.safetensors").write_bytes(b"model")
    assert reusable_tasks((tmp_path,), tasks, frozen) == {target.key: final.resolve()}

    config["TOTAL_TIMESTEPS"] = 10_000_000
    (final / "config.json").write_text(json.dumps(config))
    assert reusable_tasks((tmp_path,), tasks, frozen) == {}


def test_monitor_parses_progress_and_reused_anchor(tmp_path):
    assert parse_latest_step("steps=1.25e+07") == 12_500_000
    tasks = task_matrix(
        families=("heterogeneous",), agent_counts=(8,), seeds=(1,)
    )
    target = tasks[0]
    (tmp_path / "reused_runs.json").write_text(
        json.dumps({"runs": {"|".join(map(str, target.key)): "/checkpoint"}})
    )
    assert reused_run_names(tmp_path, tasks) == {target.run_name}
