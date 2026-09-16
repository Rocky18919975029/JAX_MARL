import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_mabrax_alignment.py"
CALIBRATOR = ROOT / "scripts" / "calibrate_mabrax_cka.py"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_alignment_distances_have_expected_invariances():
    jnp = pytest.importorskip("jax.numpy")
    from baselines.MAPPO.alignment_utils import (
        directional_subspace_containment,
        layernorm_mse_distance,
        linear_cka_distance,
    )

    source = jnp.asarray(
        [[1.0, -1.0, 0.5], [0.2, 1.3, -0.4], [-0.7, 0.1, 1.2], [1.5, 0.4, -0.9]]
    )
    rotation = jnp.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    rotated = source @ rotation
    mask = jnp.ones((source.shape[0],))
    assert float(layernorm_mse_distance(source, source, mask)) < 1e-7
    assert float(linear_cka_distance(source, source, mask)) < 1e-6
    assert float(layernorm_mse_distance(source, rotated, mask)) > 0.1
    assert float(linear_cka_distance(source, rotated, mask)) < 1e-5
    assert float(directional_subspace_containment(source, rotated, mask)[0]) < 5e-3


def test_containment_is_directional_and_masked():
    jnp = pytest.importorskip("jax.numpy")
    from baselines.MAPPO.alignment_utils import directional_subspace_containment

    grid = jnp.linspace(-2.0, 2.0, 128)
    full = jnp.stack((grid, grid**2, jnp.sin(grid)), axis=-1)
    rank_two = full.at[:, 2].set(0.0)
    mask = jnp.ones((128,))
    contained = directional_subspace_containment(rank_two, full, mask)[0]
    not_contained = directional_subspace_containment(full, rank_two, mask)[0]
    assert float(contained) < 5e-3
    assert float(not_contained) > 0.2

    changed = full.at[-8:].set(1e6)
    partial_mask = mask.at[-8:].set(0.0)
    np.testing.assert_allclose(
        directional_subspace_containment(full, rank_two, partial_mask)[0],
        directional_subspace_containment(changed, rank_two, partial_mask)[0],
        atol=1e-6,
    )


def test_representation_distance_never_pools_nps_agents():
    jnp = pytest.importorskip("jax.numpy")
    from baselines.MAPPO.alignment_utils import representation_distance

    rng = np.random.default_rng(7)
    source = jnp.asarray(rng.normal(size=(3, 2, 5, 4)), dtype=jnp.float32)
    target = source.at[:, 1].set(
        source[:, 1]
        @ jnp.asarray(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )
    )
    mask = jnp.ones(source.shape[:-1])
    result = representation_distance(source, target, mask, "linear_cka", agent_axis=1)
    assert float(result) < 1e-5


def test_formal_matrix_is_56_unique_runs():
    runner = load_module(RUNNER, "run_mabrax_alignment_test")
    tasks = runner.task_matrix(("ps", "nps"), (1, 2, 3, 4), 0.25)
    assert len(tasks) == 56
    assert len({task.run_name for task in tasks}) == 56
    assert sum(task.mode == "none" for task in tasks) == 8


def test_containment_can_be_selected_without_changing_default_matrix():
    runner = load_module(RUNNER, "run_mabrax_alignment_dsc_test")
    tasks = runner.task_matrix(
        ("nps",),
        (1, 2, 3, 4),
        cka_coefficient=None,
        distances=("containment",),
        containment_coefficient=0.3,
    )
    assert len(tasks) == 16
    aligned = [task for task in tasks if task.mode != "none"]
    assert all(task.distance == "containment" for task in aligned)
    assert all("_dsc-" in task.run_name for task in aligned)


def test_calibration_uses_all_four_direction_recipient_cells():
    calibrator = load_module(CALIBRATOR, "calibrate_mabrax_cka_test")
    cells = [
        {
            "ln_mse_cross_to_rl_ratio": value,
            "linear_cka_cross_to_rl_ratio": value * 2,
        }
        for value in (1.0, 2.0, 3.0, 4.0)
    ]
    assert np.isclose(calibrator.pooled_rms_coefficient(cells), 0.05)
