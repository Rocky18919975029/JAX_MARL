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
