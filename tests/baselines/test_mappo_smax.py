import os
import subprocess
import sys

import pytest


def run_script(script_path, *args):
    result = subprocess.run(
        [sys.executable, script_path, *args], capture_output=True, text=True
    )
    return result


def test_script_with_arguments():
    for package in ("jax", "flax", "distrax", "optax", "hydra", "wandb"):
        pytest.importorskip(package)
    script_path = os.path.join("baselines/MAPPO/mappo_rnn_smax.py")
    result = run_script(
        script_path,
        "TOTAL_TIMESTEPS=512", "NUM_ENVS=4", "NUM_STEPS=32",
        "NUM_MINIBATCHES=2", "UPDATE_EPOCHS=1", "WANDB_MODE=disabled",
    )
    assert result.returncode == 0, result.stderr[-4000:]
