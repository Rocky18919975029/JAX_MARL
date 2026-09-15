import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.h1_latent_distortion import REFERENCE_PROTOCOL


def test_latent_cli_writes_only_canonical_baseline_free_outputs(tmp_path):
    episodes, timesteps, agents, score_dim = 20, 4, 2, 3
    rng = np.random.default_rng(37)
    diagnostic = tmp_path / "diagnostic"
    diagnostic.mkdir()
    reward = rng.normal(size=(episodes, timesteps, agents)).astype(np.float32)
    mc_return = np.zeros_like(reward)
    for timestep in range(timesteps - 1, -1, -1):
        mc_return[:, timestep] = reward[:, timestep]
        if timestep + 1 < timesteps:
            mc_return[:, timestep] += 0.99 * mc_return[:, timestep + 1]
    arrays = {
        "active": np.ones((episodes, timesteps), dtype=bool),
        "alive": np.ones((episodes, timesteps, agents), dtype=bool),
        "diagnostic_episode_id": np.arange(episodes, dtype=np.int32),
        "reward": reward,
        "value": rng.normal(size=(episodes, timesteps, agents)).astype(np.float32),
        "global_done": np.pad(
            np.ones((episodes, 1), dtype=bool), ((0, 0), (timesteps - 1, 0))
        ),
        "mc_return": mc_return,
        "actor_score": rng.normal(size=(episodes, timesteps, agents, score_dim)).astype(
            np.float32
        ),
    }
    np.savez_compressed(diagnostic / "episodes_0000.npz", **arrays)
    metadata = {
        "shards": [{"path": "episodes_0000.npz", "episodes": episodes}],
        "run_id": "fixture",
        "run_name": "H1-fixture-nps-c_to_a_cka-seed1",
        "map_name": "fixture_map",
        "training_seed": 1,
        "checkpoint_env_step": 500_000,
        "checkpoint_nominal_env_step": 500_000,
        "num_agents": agents,
        "actor_parameter_sharing": False,
        "condition": "c_to_a_cka",
        "align_distance": "linear_cka",
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "training_rollout_steps": 2,
        "protocol_version": "h1-v1.0",
        "git_commit": "fixture-commit",
    }
    (diagnostic / "metadata.json").write_text(json.dumps(metadata))
    output = diagnostic / "latent_metrics.csv"
    script = Path(__file__).resolve().parents[1] / "scripts" / "h1_latent_distortion.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--diagnostics-dir",
            str(diagnostic),
            "--output-csv",
            str(output),
            "--fisher-ridge-absolute",
            "0.001",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads((diagnostic / "latent_summary.json").read_text())
    assert summary["reference_protocol"] == REFERENCE_PROTOCOL
    assert summary["reference_baseline"] == "none"
    assert summary["reference_has_bootstrap"] is False
    assert summary["critic_signal"] == "unnormalized_training_matched_gae"
    assert summary["fisher_ridge_absolute"] == 0.001
    assert summary["heldout_episodes"] == episodes
    with np.load(diagnostic / "reference_signals.npz") as signals:
        assert np.array_equal(signals["complete_mc_return"], mc_return[:, :, 0])
        assert "training_matched_gae" in signals
    with output.open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == agents
    assert {row["reference_protocol"] for row in rows} == {REFERENCE_PROTOCOL}
    with (diagnostic / "mc_convergence.csv").open(newline="") as file:
        convergence = list(csv.DictReader(file))
    assert {int(row["episode_budget"]) for row in convergence} == {5, 10, 20}
