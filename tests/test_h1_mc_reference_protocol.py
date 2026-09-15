import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.h1_latent_distortion import (
    CONTROL_VARIATE_PROTOCOL,
    REFERENCE_PROTOCOL,
)


def test_latent_cli_uses_raw_mc_primary_and_separate_crossfit_sensitivity(tmp_path):
    episodes, timesteps, agents, score_dim = 20, 3, 2, 2
    rng = np.random.default_rng(37)
    diagnostic = tmp_path / "diagnostic"
    diagnostic.mkdir()
    mc_return = rng.normal(size=(episodes, timesteps, agents)).astype(np.float32)
    arrays = {
        "active": np.ones((episodes, timesteps), dtype=bool),
        "alive": np.ones((episodes, timesteps, agents), dtype=bool),
        "diagnostic_episode_id": np.arange(episodes, dtype=np.int32),
        "world_state": rng.normal(size=(episodes, timesteps, agents, 4)).astype(
            np.float32
        ),
        "mc_return": mc_return,
        "actor_score": rng.normal(size=(episodes, timesteps, agents, score_dim)).astype(
            np.float32
        ),
        "gae_raw": rng.normal(size=(episodes, timesteps, agents)).astype(np.float32),
        "state_unit_types": np.zeros((episodes, timesteps, agents), dtype=np.int32),
    }
    np.savez_compressed(diagnostic / "episodes_0000.npz", **arrays)
    metadata = {
        "shards": [{"path": "episodes_0000.npz", "episodes": episodes}],
        "run_id": "fixture",
        "run_name": "H1-fixture-nps-c_to_a_cka-seed1",
        "map_name": "fixture_map",
        "training_seed": 1,
        "checkpoint_env_step": 500_000,
        "num_agents": agents,
        "actor_parameter_sharing": False,
        "condition": "c_to_a_cka",
        "protocol_version": "h1-v1.0",
        "git_commit": "fixture-commit",
    }
    (diagnostic / "metadata.json").write_text(json.dumps(metadata))
    output = diagnostic / "compatibility_mc_metrics.csv"
    script = Path(__file__).resolve().parents[1] / "scripts" / "h1_latent_distortion.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--diagnostics-dir",
            str(diagnostic),
            "--output-csv",
            str(output),
            "--reference-width",
            "4",
            "--minibatch-samples-per-agent",
            "32",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads((diagnostic / "latent_distortion_mc_summary.json").read_text())
    assert summary["reference_protocol"] == REFERENCE_PROTOCOL
    assert summary["reference_signal"] == "complete_discounted_mc_return"
    assert summary["reference_baseline"] == "none"
    assert summary["reference_has_bootstrap"] is False
    assert summary["mc_convergence_episode_budgets"] == [5, 10, 20]
    assert summary["reference_base_episode_budget_m"] == 5
    assert (
        summary["cross_fitted_state_baseline_sensitivity"]["reference_protocol"]
        == CONTROL_VARIATE_PROTOCOL
    )
    with np.load(diagnostic / "reference_signals_mc.npz") as signals:
        assert np.array_equal(signals["mc_return"], mc_return[:, :, 0])
        assert not np.array_equal(
            signals["mc_return"], signals["cross_fitted_control_variate_signal"]
        )
    with output.open(newline="") as file:
        primary_rows = list(csv.DictReader(file))
    assert {row["reference_protocol"] for row in primary_rows} == {REFERENCE_PROTOCOL}
    with (diagnostic / "compatibility_crossfit_sensitivity_metrics.csv").open(
        newline=""
    ) as file:
        sensitivity_rows = list(csv.DictReader(file))
    assert {row["reference_protocol"] for row in sensitivity_rows} == {
        CONTROL_VARIATE_PROTOCOL
    }
    with (diagnostic / "mc_reference_convergence.csv").open(newline="") as file:
        convergence_rows = list(csv.DictReader(file))
    assert {int(row["episode_budget"]) for row in convergence_rows} == {5, 10, 20}
    assert {row["episode_budget_label"] for row in convergence_rows} == {
        "M",
        "2M",
        "4M",
    }
    assert {row["run_id"] for row in convergence_rows} == {"fixture"}
