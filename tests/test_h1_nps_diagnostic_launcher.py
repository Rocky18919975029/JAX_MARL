import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_h1_nps_diagnostics import phase_plan, verify_matched_frozen_configs


def test_canonical_pipeline_reuses_collection_and_has_no_legacy_phases(tmp_path):
    args = SimpleNamespace(
        mse_root=tmp_path / "mse",
        cka_root=tmp_path / "cka",
        analysis_root=tmp_path / "analysis",
        gpus="0,1,2,3",
        max_runs_per_gpu=1,
        fisher_ridge_absolute=1e-3,
        anchors=256,
        continuations=32,
        bellman_heads=32,
    )
    phases = phase_plan(args)
    names = [phase.name for phase in phases]
    assert names == [
        "ln_mse-latent",
        "linear_cka-latent",
        "ln_mse-decision",
        "linear_cka-decision",
        "ln_mse-bellman",
        "linear_cka-bellman",
        "analysis",
        "figures",
    ]
    command_text = "\n".join(" ".join(phase.command) for phase in phases)
    assert "--stages collect" not in command_text
    assert "deterministic" not in command_text
    assert "--delta-dec" not in command_text
    assert "--delta-bell" not in command_text
    assert str(Path(args.mse_root).resolve()) in command_text
    assert str(Path(args.cka_root).resolve()) in command_text


def test_frozen_training_configs_are_matched_before_recompute(tmp_path):
    roots = (tmp_path / "mse", tmp_path / "cka")
    for root in roots:
        directory = root / "protocol"
        directory.mkdir(parents=True)
        (directory / "frozen_training_config.json").write_text(
            json.dumps(
                {"training_config": {"MATCHED_COMPARISON": True, "NUM_STEPS": 128}}
            )
        )
    assert len(verify_matched_frozen_configs(*roots)) == 64
    (roots[1] / "protocol" / "frozen_training_config.json").write_text(
        json.dumps({"training_config": {"MATCHED_COMPARISON": True, "NUM_STEPS": 64}})
    )
    with pytest.raises(RuntimeError, match="not matched"):
        verify_matched_frozen_configs(*roots)
