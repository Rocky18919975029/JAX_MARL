"""Selected historical outputs remain untouched while the ten-seed view grows."""

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import smax_four_method as control
from scripts import organize_smax_first_four_panel as organizer
from scripts import reconcile_smax_first_four_panel_commits as commit_audit
from scripts.organize_smax_first_four_panel import apply_plan, extension_plan
from scripts.run_smax_first_four_panel_tuned import FIGURE_ID, launch_args, load_pair


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_collection_links_are_idempotent_and_preserve_source(tmp_path: Path) -> None:
    source = tmp_path / "original.jsonl"
    source.write_text('{"env_step": 1}\n', encoding="utf-8")
    entry = {
        "task": "10m_vs_11m", "method": "none", "seed": 1,
        "origin": "first_four_panel_v1", "run_name": "original-run",
        "source_root": str(tmp_path), "outputs": {"metrics.jsonl": str(source)},
    }
    root = tmp_path / "collection"
    apply_plan(root, [entry], {"10m_vs_11m": {"commit": "old"}})
    apply_plan(root, [entry], {"10m_vs_11m": {"commit": "old"}})
    linked = root / "10m_vs_11m/runs/none/seed_01/metrics.jsonl"
    assert linked.is_symlink() and linked.resolve() == source
    assert source.read_text() == '{"env_step": 1}\n'
    new_entry = dict(entry, seed=5, origin="six_seed_extension", run_name="new-run")
    assert apply_plan(root, [entry, new_entry], {"10m_vs_11m": {"commit": "old"}}) == 2
    assert apply_plan(root, [entry], {"10m_vs_11m": {"commit": "old"}}) == 2
    assert len(json.loads((root / "collection_index.json").read_text())["runs"]) == 2
    other = tmp_path / "different.jsonl"
    other.write_text("other", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source differs"):
        apply_plan(root, [dict(entry, outputs={"metrics.jsonl": str(other)})],
                   {"10m_vs_11m": {"commit": "old"}})


def test_extension_manifest_must_match_frozen_six_seed_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = "10m_vs_11m"
    root = tmp_path / "collection" / task / "extension_6seed"
    none, arec, paths = load_pair(task)
    args = SimpleNamespace(
        task=task, seed_start=5, seed_count=6, run_root=root,
        gpus=("0",), max_runs_per_gpu=1, project="test",
        wandb_mode="disabled", dry_run=True,
    )
    runs = control.make_grid(launch_args(args, none, arec, paths, None))
    manifest = {
        "protocol": control.PROTOCOL, "map_name": task,
        "seed_start": 5, "seed_count": 6, "methods": ["none", "arec"],
        "gpus": ["0"], "max_runs_per_gpu": 1, "project": "test",
        "wandb_mode": "disabled", "git_commit": "historical-test",
        "tuned_selection": {"figure_id": FIGURE_ID},
        "runs": [dict(asdict(run), name=run.name) for run in runs],
    }
    _write(root / "experiment_manifest.json", manifest)
    first = runs[0]
    _write(root / "status" / f"{first.name}.json", {"status": "completed"})
    (root / "metrics" / f"{first.name}.jsonl").parent.mkdir(parents=True)
    (root / "metrics" / f"{first.name}.jsonl").write_text("{}\n")
    (root / "logs" / f"{first.name}.log").parent.mkdir(parents=True)
    (root / "logs" / f"{first.name}.log").write_text("done\n")
    checkpoint = control.checkpoint_dir(root, "test", first).parent
    checkpoint.mkdir(parents=True)
    monkeypatch.setattr(control, "validate_artifacts", lambda *_args: None)
    entries, counts = extension_plan(tmp_path / "collection")
    assert counts[task] == 1 and len(entries) == 1
    assert entries[0]["seed"] == 5 and entries[0]["method"] == "none"
    assert entries[0]["outputs"]["checkpoints"] == str(checkpoint)

    backup = root / "reconciliation" / "original_failed_status" / f"{first.name}.json"
    _write(backup, {"status": "failed"})
    _write(root / "status" / f"{first.name}.json", {
        "status": "completed", "validated_git_commit": "later-test",
        "commit_reconciliation": {
            "manifest_git_commit": "historical-test",
            "checkpoint_git_commit": "later-test",
            "nontraining_changes": ["docs/report.md"],
            "original_status_backup": str(backup),
        },
    })
    monkeypatch.setattr(commit_audit, "verify_commit_equivalence",
                        lambda *_args: ["docs/report.md"])
    entries, counts = extension_plan(tmp_path / "collection")
    assert counts[task] == 1 and entries[0]["training_git_commit"] == "later-test"

    manifest["runs"][0]["lr"] = 0.123
    _write(root / "experiment_manifest.json", manifest)
    with pytest.raises(RuntimeError, match="differs from frozen YAML"):
        extension_plan(tmp_path / "collection")


def test_historical_plan_omits_unselected_sweep_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = "10m_vs_11m"
    source = tmp_path / "original_sweep"
    none, arec, _ = load_pair(task)
    runs = []
    for method in ("none", "arec"):
        for seed in range(1, 5):
            name = f"selected-{method}-seed{seed}"
            runs.append({
                "run_name": name, "condition": ("none" if method == "none"
                                                   else "actor_score_recovery"),
                "seed": seed,
                "coef": (0.0 if method == "none" else arec["config"]["ACTOR_SCORE_RECOVERY_COEF"]),
                "q_steps": (0 if method == "none" else arec["config"]["ACTOR_SCORE_RECOVERY_Q_STEPS"]),
                "q_learning_rate": 0.001, "fisher_ridge": 0.001,
            })
            _write(source / "status" / f"{name}.json", {"status": "completed"})
            (source / "metrics" / f"{name}.jsonl").parent.mkdir(parents=True, exist_ok=True)
            (source / "metrics" / f"{name}.jsonl").write_text("{}\n")
            final = source / "checkpoints" / "project" / f"{name}-id" / "final"
            _write(final / "config.json", {"GIT_COMMIT": "old-code"})
            (final / "model.safetensors").write_bytes(b"weights")
    runs.append({
        "run_name": "unselected-arec", "condition": "actor_score_recovery",
        "seed": 1, "coef": 0.123, "q_steps": 8,
        "q_learning_rate": 0.001, "fisher_ridge": 0.001,
    })
    _write(source / "experiment_manifest.json", {"runs": runs})
    monkeypatch.setattr(organizer, "TASKS", (task,))
    monkeypatch.setattr(organizer, "verify_historical",
                        lambda *_args: {"historical_source_root": str(source)})
    entries, _ = organizer.historical_plan(tmp_path)
    assert len(entries) == 8
    assert {row["seed"] for row in entries} == {1, 2, 3, 4}
    assert {row["method"] for row in entries} == {"none", "arec"}
    assert all(row["training_git_commit"] == "old-code" for row in entries)
