"""Commit-drift recovery must preserve evidence and reject training-code changes."""

import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

from scripts import reconcile_smax_first_four_panel_commits as recovery
from scripts import smax_four_method as control


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_only_unrelated_commit_changes_are_accepted(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "test@example.org")
    _git(repository, "config", "user.name", "Test")
    training = repository / "scripts/train.py"
    training.parent.mkdir()
    training.write_text("training = 1\n")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "original training")
    original = _git(repository, "rev-parse", "HEAD")
    manifest = {
        "git_commit": original,
        "source_sha256": {
            "scripts/train.py": hashlib.sha256(training.read_bytes()).hexdigest(),
        },
    }
    docs = repository / "docs/report.md"
    docs.parent.mkdir()
    docs.write_text("report\n")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "add report")
    unrelated = _git(repository, "rev-parse", "HEAD")
    assert recovery.verify_commit_equivalence(
        manifest, unrelated, repository=repository,
    ) == ["docs/report.md"]
    training.write_text("training = 2\n")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "change training")
    changed = _git(repository, "rev-parse", "HEAD")
    with pytest.raises(RuntimeError, match="may affect SMAX training"):
        recovery.verify_commit_equivalence(manifest, changed, repository=repository)


def test_recovery_preserves_original_failed_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = tmp_path / "collection"
    run = control.Run(
        map_name=recovery.TASKS[0], seed=5, method="none",
        timesteps=20, lr=0.002, epochs=4, num_envs=1,
        num_minibatches=1, num_steps=4,
    )
    final = tmp_path / "final"
    _json(final / "metadata.json", {"git_commit": "new"})
    _json(final / "config.json", {
        "GIT_COMMIT": "new", "PROTOCOL_VERSION": control.PROTOCOL,
    })
    (final / "model.safetensors").write_bytes(b"model")
    monkeypatch.setattr(control, "checkpoint_dir", lambda *_args: final)
    monkeypatch.setattr(control, "validate_artifacts", lambda *_args: None)
    monkeypatch.setattr(recovery, "verify_commit_equivalence", lambda *_args: ["docs/report.md"])
    for task in recovery.TASKS:
        root = collection / task / "extension_6seed"
        runs = [dict(asdict(run), name=run.name)] if task == recovery.TASKS[0] else []
        runs += [{"name": f"synthetic-{task}-{i}"} for i in range(12 - len(runs))]
        _json(root / "experiment_manifest.json", {
            "protocol": control.PROTOCOL, "map_name": task,
            "seed_start": 5, "seed_count": 6,
            "methods": ["none", "arec"], "project": "test",
            "git_commit": "old", "runs": runs,
        })
        for row in runs:
            _json(root / "status" / f"{row['name']}.json", {
                "status": "failed" if row["name"] == run.name else "completed",
                "exit_code": 0,
                "artifact_error": (
                    "checkpoint protocol/code mismatch" if row["name"] == run.name else None
                ),
            })
    status_path = collection / recovery.TASKS[0] / "extension_6seed/status" / f"{run.name}.json"
    assert len(recovery.reconcile(collection, apply=False)) == 1
    assert json.loads(status_path.read_text())["status"] == "failed"
    assert len(recovery.reconcile(collection, apply=True)) == 1
    updated = json.loads(status_path.read_text())
    assert updated["status"] == "completed"
    assert updated["validated_git_commit"] == "new"
    backup = Path(updated["commit_reconciliation"]["original_status_backup"])
    assert json.loads(backup.read_text())["status"] == "failed"
    assert recovery.reconcile(collection, apply=False) == []
