#!/usr/bin/env python3
"""Audit and recover successful SMAX runs rejected only for a Git-commit drift.

The launcher freezes a manifest commit when it starts. If the checkout is
pulled before later workers start, their checkpoints record the newer commit.
This tool accepts that difference only when Git proves that no training source
changed. Original failed statuses are preserved before any status is updated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

try:
    from scripts import smax_four_method as control
    from scripts.run_smax_first_four_panel_tuned import TASKS
except ModuleNotFoundError:  # Direct execution from scripts/.
    import smax_four_method as control
    from run_smax_first_four_panel_tuned import TASKS


ALLOWED_NONTRAINING_PREFIXES = ("docs/", "tests/", "experiments/harl_dexhands/")
ALLOWED_NONTRAINING_FILES = frozenset({"scripts/report_smax_first_four_panel_10seed.py"})


def _git(*args: str, repository: Path = control.REPO) -> bytes:
    result = subprocess.run(
        ["git", *args], cwd=repository, check=True, capture_output=True,
    )
    return result.stdout


def verify_commit_equivalence(
    manifest: dict, checkpoint_commit: str, *, repository: Path = control.REPO,
) -> list[str]:
    """Require a descendant commit with identical frozen training source bytes."""
    manifest_commit = manifest["git_commit"]
    if checkpoint_commit == manifest_commit:
        raise RuntimeError("No commit mismatch to reconcile")
    if not checkpoint_commit or checkpoint_commit == "unknown":
        raise RuntimeError("Checkpoint Git commit is unavailable")
    _git("cat-file", "-e", manifest_commit + "^{commit}", repository=repository)
    _git("cat-file", "-e", checkpoint_commit + "^{commit}", repository=repository)
    _git("merge-base", "--is-ancestor", manifest_commit, checkpoint_commit,
         repository=repository)
    diff = _git(
        "diff", "--name-only", "--no-renames", manifest_commit, checkpoint_commit,
        repository=repository,
    ).decode().splitlines()
    disallowed = [
        path for path in diff
        if path not in ALLOWED_NONTRAINING_FILES
        and not path.startswith(ALLOWED_NONTRAINING_PREFIXES)
    ]
    if disallowed:
        raise RuntimeError(f"Tracked changes may affect SMAX training: {disallowed}")
    sources = manifest.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise RuntimeError("Manifest has no frozen training-source hashes")
    for path, expected_sha in sources.items():
        for commit in (manifest_commit, checkpoint_commit):
            source = _git("show", f"{commit}:{path}", repository=repository)
            if hashlib.sha256(source).hexdigest() != expected_sha:
                raise RuntimeError(f"Frozen source differs at {commit}:{path}")
    return diff


def audit_candidate(root: Path, manifest: dict, row: dict) -> dict | None:
    name = row["name"]
    status_path = root / "status" / f"{name}.json"
    if not status_path.is_file():
        raise RuntimeError(f"Missing status for {name}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") == "completed":
        return None
    if (status.get("status") != "failed" or status.get("exit_code") != 0
            or status.get("artifact_error") != "checkpoint protocol/code mismatch"):
        raise RuntimeError(f"Not a code-identity-only successful run: {name}: {status}")
    run = control.Run(**{key: row[key] for key in control.Run.__dataclass_fields__})
    final = control.checkpoint_dir(root, manifest["project"], run)
    metadata = json.loads((final / "metadata.json").read_text(encoding="utf-8"))
    config = json.loads((final / "config.json").read_text(encoding="utf-8"))
    actual_commit = metadata.get("git_commit")
    if actual_commit != config.get("GIT_COMMIT"):
        raise RuntimeError(f"Checkpoint metadata/config Git commit mismatch: {name}")
    changed = verify_commit_equivalence(manifest, actual_commit)
    issue = control.validate_artifacts(root, manifest["project"], run, actual_commit)
    if issue is not None:
        raise RuntimeError(f"Other artifact failure remains for {name}: {issue}")
    for checkpoint in final.parent.iterdir():
        if not (checkpoint / "model.safetensors").is_file():
            continue
        saved = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        saved_meta = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
        if (saved.get("GIT_COMMIT") != actual_commit
                or saved.get("PROTOCOL_VERSION") != manifest["protocol"]
                or saved_meta.get("git_commit") != actual_commit):
            raise RuntimeError(f"Inconsistent saved checkpoint identity: {checkpoint}")
    return {
        "run_name": name, "status_path": status_path,
        "manifest_git_commit": manifest["git_commit"],
        "checkpoint_git_commit": actual_commit,
        "nontraining_changes": changed,
    }


def reconcile(collection_root: Path, *, apply: bool) -> list[dict]:
    candidates = []
    for task in TASKS:
        root = collection_root / task / "extension_6seed"
        manifest = json.loads((root / "experiment_manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("protocol") != control.PROTOCOL
                or manifest.get("map_name") != task
                or manifest.get("seed_start") != 5
                or manifest.get("seed_count") != 6
                or manifest.get("methods") != ["none", "arec"]
                or len(manifest.get("runs", [])) != 12):
            raise RuntimeError(f"Not the frozen six-seed extension: {root}")
        for row in manifest["runs"]:
            candidate = audit_candidate(root, manifest, row)
            if candidate is not None:
                candidate["root"] = root
                candidates.append(candidate)
    for item in candidates:
        print(
            f"AUDITED {item['run_name']} "
            f"{item['manifest_git_commit'][:10]} -> {item['checkpoint_git_commit'][:10]} "
            f"unrelated_files={len(item['nontraining_changes'])}",
            flush=True,
        )
    print(f"Audited {len(candidates)} successful runs with Git-identity drift", flush=True)
    if not apply:
        print("Dry run: no status files changed; add --apply after reviewing the audit", flush=True)
        return candidates
    for item in candidates:
        original = item["status_path"]
        backup = item["root"] / "reconciliation" / "original_failed_status" / original.name
        if backup.exists():
            raise RuntimeError(f"Refusing to overwrite previous status backup: {backup}")
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, backup)
        status = json.loads(original.read_text(encoding="utf-8"))
        status.update({
            "status": "completed", "artifact_error": None,
            "validated_git_commit": item["checkpoint_git_commit"],
            "commit_reconciliation": {
                "manifest_git_commit": item["manifest_git_commit"],
                "checkpoint_git_commit": item["checkpoint_git_commit"],
                "nontraining_changes": item["nontraining_changes"],
                "original_status_backup": str(backup),
                "reason": "process exit 0; only post-run manifest/checkpoint Git commit check failed",
            },
        })
        control.atomic_json(original, status)
    print(f"Reconciled {len(candidates)} statuses; original failures remain in reconciliation/original_failed_status", flush=True)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    reconcile(args.collection_root.expanduser().resolve(), apply=args.apply)


if __name__ == "__main__":
    main()
