"""End-to-end synthetic checks for the frozen ten-seed four-panel report."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import organize_smax_first_four_panel as organizer
from scripts import report_smax_first_four_panel_10seed as report


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class TenSeedReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.collection = self.root / "collection"
        self.output = self.collection / "report_10seed"
        self.profiles = report.frozen_profiles()
        entries = []
        for task in report.TASKS:
            for method in report.METHODS:
                self.profiles[task][method] = {
                    **self.profiles[task][method],
                    "TOTAL_TIMESTEPS": 20, "NUM_ENVS": 1, "NUM_STEPS": 4,
                    "LR": 0.002, "UPDATE_EPOCHS": 4,
                    "ACTOR_SCORE_RECOVERY_COEF": 0 if method == "none" else 0.0001,
                    "ACTOR_SCORE_RECOVERY_Q_STEPS": 4,
                    "ACTOR_SCORE_RECOVERY_Q_LR": 0.001,
                    "ACTOR_SCORE_RECOVERY_FISHER_RIDGE": 0.001,
                }
                for seed in report.SEEDS:
                    origin = "first_four_panel_v1" if seed <= 4 else "six_seed_extension"
                    name = f"{task}-{method}-seed{seed}"
                    source = self.root / "sources" / name
                    condition = (
                        report.CONDITION[method] if seed <= 4 else method
                    )
                    source_run = {"seed": seed}
                    if seed <= 4:
                        source_run.update(run_name=name, condition=condition, map_name=task)
                        manifest = {"runs": [source_run]}
                    else:
                        source_run.update(name=name, method=method)
                        manifest = {"map_name": task, "effective_timesteps": 20,
                                    "runs": [source_run]}
                    manifest_path = source / "experiment_manifest.json"
                    status_path = source / "status.json"
                    metrics_path = source / "metrics.jsonl"
                    checkpoint_parent = source / "checkpoints"
                    write_json(manifest_path, manifest)
                    write_json(status_path, {"status": "completed"})
                    reward = 1.0 + 0.01 * seed + (0.2 if method == "arec" else 0.0)
                    metrics_path.write_text("\n".join(
                        json.dumps({"env_step": step, "returns": reward})
                        for step in (4, 8, 12, 16, 20)
                    ) + "\n", encoding="utf-8")
                    for step in (4, 8, 12, 16, 20):
                        checkpoint = checkpoint_parent / (
                            "final" if step == 20 else f"step_{step:012d}"
                        )
                        write_json(checkpoint / "metadata.json", {"nominal_env_step": step})
                        write_json(checkpoint / "config.json", {
                            **self.profiles[task][method],
                            "SEED": seed, "EXPERIMENT_CONDITION": condition,
                            "GIT_COMMIT": "test-commit",
                        })
                        (checkpoint / "model.safetensors").write_bytes(b"test")
                    entries.append({
                        "task": task, "method": method, "seed": seed,
                        "origin": origin, "run_name": name,
                        "source_root": str(source),
                        "training_git_commit": "test-commit",
                        "outputs": {
                            "source_manifest.json": str(manifest_path),
                            "status.json": str(status_path),
                            "metrics.jsonl": str(metrics_path),
                            "checkpoints": str(checkpoint_parent),
                        },
                    })
        organizer.apply_plan(self.collection, entries, {"test": "synthetic"})

    def test_preflight_requires_all_eighty_selected_runs(self) -> None:
        with patch.object(report, "frozen_profiles", return_value=self.profiles):
            _, entries, jobs, (selections, checkpoint_steps) = report.prepare(
                self.collection, self.output
            )
        self.assertEqual(len(entries), 80)
        self.assertEqual(len(jobs), 400)
        self.assertEqual(selections["10m_vs_11m"]["seeds"], tuple(range(1, 11)))
        self.assertEqual(checkpoint_steps["10m_vs_11m", "none", 1], (4, 8, 12, 16, 20))
        index_path = self.collection / "collection_index.json"
        index = report.read_json(index_path)
        index["runs"].pop()
        write_json(index_path, index)
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            report.load_collection(self.collection)

    def test_preflight_rejects_unmatched_config_and_training_grid(self) -> None:
        with patch.object(report, "frozen_profiles", return_value=self.profiles):
            _, entries = report.load_collection(self.collection)
            entry = entries["10m_vs_11m", "arec", 10]
            checkpoint = Path(entry["outputs"]["checkpoints"]) / "final" / "config.json"
            config = report.read_json(checkpoint)
            config["ACTOR_SCORE_RECOVERY_Q_STEPS"] = 99
            write_json(checkpoint, config)
            with self.assertRaisesRegex(RuntimeError, "ACTOR_SCORE_RECOVERY_Q_STEPS"):
                report.prepare(self.collection, self.output)
            config["ACTOR_SCORE_RECOVERY_Q_STEPS"] = 4
            write_json(checkpoint, config)
            metrics = Path(entry["outputs"]["metrics.jsonl"])
            rows = [json.loads(line) for line in metrics.read_text().splitlines()]
            metrics.write_text(
                "\n".join(json.dumps(row) for row in rows if row["env_step"] in (4, 20)) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "grids are not sufficiently matched"):
                report.prepare(self.collection, self.output)

    def test_extension_final_uses_effective_whole_rollout_budget(self) -> None:
        _, entries = report.load_collection(self.collection)
        entry = entries["10m_vs_11m", "none", 5]
        source_manifest = Path(entry["outputs"]["source_manifest.json"])
        manifest = report.read_json(source_manifest)
        manifest["effective_timesteps"] = 19
        write_json(source_manifest, manifest)
        checkpoint_parent = Path(entry["outputs"]["checkpoints"])
        for checkpoint in checkpoint_parent.iterdir():
            config = report.read_json(checkpoint / "config.json")
            config["TOTAL_TIMESTEPS"] = 19
            write_json(checkpoint / "config.json", config)
        final = checkpoint_parent / "final" / "metadata.json"
        metadata = report.read_json(final)
        metadata["nominal_env_step"] = 19
        write_json(final, metadata)
        profile = self.profiles["10m_vs_11m"]["none"]
        self.assertEqual(report.last_five(entry, profile)[-1][0], 19)

    def test_render_and_table_use_ten_seeds_and_heldout_last_five(self) -> None:
        _, selected = report.load_collection(self.collection)
        for entry in selected.values():
            if entry["origin"] != "six_seed_extension":
                continue
            source_manifest = Path(entry["outputs"]["source_manifest.json"])
            manifest = report.read_json(source_manifest)
            manifest["effective_timesteps"] = 19
            write_json(source_manifest, manifest)
            checkpoint_parent = Path(entry["outputs"]["checkpoints"])
            for checkpoint in checkpoint_parent.iterdir():
                config = report.read_json(checkpoint / "config.json")
                config["TOTAL_TIMESTEPS"] = 19
                write_json(checkpoint / "config.json", config)
            final = checkpoint_parent / "final" / "metadata.json"
            metadata = report.read_json(final)
            metadata["nominal_env_step"] = 19
            write_json(final, metadata)
        with patch.object(report, "frozen_profiles", return_value=self.profiles):
            _, entries, jobs, _ = report.prepare(self.collection, self.output)
            for job in jobs:
                method = "none" if job.condition == "none" else "arec"
                origin = entries[job.task, method, job.seed]["origin"]
                condition = report.CONDITION[method] if origin == "first_four_panel_v1" else method
                write_json(job.output, {
                    "checkpoint": str(job.checkpoint), "episodes": 256,
                    "eval_seed": job.eval_seed, "policy": "stochastic",
                    "map_name": job.task, "training_seed": job.seed,
                    "checkpoint_nominal_env_step": job.nominal_step,
                    "return_mean": 2.0 + 0.01 * job.seed + (0.5 if method == "arec" else 0),
                    "num_envs": 128, "condition": condition,
                })
            path = report.report(SimpleNamespace(
                collection_root=self.collection, output_root=self.output,
                reuse_evaluation_root=None, evaluate_missing=False,
                eval_episodes=256, eval_num_envs=128, eval_policy="stochastic",
                gpus=("0",), max_runs_per_gpu=1,
                bootstrap_samples=200, bootstrap_seed=17,
            ))
        self.assertTrue(path.is_file())
        self.assertTrue(path.with_suffix(".pdf").is_file())
        self.assertTrue(path.with_suffix(".svg").is_file())
        summary = report.read_json(self.output / "report_manifest.json")
        self.assertEqual(summary["seeds"], list(range(1, 11)))
        self.assertIn("different code commits", summary["code_cohort_warning"])
        self.assertIn("last 5", (self.output / "summary_table.md").read_text())
        with (self.output / "summary_all_tasks.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 8)
        arec = next(row for row in rows if row["task"] == "10m_vs_11m" and row["condition"] != "none")
        self.assertEqual(int(arec["n_seeds"]), 10)
        self.assertIn("varies slightly", arec["final_checkpoint_steps"])
        self.assertAlmostEqual(float(arec["delta_final_eval_return_last5_ckpt_vs_none_mean"]), 0.5)
        self.assertAlmostEqual(float(arec["delta_return_auc_vs_none_mean"]), 0.2)


if __name__ == "__main__":
    unittest.main()
