"""CPU-only checks for the CKA diagnostic launcher and locked protocol."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.eval_h1_checkpoints import CHECKPOINT_SPECS
from scripts.run_h1_cka_diagnostics import (
    CONDITIONS,
    MAPS,
    SEEDS,
    phase_plan,
    verify_matrix,
)


class TestCKADiagnosticsLauncher(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run_dirs = {}
        for task in MAPS:
            for condition in CONDITIONS:
                for seed in SEEDS:
                    run = (
                        self.root
                        / "checkpoints"
                        / "h1-smax-reduced-nps-4condition"
                        / f"H1-reduced-{task}-nps-{condition}-lam0p35-seed{seed}-fake"
                    )
                    self.run_dirs[(task, condition, seed)] = run
                    config = {
                        "MAP_NAME": task,
                        "EXPERIMENT_CONDITION": condition,
                        "SEED": seed,
                        "PROTOCOL_VERSION": "h1-v1.0",
                        "MATRIX_PROFILE": "reduced-nps-cka",
                        "ACTOR_PARAMETER_SHARING": False,
                        "MATCHED_COMPARISON": True,
                        "ALIGN_DISTANCE": "linear_cka",
                        "ALIGN_TARGET_SHUFFLE": False,
                        "ALIGN_MODE": condition.removesuffix("_cka"),
                    }
                    for name, _ in CHECKPOINT_SPECS:
                        checkpoint = run / name
                        checkpoint.mkdir(parents=True)
                        (checkpoint / "model.safetensors").write_bytes(b"fixture")
                    (run / "initial" / "config.json").write_text(json.dumps(config))

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_16_run_128_checkpoint_matrix(self):
        self.assertEqual(verify_matrix(self.root), 128)

    def test_rejects_missing_preregistered_checkpoint(self):
        (self.run_dirs[(MAPS[0], CONDITIONS[0], SEEDS[0])] / "final" / "model.safetensors").unlink()
        with self.assertRaisesRegex(RuntimeError, "missing"):
            verify_matrix(self.root)

    def test_rejects_changed_distance_or_actor_variant(self):
        config_path = self.run_dirs[(MAPS[0], CONDITIONS[0], SEEDS[0])] / "initial" / "config.json"
        config = json.loads(config_path.read_text())
        config["ALIGN_DISTANCE"] = "ln_mse"
        config["ACTOR_PARAMETER_SHARING"] = True
        config_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(RuntimeError, "Unexpected CKA checkpoint config"):
            verify_matrix(self.root)

    def test_budget_and_resume_flags(self):
        plan = phase_plan(self.root, "0,1,2,3", 2)
        phases = {phase.name: phase.command for phase in plan}
        self.assertEqual(tuple(phases), (
            "collect", "latent", "decision", "bellman", "merge",
            "deterministic-eval", "performance", "mechanisms", "figures",
        ))
        self.assertEqual(phases["collect"][phases["collect"].index("--max-runs-per-gpu") + 1], "2")
        for phase in ("latent", "decision", "bellman"):
            self.assertEqual(phases[phase][phases[phase].index("--max-runs-per-gpu") + 1], "1")
        for phase, option, value in (
            ("collect", "--episodes", "512"),
            ("collect", "--batch-size", "64"),
            ("decision", "--anchors", "256"),
            ("decision", "--continuations", "32"),
            ("bellman", "--bellman-heads", "32"),
            ("deterministic-eval", "--episodes", "256"),
            ("deterministic-eval", "--num-envs", "128"),
        ):
            self.assertEqual(phases[phase][phases[phase].index(option) + 1], value)
        self.assertFalse(any("--rerun" in cmd or "--all-checkpoints" in cmd for cmd in phases.values()))

    def test_cli_dry_run_has_no_output_side_effects(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_h1_cka_diagnostics.py"
        result = subprocess.run(
            [sys.executable, str(script), "--run-root", str(self.root), "--dry-run"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("16 runs, 128 preregistered checkpoints", result.stdout)
        self.assertIn("decision:", result.stdout)
        self.assertFalse((self.root / ".cka_diagnostics_pipeline.lock").exists())
        self.assertFalse((self.root / "diagnostics_raw").exists())


if __name__ == "__main__":
    unittest.main()
