"""ARec protocol and lightweight actor-path checks without Isaac Gym."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # The protocol checks still run without the HARL runtime.
    torch = None
    nn = None


ROOT = Path(__file__).resolve().parents[1]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ProtocolTests(unittest.TestCase):
    def test_six_matched_conditions_preserve_baseline_names(self):
        protocol = load(
            ROOT / "experiments/harl_dexhands/protocol.py", "test_dex_arec_protocol"
        )
        tasks = protocol.task_matrix(
            protocol.ALGORITHMS, (1,), conditions=protocol.CONDITIONS
        )
        self.assertEqual(len(tasks), 6)
        self.assertEqual(len({task.name for task in tasks}), 6)
        self.assertIn(
            "HARL-ShadowHandOver-nps-happo-seed1", {task.name for task in tasks}
        )
        self.assertTrue(
            all("-arec-lam" in task.name for task in tasks if task.condition == "arec")
        )

    def test_arec_config_does_not_change_the_base_training_protocol(self):
        protocol = load(
            ROOT / "experiments/harl_dexhands/protocol.py", "test_dex_arec_config"
        )
        source = {
            "main_args": {"algo": "happo", "env": "dexhands"},
            "algo_args": {
                "seed": {"seed_specify": False, "seed": 1},
                "algo": {"share_param": False, "clip_param": 0.2},
                "eval": {"use_eval": False},
                "logger": {"log_dir": ""},
                "train": {"model_dir": None, "num_env_steps": 100},
                "model": {"hidden_sizes": [256, 256, 256]},
            },
            "env_args": {"task": "ShadowHandOver"},
        }
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(source))
            baseline = protocol.load_matched_config(
                path, "mappo", 1, Path(directory), condition="none"
            )
            arec = protocol.load_matched_config(
                path, "mappo", 1, Path(directory), condition="arec"
            )
        self.assertEqual(baseline[0], arec[0])
        self.assertEqual(baseline[2], arec[2])
        self.assertEqual(baseline[1]["train"], arec[1]["train"])
        self.assertEqual(baseline[1]["model"], arec[1]["model"])
        self.assertNotIn("arec_coef", baseline[1]["algo"])
        self.assertEqual(arec[1]["algo"]["arec_coef"], 0.0001)

    def test_four_seed_lambda_grid_has_twelve_runs_per_seed(self):
        protocol = load(
            ROOT / "experiments/harl_dexhands/protocol.py", "test_dex_arec_grid"
        )
        grid = protocol.parse_positive_floats("0.00003,0.0001,0.0003")
        tasks = protocol.task_matrix(
            protocol.ALGORITHMS,
            (1, 2, 3, 4),
            conditions=("none", "arec"),
            arec_coefs=grid,
        )
        self.assertEqual(len(tasks), 48)
        self.assertEqual(len({task.name for task in tasks}), 48)
        for seed in (1, 2, 3, 4):
            cohort = [task for task in tasks if task.seed == seed]
            self.assertEqual(len(cohort), 12)
            for algorithm in protocol.ALGORITHMS:
                candidates = [task for task in cohort if task.algorithm == algorithm]
                self.assertEqual(
                    sum(task.condition == "none" for task in candidates), 1
                )
                self.assertEqual(
                    {task.arec_coef for task in candidates if task.condition == "arec"},
                    set(grid),
                )
        for invalid in ("", "1e-4,1e-4", "0,1e-4", "nan,1e-4"):
            with self.assertRaises(ValueError):
                protocol.parse_positive_floats(invalid)

    def test_launcher_freezes_grid_and_monitor_discovers_it(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            harl_root = Path(directory) / "harl"
            config = (
                harl_root / "tuned_configs/dexhands/ShadowHandOver/happo/config.json"
            )
            config.parent.mkdir(parents=True)
            config.write_text(
                json.dumps({"algo_args": {"train": {"num_env_steps": 50_000_000}}})
            )
            command = [
                sys.executable,
                str(ROOT / "experiments/harl_dexhands/run_matrix.py"),
                "--run-root",
                str(root),
                "--harl-root",
                str(harl_root),
                "--algorithms",
                "happo,mappo,madpo",
                "--conditions",
                "none,arec",
                "--seeds",
                "1-4",
                "--arec-coefs",
                "0.00003,0.0001,0.0003",
                "--gpus",
                "0,1,2,3",
                "--max-runs-per-gpu",
                "2",
                "--dry-run",
            ]
            output = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertIn("total=48", output.stdout)
            manifest = json.loads((root / "experiment_manifest.json").read_text())
            self.assertEqual(len(manifest["runs"]), 48)
            self.assertEqual(manifest["study_spec"]["arec_coefs"], [3e-5, 1e-4, 3e-4])
            self.assertEqual(output.stdout.count("--arec-coef 3e-05"), 12)
            self.assertEqual(output.stdout.count("--arec-coef 0.0001"), 24)
            self.assertEqual(output.stdout.count("--arec-coef 0.0003"), 12)
            monitor = load(
                ROOT / "experiments/harl_dexhands/monitor.py", "test_dex_arec_monitor"
            )
            rows = monitor.load_manifest_rows(root)
            self.assertEqual(len(rows), 48)
            self.assertTrue(all(row["status"] == "pending" for row in rows))
            self.assertTrue(all(row["total"] == 50_000_000 for row in rows))
            modified = [
                *command[:-1],
                "--arec-coefs",
                "0.00001,0.0001,0.001",
                "--dry-run",
            ]
            retry = subprocess.run(modified, capture_output=True, text=True)
            self.assertNotEqual(retry.returncode, 0)
            self.assertIn("manifest differs", retry.stderr)

            overcommitted = [
                *command[:-3],
                "--max-runs-per-gpu",
                "3",
                "--dry-run",
            ]
            rejected = subprocess.run(overcommitted, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("limited to two concurrent runs per GPU", rejected.stderr)

    def test_launcher_completes_each_seed_and_retries_failures_without_stopping(self):
        import tempfile

        launcher = load(
            ROOT / "experiments/harl_dexhands/run_matrix.py",
            "test_dex_arec_failure_continuation",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            harl_root = Path(directory) / "harl"
            config = (
                harl_root / "tuned_configs/dexhands/ShadowHandOver/happo/config.json"
            )
            config.parent.mkdir(parents=True)
            config.write_text(
                json.dumps({"algo_args": {"train": {"num_env_steps": 2400}}})
            )
            attempts = {}
            invocation_order = []

            def fake_worker(command, **_kwargs):
                name = command[command.index("--run-name") + 1]
                seed = int(command[command.index("--seed") + 1])
                attempts[name] = attempts.get(name, 0) + 1
                invocation_order.append((seed, name, attempts[name]))
                failed_once = name.endswith("happo-seed1") and attempts[name] == 1
                state = "failed" if failed_once else "completed"
                status = root / "status" / f"{name}.json"
                status.parent.mkdir(parents=True, exist_ok=True)
                status.write_text(json.dumps({"run_name": name, "status": state}))
                metric = root / "metrics" / f"{name}.jsonl"
                metric.parent.mkdir(parents=True, exist_ok=True)
                metric.write_text('{"attempt": %d}\n' % attempts[name])
                return types.SimpleNamespace(returncode=int(failed_once))

            arguments = [
                "run_matrix.py", "--run-root", str(root),
                "--harl-root", str(harl_root),
                "--algorithms", "happo,mappo,madpo",
                "--conditions", "none,arec",
                "--seeds", "1-2",
                "--arec-coefs", "0.00003,0.0001,0.0003",
                "--gpus", "0", "--max-runs-per-gpu", "1",
                "--wandb-mode", "disabled",
            ]
            with patch.object(sys, "argv", arguments), patch.object(
                launcher.subprocess, "run", side_effect=fake_worker
            ):
                launcher.main()

            events = (root / "launcher.log").read_text()
            self.assertEqual(len(attempts), 24)
            self.assertEqual(sum(attempts.values()), 25)
            self.assertNotIn("Stopping new launches", events)
            self.assertIn("SEED 1 retry=1", events)
            self.assertIn("SEED 1 finished completed=12/12", events)
            self.assertIn("SEED 2 finished completed=12/12", events)
            self.assertTrue(all(seed == 1 for seed, _, _ in invocation_order[:13]))
            failed_run = "HARL-ShadowHandOver-nps-happo-seed1"
            archive = root / "failed_attempts" / failed_run / "attempt_01"
            self.assertEqual(json.loads((archive / "status.json").read_text())["status"], "failed")
            self.assertEqual((archive / "metrics.jsonl").read_text(), '{"attempt": 1}\n')
            self.assertEqual(
                (root / "metrics" / f"{failed_run}.jsonl").read_text(),
                '{"attempt": 2}\n',
            )

    def test_live_worker_gpu_is_reserved_and_identity_checked(self):
        import tempfile

        launcher = load(
            ROOT / "experiments/harl_dexhands/run_matrix.py",
            "test_dex_arec_live_worker",
        )
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory) / "1234"
            proc.mkdir()
            run_name = "HARL-ShadowHandOver-nps-happo-seed2"
            (proc / "stat").write_text("1234 (python) R 1 1 1\n")
            (proc / "cmdline").write_bytes(
                f"python\0train.py\0--run-name\0{run_name}\0".encode()
            )
            (proc / "environ").write_bytes(b"CUDA_VISIBLE_DEVICES=3\0")
            status = {"status": "running", "pid": 1234}
            self.assertEqual(
                launcher.running_gpu(run_name, status, ("0", "1", "2", "3"), Path(directory)),
                "3",
            )
            with self.assertRaisesRegex(RuntimeError, "different title"):
                launcher.running_gpu("another-run", status, ("3",), Path(directory))
            with self.assertRaisesRegex(RuntimeError, "select its GPU"):
                launcher.running_gpu(run_name, status, ("0",), Path(directory))

    def test_resume_keeps_live_worker_and_counts_its_gpu_slot(self):
        import tempfile
        import threading

        launcher = load(
            ROOT / "experiments/harl_dexhands/run_matrix.py",
            "test_dex_arec_external_slot",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            harl_root = Path(directory) / "harl"
            config = harl_root / "tuned_configs/dexhands/ShadowHandOver/happo/config.json"
            config.parent.mkdir(parents=True)
            config.write_text(json.dumps({"algo_args": {"train": {"num_env_steps": 2400}}}))
            external_name = "HARL-ShadowHandOver-nps-happo-seed2"
            external_status = root / "status" / f"{external_name}.json"
            external_status.parent.mkdir(parents=True)
            external_status.write_text(
                json.dumps({"run_name": external_name, "status": "running", "pid": 1234})
            )
            done = threading.Event()
            starts = []

            def finish_external():
                external_status.write_text(
                    json.dumps({"run_name": external_name, "status": "completed"})
                )
                done.set()

            def fake_worker(command, **_kwargs):
                name = command[command.index("--run-name") + 1]
                starts.append((name, done.is_set()))
                path = root / "status" / f"{name}.json"
                path.write_text(json.dumps({"run_name": name, "status": "completed"}))
                return types.SimpleNamespace(returncode=0)

            arguments = [
                "run_matrix.py", "--run-root", str(root),
                "--harl-root", str(harl_root),
                "--algorithms", "happo", "--conditions", "none",
                "--seeds", "1-2", "--gpus", "0", "--max-runs-per-gpu", "1",
                "--wandb-mode", "disabled",
            ]
            timer = threading.Timer(0.05, finish_external)
            with patch.object(sys, "argv", arguments), patch.object(
                launcher, "running_gpu", return_value="0"
            ), patch.object(
                launcher, "process_is_running", side_effect=lambda _pid: not done.is_set()
            ), patch.object(
                launcher.subprocess, "run", side_effect=fake_worker
            ):
                timer.start()
                try:
                    launcher.main()
                finally:
                    timer.join()
            self.assertEqual(starts, [("HARL-ShadowHandOver-nps-happo-seed1", True)])
            self.assertIn(f"KEEP  {external_name}", (root / "launcher.log").read_text())

    def test_launcher_rejects_incomplete_wandb_before_dispatch(self):
        from unittest.mock import patch

        launcher = load(
            ROOT / "experiments/harl_dexhands/run_matrix.py",
            "test_dex_arec_wandb_preflight",
        )
        incomplete = types.ModuleType("wandb")
        with patch.dict(sys.modules, {"wandb": incomplete}):
            with self.assertRaisesRegex(RuntimeError, "no callable init"):
                launcher.verify_wandb()


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ActorPathTests(unittest.TestCase):
    def _module(self):
        class BaseActor:
            def __init__(self, args, obs_space, act_space, device):
                self.tpdv = {"dtype": torch.float32, "device": device}
                self.device = device
                self.use_recurrent_policy = False
                self.use_naive_recurrent_policy = False
                self.use_policy_active_masks = True
                self.action_aggregation = "prod"
                self.clip_param = 0.2
                self.entropy_coef = 0.01
                self.use_max_grad_norm = False
                self.actor = ToyPolicy()
                self.actor_optimizer = torch.optim.Adam(
                    self.actor.parameters(), lr=0.001
                )

            def evaluate_actions(self, obs, rnn, actions, masks, available, active):
                latent = self.actor.base(torch.as_tensor(obs, dtype=torch.float32))
                distribution = self.actor.act.action_out(latent)
                entropy = (
                    distribution.entropy() * torch.as_tensor(active).squeeze(-1)
                ).sum() / torch.as_tensor(active).sum()
                return (
                    distribution.log_probs(torch.as_tensor(actions)),
                    entropy,
                    distribution,
                )

        class FakeHAPPO(BaseActor):
            pass

        class FakeMAPPO(BaseActor):
            pass

        class FakeMADPO(FakeHAPPO):
            pass

        class ToyDistribution:
            def __init__(self, mean):
                self.distribution = torch.distributions.Normal(mean, 0.5)

            def log_probs(self, actions):
                return self.distribution.log_prob(actions)

            def entropy(self):
                return self.distribution.entropy().sum(-1)

        class ToyPolicy(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_sizes = [3]
                self.base = nn.Linear(2, 3)
                self.act = nn.Module()
                self.act.action_out = nn.Linear(3, 2)
                linear = self.act.action_out
                self.act.action_out.forward = lambda latent: ToyDistribution(
                    nn.Linear.forward(linear, latent)
                )

        modules = {}
        for name in (
            "harl",
            "harl.algorithms",
            "harl.algorithms.actors",
            "harl.algorithms.actors.happo",
            "harl.algorithms.actors.mappo",
            "harl.utils",
            "harl.utils.envs_tools",
            "harl.utils.models_tools",
            "experiments.harl_dexhands.madpo",
        ):
            modules[name] = types.ModuleType(name)
        modules["harl.algorithms.actors.happo"].HAPPO = FakeHAPPO
        modules["harl.algorithms.actors.mappo"].MAPPO = FakeMAPPO
        modules["harl.utils.envs_tools"].check = lambda value: torch.as_tensor(value)
        modules["harl.utils.models_tools"].get_grad_norm = (
            lambda parameters: torch.linalg.vector_norm(
                torch.stack([p.grad.norm() for p in parameters if p.grad is not None])
            )
        )
        modules["experiments.harl_dexhands.madpo"].MADPO = FakeMADPO
        with patch.dict(sys.modules, modules):
            return load(
                ROOT / "experiments/harl_dexhands/arec.py", "test_dex_arec_runtime"
            )

    def test_fisher_and_minibatch_alignment(self):
        module = self._module()
        scores = torch.randn(12, 3)
        matrix = module.fisher_inverse_sqrt(scores, 0.001)
        self.assertTrue(torch.isfinite(matrix).all())
        self.assertTrue(torch.allclose(matrix, matrix.T, atol=1e-5))
        buffer = types.SimpleNamespace(
            obs=np.arange(18, dtype=np.float32).reshape(3, 3, 2),
            rnn_states=np.zeros((3, 3, 1, 3), dtype=np.float32),
            actions=np.zeros((2, 3, 2), dtype=np.float32),
            masks=np.ones((3, 3, 1), dtype=np.float32),
            active_masks=np.ones((3, 3, 1), dtype=np.float32),
            action_log_probs=np.zeros((2, 3, 2), dtype=np.float32),
            available_actions=None,
            factor=np.ones((2, 3, 1), dtype=np.float32),
        )
        teacher = buffer.obs[:-1, :, :1].copy()
        sample = next(
            module.RecoveryBufferView(buffer, teacher).feed_forward_generator_actor(
                np.ones((2, 3, 1), dtype=np.float32), 1
            )
        )
        self.assertEqual(len(sample), 10)
        np.testing.assert_array_equal(sample[0][:, :1], sample[-1])

    def test_recovery_head_initialization_preserves_baseline_rng(self):
        module = self._module()
        Box = type("Box", (), {"shape": (2,)})
        args = {
            "arec_coef": 0.1,
            "arec_q_steps": 2,
            "arec_q_lr": 0.001,
            "arec_fisher_ridge": 0.001,
        }
        torch.manual_seed(37)
        baseline_rng = torch.random.get_rng_state().clone()
        module.ARecHAPPO(args, None, Box(), torch.device("cpu"))
        after_arec = torch.random.get_rng_state()
        torch.random.set_rng_state(baseline_rng)
        module.ARecHAPPO.__mro__[2](args, None, Box(), torch.device("cpu"))
        after_baseline_actor = torch.random.get_rng_state()
        self.assertTrue(torch.equal(after_arec, after_baseline_actor))

    def test_happo_and_mappo_actor_update_do_not_change_q_or_critic(self):
        module = self._module()
        torch.manual_seed(7)
        Box = type("Box", (), {"shape": (2,)})
        buffer = types.SimpleNamespace(
            obs=np.random.default_rng(7).normal(size=(3, 4, 2)).astype(np.float32),
            rnn_states=np.zeros((3, 4, 1, 3), dtype=np.float32),
            actions=np.random.default_rng(8).normal(size=(2, 4, 2)).astype(np.float32),
            masks=np.ones((3, 4, 1), dtype=np.float32),
            active_masks=np.ones((3, 4, 1), dtype=np.float32),
            action_log_probs=np.zeros((2, 4, 2), dtype=np.float32),
            available_actions=None,
            factor=np.ones((2, 4, 1), dtype=np.float32),
        )
        for actor_class, factor in (
            (module.ARecHAPPO, buffer.factor),
            (module.ARecMAPPO, None),
        ):
            with self.subTest(actor=actor_class.__name__):
                buffer.factor = factor
                actor = actor_class(
                    {
                        "arec_coef": 0.1,
                        "arec_q_steps": 2,
                        "arec_q_lr": 0.001,
                        "arec_fisher_ridge": 0.001,
                    },
                    None,
                    Box(),
                    torch.device("cpu"),
                )
                critic_latent = torch.randn(8, 3, requires_grad=True)
                actor.prepare_recovery(buffer, critic_latent)
                q_before = [
                    parameter.detach().clone()
                    for parameter in actor.arec_q.parameters()
                ]
                actor_before = [
                    parameter.detach().clone() for parameter in actor.actor.parameters()
                ]
                sample = next(
                    module.RecoveryBufferView(
                        buffer, actor.arec_teacher
                    ).feed_forward_generator_actor(
                        np.ones((2, 4, 1), dtype=np.float32), 1
                    )
                )
                actor.update(sample)
                self.assertTrue(
                    any(
                        not torch.equal(a, b)
                        for a, b in zip(actor_before, actor.actor.parameters())
                    )
                )
                self.assertTrue(
                    all(
                        torch.equal(a, b)
                        for a, b in zip(q_before, actor.arec_q.parameters())
                    )
                )
                self.assertIsNone(critic_latent.grad)


if __name__ == "__main__":
    unittest.main()
