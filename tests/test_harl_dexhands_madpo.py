import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments" / "harl_dexhands" / "protocol.py"
RUN_MATRIX = ROOT / "experiments" / "harl_dexhands" / "run_matrix.py"
TRAIN = ROOT / "experiments" / "harl_dexhands" / "train.py"
MADPO = ROOT / "experiments" / "harl_dexhands" / "madpo.py"
RUNNER = ROOT / "experiments" / "harl_dexhands" / "runner.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_shadowhandover_happo_config_is_the_frozen_common_protocol():
    path = (
        ROOT
        / "third_party"
        / "HARL"
        / "tuned_configs"
        / "dexhands"
        / "ShadowHandOver"
        / "happo"
        / "config.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["main_args"] == {
        "algo": "happo",
        "env": "dexhands",
        "exp_name": "report",
        "load_config": "",
    }
    assert payload["algo_args"]["train"]["num_env_steps"] == 50_000_000
    assert payload["algo_args"]["train"]["n_rollout_threads"] == 256
    assert payload["algo_args"]["train"]["episode_length"] == 75
    assert payload["algo_args"]["model"]["hidden_sizes"] == [256, 256, 256]
    assert payload["algo_args"]["algo"]["share_param"] is False


def test_matched_matrix_has_three_algorithms_and_four_seeds():
    protocol = load(PROTOCOL, "test_dex_protocol")
    tasks = protocol.task_matrix(protocol.ALGORITHMS, (1, 2, 3, 4))
    assert len(tasks) == 12
    assert len({task.name for task in tasks}) == 12
    assert {task.algorithm for task in tasks} == {"happo", "mappo", "madpo"}
    assert {task.seed for task in tasks} == {1, 2, 3, 4}
    assert all(
        "div1000-w0p05-sig1-k1024" in task.name
        for task in tasks
        if task.algorithm == "madpo"
    )


def test_config_changes_only_algorithm_specific_madpo_fields(tmp_path):
    protocol = load(PROTOCOL, "test_dex_protocol_config")
    source = (
        ROOT
        / "third_party"
        / "HARL"
        / "tuned_configs"
        / "dexhands"
        / "ShadowHandOver"
        / "happo"
        / "config.json"
    )
    configs = {
        algorithm: protocol.load_matched_config(source, algorithm, 7, tmp_path / "logs")
        for algorithm in protocol.ALGORITHMS
    }
    for algorithm, (main, algo, env) in configs.items():
        assert main["algo"] == algorithm
        assert main["env"] == "dexhands"
        assert env == configs["happo"][2]
        assert algo["algo"]["share_param"] is False
        assert algo["seed"]["seed"] == 7
        assert algo["eval"]["use_eval"] is False
        assert algo["train"] == configs["happo"][1]["train"]
        assert algo["model"] == configs["happo"][1]["model"]
    assert configs["madpo"][1]["algo"]["div_coef"] == 1000.0
    assert configs["madpo"][1]["algo"]["div_max_samples"] == 1024


def test_paper_profile_changes_madpo_only(tmp_path):
    protocol = load(PROTOCOL, "test_dex_paper_config")
    source = (
        ROOT
        / "third_party"
        / "HARL"
        / "tuned_configs"
        / "dexhands"
        / "ShadowHandOver"
        / "happo"
        / "config.json"
    )
    for algorithm in ("happo", "mappo"):
        old = protocol.load_matched_config(source, algorithm, 7, tmp_path)
        new = protocol.load_matched_config(
            source, algorithm, 7, tmp_path, madpo_profile="paper2024"
        )
        assert old == new
    _, algo, env = protocol.load_matched_config(
        source, "madpo", 7, tmp_path, madpo_profile="paper2024"
    )
    assert env["task"] == "ShadowHandOver"
    assert algo["train"]["num_env_steps"] == 40_000_000
    assert algo["train"]["n_rollout_threads"] == 128
    assert algo["train"]["episode_length"] == 75
    assert algo["model"]["hidden_sizes"] == [256, 256, 256]
    assert algo["algo"]["gamma"] == 0.99
    assert algo["algo"]["div_weight"] == 0.2
    assert algo["algo"]["div_sigma"] == 500.0
    assert algo["algo"]["div_coef"] == 1000.0
    assert algo["algo"]["div_max_samples"] == 4000


def test_new_study_uses_paper_profile_and_legacy_study_is_resumable(tmp_path):
    launcher = load(RUN_MATRIX, "test_dex_profile_resolution")
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    assert launcher.resolve_madpo_profile(new_root, "auto") == "paper2024"
    old_root.mkdir()
    (old_root / "experiment_manifest.json").write_text(
        json.dumps({"study_spec": {"protocol_version": "old"}})
    )
    assert launcher.resolve_madpo_profile(old_root, "auto") == "legacy"
    command = [
        sys.executable,
        str(RUN_MATRIX),
        "--run-root",
        str(new_root),
        "--algorithms",
        "mappo,madpo",
        "--conditions",
        "none,arec",
        "--seeds",
        "1",
        "--gpus",
        "0",
        "--wandb-mode",
        "disabled",
        "--dry-run",
    ]
    result = __import__("subprocess").run(
        command, check=True, capture_output=True, text=True
    )
    manifest = json.loads((new_root / "experiment_manifest.json").read_text())
    spec = manifest["study_spec"]
    assert spec["madpo_profile"] == "paper2024"
    assert spec["num_env_steps_by_algorithm"] == {
        "mappo": 50_000_000,
        "madpo": 40_000_000,
    }
    assert spec["n_rollout_threads_by_algorithm"] == {"mappo": 256, "madpo": 128}
    assert len(spec["madpo_paper_config_sha256"]) == 64
    assert "madpo-div1000-w0p2-sig500-k4000" in result.stdout
    assert "mappo-arec" in result.stdout
    assert "--madpo-profile paper2024" in result.stdout
    monitor = load(
        ROOT / "experiments/harl_dexhands/monitor.py", "test_dex_paper_monitor"
    )
    rows = monitor.load_manifest_rows(new_root)
    assert {row["total"] for row in rows if "-madpo-" in row["name"]} == {40_000_000}
    assert {row["total"] for row in rows if "-mappo-" in row["name"]} == {50_000_000}
    changed = __import__("subprocess").run(
        command + ["--madpo-profile", "legacy"], capture_output=True, text=True
    )
    assert changed.returncode != 0
    assert "manifest differs" in changed.stderr


def test_import_order_and_reference_policy_safety_are_explicit():
    train_source = TRAIN.read_text(encoding="utf-8")
    assert train_source.index("import isaacgym") < train_source.index(
        "from harl.algorithms.actors import ALGO_REGISTRY"
    )
    madpo_source = MADPO.read_text(encoding="utf-8")
    assert "with torch.no_grad():" in madpo_source
    assert "return log_probs.detach()" in madpo_source
    assert "Non-finite MADPO" in madpo_source
    assert "old_actor_buffer" not in madpo_source
    runner_source = RUNNER.read_text(encoding="utf-8")
    assert "copy.deepcopy(self.actor[agent_id].actor)" in runner_source
    assert "previous_agent_id = agent_id" in runner_source
    assert "self.actor[agent_id - 1]" not in runner_source


def test_ccsd_zero_for_identical_paired_empirical_distributions():
    torch = pytest.importorskip("torch")
    from experiments.harl_dexhands.ccsd import conditional_cs_divergence

    generator = torch.Generator().manual_seed(5)
    observations = torch.randn(128, 7, generator=generator)
    statistics = torch.randn(128, 3, generator=generator, requires_grad=True)
    divergence, metadata = conditional_cs_divergence(
        observations,
        observations,
        statistics,
        statistics.detach(),
        max_samples=64,
        generator=torch.Generator().manual_seed(9),
        paired_samples=True,
    )
    assert torch.isfinite(divergence)
    assert abs(float(divergence.detach())) < 1e-5
    assert int(metadata["current_samples"]) == 64
    divergence.backward()
    assert statistics.grad is not None
    assert torch.isfinite(statistics.grad).all()


def test_reference_fork_is_not_vendored_or_used_as_runtime():
    assert not (ROOT / "third_party" / "MADPO").exists()
    source = RUN_MATRIX.read_text(encoding="utf-8")
    assert 'REPO_ROOT / "third_party" / "HARL"' in source


def test_launcher_restores_conda_runtime_library_path(monkeypatch):
    launcher = load(RUN_MATRIX, "test_dex_launcher_environment")
    monkeypatch.setenv("CONDA_PREFIX", "/tmp/harl-dex")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/cuda/lib64:/tmp/harl-dex/lib")
    environment = launcher.training_environment("2")
    assert environment["CUDA_VISIBLE_DEVICES"] == "2"
    assert environment["LD_LIBRARY_PATH"].split(":") == [
        "/tmp/harl-dex/lib",
        "/opt/cuda/lib64",
    ]
