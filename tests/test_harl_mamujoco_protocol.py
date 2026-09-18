import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUN_MATRIX_PATH = ROOT / "experiments" / "harl_mamujoco" / "run_matrix.py"
NPS_CORE_PATH = ROOT / "experiments" / "harl_mamujoco" / "run_nps_core_matrix.py"
NPS_MONITOR_PATH = ROOT / "experiments" / "harl_mamujoco" / "monitor_nps_core_matrix.py"


def load_run_matrix():
    spec = importlib.util.spec_from_file_location("harl_run_matrix", RUN_MATRIX_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_nps_core_launcher():
    spec = importlib.util.spec_from_file_location("harl_nps_core", NPS_CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_nps_core_monitor():
    spec = importlib.util.spec_from_file_location("harl_nps_monitor", NPS_MONITOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tuned_humanoid_config_is_pinned():
    path = (
        ROOT
        / "third_party"
        / "HARL"
        / "tuned_configs"
        / "mamujoco"
        / "Humanoid-v2-17x1"
        / "mappo"
        / "config.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["main_args"]["algo"] == "mappo"
    assert payload["main_args"]["env"] == "mamujoco"
    assert payload["env_args"]["agent_conf"] == "17x1"
    assert payload["algo_args"]["train"]["num_env_steps"] == 10_000_000
    assert payload["algo_args"]["model"]["hidden_sizes"] == [128, 128, 128]
    assert payload["algo_args"]["algo"]["actor_num_mini_batch"] == 1
    assert payload["algo_args"]["algo"]["critic_num_mini_batch"] == 1


def test_formal_matrix_has_one_distance_free_baseline():
    module = load_run_matrix()
    tasks = module.task_matrix((1, 2, 3, 4), 0.37)
    assert len(tasks) == 56
    assert len({task.name for task in tasks}) == 56
    baselines = [task for task in tasks if task.mode == "none"]
    assert len(baselines) == 8
    assert all(task.distance == "ln_mse" for task in baselines)
    for actor_label in ("ps", "nps"):
        selected = [task for task in tasks if task.actor_label == actor_label]
        assert {task.seed for task in selected} == {1, 2, 3, 4}
        assert {task.condition for task in selected} == {
            "none",
            "c_to_a_mse",
            "a_to_c_mse",
            "joint_mse",
            "c_to_a_cka",
            "a_to_c_cka",
            "joint_cka",
        }


def test_nps_core_matrix_has_exactly_three_conditions_and_four_seeds():
    module = load_run_matrix()
    tasks = module.task_matrix(
        (1, 2, 3, 4),
        0.2258099635,
        distances=("ln_mse", "linear_cka"),
        actor_variants=("nps",),
        directions=("c_to_a",),
    )
    assert len(tasks) == 12
    assert len({task.name for task in tasks}) == 12
    assert {task.actor_label for task in tasks} == {"nps"}
    assert {task.seed for task in tasks} == {1, 2, 3, 4}
    assert {task.condition for task in tasks} == {
        "none",
        "c_to_a_mse",
        "c_to_a_cka",
    }


def test_nps_core_launcher_delegates_fixed_confirmatory_matrix(tmp_path):
    module = load_nps_core_launcher()
    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}", encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "run_root": tmp_path / "runs",
            "harl_root": tmp_path / "HARL",
            "cka_calibration": calibration,
            "gpus": "0,1,2,3",
            "max_runs_per_gpu": 3,
            "checkpoint_interval_steps": 500_000,
            "wandb_project": "test-project",
            "upload_checkpoints": False,
            "dry_run": True,
        },
    )()
    command = module.build_delegate_command(args)
    assert command[0] == sys.executable
    assert command[1] == str(module.GENERAL_LAUNCHER)
    assert command[command.index("--actor-variants") + 1] == "nps"
    assert command[command.index("--directions") + 1] == "c_to_a"
    assert command[command.index("--distances") + 1] == "ln_mse,linear_cka"
    assert command[command.index("--seeds") + 1] == "1-4"
    assert command[command.index("--max-runs-per-gpu") + 1] == "3"
    assert command[-1] == "--dry-run"


def test_nps_core_monitor_reads_live_and_checkpoint_progress(tmp_path):
    module = load_nps_core_monitor()
    root = tmp_path / "runs"
    status = root / "status"
    status.mkdir(parents=True)
    live_name = "HARL-Humanoid-v2-17x1-nps-none-lam0p1-seed1"
    legacy_name = "HARL-Humanoid-v2-17x1-nps-c_to_a_mse-lam0p1-seed1"
    (status / f"{live_name}.json").write_text(
        json.dumps(
            {
                "status": "running",
                "run_name": live_name,
                "env_steps": 1_250_000,
                "total_env_steps": 10_000_000,
            }
        ),
        encoding="utf-8",
    )
    (status / f"{legacy_name}.json").write_text(
        json.dumps({"status": "running", "run_name": legacy_name}),
        encoding="utf-8",
    )
    metadata = root / "checkpoints" / legacy_name / "step_000002000000"
    metadata.mkdir(parents=True)
    (metadata / "metadata.json").write_text(
        json.dumps({"environment_steps": 2_000_000}), encoding="utf-8"
    )
    rows = module.load_rows(root, 10_000_000)
    assert {row["run_name"]: row["steps"] for row in rows} == {
        live_name: 1_250_000,
        legacy_name: 2_000_000,
    }


def test_containment_is_opt_in_and_named_dsc():
    module = load_run_matrix()
    tasks = module.task_matrix(
        (1, 2, 3, 4),
        cka_coefficient=None,
        distances=("containment",),
        containment_coefficient=0.25,
    )
    assert len(tasks) == 32
    aligned = [task for task in tasks if task.mode != "none"]
    assert all(task.distance == "containment" for task in aligned)
    assert all(task.condition.endswith("_dsc") for task in aligned)


def test_alignment_implementation_has_no_cosine_objective():
    source = (ROOT / "experiments" / "harl_mamujoco" / "alignment.py").read_text(
        encoding="utf-8"
    )
    assert "cosine_distance" not in source
    assert "critic_old.detach()" in source
    assert "actor_old.detach()" in source
    assert "source_centered.transpose(0, 1) @ target_centered" in source
    assert "torch.linalg.solve" in source
    assert "source_effective_rank" in source


def test_torch_containment_is_directional():
    torch = pytest.importorskip("torch")
    from experiments.harl_mamujoco.alignment import (
        directional_subspace_containment,
    )

    grid = torch.linspace(-2.0, 2.0, 128)
    full = torch.stack((grid, grid.square(), grid.sin()), dim=-1)
    rank_two = full.clone()
    rank_two[:, 2] = 0.0
    mask = torch.ones(128)
    contained = directional_subspace_containment(rank_two, full, mask)[0]
    not_contained = directional_subspace_containment(full, rank_two, mask)[0]
    assert float(contained) < 5e-3
    assert float(not_contained) > 0.2


def test_runner_records_matched_nps_scaling_and_ep_pairing():
    source = (ROOT / "experiments" / "harl_mamujoco" / "runner.py").read_text(
        encoding="utf-8"
    )
    assert "grad * self.num_agents" in source
    assert "self._initialize_matched_networks()" in source
    assert "actor.actor.load_state_dict(template.actor.state_dict())" in source
    assert 'self.state_type != "EP"' in source
    assert '"one EP critic latent repeated over 17 agents"' in source
    assert 'outputs["actor_objective"] + outputs["critic_objective"]' in source
