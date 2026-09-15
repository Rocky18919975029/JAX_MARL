import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_MATRIX_PATH = ROOT / "experiments" / "harl_mamujoco" / "run_matrix.py"


def load_run_matrix():
    spec = importlib.util.spec_from_file_location("harl_run_matrix", RUN_MATRIX_PATH)
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


def test_alignment_implementation_has_no_cosine_objective():
    source = (ROOT / "experiments" / "harl_mamujoco" / "alignment.py").read_text(
        encoding="utf-8"
    )
    assert "cosine_distance" not in source
    assert "critic_old.detach()" in source
    assert "actor_old.detach()" in source
    assert "source_centered.transpose(0, 1) @ target_centered" in source


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
