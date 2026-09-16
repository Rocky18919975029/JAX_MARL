import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "baselines" / "MAPPO" / "mappo_ff_mabrax.py"
CONFIG = ROOT / "baselines" / "MAPPO" / "config" / "mappo_ff_mabrax.yaml"
EVAL_SOURCE = ROOT / "baselines" / "MAPPO" / "eval_mappo_ff_mabrax.py"


def test_mappo_mabrax_source_is_valid_python():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert {"MABraxWorldStateWrapper", "ActorFF", "CriticFF", "Transition"} <= classes


def test_mappo_mabrax_is_continuous_and_centralized():
    source = SOURCE.read_text(encoding="utf-8")
    assert "distrax.MultivariateNormalDiag" in source
    assert "distrax.Categorical" not in source
    assert "state.obs[None, :]" in source
    assert 'last_obs["world_state"]' in source
    assert "actor_network.apply" in source
    assert "critic_network.apply" in source


def test_mappo_mabrax_has_matched_ps_nps_alignment_protocol():
    source = SOURCE.read_text(encoding="utf-8")
    for token in (
        'config["ACTOR_PARAMETER_SHARING"]',
        'config["MATCHED_COMPARISON"]',
        'config["ALIGN_MODE"]',
        'config["ALIGN_DISTANCE"]',
        "vmapped_optimizer",
        "representation_distance",
        "actor_latent_old",
        "critic_latent_old",
        "jax.lax.stop_gradient",
    ):
        assert token in source


def test_mappo_mabrax_checkpoint_evaluator_is_valid_python():
    ast.parse(EVAL_SOURCE.read_text(encoding="utf-8"))
    source = EVAL_SOURCE.read_text(encoding="utf-8")
    assert "deterministic_mean" in source
    assert "load_params" in source


def test_halfcheetah_config_uses_paper_hyperparameters():
    config = CONFIG.read_text(encoding="utf-8")
    expected = {
        "LR": "0.0006",
        "NUM_ENVS": "64",
        "NUM_STEPS": "300",
        "TOTAL_TIMESTEPS": "100000000",
        "GAE_LAMBDA": "1.0",
        "ENT_COEF": "0.0045",
        "VF_COEF": "0.14",
        "ENV_NAME": '"halfcheetah_6x1"',
        "ACTOR_PARAMETER_SHARING": "true",
        "MATCHED_COMPARISON": "true",
        "ALIGN_MODE": '"none"',
        "ALIGN_DISTANCE": '"ln_mse"',
    }
    for key, value in expected.items():
        assert re.search(rf'^"{key}":\s*{re.escape(value)}\s*$', config, re.MULTILINE)
