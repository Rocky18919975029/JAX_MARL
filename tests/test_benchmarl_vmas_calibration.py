import argparse
import sys
import types

from experiments.benchmarl_vmas import calibrate_cka


def test_calibration_namespace_supplies_training_metadata(monkeypatch, tmp_path):
    captured = {}

    class FakeExperiment:
        collector = ()

        def close(self):
            pass

    def fake_build_experiment(args):
        captured.update(vars(args))
        return FakeExperiment()

    fake_train_module = types.ModuleType(
        "experiments.benchmarl_vmas.train_alignment"
    )
    fake_train_module.build_experiment = fake_build_experiment
    monkeypatch.setitem(
        sys.modules,
        "experiments.benchmarl_vmas.train_alignment",
        fake_train_module,
    )
    args = argparse.Namespace(pilot_seed=9001, minibatches=8)

    try:
        calibrate_cka.run_cell(args, "discovery", tmp_path / "cell.json")
    except StopIteration:
        pass

    assert captured["cka_calibration_coef"] is None
    assert captured["cka_multiplier"] is None
    assert captured["experiment_stage"] == "calibration"
    assert captured["task"] == "discovery"
