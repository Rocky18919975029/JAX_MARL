import json

from experiments.benchmarl_vmas.analyze import load_records


def test_analysis_loads_completed_runs_without_pooling_tasks(tmp_path):
    root = tmp_path / "runs"
    status_root = root / "status"
    output = root / "benchmarl_runs" / "example" / "generated"
    status_root.mkdir(parents=True)
    output.mkdir(parents=True)
    status = {
        "status": "completed",
        "task": "discovery",
        "condition": "none",
        "seed": 1,
        "benchmarl_output": str(output),
    }
    (status_root / "example.json").write_text(json.dumps(status))
    evaluation = {
        "vmas": {
            "discovery": {
                "alignmentmappo": {
                    "seed_1": {
                        "absolute_metrics": {},
                        "step_1": {
                            "step_count": 120000,
                            "return": [1.0, 3.0],
                        },
                    }
                }
            }
        }
    }
    (output / "evaluation.json").write_text(json.dumps(evaluation))
    records = load_records(root)
    assert records == [
        {
            "task": "discovery",
            "condition": "none",
            "seed": 1,
            "env_step": 120000,
            "return_mean": 2.0,
            "return_std_episode": 2**0.5,
            "evaluation_episodes": 2,
        }
    ]
