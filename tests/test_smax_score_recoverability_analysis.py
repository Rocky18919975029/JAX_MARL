import csv
import json

from scripts.analyze_smax_score_recoverability import collect, plot, summarize


def test_task_step_analysis_produces_aligned_vector_and_raster_figures(tmp_path):
    task = "10m_vs_11m"
    conditions = ("none", "c_to_a_mse", "c_to_a_cka")
    steps = (2_000_000, 5_000_000, 8_000_000, 10_000_000)
    protocol = {
        "protocol": "smax-score-recoverability-resmlp-v2.0",
        "tasks": [task],
        "conditions": list(conditions),
        "seeds": [1, 2],
        "selected_budgets": {task: 10_000_000},
        "checkpoint_plan": {
            task: [
                {
                    "requested_fraction": fraction,
                    "actual_fraction": step / 10_000_000,
                    "env_step": step,
                }
                for fraction, step in zip((0.25, 0.5, 0.75, 1.0), steps)
            ]
        },
    }
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    for step in steps:
        for index, condition in enumerate(conditions):
            for seed in (1, 2):
                output = (
                    tmp_path
                    / "runs"
                    / task
                    / condition
                    / f"seed_{seed}"
                    / f"step_{step:012d}"
                )
                output.mkdir(parents=True)
                base = 1.2 + 0.1 * index + 0.01 * seed
                (output / "summary.json").write_text(
                    json.dumps(
                        {
                            "protocol": protocol["protocol"],
                            "task": task,
                            "condition": condition,
                            "training_seed": seed,
                            "checkpoint_env_step": step,
                            "num_agents": 1,
                            "on_policy_episode_return_mean": base * step / 1e6,
                            "on_policy_episode_return_se": 0.1,
                            "epsilon_rec": base,
                            "epsilon_rec_normalized": base,
                            "fit_epsilon_rec_normalized": base / 2,
                            "validation_epsilon_rec_normalized": base / 1.5,
                            "estimator_failure_agents_gt_one": 1,
                            "episodes": 20,
                            "fit_samples_per_agent": 32,
                            "validation_samples_per_agent": 12,
                            "test_samples_per_agent": 12,
                            "fisher_ridge_absolute": 0.001,
                        }
                    )
                )
                with (output / "agent_metrics.csv").open("w", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=("task", "agent_id"))
                    writer.writeheader()
                    writer.writerow({"task": task, "agent_id": 0})
    loaded, agent_rows, seed_rows = collect(tmp_path)
    assert len(agent_rows) == 24
    assert len(seed_rows) == 24
    summary = summarize(loaded, seed_rows)
    assert len(summary) == 12
    assert all(row["estimator_failure_seeds_gt_one"] == 2 for row in summary)
    plot(loaded, summary, tmp_path / "analysis")
    for suffix in ("png", "pdf", "svg"):
        assert (
            tmp_path / "analysis" / task / f"return-vs-score-recoverability.{suffix}"
        ).is_file()
