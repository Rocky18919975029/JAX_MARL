# First four-panel figure: tuned NPS none and ARec configurations

Eight committed YAML snapshots live in `configs/smax_first_four_panel/`: one
`none` and one `arec` file for each of the four tasks. They encode the
**first** four-panel report (`actor_score_recovery_best_return_report_v1`), not
the later 6s9z tuned-baseline or SMACv2 q-fit sweeps.

| Task | Nominal steps | ARec coefficient | q steps | Paired none source |
| --- | ---: | ---: | ---: | --- |
| `10m_vs_11m` | 10M | 0.000003 | 4 | Original task sweep |
| `3s5z_vs_3s6z` | 20M | 0.0003 | 4 | Original task sweep |
| `6s9z_vs_6s10z` | 20M | 0.0001 | 8 | Original task sweep |
| `smacv2_10_units` | 10M | 0.00003 | 4 | Original task sweep |

All eight YAML files freeze the effective model/PPO/environment hyperparameters
as well as the condition-specific values. They use independent actor networks
(`ACTOR_PARAMETER_SHARING=false`), matched NPS initialization, LR `0.002`,
four PPO epochs, 128 environments × 128 rollout steps, four minibatches, and
seeds 1–4. The only algorithmic difference in a task pair is ARec. Its q
learning rate and Fisher ridge are both `0.001`. The *nominal* budgets are 10M
or 20M; the current launcher records and uses the same whole-update effective
step count (9,994,240 or 19,988,480 with this rollout size).

`scripts/run_smax_first_four_panel_tuned.py` loads the two YAML files for one
task. Unless explicitly bypassed, it validates the old report's selected ARec
cell, source-root identity, old sweep settings, completed seeds, and all
effective hyperparameters in each source final checkpoint. It then invokes
the immutable `smax_four_method.py` control plane for exactly eight runs. The
new run manifest records the current Git commit, source hashes, the two YAML
hashes, generated commands, package/GPU environment, and the historical
verification hashes. A nonempty root without its matching immutable manifest
is refused. Do not use `--skip-historical-validation` for the formal replay.

On the server, pull the new code and activate `jaxmarl`, then validate the
source and preview commands for one task:

```bash
python scripts/run_smax_first_four_panel_tuned.py --task 10m_vs_11m --run-root /home/data/zeshenghong/JaxMARL/h1_smax_runs/first_four_panel_tuned_10m_v1 --historical-matrix-root /home/data/zeshenghong/JaxMARL/h1_smax_runs --gpus 0,1,2,3 --max-runs-per-gpu 2 --dry-run
```

Remove `--dry-run` to train. Use a different fresh run root for each task.
The existing monitor works on every such root:

```bash
python scripts/smax_four_method.py status --run-root /home/data/zeshenghong/JaxMARL/h1_smax_runs/first_four_panel_tuned_10m_v1
```

This replays the selected *configuration* on the current cleaned four-method
implementation. It is not a bitwise replay of the historical Git commit or
GPU environment; both historical and current identities are retained in the
manifest so discrepancies are auditable. `EXPERIMENT_CONDITION` was renamed
from historical `actor_score_recovery` to current `arec`; the verification
explicitly accounts for this metadata-only rename. For a strict old-code
replay, the historical commit and dependency environment must also be used.
