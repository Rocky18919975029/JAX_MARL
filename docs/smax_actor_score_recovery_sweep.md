# SMAX actor-score-recovery tuning with a matched isolated baseline

The sweep launcher is separate from the existing single-coefficient formal
launcher and never modifies its manifest or checkpoints. It runs NPS `none`
once per map/seed and actor-side score recovery for every requested
`lambda × q_steps × q_learning_rate × Fisher_ridge` cell. Every cell on a map
uses the same MAPPO budget, seed, PPO settings and W&B project. The isolated
run disables latent alignment, critic-side recovery, actor-side recovery and
oracle loss. The default scan varies the two most consequential controls,
`lambda` and the number of q fitting steps; q learning rate and Fisher ridge
are fixed by default but can be varied explicitly.

Use a fresh root for each frozen grid. The launcher skips completed or active
runs and records the exact matrix in `experiment_manifest.json`. Its monitor
shows every run, including `none`.

## Smoke test on the server

```bash
cd ~/JaxMARL
git pull --ff-only
conda activate jaxmarl
unset LD_LIBRARY_PATH

export AREC_SWEEP_SMOKE=/home/data/zeshenghong/JaxMARL/h1_smax_runs/actor_score_recovery_sweep_smoke_v1
python scripts/run_smax_actor_score_recovery_sweep.py \
  --run-root "$AREC_SWEEP_SMOKE" \
  --maps 10m_vs_11m \
  --seeds 9001 \
  --coefs 0.0001 \
  --q-steps-grid 8 \
  --total-timesteps 16384 \
  --gpus 0,1 \
  --wandb-mode disabled
python scripts/monitor_smax_actor_score_recovery_sweep.py \
  --run-root "$AREC_SWEEP_SMOKE"
```

## Exploratory screening grid

Do not run this in parallel with another full-GPU matrix unless capacity has
been checked. The default map budgets are 10M and 20M steps respectively.
For a shorter screening grid, pass `--budget-fraction 0.5` and treat its
ranking as provisional; some SMAX policies improve late. The following
full-budget grid has 28 runs: two maps, pilot seeds 9001–9002, one paired
isolated run and six intervention settings per map/seed.

```bash
export AREC_SWEEP_ROOT=/home/data/zeshenghong/JaxMARL/h1_smax_runs/actor_score_recovery_sweep_pilot_v1
mkdir -p "$AREC_SWEEP_ROOT"
nohup python scripts/run_smax_actor_score_recovery_sweep.py \
  --run-root "$AREC_SWEEP_ROOT" \
  --maps 10m_vs_11m,3s5z_vs_3s6z \
  --seeds 9001-9002 \
  --coefs 0.00003,0.0001,0.0003 \
  --q-steps-grid 4,8 \
  --q-learning-rates 0.001 \
  --fisher-ridges 0.001 \
  --gpus 0,1,2,3 \
  --max-runs-per-gpu 1 \
  --project jaxmarl-smax-actor-score-recovery \
  > "$AREC_SWEEP_ROOT/launcher.stdout" 2>&1 &

watch -n 5 "python scripts/monitor_smax_actor_score_recovery_sweep.py --run-root '$AREC_SWEEP_ROOT'"
```

After all runs finish:

```bash
python scripts/analyze_smax_actor_score_recovery_sweep.py \
  --run-root "$AREC_SWEEP_ROOT"
```

The analyzer writes separate CSVs for each map and `analysis/selection.json`.
It ranks settings by the **training** win-rate AUC improvement paired by
screening seed against `none`, with return AUC as a tie-breaker. The last-five
metric is the last five *logged updates*, not five held-out checkpoints. This
is a tuning diagnostic, not a paper result. If no setting beats `none`, the
selection file explicitly says not to promote a candidate. Selected settings
must be evaluated at full budget on unused training seeds (for example 1–4),
with their own matched `none` runs; do not report the winning pilot seeds as
an unbiased confirmatory estimate. Maps are selected and reported separately.

For confirmation, run the same launcher in a new root for each map with
`--seeds 1-4 --conditions none,actor_score_recovery`, a single selected
`--coefs` value and a single selected value for each other tuned parameter.
Omit `--budget-fraction` so the full 10M/20M map budget is used. Separate
roots are necessary if the two maps select different hyperparameters.

To backfill only the full-budget isolated runs for an already-running
single-coefficient formal matrix, use a **different root** and
`--conditions none --seeds 1-4`; this does not touch the active actor-side
workers. The same code's other defaults match the existing formal launcher.
