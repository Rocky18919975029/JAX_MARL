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

The confirmation wrapper performs this selection and launch automatically.
It recomputes `selection.json` from the completed pilot, checks that seeds
1–4 were not used for tuning, freezes a `confirmation_plan.json`, and runs
the two task-specific eight-run matrices **sequentially** so they do not
compete for the same GPUs. Each task has its own isolated runs, checkpoints,
metrics, and launcher log. It refuses a task whose best pilot setting does
not beat its paired isolated win-rate AUC unless explicitly overridden.
Within each task, the confirmation default is two concurrent runs per GPU
(eight runs across four GPUs); this changes scheduling only, not the training
protocol or selected hyperparameters.

```bash
export AREC_CONFIRM_ROOT=/home/data/zeshenghong/JaxMARL/h1_smax_runs/actor_score_recovery_confirm_4seed_v1
mkdir -p "$AREC_CONFIRM_ROOT"

python scripts/run_smax_actor_score_recovery_confirmation.py \
  --sweep-root "$AREC_SWEEP_ROOT" \
  --run-root "$AREC_CONFIRM_ROOT" \
  --seeds 1-4 \
  --gpus 0,1,2,3 \
  --max-runs-per-gpu 2 \
  --dry-run

nohup python scripts/run_smax_actor_score_recovery_confirmation.py \
  --sweep-root "$AREC_SWEEP_ROOT" \
  --run-root "$AREC_CONFIRM_ROOT" \
  --seeds 1-4 \
  --gpus 0,1,2,3 \
  --max-runs-per-gpu 2 \
  --project jaxmarl-smax-actor-score-recovery \
  > "$AREC_CONFIRM_ROOT/launcher.stdout" 2>&1 &
```

Monitor the active task with
`python scripts/monitor_smax_actor_score_recovery_sweep.py --run-root "$AREC_CONFIRM_ROOT/10m_vs_11m"`;
after it completes, use the same command with `3s5z_vs_3s6z`. Expect
`COMPLETED=8 FAILED=0` for each task. The selection is based on **pilot
training** win-rate AUC, not final performance or held-out evaluation; the
four new seeds provide the confirmatory comparison.

To backfill only the full-budget isolated runs for an already-running
single-coefficient formal matrix, use a **different root** and
`--conditions none --seeds 1-4`; this does not touch the active actor-side
workers. The same code's other defaults match the existing formal launcher.

## One seed-aggregated sweep figure

After the 28-run pilot matrix finishes, generate a single two-panel figure:

```bash
python scripts/plot_smax_actor_score_recovery_sweep.py \
  --run-root "$AREC_SWEEP_ROOT"
```

The output is `analysis/smax-actor-score-recovery-sweep-win-rate.png`, with
vector PDF/SVG masters and a plotted-data CSV beside it. The 10m and 3s5z
tasks occupy separate panels and are **never averaged together**. Each panel
contains the paired isolated baseline plus all six `(lambda, q_steps)`
settings. Each curve is the mean over training seeds at the same exact
`env_step`. The shaded band is a pointwise 95% percentile interval from
20,000 resamples of whole training seeds (fixed random seed). Missing or
non-finite steps are excluded; curves are neither smoothed nor interpolated.
The baseline is neutral gray, lambda is encoded by color and marker, and q
fitting steps by line style. The figure's metadata JSON records the bootstrap
protocol.

This pilot has only two seeds, so its bootstrap bands are exploratory: they
often span essentially the two observed seed curves and should not be cited
as confirmatory 95% uncertainty. For a paper result, rerun the selected
setting and a matched isolated baseline on unused seeds 1–4, then make a
separate figure from that confirmation root.

## Denser 10m_vs_11m tuning on seeds 1–4

If the four-seed confirmation is inconclusive, use a **new root and W&B
project** for a targeted full-budget grid. The five coefficients below refine
the low-lambda region where the pilot was most promising; `q_steps=2,4,8`
tests probe fitting strength while keeping q learning rate and Fisher ridge
fixed. This is 5 × 3 × 4 = 60 recovery runs plus four paired isolated runs,
each at 10M steps. The two-run-per-GPU limit means at most eight workers at
once. Do not start this while the previous full-GPU matrix is still active.

```bash
export AREC_DETAIL_ROOT=/home/data/zeshenghong/JaxMARL/h1_smax_runs/actor_score_recovery_10m_detail_4seed_v1
mkdir -p "$AREC_DETAIL_ROOT"

python scripts/run_smax_actor_score_recovery_sweep.py \
  --run-root "$AREC_DETAIL_ROOT" \
  --maps 10m_vs_11m --seeds 1-4 \
  --conditions none,actor_score_recovery \
  --coefs 0.000003,0.00001,0.00002,0.00003,0.00005 \
  --q-steps-grid 2,4,8 \
  --q-learning-rates 0.001 --fisher-ridges 0.001 \
  --total-timesteps 10000000 \
  --gpus 0,1,2,3 --max-runs-per-gpu 2 \
  --project jaxmarl-smax-arec-10m-detail4seed \
  --dry-run

nohup python scripts/run_smax_actor_score_recovery_sweep.py \
  --run-root "$AREC_DETAIL_ROOT" \
  --maps 10m_vs_11m --seeds 1-4 \
  --conditions none,actor_score_recovery \
  --coefs 0.000003,0.00001,0.00002,0.00003,0.00005 \
  --q-steps-grid 2,4,8 \
  --q-learning-rates 0.001 --fisher-ridges 0.001 \
  --total-timesteps 10000000 \
  --gpus 0,1,2,3 --max-runs-per-gpu 2 \
  --project jaxmarl-smax-arec-10m-detail4seed \
  > "$AREC_DETAIL_ROOT/launcher.stdout" 2>&1 &
```

After completion, run `analyze_smax_actor_score_recovery_sweep.py` on this
root. Its per-task CSV now reports seed-paired AUC and last-five-logged-update
win-rate differences against isolated, plus the number of seeds with positive
differences. The last-five measure is **not** a held-out final-checkpoint
evaluation. Review both AUC and final-performance columns; a large mean
driven by one seed is not a robust win. If these four seeds are used to choose
another hyperparameter setting, they become **tuning seeds**. Any subsequent
unbiased confirmation requires new, unused seeds (for example 5–8); do not
reuse seeds 1–4 to claim an independent confirmation.

## One-shot plot while a sweep is still running

The completed-run plot above deliberately rejects unfinished runs. For an
in-progress matrix, take a fresh snapshot whenever desired (no background
plotting process is needed):

```bash
AREC_DETAIL_ROOT="/home/data/zeshenghong/JaxMARL/h1_smax_runs/actor_score_recovery_10m_detail_4seed_v1"
python scripts/plot_smax_actor_score_recovery_live.py --run-root "$AREC_DETAIL_ROOT"
```

The command overwrites PNG/PDF/SVG, plotted-data CSV, coverage CSV, and
metadata JSON under `$AREC_DETAIL_ROOT/analysis/live/10m_vs_11m/`. The PNG is
`smax-arec-sweep-live-win-rate.png`. Run the command again later to refresh
from the latest locally logged metrics. The figure has separate q-step panels,
each with the isolated baseline and the five coefficient curves. If a sweep
root contains multiple tasks, it writes a separate figure for each task.

For each curve, the script uses running/completed seeds with at least two
finite logged updates, intersecting their **exact** `env_step` grids. It
resamples whole training seeds for a pointwise percentile 95% bootstrap band;
one-seed curves have no band. Missing, pending, or failed runs are never
silently counted as zero. The per-curve CSV records the actual seed IDs and
`n_seeds`, because different in-progress curves can have different cohort
sizes and horizons. These are exploratory live snapshots, not a matched
four-seed comparison until all four seeds finish. There is no smoothing,
interpolation, or cross-task averaging.
