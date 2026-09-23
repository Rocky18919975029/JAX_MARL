# 6s9z_vs_6s10z isolated MAPPO PPO sweep

This is an exploratory NPS/`none` baseline sweep at **20M environment steps**
on the same seeds 1–4 as the actor-score-recovery comparison. It changes only
PPO learning rate and update epochs. The six cells are
`LR={0.0005,0.001,0.002} × UPDATE_EPOCHS={2,4}`. The existing baseline
`(0.002,4)` is included as an incumbent. `NUM_ENVS=128`,
`NUM_MINIBATCHES=4`, action/agent setup, learning-rate annealing, clipping,
entropy coefficient, checkpoint interval and the other benchmark settings
remain unchanged. All alignment, oracle and score-recovery losses are off.

The launcher stores a frozen `experiment_manifest.json`, keeps run names unique
per hyperparameter cell and seed, and refuses a mismatched rerun in the same
root. Each run has its own logs, metrics and checkpoints. A failed run can be
retried by relaunching with the same command. If partial metrics are present,
the launcher moves them to `retries/` before retrying; it does not delete them.

## Server smoke test

Use a separate root from every existing ARec experiment. The shell commands
are intentionally single-line to avoid lost continuation backslashes.

```bash
cd ~/JaxMARL
git pull --ff-only
conda activate jaxmarl
unset LD_LIBRARY_PATH
python -c 'import jax; print(jax.__version__, jax.devices())'
export BASE_SWEEP_SMOKE=/home/data/zeshenghong/JaxMARL/h1_smax_runs/none_6s9z_ppo_sweep_smoke_v1
python scripts/run_smax_nps_baseline_sweep.py --run-root "$BASE_SWEEP_SMOKE" --seeds 1 --learning-rates 0.001,0.002 --update-epochs-grid 4 --total-timesteps 16384 --gpus 0,1 --max-runs-per-gpu 1 --wandb-mode disabled
python scripts/monitor_smax_nps_baseline_sweep.py --run-root "$BASE_SWEEP_SMOKE"
```

Expect `COMPLETED=2`, `FAILED=0` before launching the full matrix.

## Full four-seed sweep

Do not run this concurrently with other full-GPU matrices unless capacity has
been checked. The default limit is two simultaneous runs per GPU, eight in
total. Each configuration receives the same 20M-step budget.

```bash
export BASE_SWEEP_ROOT=/home/data/zeshenghong/JaxMARL/h1_smax_runs/none_6s9z_ppo_sweep_4seed_v1
mkdir -p "$BASE_SWEEP_ROOT"
nohup python scripts/run_smax_nps_baseline_sweep.py --run-root "$BASE_SWEEP_ROOT" --seeds 1-4 --learning-rates 0.0005,0.001,0.002 --update-epochs-grid 2,4 --total-timesteps 20000000 --gpus 0,1,2,3 --max-runs-per-gpu 2 --project jaxmarl-smax-none-6s9z-ppo-sweep > "$BASE_SWEEP_ROOT/launcher.stdout" 2>&1 &
python scripts/monitor_smax_nps_baseline_sweep.py --run-root "$BASE_SWEEP_ROOT"
```

Use `watch -n 5 "python scripts/monitor_smax_nps_baseline_sweep.py --run-root '$BASE_SWEEP_ROOT'"`
for progress. `COMPLETED=24`, `FAILED=0` confirms all cells finished.

Analyze after completion:

```bash
python scripts/analyze_smax_nps_baseline_sweep.py --run-root "$BASE_SWEEP_ROOT"
```

`analysis/seed_level.csv` records return/win-rate AUC and the mean of the
last five **logged training updates** per seed. `analysis/config_summary.csv`
reports four-seed means, SDs, minimum seed AUCs and seed-bootstrap 95% CIs.
`analysis/selection.json` shows the winner by mean return AUC, the winner by
worst-seed return AUC, and the incumbent. The latter two are diagnostics: do
not pick a flat but poor baseline merely because its variance is small.

This grid is tuned on seeds 1–4. Any claim that ARec beats a tuned baseline
requires rerunning both conditions with the **same selected PPO settings** on
fresh confirmation seeds; the existing ARec curves used `(0.002,4)` and are
not a matched comparison against another baseline cell.
