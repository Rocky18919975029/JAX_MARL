# SMAX action-conditioned linear score recoverability

This is an offline measurement of **frozen, final-checkpoint NPS policies**. It
reuses the `collected/` trajectories and frozen fit/test protocol produced by
`run_smax_score_recoverability.py`; it does not train MAPPO, collect new
trajectories, or fit the earlier MLP probe. Use a separate output directory so
the previous measurement remains untouched.

For agent `i`, action `a`, fit **only on fit episodes**:

    u = (F_fit + xi I)^(-1/2) score
    z = (z_critic - mean_fit) / (std_fit + 1e-6)
    psi_i(z, a) = W_i,a z + b_i,a

`F_fit` is the existing empirical Fisher from fit scores and `xi` is the
existing frozen absolute Fisher ridge. Every `(agent, action)` gets its own
linear readout. The normal equations use a fixed probe ridge (default
`1e-3`), with an **unpenalized intercept**. No SGD or test-set model selection
is used. The same fit-only Fisher map and critic statistics are applied to
test episodes. Actor and critic networks are never updated.

An action is fit only when its fit-sample count is at least `critic latent
dimension + 1` (129 for the usual 128-D latent). Unsupported test actions use
the zero predictor; their counts and fractions are reported in
`action_support.csv`, `agent_metrics.csv`, and the summaries. This rule is
fixed across tasks and conditions. Check `fallback_test_fraction` before
interpreting any comparison.

Each agent's reported metric is the **held-out**, un-clipped ratio

    sum_test ||u - psi_i(z_critic,a)||^2 / sum_test ||u||^2.

The seed score is the unweighted agent mean. A finite fitted probe can score
**above 1** on test episodes even though the theoretical infimum over all
predictors cannot; such a value is intentionally retained and flagged by
comparison with the zero-predictor baseline `1.0`. It is a generalization
failure, not evidence that the theoretical optimum exceeds 1. The notation is
`epsilon_Rec^lin`, not the unrestricted `epsilon_Rec`.

## Server smoke test

After pulling the code and activating `jaxmarl`:

```bash
cd ~/JaxMARL
git pull --ff-only
conda activate jaxmarl
unset LD_LIBRARY_PATH
export SOURCE_ROOT=/home/data/zeshenghong/JaxMARL/h1_smax_runs/score_recoverability_formal_v1
export LINEAR_SMOKE_ROOT=/home/data/zeshenghong/JaxMARL/h1_smax_runs/score_recoverability_linear_smoke_v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/run_smax_linear_score_recoverability.py \
  --source-root "$SOURCE_ROOT" \
  --run-root "$LINEAR_SMOKE_ROOT" \
  --tasks 10m_vs_11m \
  --seeds 1 \
  --workers 2
```

Inspect `analysis/10m_vs_11m/` in the smoke root, especially
`action_support.csv`, `agent_level.csv`, and the figure. The source root must
be the directory with `protocol.json`, `sample_counts.json`, and `runs/` from
the completed original measurement. If yours has a different name, set
`SOURCE_ROOT` accordingly.

## Full 3-task × 3-condition × 4-seed matrix

```bash
export LINEAR_ROOT=/home/data/zeshenghong/JaxMARL/h1_smax_runs/score_recoverability_linear_v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/run_smax_linear_score_recoverability.py \
  --source-root "$SOURCE_ROOT" \
  --run-root "$LINEAR_ROOT" \
  --workers 4
```

The runner resumes completed cells and freezes its protocol. Each task has
its own figure and tables in `analysis/<task>/`; the root tables contain
task-labelled rows but **never pool different tasks**. A 95% percentile CI is
bootstrapped over training seeds, not agents or transitions.

For a numerical-stability check only, run the same command with separate
`--run-root` directories and `--ridge 1e-4`, `1e-3`, and `1e-2`. Do not choose
a ridge independently per condition based on held-out results. The fixed
primary value is `1e-3`.
