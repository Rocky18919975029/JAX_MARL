# H1 NPS representation-interaction experiment

This is the only active H1 analysis protocol in this repository. It uses
non-parameter-sharing (NPS) actors and treats LN-MSE and Linear CKA as
co-primary alignment-distance strata.
The training checkpoints retain their `h1-v1.0` metadata; the new downstream
measurement/analysis protocol is `h1-nps-two-distance-v2.0`.

## Fixed matrix

- Tasks: `10m_vs_11m`, `smacv2_10_units`
- Actor parameterization: NPS only
- Conditions in each distance stratum: `none`, `a_to_c`, `c_to_a`
- Training seeds: `1,2,3,4`
- Alignment coefficient: the coefficient already frozen for each distance
- Checkpoints: `0, 0.5M, 1M, 2M, 4M, 6M, 8M, 10M` environment steps

The `none` policy is independent of alignment distance. The exact same NPS
LN-MSE `none` runs and held-out data are therefore reused as the Linear CKA
baseline. They are not retrained and are not counted as new experimental
replicates. `joint`, `reciprocal`, parameter-sharing, shuffled controls, and
deterministic post-hoc evaluation are outside this H1 analysis.
There are 40 unique trained runs (24 LN-MSE + 16 CKA), displayed as 48
distance-stratified condition cells because the 8 `none` runs appear as the
same comparator in both distance strata—not as independent observations.
The Linear CKA coefficient was fixed by the earlier return-independent pilot
gradient-scale calibration (pooled across PS/NPS pilot cells); it is not
reselected for this NPS subset. The analyzer verifies LN-MSE uses `0.1` and
all selected CKA runs use one identical coefficient.

The existing held-out collection is sufficient:

- LN-MSE root: 192 selected checkpoints (the 64 `joint` checkpoints are ignored)
- Linear CKA root: 128 selected checkpoints
- Each checkpoint: 512 complete, independently sampled stochastic episodes

## Canonical measurements

For agent `i`, let

```
s_i,t = grad_{z^A_i,t} log pi_i(a_i,t | z^A_i,t)
```

The reference gradient uses raw complete Monte Carlo return-to-go and no
baseline or bootstrap:

```
g_ref_i = mean(s_i,t * G_t)
```

The critic-induced gradient uses raw, unnormalized GAE reconstructed with the
checkpoint's training `GAMMA`, `GAE_LAMBDA`, `NUM_STEPS`, termination mask, and
rollout-boundary bootstrap:

```
g_critic_i = mean(s_i,t * A_GAE_i,t)
```

With a single preregistered absolute ridge `xi` shared by every task,
condition, distance, seed, and checkpoint:

```
epsilon_Lat = sum_i (g_ref_i - g_critic_i)^T
                    (F_i + xi I)^-1
                    (g_ref_i - g_critic_i)
F_i = mean(s_i,t s_i,t^T)
```

There is no relative latent-distortion statistic in H1. Nested `M,2M,4M`
episode subsets are saved only as a Monte Carlo convergence audit.

Actor decision sufficiency fits one independent probe per agent on
episode-disjoint data and reports

```
epsilon_Dec = 1 - mean KendallTau(
    counterfactual action values,
    probe-predicted action values
)
```

Lower is better. Critic Bellman compatibility uses a fixed bank of value heads,
episode-disjoint decoder fitting, and reports the untruncated worst held-out
latent residual:

```
epsilon_Bell = max_m E_test[(g_m(z_t^C) - y_m,t)^2]
y_m,t = r_t + gamma * (1 - done_t) * f_m(z_t+1^C)
```

The performance measurement is undiscounted episode return from those same 512
held-out stochastic episodes. Thus performance and all three representation
criteria use one frozen-checkpoint data protocol.

## Required preregistration values

The values fixed before recomputation are:

- `FISHER_RIDGE_ABSOLUTE=0.001`: the fixed absolute Fisher ridge `xi`
- `DELTA_DEC=0.05`: decision non-inferiority tolerance
- `DELTA_BELL=0.005`: Bellman non-inferiority tolerance

The launcher requires all three explicitly and writes them to
`recompute_manifest.json` before processing and `analysis_protocol.json` in
the final analysis. It never guesses them or tunes them from return.

## Recompute from the existing collections

```bash
cd ~/JaxMARL
git pull --ff-only
conda activate jaxmarl
unset LD_LIBRARY_PATH

export MSE_ROOT="/home/data/zeshenghong/JaxMARL/h1_smax_runs/reduced_nps_4condition"
export CKA_ROOT="/home/data/zeshenghong/JaxMARL/h1_smax_runs/cka_distance_robustness"
export H1_ANALYSIS_ROOT="/home/data/zeshenghong/JaxMARL/h1_smax_runs/nps_mse_cka_h1"
mkdir -p "$H1_ANALYSIS_ROOT"

export FISHER_RIDGE_ABSOLUTE=0.001
export DELTA_DEC=0.05
export DELTA_BELL=0.005

python scripts/run_h1_nps_diagnostics.py \
  --mse-root "$MSE_ROOT" \
  --cka-root "$CKA_ROOT" \
  --analysis-root "$H1_ANALYSIS_ROOT" \
  --fisher-ridge-absolute "$FISHER_RIDGE_ABSOLUTE" \
  --delta-dec "$DELTA_DEC" \
  --delta-bell "$DELTA_BELL" \
  --gpus 0,1,2,3 \
  --dry-run

nohup python -u scripts/run_h1_nps_diagnostics.py \
  --mse-root "$MSE_ROOT" \
  --cka-root "$CKA_ROOT" \
  --analysis-root "$H1_ANALYSIS_ROOT" \
  --fisher-ridge-absolute "$FISHER_RIDGE_ABSOLUTE" \
  --delta-dec "$DELTA_DEC" \
  --delta-bell "$DELTA_BELL" \
  --gpus 0,1,2,3 \
  > "$H1_ANALYSIS_ROOT/pipeline.stdout" 2>&1 &
```

The pipeline never recollects trajectories. It skips a completed canonical
stage by its output marker and runs the four selected GPUs in parallel.
Before any processing, it also verifies both roots have identical frozen
optimization configs and records their shared digest in the recompute manifest.

Monitor:

```bash
tail -f "$H1_ANALYSIS_ROOT/pipeline.stdout"

watch -n 10 '
for ROOT in "'$MSE_ROOT'" "'$CKA_ROOT'"; do
  echo "===== $ROOT ====="
  printf "latent:   "; find "$ROOT/diagnostics_raw" -name latent_summary.json | wc -l
  printf "decision: "; find "$ROOT/diagnostics_raw" -name decision_summary.json | wc -l
  printf "bellman:  "; find "$ROOT/diagnostics_raw" -name bellman_summary.json | wc -l
done
nvidia-smi
'
```

Expected selected totals are `192/192/192` for LN-MSE and `128/128/128` for
Linear CKA. LN-MSE may contain extra raw `joint` collections, but canonical
markers for those runs are neither required nor generated.

## Outputs and decision rule

The combined analysis directory contains:

- `analysis_protocol.json`: frozen scope, ridge, tolerances, roots, and seed unit
- `recompute_manifest.json`: locked downstream settings for resumable processing
- `checkpoint_metrics.csv`: all four absolute measurements per seed/checkpoint
  plus nested `M,2M,4M` reference-direction and distortion convergence audits
- `curve_summary.csv`: seed mean, SD, SE, and 95% seed-bootstrap CI
- `paired_effects.csv`: within-seed alignment-minus-`none` differences
- `h1_signature.csv`: checkpoint-level confirmatory signature
- `seed_return_auc.csv` and `return_auc_paired_summary.csv`
- `temporal_precedence_points.csv` and `temporal_precedence_summary.csv`
- `figures/*.png` and `figures/*.pdf`: one four-panel figure per task/distance

The statistical unit is always the training seed. Episodes and agent-time
samples are never treated as independent experimental replicates.

For a condition whose return improves over paired `none`, H1 requires:

```
Delta epsilon_Lat < 0
upper_CI(Delta epsilon_Dec) <= DELTA_DEC
upper_CI(Delta epsilon_Bell) <= DELTA_BELL
```

`h1_signature.csv` records both mean-direction and stricter 95%-CI versions.
Failure of a low-performing alignment is not itself evidence against H1; the
test is whether a performance-improving condition exhibits the complete
three-metric signature.
