# H1 NPS representation-interaction experiment

This is the only active H1 analysis protocol in this repository. It uses
non-parameter-sharing (NPS) actors and treats LN-MSE and Linear CKA as
co-primary alignment-distance strata.
The training checkpoints retain their `h1-v1.0` metadata; the new downstream
measurement/analysis protocol is `h1-nps-two-distance-v2.1`.

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

With a single fixed absolute numerical ridge `xi` shared by every task,
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

## Fixed numerical setting and interpretation

`FISHER_RIDGE_ABSOLUTE=0.001` is a numerical-stability setting used uniformly
in every Fisher solve. It is not an H1 acceptance threshold.
Conceptual tolerances `delta_A` and `delta_C` may describe functionally
acceptable loss of actor or critic information in the theory, but this
analysis does not turn them into numerical pass/fail cutoffs.

The prior v2.0 recomputation used `DELTA_DEC=0.05` and `DELTA_BELL=0.005` in a
binary signature table. Those values remain historical metadata for that run;
they are not used in the v2.1 mechanism interpretation.

The measurement definitions are unchanged from the ongoing v2.0 recomputation.
Let it finish before pulling this code on the server. Afterward, reuse its
canonical latent, decision, and Bellman summaries and regenerate only the
analysis and figures in a new directory:

## Reanalyze the completed measurements

```bash
cd ~/JaxMARL
git pull --ff-only
conda activate jaxmarl
unset LD_LIBRARY_PATH

export MSE_ROOT="/home/data/zeshenghong/JaxMARL/h1_smax_runs/reduced_nps_4condition"
export CKA_ROOT="/home/data/zeshenghong/JaxMARL/h1_smax_runs/cka_distance_robustness"
export H1_TRENDS_ROOT="/home/data/zeshenghong/JaxMARL/h1_smax_runs/nps_mse_cka_h1_trends_v2p1"

python scripts/analyze_h1_mechanisms.py \
  --mse-root "$MSE_ROOT" \
  --cka-root "$CKA_ROOT" \
  --output-root "$H1_TRENDS_ROOT" \
  --fisher-ridge-absolute 0.001

python scripts/plot_h1_mechanisms.py \
  --analysis-root "$H1_TRENDS_ROOT"
```

This is analysis-only: it does not rerun training, collect, latent, decision,
or Bellman diagnostics. It rejects incomplete canonical summaries and Fisher
ridge mismatches rather than silently mixing protocols. For a fresh dataset
with incomplete downstream markers, use `run_h1_nps_diagnostics.py` with the
same roots, ridge, and a fresh analysis directory; its four-card launcher
remains resumable and no longer accepts decision/Bellman tolerances.

For a separate full downstream recomputation, monitor its stage markers:

```bash
for ROOT in "$MSE_ROOT" "$CKA_ROOT"; do
  echo "===== $ROOT ====="
  printf "latent:   "; find "$ROOT/diagnostics_raw" -name latent_summary.json | wc -l
  printf "decision: "; find "$ROOT/diagnostics_raw" -name decision_summary.json | wc -l
  printf "bellman:  "; find "$ROOT/diagnostics_raw" -name bellman_summary.json | wc -l
done
```

Expected selected totals are `192/192/192` for LN-MSE and `128/128/128` for
Linear CKA. LN-MSE may contain extra raw `joint` collections, but canonical
markers for those runs are neither required nor generated.

## Outputs and claim–evidence interpretation

The combined analysis directory contains:

- `analysis_protocol.json`: scope, fixed ridge, roots, and seed unit
- `checkpoint_metrics.csv`: all four absolute measurements per seed/checkpoint
  plus nested `M,2M,4M` MC-convergence and decision-probe quality diagnostics
- `curve_summary.csv`: seed mean, SD, SE, and descriptive 95% seed-bootstrap CI
- `paired_effects.csv`: within-seed alignment-minus-`none` differences, without
  condition pass/fail labels
- `decision_probe_audit.csv`: Kendall tau, pairwise accuracy, top-1 agreement,
  and test-anchor count for every seed/checkpoint
- `seed_return_auc.csv` and `return_auc_paired_summary.csv`
- `temporal_precedence_points.csv` and `temporal_precedence_summary.csv`
- `figures/*.png` and `figures/*.pdf`: absolute and paired curves showing each
  seed plus mean/interval, and a separate decision-probe validity figure

The statistical unit is always the training seed. Episodes and agent-time
samples are never treated as independent experimental replicates.

For a performance-improving alignment, inspect whether its seed-paired
`epsilon_Lat` difference is usually negative at the same checkpoints and
whether `epsilon_Dec` and `epsilon_Bell` lack clear, sustained worsening.
This is a mechanism trend, not a binary non-inferiority trial or a causal
proof. A gain without latent improvement cannot be explained by H1; a latent
gain accompanied by persistent actor-decision or critic-Bellman degradation
cannot support the phrase “without sacrificing.”

Measurement validity is separate from trend direction. Random action ordering
has expected Kendall tau `0`, pairwise accuracy `0.5`, and hence
`epsilon_Dec=1`. If the decision probe stays near those levels, an unchanged
`epsilon_Dec` does not establish preserved decision sufficiency. The validity
figures show these reference lines; top-1 agreement is reported descriptively
because its random baseline depends on the number of legal actions.

## Offline SMACv2 slot-by-type audit

SMACv2 samples a unit type for each actor slot at episode reset.  The NPS
actor module assigned to slot `i` is fixed, but the unit type occupying that
slot changes across episodes.  The original diagnostic pooled all valid
samples within slot before constructing its gradient mismatch and Fisher
matrix.  The audit therefore reports both

```
epsilon_Lat_slot      = sum_i D_i
epsilon_Lat_slot_type = sum_i sum_k p_i,k D_i,k
```

where `p_i,k` is the valid-sample frequency of type `k` inside slot `i`.
It also computes held-out Linear CKA in the same two ways:

```
d_CKA_slot      = mean_i (1 - CKA_i)
d_CKA_slot_type = mean_i sum_k p_i,k (1 - CKA_i,k)
```

Slots are never pooled, so independent actor encoders remain separate.  The
calculation streams sufficient statistics from the existing episode shards;
it does not recollect trajectories, alter checkpoints, or overwrite canonical
H1 summaries.  `10m_vs_11m` is included as a mandatory one-type control: both
versions of each metric must agree there to numerical precision.  When the
canonical baseline-free MC latent summary is present, the new streaming
slot-pooled result must also reproduce it before the audit can finish.

Run the resumable offline audit with:

```bash
cd ~/JaxMARL
conda activate jaxmarl
unset LD_LIBRARY_PATH

export H1_ROOT="/home/data/zeshenghong/JaxMARL/h1_smax_runs"
export MSE_ROOT="$H1_ROOT/reduced_nps_4condition"
export CKA_ROOT="$H1_ROOT/cka_distance_robustness"
export TYPE_AUDIT_ROOT="$H1_ROOT/nps_slot_type_audit"
mkdir -p "$TYPE_AUDIT_ROOT"

nohup python scripts/run_h1_type_conditioning_audit.py \
  --mse-root "$MSE_ROOT" \
  --cka-root "$CKA_ROOT" \
  --output-root "$TYPE_AUDIT_ROOT" \
  --maps 10m_vs_11m,smacv2_10_units \
  --seeds 1,2,3,4 \
  --fisher-ridge-absolute 0.001 \
  --workers 4 \
  > "$TYPE_AUDIT_ROOT/pipeline.stdout" 2>&1 &
```

Completed checkpoint JSON files are cache markers and are skipped on restart.
The audit writes `tables/checkpoint_type_conditioning.csv`,
`tables/curve_summary.csv`, `tables/paired_seed_differences.csv`,
`tables/paired_curve_summary.csv`, and one absolute/paired figure per
task-distance combination under `figures/`.  The distance-free `none` rows are
reused exactly in the Linear CKA stratum rather than retrained or recomputed.
