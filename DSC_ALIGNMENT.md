# Directional Subspace Containment alignment

`ALIGN_DISTANCE=containment` implements Directional Subspace Containment
(DSC) as a third representation objective alongside `ln_mse` and
`linear_cka`.

For each agent-specific sample pool, DSC centers the alive source and target
latents, forms their feature covariances, and minimizes

\[
1-
\frac{\operatorname{tr}[(C_A+\alpha_A I)^{-1}C_{AC}
(C_C+\alpha_C I)^{-1}C_{AC}^{\top}]}
{\operatorname{tr}[(C_A+\alpha_A I)^{-1}C_A]+\varepsilon}.
\]

Here `source` is the representation receiving the directional gradient. For
`ALIGN_MODE=c_to_a`, source is the current actor latent and target is the
stored, stop-gradient rollout critic latent. The objective therefore asks how
much of the actor's effective subspace is contained in the critic subspace.
No per-sample LayerNorm is used.

The fixed numerical settings are:

```text
ALIGN_CONTAINMENT_RIDGE_RATIO=1e-3
ALIGN_CONTAINMENT_EPS=1e-6
```

These are covariance-solve parameters and are unrelated to the Fisher ridge
used by post-training diagnostics. PS and NPS both compute the loss separately
inside each agent slot. SMAX additionally defaults to `(slot, unit_type)`
groups with at least 64 alive samples:

```text
ALIGN_CONTAINMENT_GROUP_BY_UNIT_TYPE=true
ALIGN_CONTAINMENT_MIN_GROUP_SAMPLES=64
```

Training logs include `containment_similarity`,
`containment_source_effective_rank`, the weighted alignment objective, and
separate RL/alignment gradient norms. Checkpoint metadata records all DSC
ridge and grouping settings.

## Gradient-scale calibration

Do not select the DSC coefficient from returns. Match its initial
cross-gradient scale to LN-MSE at lambda 0.1 using an independent pilot seed.

MABrax HalfCheetah 6x1:

```bash
python scripts/calibrate_mabrax_cka.py \
  --target-distance containment \
  --output-root "$DSC_CAL_ROOT" \
  --pilot-seed 9001 \
  --gpus 0,1,2,3
```

SMAX:

```bash
python scripts/calibrate_h1_cka.py \
  --target-distance containment \
  --frozen-config "$FROZEN_CONFIG" \
  --output-root "$DSC_CAL_ROOT" \
  --maps 3s5z_vs_3s6z \
  --actor-variants nps \
  --pilot-seed 9001 \
  --gpus 0,1,2,3
```

HARL Humanoid 17x1:

```bash
python experiments/harl_mamujoco/calibrate_cka.py \
  --target-distance containment \
  --run-root "$DSC_CAL_ROOT" \
  --pilot-seed 9001 \
  --gpus 0,1,2,3
```

Each command writes `containment_gradient_calibration.json`.

## Opt-in launch

Existing launchers retain their original MSE/CKA defaults. DSC is added only
when explicitly selected, so existing manifests and resumable runs remain
unchanged.

For example, a DSC-only MABrax matrix uses:

```bash
python scripts/run_mabrax_alignment.py \
  --run-root "$DSC_RUN_ROOT" \
  --distances containment \
  --containment-calibration \
    "$DSC_CAL_ROOT/containment_gradient_calibration.json" \
  --actor-variants ps,nps \
  --seeds 1-4 \
  --gpus 0,1,2,3 \
  --max-runs-per-gpu 4
```

The generated condition/run-name suffix is `_dsc`.

A four-seed, NPS-only DSC matrix on SMAX `3s5z_vs_3s6z` uses:

```bash
python scripts/run_smax_alignment_benchmark.py \
  --run-root "$DSC_RUN_ROOT" \
  --frozen-config "$FROZEN_CONFIG" \
  --distances containment \
  --containment-calibration \
    "$DSC_CAL_ROOT/containment_gradient_calibration.json" \
  --maps 3s5z_vs_3s6z \
  --actor-variants nps \
  --seeds 1-4 \
  --reuse-root "$H1_ROOT" \
  --gpus 0,1,2,3 \
  --max-runs-per-gpu 4
```

This matrix contains one distance-free `none` baseline and the three DSC
directions (`c_to_a`, `a_to_c`, and `joint`) per seed. An exactly matching
completed baseline under `H1_ROOT` is reused rather than retrained.
