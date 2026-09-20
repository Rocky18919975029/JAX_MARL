# SMAX independent-reference oracle training

The v2 protocol compares matched NPS MAPPO against MAPPO with the additional
objective

\[
\epsilon_{\mathrm{Lat}}
=\sum_i (g_i^{\mathrm{ref}}-g_i^{\mathrm{GAE}})^\top
(F_i^{\mathrm{ref}}+\xi I)^{-1}
(g_i^{\mathrm{ref}}-g_i^{\mathrm{GAE}}).
\]

The two estimates no longer reuse the same rollout:

- `g_GAE` uses the ordinary PPO minibatch and its raw, unnormalised training
  GAE.
- Before each PPO update, `g_ref` uses fresh environments sampled by the frozen
  pre-update policy. The default horizon guarantees at least four times as many
  complete reference transitions in every minibatch.
- Reference returns never bootstrap. The unfinished final episode in every
  reference environment is masked out.
- The default reference signal is `G_MC - b_cf(s)`. The baseline is an affine
  calibration of the frozen pre-update centralized critic, fitted with two-fold
  cross-fitting across independent environments. A held-out transition never
  helps fit its own baseline. The baseline is action-independent, so it is a
  control variate rather than a bootstrap.
- MC returns, the frozen baseline, and raw GAE are all stop-gradient.
- The reference Fisher is estimated only from the independent reference data.
- Old-policy importance ratios keep the fixed rollout valid across PPO epochs.

The run logs `ppo_env_step`, `oracle_env_step`, and `total_env_step` separately.
Performance plots may use `ppo_env_step`, but compute/sample-cost comparisons
must also report `total_env_step`.

## Smoke test

Use a new output root; v1 artifacts and coefficients are incompatible.

```bash
cd ~/JaxMARL
git pull --ff-only
conda activate jaxmarl
unset LD_LIBRARY_PATH

export SMOKE_ROOT="/home/data/zeshenghong/JaxMARL/smax_oracle_independent_ref_smoke_v2p1"
mkdir -p "$SMOKE_ROOT"

nohup python scripts/run_smax_oracle_training.py \
  --run-root "$SMOKE_ROOT" \
  --map-name 10m_vs_11m \
  --seeds 1 \
  --conditions none,oracle_latent_distortion \
  --oracle-coef 1 \
  --fisher-ridge 0.001 \
  --reference-multiplier 4 \
  --reference-baseline crossfit_linear_critic \
  --total-timesteps 16384 \
  --update-epochs 1 \
  --gpus 0,1 \
  --max-runs-per-gpu 1 \
  --wandb-mode disabled \
  > "$SMOKE_ROOT/training.stdout" 2>&1 &

echo "Smoke manager PID: $!"
watch -n 5 "python scripts/monitor_smax_oracle_training.py --run-root '$SMOKE_ROOT'"
```

After completion, audit the estimator:

```bash
python - "$SMOKE_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
path = next((root / "metrics").glob("*oracle_latent_distortion*.jsonl"))
row = json.loads(path.read_text().splitlines()[-1])
for key in (
    "oracle_reference_valid_samples",
    "oracle_critic_valid_samples",
    "oracle_reference_to_critic_sample_ratio",
    "oracle_reference_mc_return_std",
    "oracle_reference_baselined_advantage_std",
    "oracle_reference_baseline_variance_reduction",
    "oracle_epsilon_lat",
    "oracle_actor_to_rl_grad_ratio_mean",
    "ppo_env_step",
    "oracle_env_step",
    "total_env_step",
):
    print(f"{key}: {row[key]}")
PY
```

The smoke test passes only if both runs finish, the reference/critic effective
sample ratio is at least four, and all oracle metrics are finite. A negative
variance-reduction value is allowed diagnostically—it means the frozen critic
is a poor control variate at that checkpoint and the `zero` baseline should be
retained as an ablation.

## Coefficient pilot

Do not reuse the v1 coefficient. First run the v2 estimator with coefficient
one, zero learning rate, and one update. The resulting gradient ratio is a
scale diagnostic, not a theoretically privileged target. If a coefficient is
selected, report the chosen target ratio and include a small sensitivity sweep.

```bash
export ORACLE_ROOT="/home/data/zeshenghong/JaxMARL/smax_oracle_10m_vs_11m_v2"
export PILOT_ROOT="$ORACLE_ROOT/gradient_calibration"
mkdir -p "$PILOT_ROOT"

nohup python scripts/run_smax_oracle_training.py \
  --run-root "$PILOT_ROOT" \
  --map-name 10m_vs_11m \
  --seeds 9001 \
  --conditions oracle_latent_distortion \
  --oracle-coef 1 \
  --reference-multiplier 4 \
  --reference-baseline crossfit_linear_critic \
  --total-timesteps 16384 \
  --update-epochs 1 \
  --learning-rate 0 \
  --gpus 0 \
  --max-runs-per-gpu 1 \
  --wandb-mode disabled \
  > "$PILOT_ROOT/launcher.stdout" 2>&1 &
```
