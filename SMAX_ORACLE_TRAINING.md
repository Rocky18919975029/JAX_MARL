# SMAX oracle latent-distortion training

This protocol compares matched NPS MAPPO against MAPPO with the additional
objective

\[
\epsilon_{\mathrm{Lat}}
=\sum_i (g_i^{\mathrm{MC}}-g_i^{\mathrm{GAE}})^\top
(F_i+\xi I)^{-1}(g_i^{\mathrm{MC}}-g_i^{\mathrm{GAE}}).
\]

At every rollout, `g_MC` uses baseline-free discounted Monte Carlo return to
go and `g_GAE` uses the unnormalized GAE reconstructed by the training critic.
Only transitions with an observed episode termination inside the current
rollout are used; the incomplete suffix is never bootstrapped. Both scalar
signals are explicitly stop-gradient. The oracle objective differentiates
only through the current actor latent score `d log pi(a|z) / dz`.

## 1. Reward-free coefficient calibration

The pilot uses one rollout, zero learning rate, and no return-based model
selection. It selects the coefficient whose oracle/PPO actor-gradient norm
ratio is 0.1.

```bash
cd ~/JaxMARL
conda activate jaxmarl
unset LD_LIBRARY_PATH

export ORACLE_ROOT="/home/data/zeshenghong/JaxMARL/smax_oracle_10m_vs_11m"
export PILOT_ROOT="$ORACLE_ROOT/gradient_calibration"
mkdir -p "$PILOT_ROOT"

nohup python scripts/run_smax_oracle_training.py \
  --run-root "$PILOT_ROOT" \
  --map-name 10m_vs_11m \
  --seeds 9001 \
  --conditions oracle_latent_distortion \
  --oracle-coef 1 \
  --total-timesteps 16384 \
  --update-epochs 1 \
  --learning-rate 0 \
  --gpus 0 \
  --max-runs-per-gpu 1 \
  --wandb-mode disabled \
  > "$PILOT_ROOT/launcher.stdout" 2>&1 &

wait $!

python scripts/select_smax_oracle_coef.py \
  --metrics "$PILOT_ROOT/metrics/SMAX-ORACLE-10m_vs_11m-nps-oracle_latent_distortion-seed9001.jsonl" \
  --output "$PILOT_ROOT/oracle_gradient_calibration.json" \
  --target-gradient-ratio 0.1
```

## 2. Four-seed comparison

```bash
export FORMAL_ROOT="$ORACLE_ROOT/formal_4seed"
export ORACLE_COEF="$(python - "$PILOT_ROOT/oracle_gradient_calibration.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["oracle_distortion_coef"])
PY
)"

mkdir -p "$FORMAL_ROOT"

nohup python scripts/run_smax_oracle_training.py \
  --run-root "$FORMAL_ROOT" \
  --map-name 10m_vs_11m \
  --seeds 1-4 \
  --conditions none,oracle_latent_distortion \
  --oracle-coef "$ORACLE_COEF" \
  --fisher-ridge 0.001 \
  --total-timesteps 10000000 \
  --gpus 0,1,2,3 \
  --max-runs-per-gpu 2 \
  --project jaxmarl-smax-10m-vs-11m-oracle \
  > "$FORMAL_ROOT/training.stdout" 2>&1 &
```

Monitor all eight runs:

```bash
watch -n 5 "python scripts/monitor_smax_oracle_training.py --run-root '$FORMAL_ROOT'"
```

In W&B, filter `MAP_NAME=10m_vs_11m`, group by
`EXPERIMENT_CONDITION`, use `env_step` on the x-axis, and compare `returns` or
`win_rate`. The two groups are `none` and `oracle_latent_distortion`.

After all runs finish, generate the matched two-condition figure and its CSV:

```bash
python scripts/plot_smax_oracle_training.py --run-root "$FORMAL_ROOT"
```
