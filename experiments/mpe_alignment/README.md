# MPE Simple Spread-5 alignment experiment

This experiment uses the repository's tuned MAPPO MPE configuration as its
frozen base (`baselines/MAPPO/config/mappo_homogenous_rnn_mpe.yaml`) and changes
only the environment cardinality and the representation interaction.

The confirmatory matrix is:

- task: `MPE_simple_spread_v3` with 5 agents and 5 landmarks;
- actor parameterization: non-parameter-sharing (one actor network per agent);
- centralized recurrent MAPPO critic;
- conditions: isolated, C→A LN-MSE, and C→A Linear CKA;
- seeds: 1–4;
- total: 12 runs.

NPS actors start from identical matched parameters. Minibatches permute the
environment axis once and retain every agent, so different actor coordinate
systems are never pooled. Linear CKA and LN-MSE are computed independently for
each actor and then averaged. The critic latent used by C→A is the frozen
rollout latent (`stop_gradient`).

## Server workflow

Activate the existing JaxMARL environment, then calibrate CKA without using
return:

```bash
cd ~/JaxMARL
conda activate jaxmarl
unset LD_LIBRARY_PATH

export MPE_ROOT="/home/data/zeshenghong/JaxMARL/mpe_simple_spread5_alignment"
export MPE_CAL_ROOT="$MPE_ROOT/cka_gradient_calibration"

nohup python experiments/mpe_alignment/calibrate_cka.py \
  --output-root "$MPE_CAL_ROOT" \
  --gpus 0,1 \
  > "$MPE_CAL_ROOT/calibration.stdout" 2>&1 &

tail --retry -F "$MPE_CAL_ROOT/calibration.stdout"
```

Run a four-GPU smoke test (one short run per condition):

```bash
export MPE_CALIBRATION="$MPE_CAL_ROOT/cka_gradient_calibration.json"
export MPE_SMOKE_ROOT="$MPE_ROOT/smoke"

python experiments/mpe_alignment/run_matrix.py \
  --run-root "$MPE_SMOKE_ROOT" \
  --cka-calibration "$MPE_CALIBRATION" \
  --seeds 1 \
  --gpus 0,1,2,3 \
  --max-runs-per-gpu 1 \
  --total-timesteps 8192 \
  --wandb-mode disabled
```

After all three smoke runs complete, launch the formal matrix. The default
allows three simultaneous runs per GPU (12 total), rather than serializing one
run per GPU:

```bash
export MPE_FORMAL_ROOT="$MPE_ROOT/formal_4seed"

nohup python experiments/mpe_alignment/run_matrix.py \
  --run-root "$MPE_FORMAL_ROOT" \
  --cka-calibration "$MPE_CALIBRATION" \
  --seeds 1-4 \
  --gpus 0,1,2,3 \
  --max-runs-per-gpu 3 \
  --wandb-project jaxmarl-mpe-spread5-alignment \
  > "$MPE_FORMAL_ROOT/training.stdout" 2>&1 &
```

Monitor every run:

```bash
watch -n 5 python experiments/mpe_alignment/monitor.py \
  --run-root "$MPE_FORMAL_ROOT"
```

After all 12 runs complete, generate the seed-level CSV, summarized CSV, PNG,
and vector PDF without querying W&B:

```bash
python experiments/mpe_alignment/analyze.py \
  --run-root "$MPE_FORMAL_ROOT"
```
