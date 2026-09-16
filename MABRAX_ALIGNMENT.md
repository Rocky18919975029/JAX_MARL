# MABrax MAPPO alignment protocol

This extension runs the actor--critic representation experiment on JaxMARL
MABrax `halfcheetah_6x1`. The tuned MAPPO hyperparameters remain those in
`baselines/MAPPO/config/mappo_ff_mabrax.yaml`.

## Experimental definition

- `ps`: one shared ActorFF for all six one-dimensional controllers.
- `nps`: six complete ActorFF parameter sets, including their two-layer
  encoders, mean heads, and learned log standard deviations.
- Both variants use the same centralized CriticFF over the Brax global state.
- PS and NPS start from identical actor parameters. NPS repeats that parameter
  tree six times before applying six independent optimizers.
- Rollouts, environment permutations, minibatches, advantage normalization,
  critic samples, update epochs, and seeds are matched.
- The aligned representation is the 64-dimensional output of the second actor
  or critic hidden layer.
- Linear CKA is computed separately for every agent. NPS actor coordinate
  systems are never pooled.

The four alignment conditions are `none`, `c_to_a`, `a_to_c`, and `joint`.
Directional targets use the rollout-time latent with stop-gradient; `joint`
allows gradients into both current representations. LN-MSE uses lambda 0.1.
The Linear CKA coefficient is calibrated from initial cross-gradient scales
without consulting return.

For four seeds, the complete matrix contains:

```
2 actor parameterizations × (1 baseline + 3 LN-MSE + 3 CKA) × 4 seeds = 56 runs
```

## Server smoke tests

Run one shared baseline and all three aligned NPS paths before calibration:

```bash
cd ~/JaxMARL
conda activate jaxmarl_brax
unset LD_LIBRARY_PATH

for item in \
  "0 true  none     ln_mse" \
  "1 false c_to_a   ln_mse" \
  "2 false a_to_c   linear_cka" \
  "3 false joint    linear_cka"
do
  set -- $item
  gpu=$1
  sharing=$2
  mode=$3
  distance=$4
  label="${sharing}-${mode}-${distance}"

  CUDA_VISIBLE_DEVICES="$gpu" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  HYDRA_FULL_ERROR=1 \
  python baselines/MAPPO/mappo_ff_mabrax.py \
    ACTOR_PARAMETER_SHARING="$sharing" \
    ALIGN_MODE="$mode" \
    ALIGN_DISTANCE="$distance" \
    ALIGNMENT_COEF=0.1 \
    NUM_ENVS=4 \
    NUM_STEPS=16 \
    TOTAL_TIMESTEPS=128 \
    UPDATE_EPOCHS=1 \
    NUM_MINIBATCHES=1 \
    SAVE_CHECKPOINTS=true \
    CHECKPOINT_INTERVAL_TIMESTEPS=64 \
    CHECKPOINT_DIR=/home/data/zeshenghong/JaxMARL/mabrax_alignment_smoke/checkpoints \
    WANDB_MODE=disabled \
    PROJECT=mabrax-alignment-smoke \
    > "/tmp/mabrax-alignment-${label}.log" 2>&1 &
done
wait
```

Verify all four final checkpoints and inspect errors:

```bash
find /home/data/zeshenghong/JaxMARL/mabrax_alignment_smoke/checkpoints \
  -type f -path '*/final/model.safetensors' | wc -l

grep -HEn 'Traceback|ValueError|nan|inf' \
  /tmp/mabrax-alignment-*.log || true
```

The checkpoint count must be `4`, and the error search should print nothing.

## Gradient calibration

```bash
export MABRAX_ROOT="/home/data/zeshenghong/JaxMARL/mabrax_alignment"
export CKA_CAL_ROOT="$MABRAX_ROOT/cka_gradient_calibration"
mkdir -p "$CKA_CAL_ROOT"

nohup python scripts/calibrate_mabrax_cka.py \
  --output-root "$CKA_CAL_ROOT" \
  --pilot-seed 9001 \
  --gpus 0,1,2,3 \
  > "$CKA_CAL_ROOT/calibration.stdout" 2>&1 &

tail --retry -F "$CKA_CAL_ROOT/calibration.stdout"
```

The result is
`$CKA_CAL_ROOT/cka_gradient_calibration.json`. It contains four cells: PS/NPS
crossed with both directional gradient recipients, plus one pooled coefficient.

## Formal four-seed matrix

```bash
export FORMAL_ROOT="$MABRAX_ROOT/formal_4seed"
export CKA_CALIBRATION="$CKA_CAL_ROOT/cka_gradient_calibration.json"
mkdir -p "$FORMAL_ROOT"

nohup python scripts/run_mabrax_alignment.py \
  --run-root "$FORMAL_ROOT" \
  --cka-calibration "$CKA_CALIBRATION" \
  --seeds 1-4 \
  --gpus 0,1,2,3 \
  --max-runs-per-gpu 4 \
  --checkpoint-interval 5000000 \
  --project jaxmarl-mabrax-halfcheetah6x1-alignment \
  > "$FORMAL_ROOT/training.stdout" 2>&1 &

tail --retry -F "$FORMAL_ROOT/launcher.log"
```

The launcher starts at most 16 runs concurrently, writes deterministic status
JSONs, and skips completed runs on restart. W&B run names have the form
`MABRAX-halfcheetah_6x1-{ps|nps}-{condition}-lam{value}-seed{seed}`.

Monitor progress:

```bash
watch -n 10 '
ROOT=/home/data/zeshenghong/JaxMARL/mabrax_alignment/formal_4seed
echo "Completed: $(grep -l '\''"status": "completed"'\'' "$ROOT"/status/*.json 2>/dev/null | wc -l) / 56"
echo "Failed:    $(grep -l '\''"status": "failed"'\'' "$ROOT"/status/*.json 2>/dev/null | wc -l)"
echo "Workers:   $(pgrep -fc '\''mappo_ff_mabrax.py'\'')"
tail -n 12 "$ROOT/launcher.log"
'
```

## Held-out evaluation

```bash
python baselines/MAPPO/eval_mappo_ff_mabrax.py \
  --checkpoint /path/to/run/final \
  --episodes 256 \
  --num-envs 64 \
  --seed 10000
```

The evaluator uses the frozen Gaussian mean action, fresh reset seeds, no
environment auto-reset, and reports per-episode return and episode length with
standard errors.
