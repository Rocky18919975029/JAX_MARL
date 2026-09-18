# HARL MA-MuJoCo alignment experiment

This overlay adapts the tuned HARL MAPPO protocol for
`Humanoid-v2-17x1` without modifying the pinned upstream HARL submodule.
All training remains PyTorch/HARL; JAX is not used for MA-MuJoCo.

## Experiment definition

- task: `Humanoid-v2-17x1` (17 one-dimensional continuous-action agents);
- actor variants: parameter sharing (`ps`) and 17 independent actors (`nps`);
- one environment-provided centralized critic;
- modes: `none`, `c_to_a`, `a_to_c`, and `joint`;
- distances: per-sample LayerNorm MSE and Linear CKA;
- seeds: `1,2,3,4`;
- MSE coefficient: `0.1`;
- CKA coefficient: one global value calibrated from initial gradient scales;
- source hyperparameters: HARL's tuned
  `tuned_configs/mamujoco/Humanoid-v2-17x1/mappo/config.json`.

`none` is distance-free and is trained once per actor variant and seed. The
formal matrix therefore contains 56 unique runs, not 64 duplicated runs.

For EP state, one critic latent is produced for each environment/time sample
and paired with each of the 17 agent actor latents. Linear CKA is calculated
inside each agent's sample pool and then averaged over agents. This prevents
NPS actor coordinate systems from being mixed into one CKA batch.

The directional rules match the SMAX implementation:

- `c_to_a`: current actor latent is optimized toward the critic latent frozen
  at rollout time;
- `a_to_c`: current critic latent is optimized toward actor latents frozen at
  rollout time;
- `joint`: current actor and critic latents both receive gradients;
- `none`: no representation loss is evaluated or applied.

Actor PPO, critic value loss, and alignment are differentiated in the same
full-batch update. For NPS, averaging across agents is undone on every actor's
gradient, so every independent actor receives its full per-agent gradient;
the centralized critic continues to receive the agent-mean alignment gradient.
For a fixed seed, PS and NPS start from the same actor weights and the same
critic weights. NPS copies that actor initialization into 17 independent model
and optimizer instances, then all conditions start rollout sampling from the
same explicit RNG substream.

## Server setup

Run these commands from the server clone. They do not install or alter the
existing MuJoCo/conda environment.

```bash
cd ~/JaxMARL
git pull --ff-only
git submodule sync -- third_party/HARL
git submodule update --init --depth 1 third_party/HARL

conda activate YOUR_EXISTING_HARL_ENV

export HARL_RUN_ROOT="/home/data/zeshenghong/JaxMARL/harl_mamujoco_humanoid17x1"
mkdir -p "$HARL_RUN_ROOT"

ln -sfn "$HARL_RUN_ROOT" "$HOME/JaxMARL/harl_mamujoco_runs"
test "$HOME/JaxMARL/harl_mamujoco_runs" -ef "$HARL_RUN_ROOT" && echo "Storage link: PASS"
```

If the project-side link already exists as a real directory, move it to the
data disk before creating the symlink; do not use `ln -sfn` to replace a real
directory.

## Four-GPU smoke test

This runs two PPO updates per process, exercises PS/NPS, both distances, all
gradient recipients, checkpoint writing, and avoids W&B pollution.

```bash
cd ~/JaxMARL
conda activate YOUR_EXISTING_HARL_ENV

for spec in \
  "0 true  c_to_a ln_mse     smoke-ps-c_to_a-mse" \
  "1 true  a_to_c linear_cka smoke-ps-a_to_c-cka" \
  "2 false joint  ln_mse     smoke-nps-joint-mse" \
  "3 false joint  linear_cka smoke-nps-joint-cka"
do
  set -- $spec
  gpu="$1"; sharing="$2"; mode="$3"; distance="$4"; name="$5"
  CUDA_VISIBLE_DEVICES="$gpu" \
  python experiments/harl_mamujoco/train_alignment.py \
    --run-root "$HARL_RUN_ROOT/smoke" \
    --run-name "$name" \
    --seed 9000 \
    --actor-parameter-sharing "$sharing" \
    --align-mode "$mode" \
    --align-distance "$distance" \
    --alignment-coef 0.1 \
    --num-env-steps 4000 \
    --disable-eval \
    --checkpoint-interval-steps 0 \
    --wandb-mode disabled \
    > "$HARL_RUN_ROOT/smoke-${name}.log" 2>&1 &
  echo "GPU $gpu $name PID $!"
done
wait

grep -HEn 'Traceback|RuntimeError|ValueError|nan|inf' \
  "$HARL_RUN_ROOT"/smoke-*.log || true
find "$HARL_RUN_ROOT/smoke/checkpoints" -name completed.json | wc -l
```

The final count must be `4` and the error grep must be empty.

## Calibrate Linear CKA

The pilot uses independent seed 9001 and four cells:
PS/NPS crossed with `c_to_a`/`a_to_c`. It never reads return or selects a
coefficient by performance.

```bash
export CKA_CAL_ROOT="$HARL_RUN_ROOT/cka_gradient_calibration"

nohup python experiments/harl_mamujoco/calibrate_cka.py \
  --run-root "$CKA_CAL_ROOT" \
  --pilot-seed 9001 \
  --gpus 0,1,2,3 \
  > "$CKA_CAL_ROOT.stdout" 2>&1 &

echo "Calibration manager PID: $!"
tail --retry -F "$CKA_CAL_ROOT.stdout"
```

The result is
`$CKA_CAL_ROOT/cka_gradient_calibration.json`. It records every raw gradient
norm and the pooled-RMS global coefficient.

## Launch the formal four-seed matrix

First audit the 56 unique commands:

```bash
export FORMAL_ROOT="$HARL_RUN_ROOT/formal_4seed"
export CKA_CALIBRATION="$CKA_CAL_ROOT/cka_gradient_calibration.json"
mkdir -p "$FORMAL_ROOT"

python experiments/harl_mamujoco/run_matrix.py \
  --run-root "$FORMAL_ROOT" \
  --cka-calibration "$CKA_CALIBRATION" \
  --seeds 1-4 \
  --gpus 0,1,2,3 \
  --max-runs-per-gpu 1 \
  --dry-run > "$FORMAL_ROOT/dry_run.txt"

grep -c ' START ' "$FORMAL_ROOT/dry_run.txt"
```

Then launch online W&B runs:

```bash
nohup python experiments/harl_mamujoco/run_matrix.py \
  --run-root "$FORMAL_ROOT" \
  --cka-calibration "$CKA_CALIBRATION" \
  --seeds 1-4 \
  --gpus 0,1,2,3 \
  --max-runs-per-gpu 1 \
  --wandb-project harl-mamujoco-humanoid17x1-alignment \
  > "$FORMAL_ROOT/training.stdout" 2>&1 &

echo "Training manager PID: $!"
tail --retry -F "$FORMAL_ROOT/launcher.log"
```

Increase `--max-runs-per-gpu` only after observing CPU, RAM, GPU memory, and
simulation throughput. Unlike SMAX, MA-MuJoCo environment stepping is not a
large JAX GPU batch, so excessive process concurrency can reduce throughput.

For the focused main experiment (NPS only; isolated, C→A LN-MSE, and C→A
Linear CKA; four seeds), use the dedicated restart-safe 12-run launcher:

```bash
nohup python experiments/harl_mamujoco/run_nps_core_matrix.py \
  --run-root "$FORMAL_ROOT" \
  --cka-calibration "$CKA_CALIBRATION" \
  --gpus 0,1,2,3 \
  --max-runs-per-gpu 3 \
  --wandb-project harl-mamujoco-humanoid17x1-nps-core \
  > "$FORMAL_ROOT/training.stdout" 2>&1 &
```

The launcher generates exactly 12 unique runs, skips only runs with a
checkpoint `completed.json`, and retries interrupted or failed runs. With four
GPUs and the default three slots per GPU, all 12 runs can execute concurrently.

Progress checks:

```bash
watch -n 10 '
ROOT="/home/data/zeshenghong/JaxMARL/harl_mamujoco_humanoid17x1/formal_4seed"
printf "Completed: "; grep -l '"status": "completed"' "$ROOT"/status/*.json 2>/dev/null | wc -l
printf "Failed:    "; grep -l '"status": "failed"'    "$ROOT"/status/*.json 2>/dev/null | wc -l
printf "Workers:   "; pgrep -fc "harl_mamujoco/train_alignment.py"
tail -n 12 "$ROOT/launcher.log" 2>/dev/null
'
```

W&B run names encode task, PS/NPS, direction, distance, coefficient, and seed.
All conditions for one actor parameterization share a W&B group. Plot
`eval/return` against `env_step`, group curves by `align_mode` plus
`align_distance`, and aggregate by mean with standard-error shading.

## Checkpoints

Checkpoints are written every 500,000 environment steps plus `initial` and
`final`. Each contains actor model and optimizer state, centralized critic and
optimizer state, ValueNorm state, and auditable metadata. PS stores one actor;
NPS stores all 17 actors. Checkpoint files remain on the data disk and are not
uploaded to W&B unless `--upload-checkpoints` is explicitly supplied.
