# RWARE large-8ag: recurrent MAPPO + C→A LN-MSE / Linear CKA

This is a first four-seed comparison on the pinned Mava checkout
`9f67e612654ecb7b7d45ff8052ce9ccfc6c68d93`. It uses the RWARE
`large-8ag` optimal recurrent-MAPPO task configuration from
`optimal_rec_mappo.json` for **every** condition. The completed four MAPPO
baseline runs in `mava_high_agent_paired/formal_4seed_v1` are verified and
reused, not retrained. Eight new runs are launched: two alignment methods ×
seeds 1–4. The coefficients are predeclared first-pass values, **not** tuned
or claimed optimal: LN-MSE `0.01`; Linear CKA `0.03`. Both use epsilon `1e-8`.

The new actor and critic preserve Mava's recurrent network parameter layouts.
At the end of each rollout, the critic's 128-D GRU output is recomputed once
and detached. At every PPO minibatch the current actor's 128-D GRU output is
recomputed. Each sample is parameter-free LayerNorm-normalised over features.
LN-MSE measures coordinatewise squared error. Linear CKA separately compares
the centred sample matrices of each agent in that minibatch, and averages the
agents' `1−CKA` distances. All RWARE agents remain active; there is no SMAX
dead-unit mask. The auxiliary term is added **only to the actor PPO loss**;
the critic still receives only Mava's original clipped value loss. There is
no recovery head, Fisher transform, or ARec objective. Because Mava's actor
shares parameters across agents whereas the main SMAX protocol uses NPS
actors, this transfers the alignment direction and distance, not the full
SMAX architecture.

## Install and smoke on GPU 1

Run from the JaxMARL checkout on the server. Installation writes two **new,
untracked** entry-point files into the pinned Mava checkout; it does not edit
upstream `rec_mappo.py` or the existing ARec entry point.

```bash
cd ~/JaxMARL
git pull --ff-only
unset LD_LIBRARY_PATH
MAVA_ROOT=/home/data/zeshenghong/Mava
git -C "$MAVA_ROOT" rev-parse HEAD
install -m 0644 experiments/mava_jumanji/rec_mappo_alignment.py \
  "$MAVA_ROOT/mava/systems/ppo/anakin/rec_mappo_alignment.py"
install -m 0644 experiments/mava_jumanji/rec_mappo_alignment.yaml \
  "$MAVA_ROOT/mava/configs/default/rec_mappo_alignment.yaml"
SMOKE_ROOT=/home/data/zeshenghong/JaxMARL/mava_rware_alignment/smoke_v1
python experiments/mava_jumanji/run_rware_alignment.py run \
  --mava-root "$MAVA_ROOT" --run-root "$SMOKE_ROOT" \
  --smoke --gpus 1
python experiments/mava_jumanji/run_rware_alignment.py status \
  --run-root "$SMOKE_ROOT"
```

Both smoke jobs must complete with full evaluation logs before formal runs.
The smoke uses only two updates and is a compilation/wiring check, not a
performance comparison. If one fails, inspect its log under
`$SMOKE_ROOT/logs/` before continuing.

## Four-seed formal run

The formal launcher refuses a changed/incomplete baseline, wrong Mava commit,
uninstalled alignment source, or a previously used run root with different
settings. It records source hashes and exact commands in its manifest. It
skips completed runs when restarted, continues to the next job after a
failure, and permits only one worker per GPU because this Mava configuration
has occupied about 18.5 GiB of a 24 GiB RTX 4090.

```bash
cd ~/JaxMARL
unset LD_LIBRARY_PATH
MAVA_ROOT=/home/data/zeshenghong/Mava
SOURCE_ROOT=/home/data/zeshenghong/JaxMARL/mava_high_agent_paired/formal_4seed_v1
RUN_ROOT=/home/data/zeshenghong/JaxMARL/mava_rware_alignment/formal_4seed_v1
mkdir -p "$RUN_ROOT"
python experiments/mava_jumanji/run_rware_alignment.py run \
  --mava-root "$MAVA_ROOT" --source-root "$SOURCE_ROOT" \
  --run-root "$RUN_ROOT" --gpus 0,1,2,3 --dry-run
nohup python -u experiments/mava_jumanji/run_rware_alignment.py run \
  --mava-root "$MAVA_ROOT" --source-root "$SOURCE_ROOT" \
  --run-root "$RUN_ROOT" --gpus 0,1,2,3 \
  --mse-coef 0.01 --cka-coef 0.03 \
  > "$RUN_ROOT/launcher.stdout" 2>&1 &
watch -n 5 "python experiments/mava_jumanji/run_rware_alignment.py status --run-root '$RUN_ROOT'"
```

Run roots under `/home/data/zeshenghong` are on the server data disk (the
existing `/mnt/sda` mount). The launcher writes local Mava JSON evaluation
logs under `$RUN_ROOT/runs/`, worker logs under `logs/`, and states under
`status/`. A worker is marked completed only when it exits successfully and
has the full set of expected evaluation checkpoints. To retry failures after
inspecting the logs, repeat the formal command with `--retry-failed`.

This is exploratory first-pass coefficient testing on the same four training
seeds. AUC or final-return gains should be reported against the original
MAPPO baseline with paired seeds; selecting the best coefficient on these
same seeds is not independent confirmation.
