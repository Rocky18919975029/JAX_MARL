# MAPPO Baseline

Pure JAX MAPPO implementation, based on the PureJaxRL PPO implementation.

## 🔎 Implementation Details
General features:
* Agents are controlled by a single network architecture (either FF or RNN).
* Parameters are shared between agents.
* Each script has a `WorldStateWrapper` which provides a global `"world_state"` observation.

## 🚀 Usage

If you have cloned JaxMARL and are in the repository root, you can run the algorithms as scripts, e.g.
```
python baselines/MAPPO/mappo_rnn_smax.py
```
Each file has a distinct config file which resides within [`config`](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/MAPPO/config).
The config file contains the MAPPO hyperparameters, the environment's parameters and the `wandb` details.  Legacy configs disable `wandb` by default; the MABrax config uses online logging and can be disabled with `WANDB_MODE=disabled` for smoke tests.

### Continuous-control MABrax

`mappo_ff_mabrax.py` is a continuous-action feed-forward MAPPO baseline and
matched representation-alignment implementation. Its decentralized actor
receives each agent's padded local observation, while its centralized critic
receives the underlying Brax global observation. `ACTOR_PARAMETER_SHARING=true`
uses one shared actor; `false` uses six complete ActorFF parameter sets for
`halfcheetah_6x1`, with the centralized critic still shared. The default config
follows the tuned HalfCheetah hyperparameters in Table 7 of the JaxMARL paper.

```
python baselines/MAPPO/mappo_ff_mabrax.py
```

For a short smoke test:

```
python baselines/MAPPO/mappo_ff_mabrax.py \
  NUM_ENVS=4 NUM_STEPS=16 TOTAL_TIMESTEPS=128 \
  UPDATE_EPOCHS=1 NUM_MINIBATCHES=1 WANDB_MODE=disabled
```

MABrax is deprecated upstream and requires a legacy Brax/JAX-compatible
environment.  The algorithm deliberately fails on tasks with unequal action
dimensions; the default `halfcheetah_6x1` has six one-dimensional actors.

The supported alignment modes are `none`, `c_to_a`, `a_to_c`, and `joint`,
with either `ALIGN_DISTANCE=ln_mse` or `linear_cka`. Linear CKA uses a
MABrax-specific coefficient selected without returns by matching its initial
cross-gradient scale to LN-MSE at lambda 0.1:

```bash
python scripts/calibrate_mabrax_cka.py \
  --output-root /path/to/mabrax/cka_gradient_calibration \
  --pilot-seed 9001 --gpus 0,1,2,3
```

Run the complete 56-run matrix (PS/NPS, one distance-free baseline plus three
directions under both distances, four seeds) with deterministic W&B names and
resumable status files:

```bash
python scripts/run_mabrax_alignment.py \
  --run-root /path/to/mabrax/formal_4seed \
  --cka-calibration /path/to/mabrax/cka_gradient_calibration/cka_gradient_calibration.json \
  --seeds 1-4 --gpus 0,1,2,3 --max-runs-per-gpu 4
```

Each checkpoint contains `model.safetensors`, `config.json`, and
`metadata.json`. A held-out deterministic evaluation can be run with:

```bash
python baselines/MAPPO/eval_mappo_ff_mabrax.py \
  --checkpoint /path/to/checkpoint/final \
  --episodes 256 --num-envs 64 --seed 10000
```
