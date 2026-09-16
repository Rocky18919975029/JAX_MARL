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

`mappo_ff_mabrax.py` is a continuous-action feed-forward MAPPO baseline.  Its
decentralized shared actor receives each agent's padded local observation, while
its centralized critic receives the underlying Brax global observation.  The
default config targets `halfcheetah_6x1` and follows the tuned HalfCheetah
hyperparameters in Table 7 of the JaxMARL paper:

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
