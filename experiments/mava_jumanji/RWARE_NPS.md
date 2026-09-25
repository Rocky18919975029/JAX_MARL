# RWARE large-8ag: independent recurrent actors

This protocol compares four seed-matched methods on the same Jumanji RWARE
`large-8ag` task: independent-actor recurrent MAPPO (`none`), critic-to-actor
LN-MSE, critic-to-actor Linear CKA, and actor-side score recovery (`arec`).
All four use the pinned Mava recurrent MAPPO task-optimal PPO/environment
configuration in `optimal_rec_mappo.json`, seeds 1–4, the same centralised
critic, and a separate recurrent actor parameter tree **and Adam state** for
each agent. The only difference is the auxiliary objective. This protocol
does not reuse or compare against the earlier parameter-sharing MAPPO baseline.

The actor bank maps each agent's observations, recurrent state and action mask
through its own parameters; only the masked categorical logits are assembled
into a joint Mava distribution. Each actor's PPO gradient is averaged over
that agent's samples and clipped independently. LN-MSE and Linear CKA detach
the critic representation and update only actors. ARec fits separate
action-conditioned recovery heads against rollout Fisher-whitened policy
scores, detaches the heads and critic for its actor objective, and does not
add a latent-alignment loss. The Fisher matrix is fixed within a PPO update.

These entrypoints are intended for the pinned Mava checkout at commit
`9f67e612654ecb7b7d45ff8052ce9ccfc6c68d93`. Install the four entrypoint
and config files plus the helper into that checkout:

```bash
cd ~/JaxMARL
git pull --ff-only
MAVA_ROOT=/home/data/zeshenghong/Mava
unset LD_LIBRARY_PATH
install -m 0644 experiments/mava_jumanji/independent_recurrent_actor.py "$MAVA_ROOT/mava/systems/ppo/anakin/independent_recurrent_actor.py"
install -m 0644 experiments/mava_jumanji/rec_mappo_nps.py "$MAVA_ROOT/mava/systems/ppo/anakin/rec_mappo_nps.py"
install -m 0644 experiments/mava_jumanji/rec_mappo_nps.yaml "$MAVA_ROOT/mava/configs/default/rec_mappo_nps.yaml"
install -m 0644 experiments/mava_jumanji/rec_mappo_arec_nps.py "$MAVA_ROOT/mava/systems/ppo/anakin/rec_mappo_arec_nps.py"
install -m 0644 experiments/mava_jumanji/rec_mappo_arec_nps.yaml "$MAVA_ROOT/mava/configs/default/rec_mappo_arec_nps.yaml"
```

First check actor-bank math and all four short runs on one GPU:

```bash
"$MAVA_ROOT/.venv/bin/python" -m pytest -q tests/test_mava_jumanji_independent_actor.py
SMOKE_ROOT=/home/data/zeshenghong/JaxMARL/mava_rware_nps/smoke_v1
python experiments/mava_jumanji/run_rware_nps.py run --mava-root "$MAVA_ROOT" --run-root "$SMOKE_ROOT" --smoke --gpus 1
python experiments/mava_jumanji/run_rware_nps.py status --run-root "$SMOKE_ROOT"
```

Then launch 16 full-budget runs (four methods × four seeds), at most one per
GPU due to JAX/Mava memory use:

```bash
RUN_ROOT=/home/data/zeshenghong/JaxMARL/mava_rware_nps/formal_4seed_v1
mkdir -p "$RUN_ROOT"
nohup python -u experiments/mava_jumanji/run_rware_nps.py run --mava-root "$MAVA_ROOT" --run-root "$RUN_ROOT" --gpus 0,1,2,3 --mse-coef 0.01 --cka-coef 0.03 --arec-coef 0.0001 --q-steps 4 --q-lr 0.001 --fisher-ridge 0.001 > "$RUN_ROOT/launcher.stdout" 2>&1 &
watch -n 5 "python experiments/mava_jumanji/run_rware_nps.py status --run-root '$RUN_ROOT'"
```

The listed auxiliary coefficients are fixed first-pass choices, not tuned
benchmark hyperparameters. The launcher records the Mava revision, source
hashes, exact overrides, run status and complete evaluation-log hashes. It
does not silently resume a different manifest. A failed run is skipped while
the remaining work continues; use `--retry-failed` later to retry failures.

Local development without JAX/Mava can run the protocol tests but cannot
validate the recurrent/GPU training path; the server smoke is required before
interpreting formal results.
