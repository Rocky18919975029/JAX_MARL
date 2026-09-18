# SMAX
## Description
SMAX is a purely JAX SMAC-like environment. It, like SMAC, focuses on decentralised unit micromanagement across a range of scenarios. Each scenario features fixed teams. 

## Scenarios

| Name         | Ally Units             | Enemy Units            |
|--------------|------------------------|------------------------|
| 2s3z         | 2 stalkers & 3 zealots | 2 stalkers & 3 zealots |
| 3s5z         | 3 stalkers & 5 zealots | 3 stalkers & 5 zealots |
| 5m_vs_6m     | 5 marines              | 6 marines              |
| 10m_vs_11m   | 10 marines             | 11 marines             |
| 27m_vs_30m   | 27 marines             | 30 marines             |
| 3s5z_vs_3s6z | 3 stalkers & 5 zealots | 3 stalkers & 6 zealots |
| 3s_vs_5z     | 3 stalkers             | 5 zealots              |
| 6h_vs_8z     | 6 hydralisks           | 8 zealots              |
| smacv2_5_units | 5 randomly chosen    | 5 randomly chosen      |
| smacv2_10_units | 10 randomly chosen  | 10 randomly chosen     |
| smacv2_10_units_hetero_h0 | one random type, profile `[10]` | mirrored ally profile |
| smacv2_10_units_hetero_h1 | two random types, profile `[9, 1]` | mirrored ally profile |
| smacv2_10_units_hetero_h2 | two random types, profile `[7, 3]` | mirrored ally profile |
| smacv2_10_units_hetero_h3 | three random types, profile `[6, 2, 2]` | mirrored ally profile |
| smacv2_10_units_hetero_h4 | four random types, profile `[4, 3, 2, 1]` | mirrored ally profile |
| smacv2_10_units_hetero_h5 | six types, profile `[2, 2, 2, 2, 1, 1]` | mirrored ally profile |
| smacv2_20_units | 20 randomly chosen  | 20 randomly chosen     |

The controlled agent-count scaling suite uses matched ally counts in two
families.  Homogeneous maps are `3m_vs_4m`, `5m_vs_6m`, `8m_vs_9m`,
`10m_vs_11m`, and `15m_vs_16m`.  Heterogeneous maps are `1s2z_vs_1s3z`,
`2s3z_vs_2s4z`, `3s5z_vs_3s6z`, `4s6z_vs_4s7z`, and
`6s9z_vs_6s10z`.  The ally counts are 3, 5, 8, 10, and 15 in both families;
the enemy team always contains exactly one additional unit.

## Visualisation
You can see the example `smax_introduction.py` in the tutorials folder for an introduction to SMAX, including example visualisation. SMAX environments tick at 8 times faster than each step of the agent. This means that when visualising, we have to expand the state sequence to encompass all ticks. This is why the `state_seq` for SMAX consists of a sequence of `(key, state, actions)` -- we must have not only the state and actions, but also the exact key passed to the step function to interpolate between the different states correctly. This process means visualisation can be time consuming if done for a large number of steps.

```python
from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario
from jaxmarl.viz.visualizer import SMAXVisualizer

scenario = map_name_to_scenario("3m")
env = make(
    "HeuristicEnemySMAX",
    enemy_shoots=True,
    scenario=scenario,
    num_agents_per_team=3,
    use_self_play_reward=False,
    walls_cause_death=True,
    see_enemy_actions=False,
)

# state_seq is a list of (key_s, state, actions) tuples
# where key_s is the RNG key passed into the step function,
# state is the jax env state and actions is the actions passed
# into the step function.
viz = SMAXVisualizer(env, state_seq)

viz.animate(view=False, save_fname="output.gif")
```
