#!/usr/bin/env python3
"""Display all 6s9z isolated MAPPO PPO-sweep run progress bars."""

try:
    from scripts.monitor_smax_actor_score_recovery_sweep import main
except ModuleNotFoundError:  # Direct execution from scripts/.
    from monitor_smax_actor_score_recovery_sweep import main


if __name__ == "__main__":
    main()
