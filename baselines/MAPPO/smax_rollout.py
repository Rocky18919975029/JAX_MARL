"""Shared rollout-length helpers for SMAX evaluation and diagnostics."""


def smax_rollout_horizon(max_steps: int) -> int:
    """Return enough transitions for SMAX to emit its time-limit done flag.

    SMAX checks ``state.step >= max_steps`` before incrementing ``state.step``.
    Since reset states start at step zero, a time-limit-only episode reports done
    on transition ``max_steps + 1``.
    """

    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    return int(max_steps) + 1
