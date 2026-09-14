import pytest

from baselines.MAPPO.smax_rollout import smax_rollout_horizon


def test_smax_rollout_horizon_includes_timeout_transition():
    assert smax_rollout_horizon(100) == 101


def test_smax_rollout_horizon_rejects_negative_limit():
    with pytest.raises(ValueError, match="non-negative"):
        smax_rollout_horizon(-1)
