import numpy as np
import pytest

from scripts.audit_smax_diagnostic_collection import (
    _complete_mc_return,
    _episode_prefix_length,
)


def test_active_mask_must_be_a_single_prefix():
    assert _episode_prefix_length(np.array([True, True, False, False])) == 2
    with pytest.raises(AssertionError, match="single true prefix"):
        _episode_prefix_length(np.array([True, False, True, False]))


def test_complete_mc_return_stops_at_terminal_and_ignores_padding():
    reward = np.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [100.0, 100.0],
        ],
        dtype=np.float32,
    )
    done = np.array([False, True, True])
    active = np.array([True, True, False])
    result = _complete_mc_return(reward, done, active, gamma=0.5)
    np.testing.assert_allclose(
        result,
        np.array([[2.0, 2.0], [2.0, 2.0], [0.0, 0.0]], dtype=np.float32),
    )
