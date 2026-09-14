import numpy as np

from scripts.analyze_h1_performance import trapezoidal_integral


def test_trapezoidal_integral_is_numpy_version_compatible():
    x = np.asarray([0.0, 1.0, 2.0])
    y = np.asarray([0.0, 1.0, 2.0])

    assert trapezoidal_integral(y, x) == 2.0
