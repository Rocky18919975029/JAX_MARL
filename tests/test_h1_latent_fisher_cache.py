import numpy as np

from scripts.h1_latent_distortion import (
    fisher_metrics,
    fisher_metrics_from_statistics,
    fisher_statistics,
)


def test_cached_fisher_statistics_preserve_metrics():
    rng = np.random.default_rng(17)
    scores = rng.normal(size=(512, 8))
    reference = rng.normal(size=512)
    critic = rng.normal(size=512)

    direct = fisher_metrics(scores, reference, critic, 1e-3)
    cached = fisher_metrics_from_statistics(
        fisher_statistics(scores, reference, critic), 1e-3
    )

    for key in direct:
        assert np.allclose(direct[key], cached[key], equal_nan=True)
