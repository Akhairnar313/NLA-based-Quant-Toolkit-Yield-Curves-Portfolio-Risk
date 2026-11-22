import numpy as np

from finance.pca import pca, dominant_factor_share
from tests.helpers import call, random_spd

rng = np.random.default_rng(10)


def test_pca():
    cov = random_spd(6, rng)
    vals, vecs, explained = call(pca, cov)
    np.testing.assert_allclose(vals, np.linalg.eigvalsh(cov)[::-1], rtol=1e-8)
    np.testing.assert_allclose(cov @ vecs, vecs * vals, atol=1e-6)
    assert np.isclose(explained.sum(), 1.0)
    assert np.all(np.diff(explained) <= 1e-12)


def test_dominant_factor_share():
    n = 5
    one_factor = np.full((n, n), 0.6) + 0.4 * np.eye(n)  # every pair correlated 0.6
    # Largest eigenvalue is 1 + 0.6 (n - 1) = 3.4, so the share is 3.4 / 5.
    assert np.isclose(call(dominant_factor_share, one_factor), 0.68, rtol=1e-8)
    assert np.isclose(call(dominant_factor_share, np.eye(n)), 1 / n, rtol=1e-6)
