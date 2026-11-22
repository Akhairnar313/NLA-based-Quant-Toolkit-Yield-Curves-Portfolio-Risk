import numpy as np

from finance.simulation import correlated_normal_samples, simulate_portfolio_returns
from tests.helpers import call, random_spd

rng = np.random.default_rng(7)


def test_correlated_samples_match_target_covariance():
    cov = random_spd(4, rng)
    mean = np.array([1., -2., 0.5, 0.])
    Z = call(correlated_normal_samples, cov, 200_000, mean=mean, seed=1)
    assert Z.shape == (200_000, 4)
    np.testing.assert_allclose(Z.mean(axis=0), mean, atol=0.05)
    np.testing.assert_allclose(np.cov(Z, rowvar=False), cov, rtol=0.03, atol=0.05)


def test_correlated_samples_reproducible():
    cov = random_spd(3, rng)
    a = call(correlated_normal_samples, cov, 10, seed=3)
    b = call(correlated_normal_samples, cov, 10, seed=3)
    np.testing.assert_array_equal(a, b)


def test_portfolio_simulation_vol_and_mean():
    cov_daily = random_spd(5, rng) / 1000 / 252
    w = np.full(5, 0.2)
    sims = call(simulate_portfolio_returns, w, cov_daily, 63, 300_000, seed=2)
    assert sims.shape == (300_000,)
    # Expected simple return is 0 by construction; volatility ~ sqrt(h w' Sigma w).
    assert abs(sims.mean()) < 3e-3
    assert np.isclose(sims.std(), np.sqrt(63 * w @ cov_daily @ w), rtol=0.03)
