import numpy as np
import pytest
from scipy.optimize import minimize

from finance.portfolio import (min_variance_portfolio, min_variance_long_only,
                               tangency_portfolio, efficient_frontier)
from tests.helpers import call, random_spd

rng = np.random.default_rng(9)
COV = random_spd(5, rng) / 100
MU = np.array([0.08, 0.10, 0.03, 0.05, 0.06])
RF = 0.02


def test_min_variance_portfolio():
    w = call(min_variance_portfolio, COV)
    x = np.linalg.solve(COV, np.ones(5))
    np.testing.assert_allclose(w, x / x.sum(), rtol=1e-9)
    # No other fully invested portfolio has lower variance.
    for _ in range(200):
        v = rng.normal(size=5)
        v /= v.sum()
        assert w @ COV @ w <= v @ COV @ v + 1e-12


def test_min_variance_long_only():
    # Vols 20% and 30%, correlation 0.9: the unconstrained solution shorts the riskier one.
    cov = np.array([[0.04, 0.054, 0.0], [0.054, 0.09, 0.0], [0.0, 0.0, 0.09]])
    assert np.any(np.linalg.solve(cov, np.ones(3)) < 0)
    w = call(min_variance_long_only, cov)
    assert np.isclose(w.sum(), 1.0) and np.all(w >= 0)
    best = minimize(lambda v: v @ cov @ v, np.full(3, 1 / 3), bounds=[(0, 1)] * 3,
                    constraints={"type": "eq", "fun": lambda v: v.sum() - 1})
    assert w @ cov @ w <= best.fun * 1.05  # heuristic: within 5% of the true optimum


def test_min_variance_long_only_no_constraint_binding():
    cov = np.diag([0.04, 0.09, 0.01])
    w = call(min_variance_long_only, cov)
    x = 1 / np.diag(cov)
    np.testing.assert_allclose(w, x / x.sum(), rtol=1e-9)


def test_tangency_portfolio():
    w = call(tangency_portfolio, MU, COV, RF)
    x = np.linalg.solve(COV, MU - RF)
    np.testing.assert_allclose(w, x / x.sum(), rtol=1e-9)
    sharpe = lambda v: (v @ MU - RF) / np.sqrt(v @ COV @ v)
    for _ in range(200):
        v = rng.normal(size=5)
        v /= v.sum()
        assert sharpe(w) >= sharpe(v) - 1e-12


def test_efficient_frontier():
    targets = np.linspace(0.03, 0.12, 7)
    vols, weights = call(efficient_frontier, MU, COV, targets)
    assert vols.shape == (7,) and weights.shape == (7, 5)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    np.testing.assert_allclose(weights @ MU, targets, atol=1e-10)
    np.testing.assert_allclose(vols, np.sqrt(np.einsum("ij,jk,ik->i", weights, COV, weights)))
    # Every frontier vol is at or above the global minimum variance vol.
    x = np.linalg.solve(COV, np.ones(5))
    w_mv = x / x.sum()
    assert np.all(vols >= np.sqrt(w_mv @ COV @ w_mv) - 1e-12)
