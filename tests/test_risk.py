import numpy as np
import pytest
from scipy.stats import norm

from finance.risk import (portfolio_volatility, risk_contributions, diversification_ratio,
                          parametric_var_cvar, monte_carlo_var_cvar)
from tests.helpers import call, random_spd

rng = np.random.default_rng(8)
COV = random_spd(5, rng) / 100
W = np.array([0.4, 0.25, 0.15, 0.1, 0.1])


def test_portfolio_volatility():
    assert np.isclose(call(portfolio_volatility, W, COV), np.sqrt(W @ COV @ W))


def test_risk_contributions():
    rc = call(risk_contributions, W, COV)
    assert np.isclose(rc.sum(), 1.0)
    np.testing.assert_allclose(rc, W * (COV @ W) / (W @ COV @ W))
    # Uncorrelated equal-vol assets with equal weights: equal risk shares.
    np.testing.assert_allclose(call(risk_contributions, np.full(4, 0.25), np.eye(4)), 0.25)


def test_diversification_ratio():
    assert np.isclose(call(diversification_ratio, W, COV),
                      W @ np.sqrt(np.diag(COV)) / np.sqrt(W @ COV @ W))
    perfectly_correlated = np.outer([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])
    assert np.isclose(call(diversification_ratio, np.full(3, 1 / 3), perfectly_correlated), 1.0)


@pytest.mark.parametrize("confidence", [0.95, 0.99])
def test_parametric_var_cvar(confidence):
    mean = np.full(5, 0.0005)
    var, cvar = call(parametric_var_cvar, W, COV, confidence, horizon_days=10, mean=mean)
    s = np.sqrt(10 * W @ COV @ W)
    m = 10 * W @ mean
    z = norm.ppf(confidence)
    assert np.isclose(var, z * s - m)
    assert np.isclose(cvar, s * norm.pdf(z) / (1 - confidence) - m)
    assert cvar >= var


def test_monte_carlo_var_cvar():
    r = np.random.default_rng(1).normal(0, 0.1, 400_000)
    var, cvar = call(monte_carlo_var_cvar, r, 0.99)
    assert np.isclose(var, -np.quantile(r, 0.01), rtol=1e-3)
    assert np.isclose(var, 0.1 * norm.ppf(0.99), rtol=0.02)
    assert np.isclose(cvar, 0.1 * norm.pdf(norm.ppf(0.99)) / 0.01, rtol=0.02)
    assert cvar >= var > 0
