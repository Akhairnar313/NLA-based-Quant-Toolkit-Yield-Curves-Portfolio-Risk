"""Portfolio risk measures (Sessions 7 and 8).

A portfolio's variance is the quadratic form w^T Sigma w, so most risk measures
are linear algebra on the covariance matrix Sigma.

VaR and CVaR are returned as POSITIVE fractions of portfolio value
(0.15 = "lose 15%"), and CVaR >= VaR always.
"""

from statistics import NormalDist

import numpy as np


def portfolio_volatility(weights, cov):
    """sqrt(w^T Sigma w). Annual if cov is annualized."""
    w = np.asarray(weights, dtype=float)
    return float(np.sqrt(w @ np.asarray(cov, dtype=float) @ w))


def risk_contributions(weights, cov):
    """Each holding's share of portfolio variance; the shares sum to 1.

        RC_i = w_i (Sigma w)_i / (w^T Sigma w)

    A holding with 10% of the money can carry 40% of the risk if it is volatile
    and correlated with the rest.
    """
    w = np.asarray(weights, dtype=float)
    marginal = np.asarray(cov, dtype=float) @ w   # (Sigma w)_i: risk added per unit of asset i
    return w * marginal / (w @ marginal)


def diversification_ratio(weights, cov):
    """(w^T sigma) / sqrt(w^T Sigma w), with sigma the vector of asset volatilities.

    1 means no diversification benefit (everything perfectly correlated);
    higher is better.
    """
    w = np.asarray(weights, dtype=float)
    vols = np.sqrt(np.diag(np.asarray(cov, dtype=float)))
    return float((w @ vols) / portfolio_volatility(w, cov))


def parametric_var_cvar(weights, cov, confidence, horizon_days=1, mean=None):
    """VaR and CVaR assuming normal portfolio returns.

    With mu_p = w^T mean * h (0 if mean is None), sigma_p = sqrt(h w^T Sigma w),
    z = NormalDist().inv_cdf(confidence) and phi the standard normal density:

        VaR  = z sigma_p - mu_p
        CVaR = sigma_p phi(z) / (1 - confidence) - mu_p

    cov and mean are per period (e.g. daily); h = horizon_days periods.

    Returns
    -------
    (var, cvar) : floats
    """
    w = np.asarray(weights, dtype=float)
    sigma_p = np.sqrt(horizon_days) * portfolio_volatility(w, cov)
    mu_p = 0.0 if mean is None else horizon_days * float(w @ np.asarray(mean, dtype=float))
    std_normal = NormalDist()
    z = std_normal.inv_cdf(confidence)
    var = z * sigma_p - mu_p
    cvar = sigma_p * std_normal.pdf(z) / (1 - confidence) - mu_p
    return float(var), float(cvar)


def monte_carlo_var_cvar(returns, confidence):
    """Historical / simulated VaR and CVaR from a sample of portfolio returns.

    VaR  = minus the (1 - confidence) quantile of the returns
    CVaR = minus the average of the returns at or below that quantile

    Returns
    -------
    (var, cvar) : floats
    """
    r = np.asarray(returns, dtype=float)
    cutoff = np.quantile(r, 1 - confidence)
    return float(-cutoff), float(-r[r <= cutoff].mean())
