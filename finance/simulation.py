"""Correlated Monte Carlo via the Cholesky factor (Session 6).

Linear Transformation Property: if z ~ N(0, I) and Sigma = L L^T, then
x = mu + L z ~ N(mu, Sigma). One Cholesky factorization turns independent
normal draws into draws with any covariance you want.
"""

import numpy as np

from nla.cholesky import cholesky


def correlated_normal_samples(cov, n_samples, mean=None, seed=None):
    """Draw n_samples from N(mean, cov).

    Parameters
    ----------
    cov : (n, n) SPD ndarray
    n_samples : int
    mean : (n,) ndarray or None (zero mean)
    seed : int or None, passed to np.random.default_rng

    Returns
    -------
    (n_samples, n) ndarray, one sample per row
    """
    L = cholesky(cov)
    z = np.random.default_rng(seed).standard_normal((n_samples, L.shape[0]))
    x = z @ L.T                        # each row is L z for one draw z
    if mean is not None:
        x += np.asarray(mean, dtype=float)
    return x


def simulate_portfolio_returns(weights, cov_daily, horizon_days, n_sims, seed=42):
    """Simulated buy-and-hold portfolio SIMPLE returns over the horizon.

    Model: asset log returns over the horizon are N(m, Sigma_h) with
    Sigma_h = horizon_days * cov_daily and m = -diag(Sigma_h) / 2, so each asset's
    expected simple return is 0 (a conservative choice for risk work).
    Asset simple return = exp(log return) - 1; portfolio return = samples @ weights.

    Returns
    -------
    (n_sims,) ndarray of portfolio returns (fractions, e.g. -0.12 = lose 12%)
    """
    cov_h = np.asarray(cov_daily, dtype=float) * horizon_days
    log_r = correlated_normal_samples(cov_h, n_sims, mean=-0.5 * np.diag(cov_h), seed=seed)
    return (np.exp(log_r) - 1.0) @ np.asarray(weights, dtype=float)
