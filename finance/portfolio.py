"""Efficient portfolios (Sessions 7 and 8).

All of these come from Lagrange multiplier systems whose matrix is the
covariance Sigma, so each one is a Cholesky solve. Sigma^{-1} is never formed
explicitly: we solve Sigma x = b instead.

Weights always sum to 1. Negative weights mean short positions, except in
min_variance_long_only.
"""

import numpy as np

from nla.cholesky import cholesky_solve
from nla.lu import lu_solve


def min_variance_portfolio(cov):
    """Global minimum-variance portfolio (shorting allowed).

        w = Sigma^{-1} 1 / (1^T Sigma^{-1} 1)
    """
    x = cholesky_solve(cov, np.ones(np.asarray(cov).shape[0]))
    return x / x.sum()


def min_variance_long_only(cov):
    """Minimum-variance weights with no short positions (all w_i >= 0).

    Simple active-set heuristic: solve the unconstrained problem on the active
    assets; if some weight is negative, drop the most negative asset and solve
    again; stop when all weights are >= 0. Dropped assets get weight 0.
    (An approximation of the true long-only optimum; good enough to show the idea.)

    Returns
    -------
    (n,) ndarray, w >= 0, sums to 1
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    active = list(range(n))
    while True:
        w_sub = min_variance_portfolio(cov[np.ix_(active, active)])
        if np.all(w_sub >= -1e-12):
            break
        active.pop(int(np.argmin(w_sub)))
    w = np.zeros(n)
    w[active] = np.clip(w_sub, 0.0, None)
    return w / w.sum()


def tangency_portfolio(mu, cov, rf):
    """Tangency (maximum Sharpe ratio) portfolio of risky assets.

        w = Sigma^{-1} (mu - rf 1) / (1^T Sigma^{-1} (mu - rf 1))

    mu and cov in the same units as rf (e.g. all annual).
    """
    x = cholesky_solve(cov, np.asarray(mu, dtype=float) - rf)
    return x / x.sum()


def efficient_frontier(mu, cov, target_returns):
    """Minimum-variance portfolio for each target expected return (shorting allowed).

    Minimize w^T Sigma w subject to mu^T w = target and 1^T w = 1. With
    x = Sigma^{-1} 1 and y = Sigma^{-1} mu (two Cholesky solves), the solution is
    a combination w = lambda_1 x + lambda_2 y; the 2x2 system for the
    multipliers uses a = 1^T x, b = 1^T y = mu^T x, c = mu^T y:

        [a b] [lambda_1]   [  1   ]
        [b c] [lambda_2] = [target]

    Returns
    -------
    vols : (k,) ndarray, sqrt(w^T Sigma w) for each target
    weights : (k, n) ndarray, one portfolio per row
    """
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    targets = np.asarray(target_returns, dtype=float)
    XY = cholesky_solve(cov, np.column_stack([np.ones(len(mu)), mu]))   # both solves at once
    x, y = XY[:, 0], XY[:, 1]
    a, b, c = x.sum(), y.sum(), mu @ y
    # One 2x2 system, one right-hand side per target return.
    lambdas = lu_solve(np.array([[a, b], [b, c]]), np.vstack([np.ones(len(targets)), targets]))
    weights = np.outer(lambdas[0], x) + np.outer(lambdas[1], y)
    vols = np.sqrt(np.sum((weights @ cov) * weights, axis=1))
    return vols, weights
