"""Covariance and correlation matrices from return time series (Sessions 1 and 6).

Written with matrix products, not pandas' built-in covariance / correlation methods:

    X  = T x n matrix of returns (one row per day, one column per asset)
    Xc = X - column means          (demeaned)
    Sigma = Xc^T Xc / (T - 1)      (sample covariance)

Correlation rescales the covariance by the volatilities:
    rho = D^{-1} Sigma D^{-1},  D = diag(sqrt(diag(Sigma)))

Inputs may be a NumPy array or a pandas DataFrame.
"""

import numpy as np


def covariance_matrix(returns):
    """Sample covariance matrix (divide by T - 1) of a T x n return matrix.

    Returns
    -------
    (n, n) ndarray, symmetric
    """
    X = np.asarray(returns, dtype=float)
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / (X.shape[0] - 1)
    return (S + S.T) / 2  # exactly symmetric despite rounding


def correlation_matrix(returns):
    """Sample correlation matrix of a T x n return matrix (ones on the diagonal)."""
    corr, _ = cov_to_corr(covariance_matrix(returns))
    return corr


def cov_to_corr(cov):
    """Correlation matrix and volatility vector from a covariance matrix.

    Returns
    -------
    corr : (n, n) ndarray
    vols : (n,) ndarray, sqrt of the diagonal of cov
    """
    cov = np.asarray(cov, dtype=float)
    vols = np.sqrt(np.diag(cov))
    corr = cov / np.outer(vols, vols)
    np.fill_diagonal(corr, 1.0)
    return corr, vols


def corr_to_cov(corr, vols):
    """Covariance matrix from a correlation matrix and volatilities: D rho D."""
    vols = np.asarray(vols, dtype=float)
    return np.asarray(corr, dtype=float) * np.outer(vols, vols)
