"""Principal component analysis of covariance / correlation matrices (Session 8).

The eigenvectors of a covariance matrix are the "factors" that drive the data,
and each eigenvalue is how much variance its factor explains. For ETF returns
the first factor is usually "the market"; for daily changes in Treasury yields
the first three are level, slope and curvature.
"""

import numpy as np

from nla.eigen import power_method, qr_algorithm


def pca(cov):
    """All principal components of a covariance (or correlation) matrix.

    Returns
    -------
    eigenvalues : (n,) ndarray, decreasing
    eigenvectors : (n, n) ndarray, column j is component j
    explained : (n,) ndarray, eigenvalues / sum(eigenvalues), sums to 1
    """
    vals, vecs = qr_algorithm(np.asarray(cov, dtype=float))
    return vals, vecs, vals / vals.sum()


def dominant_factor_share(corr):
    """Share of total variation explained by the single strongest factor.

    Largest eigenvalue of the CORRELATION matrix (power method) divided by n
    (the trace of a correlation matrix is n). Near 1: everything moves together,
    so owning many tickers diversifies less than it looks.
    """
    corr = np.asarray(corr, dtype=float)
    lam, _ = power_method(corr)
    return lam / corr.shape[0]
