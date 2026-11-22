"""Cholesky decomposition and what it gives you for SPD matrices (Session 6).

A symmetric positive definite (SPD) matrix A factors as A = L L^T, with L lower
triangular and a positive diagonal. Every valid covariance matrix is SPD, so in
this project Cholesky does three jobs:

    * validity check  : the factorization breaks down (a non-positive pivot)
                        exactly when A is not SPD
    * linear solves   : A x = b  ->  L y = b, then L^T x = y  (min-variance weights)
    * simulation      : if z ~ N(0, I) then L z ~ N(0, A)   (correlated Monte Carlo)

Operation count: about n^3 / 3 for the factorization, half of LU.
"""

import numpy as np

from nla.triangular_matrix_operations import forward_substitution, backward_substitution


def _is_symmetric(A, tol):
    n = A.shape[0]
    for i in range(0, n):
        for j in range(i+1, n):
            if abs(A[i, j] - A[j, i]) > tol * max(1.0, abs(A[i, j])):
                return False
    return True


def cholesky(A, tol=1e-12):
    """Cholesky factor of an SPD matrix: A = L L^T.

    Column by column: the diagonal entry is the square root of what is left of
    A[j, j] after removing the earlier columns, and the entries below it follow
    from A[i, j] = sum_k L[i, k] L[j, k].

    Parameters
    ----------
    A : (n, n) ndarray, symmetric positive definite

    Returns
    -------
    L : (n, n) ndarray, lower triangular with a strictly positive diagonal

    Raises
    ------
    ValueError if A is not symmetric, or not positive definite (a diagonal
    pivot <= 0 appears during the factorization). A is not modified.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if not _is_symmetric(A, tol):
        raise ValueError("matrix is not symmetric")
    L = np.zeros((n, n))
    for j in range(0, n):
        s = A[j, j] - L[j, :j] @ L[j, :j]
        if s <= tol * abs(A[j, j]):
            raise ValueError("matrix is not positive definite")
        L[j, j] = np.sqrt(s)
        for i in range(j+1, n):
            L[i, j] = (A[i, j] - L[i, :j] @ L[j, :j]) / L[j, j]
    return L


def cholesky_solve(A, b):
    """Solve A x = b for SPD A: factor A = L L^T, then L y = b and L^T x = y.

    b may be a vector (n,) or a matrix (n, k) of k right-hand sides; x has the
    same shape as b. The factorization is done once and reused for every column.
    """
    L = cholesky(A)
    b = np.asarray(b, dtype=float)
    n = L.shape[0]
    B = b.reshape(n, -1)
    X = np.zeros_like(B)
    for k in range(0, B.shape[1]):
        y = forward_substitution(L, B[:, k])      # L y = b
        X[:, k] = backward_substitution(L.T, y)   # L^T x = y
    return X.reshape(b.shape)


def inverse_spd(A):
    """Inverse of an SPD matrix via Cholesky (solve A X = I, column by column)."""
    n = np.asarray(A).shape[0]
    return cholesky_solve(A, np.eye(n))


def det_spd(A):
    """Determinant of an SPD matrix: det(A) = det(L)^2 = (prod of diag(L))^2."""
    L = cholesky(A)
    det_L = 1.0
    for i in range(0, L.shape[0]):
        det_L *= L[i, i]
    return det_L ** 2


def is_spd(A, tol=1e-12):
    """True if A is symmetric (within tol) and its Cholesky factorization succeeds.

    Much cheaper than Sylvester's criterion (one n^3/3 factorization instead of n
    determinants). Compare with nla.spd_checks.is_spd_sylvester.
    """
    try:
        cholesky(A, tol)
        return True
    except ValueError:
        return False
