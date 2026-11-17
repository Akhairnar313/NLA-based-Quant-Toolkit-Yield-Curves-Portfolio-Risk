"""LU decomposition and its uses (Sessions 2-3)."""

import numpy as np

from nla.triangular import forward_substitution, backward_substitution


def lu_no_pivot(A):
    """LU decomposition without pivoting: A = L U.

    L is unit lower triangular (ones on the diagonal), U is upper triangular.
    Exists and is unique when all leading principal minors of A are non-zero.

    Operation count: 2n^3/3 + O(n^2).

    Parameters
    ----------
    A : (n, n) ndarray

    Returns
    -------
    L : (n, n) ndarray, unit lower triangular
    U : (n, n) ndarray, upper triangular
    """
    n = A.shape[0]
    U = A.astype(float)  # work on a copy so the input is not modified
    L = np.eye(n)
    for k in range(0, n-1):
        for i in range(k+1, n):
            L[i, k] = U[i, k] / U[k, k]
            for j in range(k+1, n):
                U[i, j] -= L[i, k] * U[k, j]
            U[i, k] = 0.0  # eliminated entry, set exactly to zero
    return L, U


def lu_row_pivot(A):
    """LU decomposition with row pivoting: P A = L U.

    At step i, swap in the row (among rows i..n-1) with the largest |entry|
    in column i, so every multiplier in L satisfies |L[j, i]| <= 1.

    Parameters
    ----------
    A : (n, n) ndarray, non-singular

    Returns
    -------
    P : (n, n) ndarray, permutation matrix
    L : (n, n) ndarray, unit lower triangular, |entries| <= 1
    U : (n, n) ndarray, upper triangular
    """
    raise NotImplementedError("TODO: LU decomposition with row pivoting (Session 3)")


def lu_solve(A, b):
    """Solve A x = b using P A = L U, then L y = P b and U x = y.

    Supports multiple right-hand sides: if b has shape (n, m), solve the m
    systems A X = B with a single factorization (Session 3). This is
    where the factorization pays off: 2n^3/3 once, then 2n^2 per system.

    Parameters
    ----------
    A : (n, n) ndarray, non-singular
    b : (n,) or (n, m) ndarray

    Returns
    -------
    x : same shape as b
    """
    raise NotImplementedError("TODO: linear solve via LU with row pivoting (Sessions 2-3)")


def inverse_via_lu(A):
    """Inverse of a non-singular matrix: solve A X = I with lu_solve.

    Returns
    -------
    A_inv : (n, n) ndarray
    """
    raise NotImplementedError("TODO: matrix inverse via LU (Session 3)")


def det_via_lu(A):
    """Determinant from P A = L U: det(A) = det(P) * prod(diag(U)).

    det(P) = (-1)^(number of row swaps). det(L) = 1.

    Returns
    -------
    det : float
    """
    raise NotImplementedError("TODO: determinant via LU (Session 3)")


def is_nonsingular(A, tol=1e-12):
    """Decide whether A is non-singular using the LU algorithm with row pivoting.

    A is singular exactly when the pivoting step finds a column whose
    candidate pivots are all (numerically) zero, i.e. |pivot| <= tol.

    Returns
    -------
    bool
    """
    raise NotImplementedError("TODO: identify non-singular matrices using LU (Session 3)")
