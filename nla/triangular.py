"""Triangular and bidiagonal solvers (Session 2)."""

import numpy as np


def forward_substitution(L, b):
    """Solve L x = b for a non-singular lower triangular matrix L.

    Operation count: n^2 + O(n).
    """
    n = len(b)
    x = np.zeros(n)
    x[0] = b[0] / L[0, 0]
    for i in range(1, n):
        sum = 0
        for j in range(0, i):
            sum += L[i, j] * x[j]
        x[i] = (b[i] - sum) / L[i, i]
    return x


def backward_substitution(U, b):
    """Solve U x = b for a non-singular upper triangular matrix U.

    Operation count: n^2 + O(n).
    """
    n = len(b)
    x = np.zeros(n)
    x[n-1] = b[n-1]/U[n-1, n-1]
    for i in range(n-2, -1, -1):
        sum = 0
        for j in range(i+1, n):
            sum += U[i, j] * x[j]
        x[i] = (b[i] - sum) / U[i, i]
    return x


def forward_substitution_bidiagonal(L, b):
    """Solve L x = b where L is lower bidiagonal (non-zeros only on the
    main diagonal and the first sub-diagonal).

    Only L[i, i] and L[i, i-1] may be read, so the cost is O(n)
    (about 3n operations) instead of n^2.

    Parameters
    ----------
    L : (n, n) ndarray, lower bidiagonal, non-singular
    b : (n,) ndarray

    Returns
    -------
    x : (n,) ndarray
    """
    n = len(b)
    x = np.zeros(n)
    x[0] = b[0] / L[0, 0]
    for i in range(1, n):
        x[i] = (b[i] - L[i, i-1] * x[i-1]) / L[i, i]
    return x


def backward_substitution_bidiagonal(U, b):
    """Solve U x = b where U is upper bidiagonal (non-zeros only on the
    main diagonal and the first super-diagonal).

    Only U[i, i] and U[i, i+1] may be read, so the cost is O(n).

    Parameters
    ----------
    U : (n, n) ndarray, upper bidiagonal, non-singular
    b : (n,) ndarray

    Returns
    -------
    x : (n,) ndarray
    """
    n = len(b)
    x = np.zeros(n)
    x[n-1] = b[n-1] / U[n-1, n-1]
    for i in range(n-2, -1, -1):
        x[i] = (b[i] - U[i, i+1] * x[i+1]) / U[i, i]
    return x
