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
