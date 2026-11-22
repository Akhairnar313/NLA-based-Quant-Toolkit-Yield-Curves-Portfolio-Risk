"""Triangular and bidiagonal solvers (Session 2)."""

import numpy as np


def forward_substitution(L, b):
    # Input is Lower Triangular matrix L and column vector b
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
    # Input is Upper Triangular matrix and column vector b
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
    # Input is Lower Bidiagonal matrix L and column vector b
    n = len(b)
    x = np.zeros(n)
    x[0] = b[0] / L[0, 0]
    for i in range(1, n):
        x[i] = (b[i] - L[i, i-1] * x[i-1]) / L[i, i]
    return x


def backward_substitution_bidiagonal(U, b):
    # Input is Upper Bidiagonal matrix U and column vector b
    n = len(b)
    x = np.zeros(n)
    x[n-1] = b[n-1] / U[n-1, n-1]
    for i in range(n-2, -1, -1):
        x[i] = (b[i] - U[i, i+1] * x[i+1]) / U[i, i]
    return x
