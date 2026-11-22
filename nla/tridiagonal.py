"""Linear-time methods for tridiagonal matrices leveraging bidiagonal algos."""

import numpy as np

from nla.triangular_matrix_operations import forward_substitution_bidiagonal, backward_substitution_bidiagonal


def lu_tridiagonal(A):
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))
    U[0, 0] = A[0, 0]
    for k in range(0, n-1):
        U[k, k+1] = A[k, k+1]  # super-diagonal of U is copied from A
        L[k+1, k] = A[k+1, k] / U[k, k]
        U[k+1, k+1] = A[k+1, k+1] - L[k+1, k] * U[k, k+1]
    return L, U


def solve_tridiagonal(A, b):
    L, U = lu_tridiagonal(A)
    y = forward_substitution_bidiagonal(L, b)   # L y = b
    x = backward_substitution_bidiagonal(U, y)  # U x = y
    return x
