"""LU decomposition without/with pivoting and other applications"""

import numpy as np
from nla.basic_operations import is_nonsingular


def lu_no_pivot(A):
    n = A.shape[0]
    U = A.astype(float)
    L = np.eye(n)
    for k in range(0, n-1):
        for i in range(k+1, n):
            L[i, k] = U[i, k] / U[k, k]
            for j in range(k+1, n):
                U[i, j] -= L[i, k] * U[k, j]
            U[i, k] = 0.0  # eliminated entry, set exactly to zero
    return L, U


def lu_row_pivot(A):
    n = A.shape[0]
    U = A.astype(float)
    L = np.eye(n)
    P = np.eye(n)
    for k in range(0, n-1):
        # pivot: row (among k..n-1) with the largest |entry| in column k
        p = k + np.argmax(np.abs(U[k:, k]))
        if p != k:
            U[[k, p], :] = U[[p, k], :]    # swap rows of U
            P[[k, p], :] = P[[p, k], :]    # record the swap in P
            L[[k, p], :k] = L[[p, k], :k]  # swap multipliers already computed
        for i in range(k+1, n):
            L[i, k] = U[i, k] / U[k, k]
            for j in range(k+1, n):
                U[i, j] -= L[i, k] * U[k, j]
            U[i, k] = 0.0  # eliminated entry, set exactly to zero
    return P, L, U


def lu_solve(A, b):
    n = A.shape[0]
    if not is_nonsingular(A):
        raise ValueError("matrix is singular, cannot solve A x = b")
    P, L, U = lu_row_pivot(A)
    B = b.reshape(n, -1).astype(float)  # treat a vector as a single column
    y = P @ B
    # forward substitution: L y = P b (L has unit diagonal, so no division)
    for i in range(1, n):
        for j in range(0, i):
            y[i, :] -= L[i, j] * y[j, :]
    # back substitution: U x = y
    x = y
    for i in range(n-1, -1, -1):
        for j in range(i+1, n):
            x[i, :] -= U[i, j] * x[j, :]
        x[i, :] /= U[i, i]
    return x.reshape(b.shape)


def inverse_via_lu(A):
    n = A.shape[0]
    return lu_solve(A, np.eye(n))  # columns of I are the n right-hand sides


def det_via_lu(A):
    n = A.shape[0]
    P, L, U = lu_row_pivot(A)
    # count row swaps: sort the permutation stored in P one swap at a time
    perm = list(np.argmax(P, axis=1))
    swaps = 0
    for i in range(0, n):
        while perm[i] != i:
            t = perm[i]
            perm[i], perm[t] = perm[t], perm[i]
            swaps += 1
    det = (-1.0) ** swaps
    for i in range(0, n):
        det *= U[i, i]
    return det
