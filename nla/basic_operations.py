#Basic matrix operation and checks that can be used across the project

import numpy as np


def is_nonsingular(A, tol=1e-12):
    n = A.shape[0]
    U = A.astype(float)
    for k in range(0, n):  # up to n (not n-1) so the last pivot is checked too
        p = k + np.argmax(np.abs(U[k:, k]))
        if abs(U[p, k]) <= tol:
            return False  # every candidate pivot in column k is zero
        if p != k:
            U[[k, p], :] = U[[p, k], :]
        for i in range(k+1, n):
            m = U[i, k] / U[k, k]
            for j in range(k+1, n):
                U[i, j] -= m * U[k, j]
            U[i, k] = 0.0
    return True