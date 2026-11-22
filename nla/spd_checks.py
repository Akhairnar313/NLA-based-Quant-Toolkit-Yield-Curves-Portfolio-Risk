"""Checks to understand if a matrix is spd or not"""

import numpy as np

from nla.lu import det_via_lu


def gershgorin_discs(A):
    # Gives the radius (sum of all entries of a row except diagonal entry) and center (diagonal entry)
    n = A.shape[0]
    centers = np.zeros(n)
    radii = np.zeros(n)
    for i in range(0, n):
        centers[i] = A[i, i]
        for j in range(0, n):
            if j != i:
                radii[i] += abs(A[i, j])
    return centers, radii


def is_diagonally_dominant(A, strict=True):
    #Row diagonal dominance: abs(A[i, i]) > (for strict) or >= (if weakly) sum of |A[i, j]| for every row i where j is everything except equal to i
    n = A.shape[0]
    centers, radii = gershgorin_discs(A)  # |center| vs radius is exactly the test
    for i in range(0, n):
        if strict and not abs(centers[i]) > radii[i]:
            return False
        if not strict and not abs(centers[i]) >= radii[i]:
            return False
    return True



def is_spd_sylvester(A, tol=1e-12):
    #Sylvester's criterion: a symmetric matrix is spd iff every leading principal minoris > 0
    n = A.shape[0]
    # symmetry check (up to tol)
    for i in range(0, n):
        for j in range(i+1, n):
            if abs(A[i, j] - A[j, i]) > tol:
                return False
    # every leading principal minor must be positive
    for k in range(1, n+1):
        if not det_via_lu(A[:k, :k]) > tol:
            return False
    return True
