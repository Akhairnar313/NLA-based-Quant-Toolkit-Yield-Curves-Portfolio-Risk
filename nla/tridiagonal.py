"""Linear-time methods for tridiagonal matrices (Session 3).

These power the cubic spline interpolation of interest rate curves:
the spline's linear system is tridiagonal, so it can be solved in O(n)
instead of O(n^3).
"""

import numpy as np

from nla.triangular import forward_substitution_bidiagonal, backward_substitution_bidiagonal


def lu_tridiagonal(A):
    """LU decomposition without pivoting for a tridiagonal matrix: A = L U.

    L is unit lower bidiagonal and U is upper bidiagonal, and the
    super-diagonal of U equals the super-diagonal of A. Each step touches
    only a constant number of entries, so the total cost is O(n).

    Parameters
    ----------
    A : (n, n) ndarray, tridiagonal, with an LU decomposition
        (e.g. strictly diagonally dominant)

    Returns
    -------
    L : (n, n) ndarray, unit lower bidiagonal
    U : (n, n) ndarray, upper bidiagonal
    """
    raise NotImplementedError("TODO: LU decomposition for tridiagonal matrices (Session 3)")


def solve_tridiagonal(A, b):
    """Solve A x = b for tridiagonal A in O(n):
    lu_tridiagonal, then bidiagonal forward and backward substitution.

    Parameters
    ----------
    A : (n, n) ndarray, tridiagonal
    b : (n,) ndarray

    Returns
    -------
    x : (n,) ndarray
    """
    raise NotImplementedError("TODO: tridiagonal linear solver (Session 3)")
