import numpy as np
import pytest

from nla.tridiagonal import lu_tridiagonal, solve_tridiagonal
from tests.helpers import call, random_tridiagonal

rng = np.random.default_rng(2)


@pytest.mark.parametrize("n", [1, 2, 5, 100])
def test_lu_tridiagonal(n):
    A = random_tridiagonal(n, rng)
    L, U = call(lu_tridiagonal, A)
    # L unit lower bidiagonal, U upper bidiagonal
    np.testing.assert_allclose(L, np.diag(np.ones(n)) + np.diag(np.diag(L, -1), -1))
    np.testing.assert_allclose(U, np.diag(np.diag(U)) + np.diag(np.diag(U, 1), 1))
    np.testing.assert_allclose(np.diag(U, 1), np.diag(A, 1))
    np.testing.assert_allclose(L @ U, A, atol=1e-12)


@pytest.mark.parametrize("n", [1, 2, 5, 100])
def test_solve_tridiagonal(n):
    A = random_tridiagonal(n, rng)
    b = rng.normal(size=n)
    np.testing.assert_allclose(call(solve_tridiagonal, A, b), np.linalg.solve(A, b), rtol=1e-9, atol=1e-12)


def test_solve_tridiagonal_spline_system():
    # Shape of the natural cubic spline system: 2(h_i + h_{i+1}) on the diagonal, h on the off-diagonals.
    h = np.array([0.25, 0.25, 0.5, 1.0, 1.0, 2.0])
    diag = 2 * (h[:-1] + h[1:])
    A = np.diag(diag) + np.diag(h[1:-1], -1) + np.diag(h[1:-1], 1)
    b = rng.normal(size=len(diag))
    np.testing.assert_allclose(call(solve_tridiagonal, A, b), np.linalg.solve(A, b), rtol=1e-9)
