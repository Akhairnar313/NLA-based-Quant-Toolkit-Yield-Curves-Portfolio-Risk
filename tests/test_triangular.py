import numpy as np
import pytest

from nla.triangular_matrix_operations import (
    forward_substitution,
    backward_substitution,
    forward_substitution_bidiagonal,
    backward_substitution_bidiagonal,
)
from tests.helpers import call

rng = np.random.default_rng(0)


def test_forward_substitution_bond_discount_factors():
    # Session 2 example: four bonds, cash-flow matrix is lower triangular,
    # solving gives the discount factors.
    L = np.array([
        [100., 0, 0, 0],
        [6., 106., 0, 0],
        [8., 8., 108., 0],
        [5., 5., 5., 105.],
    ])
    b = np.array([98., 104., 111., 102.])
    x = forward_substitution(L, b)
    np.testing.assert_allclose(L @ x, b)
    np.testing.assert_allclose(x, np.linalg.solve(L, b))


def test_backward_substitution_example():
    U = np.array([
        [2., -1., 0, 1.],
        [0, -1., 1., 0],
        [0, 0, 1., -1.],
        [0, 0, 0, -1.],
    ])
    b = np.array([3., 5., -3., -14.])
    np.testing.assert_allclose(backward_substitution(U, b), np.linalg.solve(U, b))


@pytest.mark.parametrize("n", [1, 2, 5, 30])
def test_triangular_random(n):
    L = np.tril(rng.uniform(-1, 1, (n, n))) + n * np.eye(n)
    U = np.triu(rng.uniform(-1, 1, (n, n))) + n * np.eye(n)
    b = rng.normal(size=n)
    np.testing.assert_allclose(forward_substitution(L, b), np.linalg.solve(L, b))
    np.testing.assert_allclose(backward_substitution(U, b), np.linalg.solve(U, b))


@pytest.mark.parametrize("n", [1, 2, 6, 50])
def test_bidiagonal(n):
    d = rng.uniform(1, 2, n)
    off = rng.uniform(-1, 1, n - 1)
    L = np.diag(d) + np.diag(off, -1)
    U = np.diag(d) + np.diag(off, 1)
    b = rng.normal(size=n)
    np.testing.assert_allclose(call(forward_substitution_bidiagonal, L, b), np.linalg.solve(L, b))
    np.testing.assert_allclose(call(backward_substitution_bidiagonal, U, b), np.linalg.solve(U, b))
