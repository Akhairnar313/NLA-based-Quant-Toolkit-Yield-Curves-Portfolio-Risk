import numpy as np
import pytest

from nla.lu import lu_no_pivot, lu_row_pivot, lu_solve, inverse_via_lu, det_via_lu, is_nonsingular
from tests.helpers import call, random_diag_dominant

rng = np.random.default_rng(1)
SIZES = [1, 2, 4, 10, 40]


def assert_unit_lower(L):
    np.testing.assert_allclose(L, np.tril(L))
    np.testing.assert_allclose(np.diag(L), 1.0)


def assert_upper(U):
    np.testing.assert_allclose(U, np.triu(U))


def assert_permutation(P):
    n = P.shape[0]
    assert set(np.unique(P)) <= {0.0, 1.0}
    np.testing.assert_allclose(P.sum(axis=0), np.ones(n))
    np.testing.assert_allclose(P.sum(axis=1), np.ones(n))


@pytest.mark.parametrize("n", SIZES)
def test_lu_no_pivot(n):
    A = random_diag_dominant(n, rng)
    L, U = call(lu_no_pivot, A)
    assert_unit_lower(L)
    assert_upper(U)
    np.testing.assert_allclose(L @ U, A, atol=1e-12)


def test_lu_no_pivot_does_not_modify_input():
    A = random_diag_dominant(5, rng)
    A_copy = A.copy()
    call(lu_no_pivot, A)
    np.testing.assert_array_equal(A, A_copy)


@pytest.mark.parametrize("n", SIZES)
def test_lu_row_pivot(n):
    A = rng.normal(size=(n, n))
    P, L, U = call(lu_row_pivot, A)
    assert_permutation(P)
    assert_unit_lower(L)
    assert_upper(U)
    assert np.all(np.abs(L) <= 1 + 1e-12), "row pivoting must keep |multipliers| <= 1"
    np.testing.assert_allclose(P @ A, L @ U, atol=1e-12)


def test_lu_row_pivot_needs_pivoting():
    # Zero in the (0, 0) position: LU without pivoting breaks down.
    A = np.array([[0., 2., 1.], [1., 1., 1.], [2., 1., 3.]])
    P, L, U = call(lu_row_pivot, A)
    np.testing.assert_allclose(P @ A, L @ U, atol=1e-12)


@pytest.mark.parametrize("n", SIZES)
def test_lu_solve_vector(n):
    A = rng.normal(size=(n, n))
    b = rng.normal(size=n)
    x = call(lu_solve, A, b)
    assert x.shape == b.shape
    np.testing.assert_allclose(x, np.linalg.solve(A, b), rtol=1e-8, atol=1e-10)


def test_lu_solve_multiple_rhs():
    A = rng.normal(size=(6, 6))
    B = rng.normal(size=(6, 4))
    X = call(lu_solve, A, B)
    assert X.shape == B.shape
    np.testing.assert_allclose(X, np.linalg.solve(A, B), rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("n", SIZES)
def test_inverse_via_lu(n):
    A = rng.normal(size=(n, n))
    np.testing.assert_allclose(call(inverse_via_lu, A), np.linalg.inv(A), rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("n", SIZES)
def test_det_via_lu(n):
    A = rng.normal(size=(n, n))
    np.testing.assert_allclose(call(det_via_lu, A), np.linalg.det(A), rtol=1e-9)


def test_det_sign_with_row_swap():
    A = np.array([[0., 1.], [1., 0.]])  # det = -1, requires one swap
    np.testing.assert_allclose(call(det_via_lu, A), -1.0)


def test_is_nonsingular():
    assert call(is_nonsingular, rng.normal(size=(5, 5)))
    singular = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    assert not call(is_nonsingular, singular)
    # Redundant security: third payoff column is the sum of the first two.
    payoff = np.array([[1., 2., 3.], [1., 0., 1.], [1., 1., 2.]])
    assert not call(is_nonsingular, payoff)
