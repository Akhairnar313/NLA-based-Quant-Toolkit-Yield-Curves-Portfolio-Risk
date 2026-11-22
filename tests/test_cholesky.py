import numpy as np
import pytest

from nla.cholesky import cholesky, cholesky_solve, inverse_spd, det_spd, is_spd
from tests.helpers import call, random_spd

rng = np.random.default_rng(4)
SIZES = [1, 2, 5, 20]
NOT_SPD = [
    np.array([[1., 2.], [2., 1.]]),                                   # indefinite
    np.array([[1., 0.9, -0.9], [0.9, 1., 0.9], [-0.9, 0.9, 1.]]),     # impossible correlations
    np.array([[1., 1.], [1., 1.]]),                                   # singular (duplicated asset)
]


@pytest.mark.parametrize("n", SIZES)
def test_cholesky(n):
    A = random_spd(n, rng)
    A_copy = A.copy()
    L = call(cholesky, A)
    np.testing.assert_allclose(L, np.tril(L))
    assert np.all(np.diag(L) > 0)
    np.testing.assert_allclose(L, np.linalg.cholesky(A), rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(A, A_copy)


@pytest.mark.parametrize("A", NOT_SPD, ids=["indefinite", "bad_corr", "singular"])
def test_cholesky_rejects_non_spd(A):
    with pytest.raises(ValueError):
        call(cholesky, A)


@pytest.mark.parametrize("n", SIZES)
def test_cholesky_solve_vector_and_matrix(n):
    A = random_spd(n, rng)
    b = rng.normal(size=n)
    B = rng.normal(size=(n, 3))
    np.testing.assert_allclose(call(cholesky_solve, A, b), np.linalg.solve(A, b), rtol=1e-9, atol=1e-12)
    X = call(cholesky_solve, A, B)
    assert X.shape == B.shape
    np.testing.assert_allclose(X, np.linalg.solve(A, B), rtol=1e-9, atol=1e-12)


def test_inverse_and_det_spd():
    A = random_spd(6, rng)
    np.testing.assert_allclose(call(inverse_spd, A), np.linalg.inv(A), rtol=1e-9, atol=1e-12)
    assert np.isclose(call(det_spd, A), np.linalg.det(A), rtol=1e-9)


def test_is_spd():
    assert call(is_spd, random_spd(5, rng))
    for A in NOT_SPD:
        assert not call(is_spd, A)
    assert not call(is_spd, np.array([[2., 1.], [0., 2.]]))  # not symmetric
