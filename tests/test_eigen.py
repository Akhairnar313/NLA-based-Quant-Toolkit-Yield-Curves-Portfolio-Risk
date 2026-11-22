import numpy as np
import pytest

from nla.eigen import (power_method, inverse_iteration, rayleigh_quotient_iteration,
                       qr_decomposition, qr_algorithm, condition_number_spd)
from tests.helpers import call, random_spd

rng = np.random.default_rng(5)


def assert_eigenpair(A, lam, v, tol=1e-6):
    assert np.isclose(np.sqrt(v @ v), 1.0, atol=1e-8), "eigenvector must have unit length"
    np.testing.assert_allclose(A @ v, lam * v, atol=tol * max(1.0, abs(lam)))


@pytest.mark.parametrize("n", [2, 5, 12])
def test_power_method(n):
    A = random_spd(n, rng)
    lam, v = call(power_method, A)
    assert np.isclose(lam, np.linalg.eigvalsh(A).max(), rtol=1e-8)
    assert_eigenpair(A, lam, v)


def test_inverse_iteration_smallest_and_shifted():
    A = random_spd(6, rng)
    eig = np.linalg.eigvalsh(A)
    lam, v = call(inverse_iteration, A)
    assert np.isclose(lam, eig.min(), rtol=1e-8)
    assert_eigenpair(A, lam, v)
    target = eig[3] + 0.1 * (eig[4] - eig[3])  # closest eigenvalue is eig[3]
    lam, v = call(inverse_iteration, A, shift=target)
    assert np.isclose(lam, eig[3], rtol=1e-8)


def test_rayleigh_quotient_iteration():
    A = random_spd(6, rng)
    lam, v = call(rayleigh_quotient_iteration, A, rng.normal(size=6))
    assert np.any(np.isclose(lam, np.linalg.eigvalsh(A), rtol=1e-9))
    assert_eigenpair(A, lam, v, tol=1e-9)


@pytest.mark.parametrize("shape", [(4, 4), (8, 5), (30, 12)])
def test_qr_decomposition(shape):
    A = rng.normal(size=shape)
    Q, R = call(qr_decomposition, A)
    n = shape[1]
    assert Q.shape == shape and R.shape == (n, n)
    np.testing.assert_allclose(Q.T @ Q, np.eye(n), atol=1e-10)
    np.testing.assert_allclose(R, np.triu(R), atol=1e-12)
    assert np.all(np.diag(R) >= 0)
    np.testing.assert_allclose(Q @ R, A, atol=1e-10)


@pytest.mark.parametrize("n", [2, 5, 10])
def test_qr_algorithm(n):
    A = random_spd(n, rng)
    vals, vecs = call(qr_algorithm, A)
    np.testing.assert_allclose(vals, np.linalg.eigvalsh(A)[::-1], rtol=1e-8)
    np.testing.assert_allclose(vecs.T @ vecs, np.eye(n), atol=1e-8)
    np.testing.assert_allclose(A @ vecs, vecs * vals, atol=1e-6)


def test_condition_number_spd():
    A = random_spd(6, rng)
    assert np.isclose(call(condition_number_spd, A), np.linalg.cond(A), rtol=1e-6)
    nearly_singular = np.array([[1., 0.999], [0.999, 1.]])
    assert call(condition_number_spd, nearly_singular) > 1000
