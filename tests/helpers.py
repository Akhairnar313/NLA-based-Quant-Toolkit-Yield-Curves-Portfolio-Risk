"""Shared test utilities.

`call` runs a routine and turns a NotImplementedError into a pytest skip,
so the suite stays green while stubs are being filled in and the skip
list doubles as a to-do list (`pytest -rs`).
"""

import numpy as np
import pytest


def call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as exc:
        pytest.skip(f"not implemented yet: {fn.__module__}.{fn.__name__} ({exc})")


def random_diag_dominant(n, rng):
    """Strictly row diagonally dominant matrix: LU without pivoting exists."""
    A = rng.uniform(-1, 1, size=(n, n))
    A[np.diag_indices(n)] = np.abs(A).sum(axis=1) + 1.0
    return A


def random_tridiagonal(n, rng, dominant=True):
    sub = rng.uniform(-1, 1, n - 1)
    sup = rng.uniform(-1, 1, n - 1)
    diag = rng.uniform(-1, 1, n)
    if dominant:
        diag = np.abs(np.r_[0, sub]) + np.abs(np.r_[sup, 0]) + 1.0
    return np.diag(diag) + np.diag(sub, -1) + np.diag(sup, 1)


def random_spd(n, rng):
    M = rng.normal(size=(n, n))
    return M @ M.T + n * np.eye(n)
