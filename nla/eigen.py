"""Eigenvalue methods and QR (Session 8).

Used by the risk engine for:
    * "one hidden factor drives X% of the risk": largest eigenvalue of the
      correlation matrix (power method)
    * how fragile the covariance matrix is: condition number of an SPD matrix,
      lambda_max / lambda_min (power method + inverse iteration)
    * PCA of ETF returns and of yield-curve changes: all eigenpairs (QR algorithm)

Iterations stop when the residual ||A v - lam v|| is small relative to |lam|:
that checks the eigenvector as well as the eigenvalue.
"""

import numpy as np

from nla.lu import lu_row_pivot
from nla.triangular_matrix_operations import forward_substitution, backward_substitution


def _norm(v):
    return np.sqrt(v @ v)


def _start_vector(n, x0):
    # A fixed pseudo-random start is almost surely not orthogonal to the wanted
    # eigenvector (an all-ones start would be, e.g. for [[1, a], [a, 1]]).
    x = np.random.default_rng(0).normal(size=n) if x0 is None else np.asarray(x0, dtype=float).copy()
    return x / _norm(x)


def _residual(A, lam, x):
    return _norm(A @ x - lam * x)


def _lu_solver(M):
    """Factor M once (P M = L U) and return a function that solves M y = x."""
    P, L, U = lu_row_pivot(M)
    return lambda x: backward_substitution(U, forward_substitution(L, P @ x))


def power_method(A, x0=None, tol=1e-10, max_iter=10_000):
    """Largest-magnitude eigenvalue of A and its unit eigenvector.

    Repeat x <- A x / ||A x||; the eigenvalue estimate is the Rayleigh quotient
    x^T A x. Stop when ||A x - lam x|| <= tol * max(1, |lam|).

    Returns
    -------
    lam : float
    v : (n,) ndarray with ||v|| = 1
    """
    A = np.asarray(A, dtype=float)
    x = _start_vector(A.shape[0], x0)
    lam = x @ A @ x
    for _ in range(max_iter):
        if _residual(A, lam, x) <= tol * max(1.0, abs(lam)):
            break
        y = A @ x
        ny = _norm(y)
        if ny == 0.0:
            return 0.0, x  # x is in the null space: eigenvalue 0
        x = y / ny
        lam = x @ A @ x
    return float(lam), x


def inverse_iteration(A, shift=0.0, x0=None, tol=1e-10, max_iter=10_000):
    """Eigenvalue of A closest to `shift`, and its unit eigenvector.

    Power method applied to (A - shift I)^{-1}: A - shift I is LU-factored once,
    and each step is one forward and one backward substitution. With shift = 0
    this gives the smallest-magnitude eigenvalue.

    Returns
    -------
    lam : float   (an eigenvalue of A itself, not of the inverse)
    v : (n,) ndarray with ||v|| = 1
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    solve = _lu_solver(A - shift * np.eye(n))
    x = _start_vector(n, x0)
    lam = x @ A @ x
    for _ in range(max_iter):
        if _residual(A, lam, x) <= tol * max(1.0, abs(lam)):
            break
        y = solve(x)
        if not np.all(np.isfinite(y)):
            break  # shift is exactly an eigenvalue: x is already its eigenvector
        x = y / _norm(y)
        lam = x @ A @ x
    return float(lam), x


def rayleigh_quotient_iteration(A, x0, tol=1e-12, max_iter=100):
    """Inverse iteration whose shift is updated every step to the Rayleigh quotient.

    Converges cubically for symmetric A, to the eigenpair "nearest" x0. The price
    is a new LU factorization every step, since the shift changes.

    Returns
    -------
    lam : float
    v : (n,) ndarray with ||v|| = 1
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    x = _start_vector(n, x0)
    lam = x @ A @ x
    for _ in range(max_iter):
        if _residual(A, lam, x) <= tol * max(1.0, abs(lam)):
            break
        with np.errstate(divide="ignore", invalid="ignore"):
            y = _lu_solver(A - lam * np.eye(n))(x)
        if not np.all(np.isfinite(y)) or _norm(y) == 0.0:
            break  # A - lam I is exactly singular: lam is an eigenvalue
        x = y / _norm(y)
        lam = x @ A @ x
    return float(lam), x


def qr_decomposition(A):
    """A = Q R with Q having orthonormal columns and R upper triangular, diag(R) >= 0.

    Householder reflections: step k reflects column k below the diagonal onto a
    multiple of e_1, zeroing it. Stable even when A is close to rank deficient,
    which the shifted QR algorithm relies on.

    Parameters
    ----------
    A : (m, n) ndarray with m >= n

    Returns
    -------
    Q : (m, n) ndarray with orthonormal columns
    R : (n, n) ndarray, upper triangular
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)
    for k in range(0, n):
        x = R[k:, k]
        nx = _norm(x)
        if nx == 0.0:
            continue
        v = x.copy()
        v[0] += (1.0 if x[0] >= 0 else -1.0) * nx   # sign choice avoids cancellation
        v /= _norm(v)
        R[k:, k:] -= 2.0 * np.outer(v, v @ R[k:, k:])   # H R,  H = I - 2 v v^T
        Q[:, k:] -= 2.0 * np.outer(Q[:, k:] @ v, v)     # Q H
        R[k+1:, k] = 0.0                                 # eliminated entries, exactly zero
    Q, R = Q[:, :n], R[:n, :]
    signs = np.where(np.diag(R) < 0, -1.0, 1.0)          # make diag(R) >= 0
    return Q * signs, R * signs[:, None]


def qr_algorithm(A, tol=1e-12, max_iter=10_000):
    """All eigenvalues and eigenvectors of a SYMMETRIC matrix by the QR algorithm.

    A_k - mu I = Q_k R_k,  A_{k+1} = R_k Q_k + mu I  (a similarity transform, so
    the eigenvalues never change). A_k tends to a diagonal matrix of eigenvalues
    and the product Q_0 Q_1 ... Q_k to the eigenvectors.

    Two standard accelerations: the Wilkinson shift mu (the eigenvalue of the
    trailing 2x2 block closer to its last entry), and deflation (once the last
    row of the active block is negligible, its diagonal entry is an eigenvalue
    and the block shrinks by one).

    Returns
    -------
    eigenvalues : (n,) ndarray, sorted in DECREASING order
    eigenvectors : (n, n) ndarray, column j belongs to eigenvalues[j]
    """
    T = np.array(A, dtype=float)
    n = T.shape[0]
    V = np.eye(n)
    scale = np.max(np.abs(T)) if n else 0.0
    hi = n - 1                       # active block is T[:hi+1, :hi+1]
    it = 0
    while hi > 0 and scale > 0:
        if np.max(np.abs(T[hi, :hi])) <= tol * scale:
            hi -= 1                  # deflate: T[hi, hi] is an eigenvalue
            continue
        if it >= max_iter:
            raise RuntimeError("QR algorithm did not converge")
        a, b, c = T[hi-1, hi-1], T[hi-1, hi], T[hi, hi]
        d = (a - c) / 2
        denom = abs(d) + np.sqrt(d*d + b*b)
        mu = c if denom == 0.0 else c - (1.0 if d >= 0 else -1.0) * b**2 / denom
        k = hi + 1
        I = np.eye(k)
        Q, R = qr_decomposition(T[:k, :k] - mu * I)
        T[:k, :k] = R @ Q + mu * I
        T[:k, :k] = (T[:k, :k] + T[:k, :k].T) / 2   # keep it exactly symmetric
        V[:, :k] = V[:, :k] @ Q
        it += 1
    vals = np.diag(T).copy()
    order = np.argsort(vals)[::-1]
    return vals[order], V[:, order]


def condition_number_spd(A):
    """2-norm condition number of an SPD matrix: lambda_max / lambda_min.

    lambda_max from the power method, lambda_min from inverse iteration with
    shift 0. A large value means the matrix is close to singular, so weights
    computed from it (e.g. min-variance) are very sensitive to the data.
    """
    lam_max, _ = power_method(A)
    lam_min, _ = inverse_iteration(A)
    return lam_max / lam_min
