import numpy as np
import pytest

from nla.spd_checks import gershgorin_discs, is_diagonally_dominant, is_spd_sylvester
from tests.helpers import call, random_spd, random_diag_dominant

rng = np.random.default_rng(3)


@pytest.mark.parametrize("n", [1, 3, 8])
def test_gershgorin_discs(n):
    A = rng.normal(size=(n, n))
    centers, radii = call(gershgorin_discs, A)
    np.testing.assert_allclose(centers, np.diag(A))
    np.testing.assert_allclose(radii, np.abs(A).sum(axis=1) - np.abs(np.diag(A)))
    # Gershgorin's theorem: every eigenvalue lies in some disc.
    for lam in np.linalg.eigvals(A):
        assert np.any(np.abs(lam - centers) <= radii + 1e-10)


def test_is_diagonally_dominant():
    assert call(is_diagonally_dominant, random_diag_dominant(6, rng))
    weak = np.array([[2., -1., -1.], [-1., 2., -1.], [-1., -1., 2.]])
    assert not call(is_diagonally_dominant, weak, strict=True)
    assert call(is_diagonally_dominant, weak, strict=False)
    assert not call(is_diagonally_dominant, np.array([[1., 2.], [0., 1.]]))


def test_is_spd_sylvester():
    assert call(is_spd_sylvester, random_spd(5, rng))
    # Correlation matrix that is NOT positive definite (pairwise valid, jointly impossible).
    bad_corr = np.array([[1., 0.9, -0.9], [0.9, 1., 0.9], [-0.9, 0.9, 1.]])
    assert not call(is_spd_sylvester, bad_corr)
    assert not call(is_spd_sylvester, np.array([[1., 2.], [0., 1.]]))  # not symmetric
    assert not call(is_spd_sylvester, np.array([[1., 0.], [0., -1.]]))
