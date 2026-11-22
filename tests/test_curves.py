import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from finance.curves import (
    cubic_spline,
    efficient_cubic_spline,
    par_bond_cash_flow_matrix,
    bootstrap_discount_factors,
    zero_rates,
    bond_price,
)
from tests.helpers import call

TENORS = np.array([1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
PAR = np.array([4.30, 4.25, 4.15, 4.00, 3.80, 3.75, 3.80, 3.95, 4.15, 4.55, 4.60]) / 100


def test_cubic_spline_matches_natural_spline():
    f = call(cubic_spline, TENORS, PAR)
    t = np.linspace(TENORS[0], TENORS[-1], 500)
    np.testing.assert_allclose(f(TENORS), PAR, atol=1e-12)
    np.testing.assert_allclose(f(t), CubicSpline(TENORS, PAR, bc_type="natural")(t), atol=1e-10)


def test_efficient_cubic_spline_matches_natural_spline():
    f = call(efficient_cubic_spline, TENORS, PAR)
    t = np.linspace(TENORS[0], TENORS[-1], 500)
    np.testing.assert_allclose(f(TENORS), PAR, atol=1e-12)
    np.testing.assert_allclose(f(t), CubicSpline(TENORS, PAR, bc_type="natural")(t), atol=1e-10)


def test_cash_flow_matrix_shape():
    grid = np.arange(1, 5) / 2
    C = call(par_bond_cash_flow_matrix, grid, np.full(4, 0.04))
    expected = np.array([
        [102., 0, 0, 0],
        [2., 102., 0, 0],
        [2., 2., 102., 0],
        [2., 2., 2., 102.],
    ])
    np.testing.assert_allclose(C, expected)


def test_flat_curve_bootstrap():
    # Flat 4% semiannual par curve -> d(t) = 1.02^(-2t), zero rate = 2 ln(1.02).
    grid = np.arange(1, 61) / 2
    C = call(par_bond_cash_flow_matrix, grid, np.full(60, 0.04))
    d = call(bootstrap_discount_factors, C, np.full(60, 100.0))
    np.testing.assert_allclose(d, 1.02 ** (-2 * grid))
    np.testing.assert_allclose(call(zero_rates, grid, d), 2 * np.log(1.02))


def test_bootstrapped_curve_reprices_par_bonds():
    grid = np.arange(1, 61) / 2
    par_grid = CubicSpline(TENORS, PAR, bc_type="natural")(grid)
    C = call(par_bond_cash_flow_matrix, grid, par_grid)
    d = call(bootstrap_discount_factors, C, np.full(60, 100.0))
    np.testing.assert_allclose(C @ d, 100.0)
    z = call(zero_rates, grid, d)
    for k in (4, 20, 60):  # 2y, 10y, 30y par bonds
        price = call(bond_price, par_grid[k - 1], grid[k - 1], grid, z)
        assert price == pytest.approx(100.0, abs=1e-6)
