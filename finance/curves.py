"""Yield curve construction and bond pricing.

Pipeline used by the dashboard's Yield Curve page:

    Treasury par yields at 1M ... 30Y           (data/treasury_yields.csv)
        |  cubic_spline            -> par yields on a semiannual grid 0.5, 1.0, ..., 30
        |  par_bond_cash_flow_matrix -> lower triangular cash-flow matrix C
        |  bootstrap_discount_factors -> solve C d = prices  (forward substitution)
        v  zero_rates               -> continuously compounded zero curve

Treat each grid point as a par bond (price 100) paying semiannual coupons
equal to its par yield. This is the standard simplification for bootstrapping
constant-maturity Treasury yields.
"""

import numpy as np

from nla.triangular_matrix_operations import forward_substitution
from nla.lu import lu_solve
from nla.tridiagonal import solve_tridiagonal


def cubic_spline(x, v):
    n = len(x) - 1
    M_bar = np.zeros((4*n, 4*n))
    b_bar = np.zeros(4*n)

    # 2.86
    M_bar[0, 2] = 2
    M_bar[0, 3] = 6*x[0]
    
    # 2.87
    M_bar[4*n-1, 4*n-2] = 2
    M_bar[4*n-1, 4*n-1] = 6*x[n]
    
    for i in range(1, n+1):
        # 2.88
        M_bar[4*i-3, 4*i-4] = 1
        M_bar[4*i-3, 4*i-3] = x[i-1]
        M_bar[4*i-3, 4*i-2] = x[i-1]**2
        M_bar[4*i-3, 4*i-1] = x[i-1]**3
        b_bar[4*i-3] = v[i-1]
        
        # 2.90
        M_bar[4*i-2, 4*i-4] = 1
        M_bar[4*i-2, 4*i-3] = x[i]
        M_bar[4*i-2, 4*i-2] = x[i]**2
        M_bar[4*i-2, 4*i-1] = x[i]**3
        b_bar[4*i-2] = v[i]
    
        if i <=n-1:
            # 2.92
            M_bar[4*i-1, 4*i-3] = 1
            M_bar[4*i-1, 4*i-2] = 2*x[i]
            M_bar[4*i-1, 4*i-1] = 3*x[i]**2

            # 2.93
            M_bar[4*i-1, 4*i+1] = -1
            M_bar[4*i-1, 4*i+2] = -2*x[i]
            M_bar[4*i-1, 4*i+3] = -3*x[i]**2

            # 2.95
            M_bar[4*i, 4*i-2] = 2
            M_bar[4*i, 4*i-1] = 6*x[i]

            # 2.96
            M_bar[4*i, 4*i+2] = -2
            M_bar[4*i, 4*i+3] = -6*x[i]

    x_bar = lu_solve(M_bar, b_bar)

    a = x_bar[0::4]
    b = x_bar[1::4]
    c = x_bar[2::4]
    d = x_bar[3::4]

    def f(t):
        t = np.asarray(t, dtype=float)
        # interval index of each t: x[i] <= t <= x[i+1] (the last node uses the last interval)
        i = np.clip(np.searchsorted(x, t, side="right") - 1, 0, n - 1)
        return a[i] + b[i]*t + c[i]*t**2 + d[i]*t**3

    return f


def efficient_cubic_spline(x, v):
    # To be implemented
    raise NotImplementedError("TODO: natural cubic spline via tridiagonal solve (Sessions 3, 6)")


def par_bond_cash_flow_matrix(maturities, par_yields, freq=2, face=100.0):
    """Cash-flow matrix of par bonds maturing on the payment grid. Gives a lower triangular matrix"""
    n = maturities.shape[0]
    C = np.zeros((n, n))
    for i in range(0, n):
        coupon = face * par_yields[i] / freq
        for j in range(0, i+1):  # coupons on every date up to and including maturity
            C[i, j] = coupon
        C[i, i] += face  # principal repaid at maturity
    return C


def bootstrap_discount_factors(C, prices):
    """Discount factors d with C d = prices, by forward substitution"""
    return forward_substitution(C, prices)


def zero_rates(t, discount_factors):
    """Continuously compounded zero rates: r(t) = -ln(d(t)) / t."""
    t = np.asarray(t, dtype=float)
    return -np.log(np.asarray(discount_factors, dtype=float)) / t


def bond_price(coupon_rate, maturity, curve_t, curve_r, freq=2, face=100.0):
    """Price a fixed-coupon bond off a zero curve.

    Cash flows at maturity - k/freq (k = 0, 1, ...) that are > 0; discount each
    with exp(-r(t) t), where r(t) is obtained from the zero curve
    (curve_t, curve_r) by cubic_spline.

    Returns
    -------
    price : float
    """
    # Payment times counting back from maturity: maturity, maturity - 1/freq, ... (> 0).
    n_pay = int(np.floor(maturity * freq + 1e-9))
    times = maturity - np.arange(n_pay) / freq
    times = times[::-1]  # ascending order
    # Zero rate at each payment time from a cubic spline of the zero curve.
    # Times outside the curve's range use the nearest end rate (flat extrapolation).
    curve_t = np.asarray(curve_t, dtype=float)
    r = cubic_spline(curve_t, np.asarray(curve_r, dtype=float))
    rates = r(np.clip(times, curve_t[0], curve_t[-1]))
    cash_flows = np.full(n_pay, face * coupon_rate / freq)
    cash_flows[-1] += face  # principal at maturity
    return float(np.sum(cash_flows * np.exp(-rates * times)))
