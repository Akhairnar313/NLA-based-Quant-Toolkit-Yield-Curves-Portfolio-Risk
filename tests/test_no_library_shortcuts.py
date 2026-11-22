"""Guard rails: the project's linear algebra comes from the hand-written `nla` package only.

1. Static: no file in nla/, finance/ or app/ mentions a library linear algebra or
   statistics routine (NumPy's linalg, cov, corrcoef, polyfit; pandas .cov()/.corr();
   SciPy, scikit-learn, statsmodels).
2. Runtime: every financial pipeline still runs when all of those are booby-trapped
   to raise, so nothing reaches them indirectly either.

NumPy remains in use for arrays, elementwise math, the `@` product, sorting and
random numbers, which is the arithmetic the course algorithms are built from.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = re.compile(
    r"\blinalg\b|\bscipy\b|\bsklearn\b|\bstatsmodels\b"
    r"|\bnp\.(cov|corrcoef|polyfit|polyval|lstsq)\b|\.cov\(|\.corr\("
)
FILES = sorted(p for pkg in ("nla", "finance", "app") for p in (ROOT / pkg).glob("*.py"))


@pytest.mark.parametrize("path", FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_no_library_linear_algebra(path):
    hits = [
        f"{i}: {line.strip()}"
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if FORBIDDEN.search(line)
    ]
    assert not hits, f"library linear algebra used in {path.name}:\n" + "\n".join(hits)


@pytest.fixture
def library_blocked(monkeypatch):
    """Make every numpy.linalg function, np.cov/corrcoef/polyfit and pandas cov/corr raise,
    and make SciPy, scikit-learn and statsmodels unimportable."""
    def trap(name):
        def f(*args, **kwargs):
            raise AssertionError(f"library routine called: {name}")
        return f
    for name in dir(np.linalg):
        obj = getattr(np.linalg, name)
        if not name.startswith("_") and callable(obj) and not isinstance(obj, type):
            monkeypatch.setattr(np.linalg, name, trap(f"numpy.linalg.{name}"))
    for name in ("cov", "corrcoef", "polyfit"):
        monkeypatch.setattr(np, name, trap(f"numpy.{name}"))
    monkeypatch.setattr(pd.DataFrame, "cov", trap("DataFrame.cov"))
    monkeypatch.setattr(pd.DataFrame, "corr", trap("DataFrame.corr"))
    for mod in ("scipy", "sklearn", "statsmodels"):
        monkeypatch.setitem(sys.modules, mod, None)


def test_pipelines_run_on_nla_only(library_blocked):
    from finance import curves, pca, portfolio, risk, simulation, stats
    from finance.data import load_prices, load_treasury_yields, returns
    from nla.cholesky import is_spd
    from nla.eigen import condition_number_spd, rayleigh_quotient_iteration
    from nla.spd_checks import is_spd_sylvester

    # Yield Curve Builder: spline -> cash-flow matrix -> bootstrap -> zero rates
    y = load_treasury_yields().iloc[-1]
    grid = np.arange(1, 61) / 2
    par_grid = curves.cubic_spline(y.index.to_numpy(float), y.to_numpy(float) / 100)(grid)
    C = curves.par_bond_cash_flow_matrix(grid, par_grid)
    d = curves.bootstrap_discount_factors(C, np.full(len(grid), 100.0))
    assert np.abs(C @ d - 100).max() < 1e-9
    assert np.all(curves.zero_rates(grid, d) > 0)

    # Risk engine on all 12 ETFs
    r = returns(load_prices(), kind="log")
    n = r.shape[1]
    w = np.full(n, 1 / n)
    cov_d = stats.covariance_matrix(r)
    cov, mu = 252 * cov_d, 252 * r.mean().to_numpy()
    corr, _ = stats.cov_to_corr(cov)
    assert is_spd(cov_d) and is_spd_sylvester(cov_d)
    assert np.isclose(risk.risk_contributions(w, cov).sum(), 1.0)
    risk.portfolio_volatility(w, cov)
    risk.diversification_ratio(w, cov)
    sims = simulation.simulate_portfolio_returns(w, cov_d, 63, 20_000)
    var, cvar = risk.monte_carlo_var_cvar(sims, 0.99)
    assert cvar >= var > 0
    risk.parametric_var_cvar(w, cov_d, 0.99, 63)
    for wts in (portfolio.min_variance_portfolio(cov), portfolio.min_variance_long_only(cov),
                portfolio.tangency_portfolio(mu, cov, 0.04)):
        assert np.isclose(wts.sum(), 1.0)
    portfolio.efficient_frontier(mu, cov, np.linspace(0.0, 0.1, 10))
    _, _, explained = pca.pca(corr)
    assert np.isclose(explained.sum(), 1.0)
    pca.dominant_factor_share(corr)
    assert condition_number_spd(cov) > 1
    rayleigh_quotient_iteration(corr, np.ones(n))
    pca.pca(stats.covariance_matrix(load_treasury_yields().dropna().diff().dropna()))
