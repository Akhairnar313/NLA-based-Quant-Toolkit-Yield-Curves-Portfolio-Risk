import numpy as np
import pandas as pd

from finance.stats import covariance_matrix, correlation_matrix, cov_to_corr, corr_to_cov
from finance.data import load_prices, returns
from tests.helpers import call

rng = np.random.default_rng(6)
X = rng.normal(size=(500, 4)) @ rng.normal(size=(4, 4))


def test_covariance_matrix():
    S = call(covariance_matrix, X)
    np.testing.assert_allclose(S, np.cov(X, rowvar=False), rtol=1e-10)
    np.testing.assert_allclose(S, S.T)


def test_covariance_accepts_dataframe():
    df = pd.DataFrame(X, columns=list("abcd"))
    np.testing.assert_allclose(call(covariance_matrix, df), df.cov().to_numpy(), rtol=1e-10)


def test_correlation_matrix():
    R = call(correlation_matrix, X)
    np.testing.assert_allclose(R, np.corrcoef(X, rowvar=False), rtol=1e-10)
    np.testing.assert_allclose(np.diag(R), 1.0)


def test_cov_corr_round_trip():
    S = np.cov(X, rowvar=False)
    corr, vols = call(cov_to_corr, S)
    np.testing.assert_allclose(vols, np.sqrt(np.diag(S)))
    np.testing.assert_allclose(corr, np.corrcoef(X, rowvar=False), rtol=1e-10)
    np.testing.assert_allclose(call(corr_to_cov, corr, vols), S, rtol=1e-10)


def test_real_etf_covariance_is_spd():
    # Sanity check on the cached data: a covariance matrix of real returns is SPD.
    r = returns(load_prices(["SPY", "QQQ", "TLT", "GLD", "DBC"]), kind="log")
    S = call(covariance_matrix, r)
    assert np.all(np.linalg.eigvalsh(S) > 0)
