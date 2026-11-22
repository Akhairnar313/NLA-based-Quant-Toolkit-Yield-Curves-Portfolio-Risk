"""Loaders for the cached market data in data/*.csv (see data/download.py)."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_treasury_yields():
    """Daily US Treasury par yields in percent.

    Returns a DataFrame indexed by date; columns are maturities in years (floats).
    """
    df = pd.read_csv(DATA_DIR / "treasury_yields.csv", index_col="date", parse_dates=True)
    df.columns = df.columns.astype(float)
    return df


def load_prices(tickers=None):
    """Daily adjusted close prices, indexed by date, one column per ticker."""
    df = pd.read_csv(DATA_DIR / "prices.csv", index_col="date", parse_dates=True)
    if tickers is not None:
        df = df[list(tickers)]
    return df.dropna()


def returns(prices, kind="simple"):
    """Period returns from a price DataFrame: 'simple' (P_t / P_{t-1} - 1) or 'log'."""
    if kind == "simple":
        r = prices / prices.shift(1) - 1
    elif kind == "log":
        r = np.log(prices / prices.shift(1))
    else:
        raise ValueError("kind must be 'simple' or 'log'")
    return r.dropna()
