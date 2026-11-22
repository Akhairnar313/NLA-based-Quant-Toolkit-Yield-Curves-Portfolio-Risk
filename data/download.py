"""One-off market data download. Run from the repo root:

    python data/download.py

Writes two CSVs that are committed to the repo, so the dashboard and
notebooks run offline and results are reproducible:

    data/prices.csv           daily adjusted close prices for a multi-asset ETF universe (Yahoo Finance)
    data/treasury_yields.csv  daily US Treasury constant-maturity (par) yields, in percent (FRED)
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
START = "2019-01-01"

# Multi-asset universe: equities (US large/small cap, international, EM),
# bonds (long/intermediate Treasuries, IG and HY credit), and real assets.
TICKERS = {
    "SPY": "US large cap equity",
    "QQQ": "US tech/growth equity",
    "IWM": "US small cap equity",
    "EFA": "Developed ex-US equity",
    "EEM": "Emerging market equity",
    "TLT": "20+ year Treasuries",
    "IEF": "7-10 year Treasuries",
    "LQD": "Investment grade corporates",
    "HYG": "High yield corporates",
    "GLD": "Gold",
    "VNQ": "US real estate (REITs)",
    "DBC": "Commodities",
}

# FRED series id -> maturity in years
TREASURY_SERIES = {
    "DGS1MO": 1 / 12,
    "DGS3MO": 0.25,
    "DGS6MO": 0.5,
    "DGS1": 1,
    "DGS2": 2,
    "DGS3": 3,
    "DGS5": 5,
    "DGS7": 7,
    "DGS10": 10,
    "DGS20": 20,
    "DGS30": 30,
}


def download_prices():
    import yfinance as yf

    raw = yf.download(list(TICKERS), start=START, auto_adjust=True, progress=False)
    prices = raw["Close"][list(TICKERS)].dropna(how="all")
    prices.index.name = "date"
    prices.to_csv(DATA_DIR / "prices.csv", float_format="%.4f")
    print(f"prices.csv: {prices.shape[0]} days x {prices.shape[1]} tickers, "
          f"{prices.index.min():%Y-%m-%d} to {prices.index.max():%Y-%m-%d}")


def download_treasury_yields():
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + ",".join(TREASURY_SERIES)
    df = pd.read_csv(url, index_col=0, parse_dates=True, na_values=".")
    df = df.loc[START:, list(TREASURY_SERIES)].dropna(how="any")
    df.columns = [f"{TREASURY_SERIES[c]:.6g}" for c in df.columns]  # column = maturity in years
    df.index.name = "date"
    df.to_csv(DATA_DIR / "treasury_yields.csv")
    print(f"treasury_yields.csv: {df.shape[0]} days x {df.shape[1]} maturities, "
          f"{df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}")


if __name__ == "__main__":
    download_treasury_yields()
    download_prices()
