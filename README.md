# Numerical Linear Algebra for Financial Applications

**Live demo: https://numerical-linear-algebra-financial-applications.streamlit.app/**

Financial risk tools (yield curves, portfolio risk, efficient frontier, VaR, PCA) built on a linear algebra
library written from scratch. The library in `nla/` implements the algorithms from Dan Stefanica's Pre-MFE
seminar *Numerical Linear Algebra for Financial Engineering*; the financial applications in `finance/` and
the Streamlit dashboard in `app/` run entirely on top of it, using real market data.

No `numpy.linalg`, SciPy, scikit-learn or statsmodels is used in `nla/`, `finance/` or `app/` (NumPy serves
only as an array container). `tests/test_no_library_shortcuts.py` enforces this. SciPy and NumPy appear
only in `tests/`, as reference answers.

## Dashboard

| Page | What it shows | Linear algebra used |
|---|---|---|
| Portfolio Risk Advisor | Risk today, tail risk (VaR/CVaR), a minimum-variance rebalance, a plain-English verdict | Cholesky, power method |
| Yield Curve | Bootstraps discount factors and zero rates from Treasury par yields, fits a cubic spline, prices a bond | LU, substitution |
| Covariance & Correlation | SPD checks, condition number, real vs simulated returns | Cholesky, Sylvester, eigen |
| Efficient Frontier | Min-variance, long-only and tangency portfolios, capital market line | LU, Cholesky |
| Value at Risk | Normal vs Monte Carlo vs historical VaR/CVaR | Cholesky |
| PCA | Factors of ETF returns; level/slope/curvature of the Treasury curve | QR algorithm |

## Structure

```
nla/        LU, tridiagonal, Cholesky, SPD checks, eigen methods, QR
finance/    yield curves, covariance, simulation, risk, portfolios, PCA
app/        Streamlit dashboard
data/       cached Treasury yields (FRED) and ETF prices (Yahoo Finance)
tests/      pytest suite
```

## Setup

Requires Python 3.13.

```bash
git clone https://github.com/Akhairnar313/Numerical-Linear-Algebra-Financial-Applications.git
cd Numerical-Linear-Algebra-Financial-Applications
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Run the tests with `pytest -rs`. Refresh the cached data with `python data/download.py`.

## Collaborators

- Atharva Khairnar
- Urshita Koshti

Parts of the project (tests, dashboard, some routines) were written with Claude (Anthropic).
