"""Dashboard for the NLA-for-finance project.

    streamlit run app/streamlit_app.py

Every number on these pages is computed with the hand-written `nla`
package. Steps whose routines are still stubs show what to implement next.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from finance import curves, pca, portfolio, risk, simulation, stats
from finance.data import load_prices, load_treasury_yields, returns
from nla.cholesky import is_spd
from nla.eigen import condition_number_spd
from nla.spd_checks import is_spd_sylvester

st.set_page_config(page_title="NLA for Finance", layout="wide")

ETF_NAMES = {
    "SPY": "US large caps (S&P 500)", "QQQ": "US tech (Nasdaq 100)", "IWM": "US small caps",
    "EFA": "Developed ex-US stocks", "EEM": "Emerging-market stocks", "TLT": "20+ year Treasuries",
    "IEF": "7–10 year Treasuries", "LQD": "Investment-grade corporate bonds", "HYG": "High-yield corporate bonds",
    "GLD": "Gold", "VNQ": "US real estate (REITs)", "DBC": "Commodities",
}
DEFAULT_HOLDINGS = {"SPY": 40_000, "QQQ": 25_000, "TLT": 15_000, "GLD": 10_000, "DBC": 10_000}
N_SIMS = 200_000


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def usd(x):
    return f"${x:,.0f}"


def todo(fn, session):
    st.info(
        f"**Next step:** implement `{fn.__module__}.{fn.__name__}` ({session}) "
        "to unlock the rest of this page.",
        icon="🛠️",
    )


def try_step(fn, session, *args, **kwargs):
    """Run a pipeline step; on NotImplementedError show a to-do and stop the page."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        todo(fn, session)
        st.stop()


@st.cache_data
def treasury_yields():
    return load_treasury_yields()


def yield_curve_page():
    st.title("Yield Curve Builder")
    st.caption(
        "US Treasury par yields → natural cubic spline (tridiagonal solver) → "
        "par bond cash-flow matrix → discount factors by forward substitution → zero curve."
    )

    ylds = treasury_yields()
    dates = ylds.index
    date = st.sidebar.select_slider(
        "Curve date", options=list(dates), value=dates[-1], format_func=lambda d: d.strftime("%Y-%m-%d")
    )
    tenors = ylds.columns.to_numpy(dtype=float)
    par = ylds.loc[date].to_numpy(dtype=float) / 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tenors, y=par * 100, mode="markers", name="Treasury par yields (market)",
                             marker=dict(size=9)))
    fig.update_layout(xaxis_title="Maturity (years)", yaxis_title="Yield (%)", height=460,
                      legend=dict(orientation="h", y=1.08), margin=dict(t=40))
    chart = st.empty()
    chart.plotly_chart(fig, width="stretch")

    # Step 1: cubic spline of par yields onto a semiannual grid.
    grid = np.arange(1, 61) / 2
    spline = try_step(curves.cubic_spline, "Sessions 3 & 6", tenors, par)
    par_grid = spline(grid)
    t_fine = np.linspace(tenors[0], tenors[-1], 400)
    fig.add_trace(go.Scatter(x=t_fine, y=spline(t_fine) * 100, name="Par curve (cubic spline)"))
    chart.plotly_chart(fig, width="stretch")

    # Step 2: bootstrap discount factors.
    C = try_step(curves.par_bond_cash_flow_matrix, "Session 2", grid, par_grid)
    d = try_step(curves.bootstrap_discount_factors, "Session 2", C, np.full(len(grid), 100.0))
    z = try_step(curves.zero_rates, "Session 2", grid, d)
    fig.add_trace(go.Scatter(x=grid, y=z * 100, name="Zero curve (cont. comp.)", line=dict(dash="dash")))
    chart.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Discount factors")
        dfig = go.Figure(go.Scatter(x=grid, y=d, mode="lines+markers", marker=dict(size=4)))
        dfig.update_layout(xaxis_title="Maturity (years)", yaxis_title="d(t)", height=360, margin=dict(t=20))
        st.plotly_chart(dfig, width="stretch")
        st.caption(f"Max repricing error of the {len(grid)} par bonds: {np.abs(C @ d - 100).max():.2e}")
    with col2:
        st.subheader("Bond pricer")
        coupon = st.number_input("Coupon rate (%)", 0.0, 20.0, 4.0, 0.125)
        maturity = st.slider("Maturity (years)", 0.5, 30.0, 10.0, 0.5)
        price = try_step(curves.bond_price, "Session 2", coupon / 100, maturity, grid, z)
        st.metric("Price per 100 face", f"{price:,.4f}")


# ----------------------------------------------------------------------------
# Shared helpers for the risk pages
# ----------------------------------------------------------------------------
@st.cache_data
def all_prices():
    return load_prices()


def ticker_picker(key, default):
    tickers = st.sidebar.multiselect("ETFs", list(ETF_NAMES), default=default, key=key,
                                     format_func=lambda t: f"{t} · {ETF_NAMES[t]}")
    if len(tickers) < 2:
        st.warning("Pick at least two ETFs in the sidebar.")
        st.stop()
    return tickers


def lookback_returns(tickers, years):
    """Daily log returns of the chosen ETFs over the last `years` years."""
    px = all_prices()[tickers].iloc[-(int(252 * years) + 1):]
    return returns(px, kind="log")


def latest_rf():
    """Latest 3-month Treasury yield (decimal) and its date, from the cached FRED data."""
    y = treasury_yields()[0.25].dropna()
    return float(y.iloc[-1]) / 100, y.index[-1]


def heatmap(M, labels, title, zmin=-1, zmax=1, fmt=".2f"):
    fig = go.Figure(go.Heatmap(z=M, x=labels, y=labels, zmin=zmin, zmax=zmax, colorscale="RdBu_r",
                               text=np.vectorize(lambda v: format(v, fmt))(M), texttemplate="%{text}"))
    fig.update_layout(title=title, height=420, margin=dict(t=40), yaxis_autorange="reversed")
    return fig


def covariance_inputs(r):
    """Annualized covariance and mean returns, with the Cholesky validity check."""
    cov_d = try_step(stats.covariance_matrix, "Sessions 1 & 6", r)
    if not try_step(is_spd, "Session 6", cov_d):
        st.error("The covariance matrix is not positive definite (Cholesky failed). "
                 "Use more history or fewer, less similar ETFs.")
        st.stop()
    return cov_d, cov_d * 252, r.mean().to_numpy() * 252


# ----------------------------------------------------------------------------
# Portfolio Risk Advisor (Urshita's advisor, without the option pricer)
# ----------------------------------------------------------------------------
def advisor_page():
    st.title("Portfolio Risk Advisor")
    st.caption("I hold this portfolio: how risky is it, where does the risk come from, how bad can the next few "
               "months get, and would a different mix of the same ETFs be safer? Built from the risk engine "
               "Urshita prototyped, running on the hand-written `nla` library.")

    sb = st.sidebar
    sb.subheader("Your holdings")
    tickers = ticker_picker("adv_tickers", list(DEFAULT_HOLDINGS))
    dollars = np.array([sb.number_input(f"{t} ($)", 0, 10_000_000, DEFAULT_HOLDINGS.get(t, 10_000), 1_000,
                                        key=f"adv_{t}") for t in tickers], dtype=float)
    if dollars.sum() <= 0:
        st.warning("Enter a positive amount for at least one holding.")
        st.stop()
    sb.subheader("Settings")
    years = sb.slider("History used (years)", 1, 7, 3)
    horizon = sb.select_slider("Horizon", options=[21, 63, 126, 252], value=63,
                               format_func=lambda d: f"{d} trading days (~{d // 21} months)")
    conf = sb.select_slider("Confidence", options=[0.90, 0.95, 0.99], value=0.99, format_func=lambda c: f"{c:.0%}")
    tol = sb.slider("Loss you can tolerate over the horizon (%)", 1, 40, 10) / 100

    total = dollars.sum()
    w = dollars / total
    r = lookback_returns(tickers, years)
    cov_d, cov, mu = covariance_inputs(r)
    months = f"{horizon // 21} months"
    tail = f"1-in-{round(1 / (1 - conf))}"

    # 1. Risk today ---------------------------------------------------------
    st.header("1 · How risky is the portfolio today?")
    corr, _ = try_step(stats.cov_to_corr, "Session 1", cov)
    vol = try_step(risk.portfolio_volatility, "Session 8", w, cov)
    rc = try_step(risk.risk_contributions, "Session 8", w, cov)
    div = try_step(risk.diversification_ratio, "Session 8", w, cov)
    factor = try_step(pca.dominant_factor_share, "Session 8", corr)
    cond = try_step(condition_number_spd, "Session 8", cov)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annual volatility", pct(vol), help="sqrt(wᵀΣw): the typical size of a year's swing")
    c2.metric("Diversification ratio", f"{div:.2f}", help="1.0 = no diversification benefit; higher is better")
    c3.metric("One factor explains", pct(factor, 0), help="Largest eigenvalue of the correlation matrix / n (power method)")
    c4.metric("Covariance condition number", f"{cond:,.0f}", help="λmax/λmin (power method + inverse iteration)")

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure([go.Bar(x=tickers, y=100 * w, name="% of money"),
                         go.Bar(x=tickers, y=100 * rc, name="% of risk")])
        fig.update_layout(title="Money vs risk contribution", yaxis_title="%", height=420, margin=dict(t=40),
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")
        st.caption("Risk share of holding i = wᵢ(Σw)ᵢ / wᵀΣw. Volatile holdings that move with everything else "
                   "carry more risk than money.")
    with col2:
        st.plotly_chart(heatmap(corr, tickers, "Correlation matrix"), width="stretch")

    # 2. Tail risk ----------------------------------------------------------
    st.header(f"2 · How bad can the next {months} get?")
    sims = try_step(simulation.simulate_portfolio_returns, "Session 6", w, cov_d, horizon, N_SIMS)
    var, cvar = try_step(risk.monte_carlo_var_cvar, "Sessions 7 & 8", sims, conf)
    var_n, _ = try_step(risk.parametric_var_cvar, "Sessions 7 & 8", w, cov_d, conf, horizon)
    p_breach = float((sims < -tol).mean())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{conf:.0%} VaR (Monte Carlo)", pct(var), f"{usd(var * total)}", delta_color="off",
              help=f"In the worst {pct(1 - conf, 0)} of scenarios you lose at least this much")
    c2.metric("CVaR (average of those losses)", pct(cvar), f"{usd(cvar * total)}", delta_color="off")
    c3.metric("VaR from the normal formula", pct(var_n), help="z·σ·√h, for comparison with the simulation")
    c4.metric(f"Chance of losing more than {pct(tol, 0)}", pct(p_breach))
    fig = go.Figure(go.Histogram(x=100 * sims, nbinsx=150, marker_color="steelblue", name="scenarios"))
    fig.add_vline(x=-100 * var, line_color="red", annotation_text=f"VaR −{pct(var)}")
    fig.add_vline(x=-100 * tol, line_color="green", line_dash="dash", annotation_text=f"tolerance −{pct(tol, 0)}",
                  annotation_position="top left")
    fig.update_layout(title=f"{N_SIMS:,} correlated Monte Carlo scenarios (Cholesky factor of the covariance)",
                      xaxis_title=f"Portfolio return over {months} (%)", height=400, margin=dict(t=40),
                      showlegend=False)
    st.plotly_chart(fig, width="stretch")

    # 3. Rebalance ----------------------------------------------------------
    st.header("3 · Alternative: rebalance to the minimum-variance mix")
    w_mv = try_step(portfolio.min_variance_long_only, "Sessions 7 & 8", cov)
    vol_mv = risk.portfolio_volatility(w_mv, cov)
    var_mv, _ = risk.monte_carlo_var_cvar(simulation.simulate_portfolio_returns(w_mv, cov_d, horizon, N_SIMS), conf)
    ret_gap = float(w @ mu - w_mv @ mu)
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = go.Figure([go.Bar(x=tickers, y=100 * w, name="current"),
                         go.Bar(x=tickers, y=100 * w_mv, name="min-variance (no shorting)")])
        fig.update_layout(title="Weights (%)", height=380, margin=dict(t=40), legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width="stretch")
    with col2:
        st.metric("Volatility", pct(vol_mv), f"{100 * (vol_mv - vol):+.1f} pts", delta_color="inverse")
        st.metric(f"{conf:.0%} VaR", pct(var_mv), f"{100 * (var_mv - var):+.1f} pts", delta_color="inverse")
        st.metric("Past average return vs current", pct(-ret_gap), delta_color="off",
                  help="Difference in average annual return over the history used: a noisy estimate, not a forecast")
    st.caption("Lowest-variance weights: solve Σx = 1 with Cholesky; drop any asset that would be shorted and solve again.")

    # 4. Verdict ------------------------------------------------------------
    st.header("4 · What this tells you")
    notes = []
    top = int(np.argmax(rc))
    if rc[top] > w[top] + 0.15:
        notes.append(f"**{tickers[top]}** is {pct(w[top], 0)} of your money but **{pct(rc[top], 0)} of your risk**: "
                     "it dominates how the portfolio behaves.")
    if factor > 0.5:
        notes.append(f"One common factor drives {pct(factor, 0)} of all the movement, so your holdings are less "
                     "diversified than they look and tend to fall together in a sell-off.")
    if var <= tol:
        notes.append(f"Even the {tail} scenario (−{pct(var)}) stays inside your {pct(tol, 0)} tolerance. "
                     "**No change is needed by this measure.**")
    else:
        cut = (var - var_mv) / var
        notes.append(f"The {tail} loss over {months} (−{pct(var)}, {usd(var * total)}) **exceeds your "
                     f"{pct(tol, 0)} tolerance**.")
        notes.append(f"Rebalancing to the minimum-variance mix cuts that loss by {pct(cut, 0)} to −{pct(var_mv)}, "
                     f"but changes what you own (and its average return was {pct(abs(ret_gap))} a year "
                     f"{'lower' if ret_gap > 0 else 'higher'} over this history).")
        if var_mv <= tol:
            notes.append("**Moving part of the way toward the minimum-variance weights would bring the tail loss "
                         "inside your tolerance.**")
        else:
            notes.append("**Even the lowest-risk mix of these ETFs exceeds your tolerance:** you'd need safer "
                         "assets (short-term Treasuries or cash) or a higher tolerance.")
    st.markdown("\n".join(f"- {n}" for n in notes))
    st.caption("Assumptions: risk estimated from past daily returns; normal shocks understate fat tails; no "
               "transaction costs or taxes. Educational analysis, not investment advice.")


# ----------------------------------------------------------------------------
# Covariance & Correlation
# ----------------------------------------------------------------------------
def covariance_page():
    st.title("Covariance & Correlation")
    st.caption("Sample covariance Σ = Xcᵀ Xc / (T − 1) from daily log returns, its validity checks, and correlated "
               "Monte Carlo samples generated with the Cholesky factor L (Σ = L Lᵀ).")
    tickers = ticker_picker("cov_tickers", ["SPY", "QQQ", "EFA", "TLT", "LQD", "GLD"])
    years = st.sidebar.slider("History used (years)", 1, 7, 3, key="cov_years")
    r = lookback_returns(tickers, years)
    cov_d, cov, _ = covariance_inputs(r)
    corr, vols = stats.cov_to_corr(cov)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SPD by Cholesky", "yes" if is_spd(cov_d) else "no", help="one n³/3 factorization")
    c2.metric("SPD by Sylvester", "yes" if try_step(is_spd_sylvester, "Session 5", cov_d) else "no",
              help="every leading principal minor > 0 (n determinants)")
    c3.metric("Condition number", f"{condition_number_spd(cov):,.0f}")
    c4.metric("Trading days", f"{len(r):,}")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(heatmap(corr, tickers, "Correlation matrix"), width="stretch")
    with col2:
        fig = go.Figure(go.Bar(x=tickers, y=100 * vols))
        fig.update_layout(title="Annual volatility (√ of the diagonal of Σ)", yaxis_title="%", height=420,
                          margin=dict(t=40))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Correlated Monte Carlo vs real returns")
    a, b = st.columns(2)
    x_t = a.selectbox("x-axis ETF", tickers, index=0)
    y_t = b.selectbox("y-axis ETF", tickers, index=min(1, len(tickers) - 1))
    i, j = tickers.index(x_t), tickers.index(y_t)
    n = min(len(r), 2000)
    sample = try_step(simulation.correlated_normal_samples, "Session 6", cov_d, n,
                      mean=r.mean().to_numpy(), seed=0)
    fig = go.Figure([
        go.Scatter(x=100 * r.iloc[-n:, i], y=100 * r.iloc[-n:, j], mode="markers", name="real daily returns",
                   marker=dict(size=4, opacity=0.5)),
        go.Scatter(x=100 * sample[:, i], y=100 * sample[:, j], mode="markers", name="simulated (L z)",
                   marker=dict(size=4, opacity=0.5)),
    ])
    fig.update_layout(xaxis_title=f"{x_t} daily return (%)", yaxis_title=f"{y_t} daily return (%)", height=480,
                      margin=dict(t=20), legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")
    sim_corr = stats.correlation_matrix(sample)[i, j]
    st.caption(f"Correlation {x_t}/{y_t}: real {corr[i, j]:.2f}, simulated {sim_corr:.2f}. The simulation matches "
               "the covariance by construction; the real cloud has fatter tails than the normal one.")


# ----------------------------------------------------------------------------
# Efficient Frontier
# ----------------------------------------------------------------------------
def frontier_page():
    st.title("Efficient Frontier")
    st.caption("Lowest-risk portfolio for every target return (a Lagrange system solved with two Cholesky solves), "
               "the minimum-variance portfolio, and the tangency portfolio with the best return per unit of risk.")
    tickers = ticker_picker("ef_tickers", ["SPY", "EFA", "EEM", "TLT", "LQD", "GLD"])
    years = st.sidebar.slider("History used (years)", 1, 7, 5, key="ef_years")
    rf_mkt, rf_date = latest_rf()
    rf = st.sidebar.slider("Risk-free rate (%)", 0.0, 8.0, round(100 * rf_mkt, 2), 0.05,
                           help=f"Default: 3-month Treasury yield on {rf_date:%Y-%m-%d}") / 100
    r = lookback_returns(tickers, years)
    _, cov, mu = covariance_inputs(r)
    vols = np.sqrt(np.diag(cov))

    w_mv = try_step(portfolio.min_variance_portfolio, "Sessions 7 & 8", cov)
    w_lo = try_step(portfolio.min_variance_long_only, "Sessions 7 & 8", cov)
    w_tan = try_step(portfolio.tangency_portfolio, "Session 7", mu, cov, rf)
    mu_mv, mu_tan = w_mv @ mu, w_tan @ mu
    hi = max(mu.max(), mu_mv) + 0.04
    targets = np.linspace(mu_mv - (hi - mu_mv), hi, 120)
    f_vol, _ = try_step(portfolio.efficient_frontier, "Sessions 7 & 8", mu, cov, targets)
    vol_of = lambda v: risk.portfolio_volatility(v, cov)

    fig = go.Figure()
    upper = targets >= mu_mv
    fig.add_trace(go.Scatter(x=100 * f_vol[upper], y=100 * targets[upper], name="efficient frontier"))
    fig.add_trace(go.Scatter(x=100 * f_vol[~upper], y=100 * targets[~upper], name="inefficient half",
                             line=dict(dash="dot", color="gray")))
    fig.add_trace(go.Scatter(x=100 * vols, y=100 * mu, mode="markers+text", text=tickers, textposition="top center",
                             name="single ETFs", marker=dict(size=9)))
    fig.add_trace(go.Scatter(x=[100 * vol_of(w_mv)], y=[100 * mu_mv], mode="markers", name="min-variance",
                             marker=dict(size=14, symbol="star")))
    fig.add_trace(go.Scatter(x=[100 * vol_of(w_lo)], y=[100 * (w_lo @ mu)], mode="markers",
                             name="min-variance, no shorting", marker=dict(size=12, symbol="diamond")))
    has_tangency = mu_mv > rf    # otherwise the formula lands on the inefficient (lower) half
    if has_tangency:
        s_tan = vol_of(w_tan)
        x_end = max(f_vol[upper].max(), s_tan)
        fig.add_trace(go.Scatter(x=[0, 100 * x_end], y=[100 * rf, 100 * (rf + (mu_tan - rf) / s_tan * x_end)],
                                 name="capital market line", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=[100 * s_tan], y=[100 * mu_tan], mode="markers", name="tangency",
                                 marker=dict(size=14, symbol="star")))
    fig.update_layout(xaxis_title="Annual volatility (%)", yaxis_title="Average annual return (%)", height=520,
                      margin=dict(t=20), xaxis_range=[0, 100 * max(f_vol[upper].max(), vols.max()) * 1.05])
    st.plotly_chart(fig, width="stretch")

    cols = {"min-variance": w_mv, "min-variance, no shorting": w_lo}
    if has_tangency:
        cols["tangency"] = w_tan
    st.dataframe(pd.DataFrame(cols, index=tickers).style.format("{:.1%}"), width="stretch")
    if not has_tangency:
        st.info(f"No tangency portfolio: the risk-free rate ({pct(rf, 2)}) is above the return of the "
                f"minimum-variance portfolio ({pct(mu_mv, 2)}). Over this history, T-bills beat the lowest-risk "
                "mix of these ETFs, so no risky portfolio improves on them. Try a lower rate or more history.")
    elif np.abs(w_tan).max() > 1.0:
        st.warning("The tangency weights are extreme (leverage and large short positions). It divides by small "
                   "differences in *average past returns*, which are very noisy estimates, so treat it as an "
                   "illustration of the math rather than a portfolio to hold.")


# ----------------------------------------------------------------------------
# Value at Risk
# ----------------------------------------------------------------------------
def var_page():
    st.title("Value at Risk")
    st.caption("VaR: the loss exceeded only (1 − confidence) of the time. CVaR: the average loss in those cases. "
               "Three estimates for an equally weighted portfolio of the chosen ETFs.")
    tickers = ticker_picker("var_tickers", ["SPY", "QQQ", "TLT", "GLD", "DBC"])
    years = st.sidebar.slider("History used (years)", 1, 7, 7, key="var_years")
    horizon = st.sidebar.select_slider("Horizon (trading days)", options=[1, 5, 21, 63], value=21, key="var_h")
    w = np.full(len(tickers), 1 / len(tickers))
    r = lookback_returns(tickers, years)
    cov_d, _, _ = covariance_inputs(r)

    sims = simulation.simulate_portfolio_returns(w, cov_d, horizon, N_SIMS)
    # Historical: real overlapping h-day returns of the same portfolio.
    hist = np.exp(r.rolling(horizon).sum().dropna().to_numpy()) - 1.0
    hist = hist @ w
    levels = np.array([0.90, 0.925, 0.95, 0.975, 0.99, 0.995])
    rows = []
    for c in levels:
        rows.append([*try_step(risk.parametric_var_cvar, "Sessions 7 & 8", w, cov_d, c, horizon),
                     *risk.monte_carlo_var_cvar(sims, c), *risk.monte_carlo_var_cvar(hist, c)])
    rows = np.array(rows)
    fig = go.Figure()
    for k, (name, color) in enumerate([("normal formula", "#1f77b4"), ("Monte Carlo", "#ff7f0e"),
                                       ("historical", "#2ca02c")]):
        fig.add_trace(go.Scatter(x=100 * levels, y=100 * rows[:, 2 * k], name=f"VaR · {name}",
                                 line=dict(color=color)))
        fig.add_trace(go.Scatter(x=100 * levels, y=100 * rows[:, 2 * k + 1], name=f"CVaR · {name}",
                                 line=dict(color=color, dash="dash")))
    fig.update_layout(xaxis_title="Confidence (%)", yaxis_title=f"Loss over {horizon} days (%)", height=480,
                      margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame(rows, index=[f"{c:.1%}" for c in levels],
                              columns=["VaR normal", "CVaR normal", "VaR MC", "CVaR MC", "VaR hist", "CVaR hist"])
                 .style.format("{:.2%}"), width="stretch")
    st.caption("Normal and Monte Carlo agree because the simulation draws normal shocks. Where the historical line "
               "sits above them at high confidence, real returns have fatter tails than the normal model assumes.")


# ----------------------------------------------------------------------------
# PCA
# ----------------------------------------------------------------------------
def pca_page():
    st.title("PCA")
    st.caption("Eigenvectors of a covariance matrix are the hidden factors behind the data; eigenvalues say how much "
               "variance each explains. All eigenpairs come from the hand-written shifted QR algorithm.")
    years = st.sidebar.slider("History used (years)", 1, 7, 3, key="pca_years")
    tab1, tab2 = st.tabs(["ETF returns", "Treasury yield curve"])
    with tab1:
        tickers = ticker_picker("pca_tickers", list(ETF_NAMES))
        corr = stats.correlation_matrix(lookback_returns(tickers, years))
        vals, vecs, ex = try_step(pca.pca, "Session 8", corr)
        share = pca.dominant_factor_share(corr)
        st.metric("First factor (power method cross-check)", pct(share), help="largest eigenvalue / n")
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Bar(x=[f"PC{k + 1}" for k in range(len(ex))], y=100 * ex))
            fig.update_layout(title="Variance explained (%)", height=400, margin=dict(t=40))
            st.plotly_chart(fig, width="stretch")
        with col2:
            k = min(3, len(tickers))
            L = vecs[:, :k] * np.where(vecs[:, :k].sum(axis=0) < 0, -1, 1)
            fig = go.Figure(go.Heatmap(z=L, x=[f"PC{i + 1}" for i in range(k)], y=tickers, colorscale="RdBu_r",
                                       zmid=0, text=np.round(L, 2), texttemplate="%{text}"))
            fig.update_layout(title="Loadings (correlation matrix eigenvectors)", height=400, margin=dict(t=40),
                              yaxis_autorange="reversed")
            st.plotly_chart(fig, width="stretch")
        st.caption("PC1 loads on everything with the same sign: that's 'the market'. Later components separate "
                   "stocks from bonds and gold.")
    with tab2:
        ylds = treasury_yields().dropna()
        dy = ylds.diff().dropna().iloc[-int(252 * years):]
        tenors = ylds.columns.to_numpy(dtype=float)
        vals, vecs, ex = try_step(pca.pca, "Session 8", stats.covariance_matrix(dy))
        V = vecs[:, :3].copy()
        V[:, 0] *= np.sign(V[:, 0].sum())                     # level: all positive
        V[:, 1] *= np.sign(V[-1, 1] - V[0, 1])                # slope: long end up
        V[:, 2] *= -np.sign(V[0, 2] + V[-1, 2] - 2 * V[len(tenors) // 2, 2])   # curvature: hump in the middle
        col1, col2 = st.columns([1, 2])
        with col1:
            for name, e in zip(["Level", "Slope", "Curvature"], ex[:3]):
                st.metric(name, pct(e))
        with col2:
            fig = go.Figure([go.Scatter(x=tenors, y=V[:, i], mode="lines+markers", name=n)
                             for i, n in enumerate(["Level", "Slope", "Curvature"])])
            fig.update_layout(xaxis_title="Maturity (years)", yaxis_title="Loading", xaxis_type="log", height=420,
                              margin=dict(t=20), legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, width="stretch")
        st.caption("Daily changes in the 11 Treasury yields: three factors explain almost all of it. Level moves "
                   "the whole curve, slope tilts it, curvature bends the middle.")


# The landing page is the fast Portfolio Risk Advisor. Streamlit only runs the selected page's
# function, so the Yield Curve page (its bond pricer solves a dense spline system, ~10 s) is
# computed only when someone clicks on it.
pages = [
    st.Page(advisor_page, title="Portfolio Risk Advisor", icon="🛡️", default=True),
    st.Page(yield_curve_page, title="Yield Curve", icon="📈"),
    st.Page(covariance_page, title="Covariance & Correlation", icon="🧮"),
    st.Page(frontier_page, title="Efficient Frontier", icon="🎯"),
    st.Page(var_page, title="Value at Risk", icon="⚠️"),
    st.Page(pca_page, title="PCA", icon="🔍"),
]
st.navigation(pages).run()
