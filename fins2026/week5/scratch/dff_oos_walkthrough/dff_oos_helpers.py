# ruff: noqa
"""Shared helpers for the Week 5 out-of-sample Data Factory Floor walkthrough.

A strict extension of the Week 4 dff_walkthrough. Week 4 built portfolios IN-SAMPLE
(choosing the weights and judging them on the same 2020-2023 data). Here we do the
honest test: at each month we choose weights using ONLY past data, then earn the next
month's returns, and roll forward. This module holds the FT-style figure helpers, the
closed-form portfolio maths (same as Week 4), and the out-of-sample backtest engine.

The engine is deliberately fast: closed-form weights and a month-by-month loop run a
full 3-year backtest in well under a second.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullFormatter

# Annualization factors. US equities trade about 252 days a year; crypto trades 365.
TRADING_DAYS = 252
CRYPTO_DAYS = 365

# Financial Times palette (matches the rest of the course figures).
FT_CREAM = "#FDF1E6"
FT_MAROON = "#990F3D"
FT_BLUE = "#0F5499"
FT_TEAL = "#2F7F73"
FT_GREY = "#6B625C"

# One colour per strategy, shared across every figure.
STRATEGY_COLORS = {
    "Equal-weight (1/N)": "#1A1A1A",
    "Minimum-variance": FT_TEAL,
    "Mean-variance (tangency)": FT_MAROON,
}


def apply_ft_style():
    """Set rcParams so the next figure follows a clean FT-style look."""
    plt.rcParams.update({
        "figure.facecolor": FT_CREAM, "axes.facecolor": FT_CREAM,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.edgecolor": "#66605C", "axes.grid": True, "grid.color": "#E2D8CF",
        "axes.axisbelow": True, "font.family": "DejaVu Sans", "font.size": 12,
    })


def ft_header(fig, title, subtitle, source):
    """Write the FT-style title / subtitle / source block onto a figure."""
    fig.text(0.012, 0.96, title, fontsize=15, fontweight="bold", color="#262A33")
    fig.text(0.012, 0.91, subtitle, fontsize=11, color=FT_GREY)
    fig.text(0.012, 0.01, source, fontsize=8, color=FT_GREY)
    fig.subplots_adjust(top=0.86, bottom=0.12)


def plain_log_yaxis(ax):
    """Plain numbers on a log y-axis (1, 10, 100), never scientific 10^2 / 10^4."""
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def save_both(frame, csv_path, parquet_path):
    """Always save the CSV; try Parquet too but do not crash on a minimal install."""
    frame.to_csv(csv_path)
    try:
        frame.to_parquet(parquet_path)
    except Exception as exc:  # e.g. no pyarrow/fastparquet installed
        print(f"  (skipped {Path(parquet_path).name}: {exc}; the CSV is saved and is enough)")


# -----------------------------------------------------------------------------
# Closed-form portfolio weights (same maths as Week 4). Fully invested (weights
# sum to 1); short positions allowed. rf is a daily scalar.
# -----------------------------------------------------------------------------

def equal_weights(n_assets):
    """The 1/N benchmark: the same fraction 1/N in every asset."""
    return np.ones(n_assets) / n_assets


def minimum_variance_weights(cov):
    """w = Sigma^{-1} 1 / (1' Sigma^{-1} 1). Lowest-variance fully-invested
    portfolio, ignoring expected returns entirely."""
    ones = np.ones(cov.shape[0])
    inv_ones = np.linalg.solve(cov, ones)
    return inv_ones / (ones @ inv_ones)


def tangency_weights(mean, cov, risk_free=0.0):
    """w = Sigma^{-1}(mu - rf 1) / (1' Sigma^{-1}(mu - rf 1)). Highest-Sharpe
    (tangency / mean-variance) fully-invested portfolio."""
    ones = np.ones(len(mean))
    excess = mean - risk_free * ones
    inv_excess = np.linalg.solve(cov, excess)
    return inv_excess / (ones @ inv_excess)


def minimum_variance_weights_long_only(cov):
    """Long-only minimum variance: minimize w' Sigma w subject to sum(w)=1 and
    w>=0. There is no closed form, so we run a small constrained solve (SLSQP with
    the analytic gradient 2 Sigma w). scipy is imported here so the long-short path
    stays dependency-free."""
    from scipy.optimize import minimize
    n = cov.shape[0]
    result = minimize(
        lambda w: w @ cov @ w, np.ones(n) / n, jac=lambda w: 2 * cov @ w,
        bounds=[(0.0, None)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(n)}],
        method="SLSQP", options={"maxiter": 300, "ftol": 1e-14},
    )
    w = np.clip(result.x, 0.0, None)
    return w / w.sum()


def tangency_weights_long_only(mean, cov, risk_free=0.0):
    """Long-only maximum-Sharpe portfolio via a convex transform: minimize
    y' Sigma y subject to (mu - rf)' y = 1 and y>=0, then rescale w = y / sum(y).
    Falls back to equal weights if no asset has a positive excess return."""
    from scipy.optimize import minimize
    n = len(mean)
    excess = mean - risk_free
    if (excess <= 0).all():
        return equal_weights(n)
    result = minimize(
        lambda y: y @ cov @ y, np.ones(n) / n, jac=lambda y: 2 * cov @ y,
        bounds=[(0.0, None)] * n,
        constraints=[{"type": "eq", "fun": lambda y: excess @ y - 1.0, "jac": lambda y: excess}],
        method="SLSQP", options={"maxiter": 300, "ftol": 1e-12},
    )
    y = np.clip(result.x, 0.0, None)
    return y / y.sum()


def blend_weights(lmbda, optimised):
    """Convex blend of the naive 1/N portfolio and an optimised weight vector,
        w = lmbda * (1/N) + (1 - lmbda) * optimised,   0 <= lmbda <= 1.
    Both inputs are long-only and sum to 1, so the blend is long-only and sums to
    1 too -- there is no new optimiser to run, just an average of two weight
    vectors we already have. lmbda is the dial: 1 is pure 1/N, 0 is pure
    optimised, 0.5 is half and half."""
    optimised = np.asarray(optimised, dtype=float)
    return lmbda * equal_weights(len(optimised)) + (1.0 - lmbda) * optimised


def inverse_vol_weights(cov):
    """Inverse-volatility weights, w_i proportional to 1/sigma_i. The simple
    risk-based portfolio -- it equalises risk only if assets were uncorrelated.
    A stepping stone to full risk parity."""
    inv_vol = 1.0 / np.sqrt(np.diag(cov))
    return inv_vol / inv_vol.sum()


def risk_parity_weights(cov):
    """Long-only risk-parity (equal-risk-contribution) weights. Each asset
    contributes the same share of portfolio variance. Solved with the convex
    form: minimise 0.5 w'Sigma w - (1/n) sum(log w_i) over w>0, then rescale to
    sum to 1 (Maillard, Roncalli and Teiletche, 2010)."""
    from scipy.optimize import minimize
    n = cov.shape[0]
    result = minimize(
        lambda w: 0.5 * w @ cov @ w - np.mean(np.log(w)),
        np.ones(n) / n, jac=lambda w: cov @ w - 1.0 / (n * w),
        bounds=[(1e-9, None)] * n, method="L-BFGS-B", options={"maxiter": 800},
    )
    w = np.clip(result.x, 1e-12, None)
    return w / w.sum()


def risk_contributions(weights, cov):
    """Each asset's share of total portfolio variance,
    RC_i = w_i (Sigma w)_i / (w' Sigma w). The shares sum to 1."""
    weights = np.asarray(weights)
    port_var = float(weights @ cov @ weights)
    return weights * (cov @ weights) / port_var


def volatility_target(returns, lookback=21, cap=2.0, target=None):
    """Volatility-managed version of a daily return series. Each day we scale the
    exposure by target / (trailing volatility), using only PAST returns, capped at
    `cap` times (Moreira and Muir, 2017). With target = the median trailing
    volatility, average exposure is about 1, so this redistributes risk over time
    rather than just levering up. Returns (managed returns, leverage path)."""
    returns = pd.Series(returns)
    trailing_vol = returns.rolling(lookback).std().shift(1)  # shift => past only
    if target is None:
        target = trailing_vol.median()
    leverage = (target / trailing_vol).clip(upper=cap).fillna(1.0)
    return leverage * returns, leverage


# The three strategies, in each constraint mode. Same keys, different weight rules.
WEIGHT_FUNCS_LONG_SHORT = {
    "Equal-weight (1/N)": lambda mu, cov: equal_weights(len(mu)),
    "Minimum-variance": lambda mu, cov: minimum_variance_weights(cov),
    "Mean-variance (tangency)": lambda mu, cov: tangency_weights(mu, cov, 0.0),
}
WEIGHT_FUNCS_LONG_ONLY = {
    "Equal-weight (1/N)": lambda mu, cov: equal_weights(len(mu)),
    "Minimum-variance": lambda mu, cov: minimum_variance_weights_long_only(cov),
    "Mean-variance (tangency)": lambda mu, cov: tangency_weights_long_only(mu, cov, 0.0),
}


def efficient_frontier(mean, cov, n_points=250, min_return=None, max_return=None):
    """(target returns, volatilities) tracing the minimum-variance frontier, using
    a = 1' S^{-1} 1, b = 1' S^{-1} mu, c = mu' S^{-1} mu and
    sigma^2(m) = (a m^2 - 2 b m + c) / (a c - b^2)."""
    ones = np.ones(len(mean))
    inv_ones = np.linalg.solve(cov, ones)
    inv_mean = np.linalg.solve(cov, mean)
    a = ones @ inv_ones
    b = ones @ inv_mean
    c = mean @ inv_mean
    determinant = a * c - b * b
    gmv_return = b / a
    low = gmv_return if min_return is None else min_return
    top = mean.max() if max_return is None else max_return
    targets = np.linspace(low, top, n_points)
    variances = (a * targets ** 2 - 2 * b * targets + c) / determinant
    return targets, np.sqrt(variances)


# -----------------------------------------------------------------------------
# Risk-and-return metrics
# -----------------------------------------------------------------------------

def annualized_stats(returns, periods_per_year, risk_free=0.0):
    """Annualized arithmetic return, volatility, and Sharpe ratio for a daily series."""
    mean_d, std_d = returns.mean(), returns.std()
    ann_return = mean_d * periods_per_year
    ann_vol = std_d * np.sqrt(periods_per_year)
    sharpe = (mean_d - risk_free) / std_d * np.sqrt(periods_per_year) if std_d > 0 else np.nan
    return ann_return, ann_vol, sharpe


def growth_of_one(returns, floor=1e-6):
    """Growth of $1, the running product of (1 + r). Floored so a portfolio that
    is wiped out (cumulative value <= 0, possible for a leveraged short book) can
    still be drawn on a chart instead of breaking it."""
    wealth = (1.0 + returns).cumprod()
    return wealth.clip(lower=floor)


def scorecard(returns, periods_per_year, risk_free=0.0):
    """One row of out-of-sample metrics for a daily portfolio return series.
    Accepts a pandas Series or a plain numpy array."""
    returns = pd.Series(returns)
    ann_return, ann_vol, sharpe = annualized_stats(returns, periods_per_year, risk_free)
    raw_wealth = (1.0 + returns).cumprod()
    wiped_out = bool((raw_wealth <= 0).any())
    total_return = -1.0 if wiped_out else float(raw_wealth.iloc[-1] - 1.0)
    drawdown = float((growth_of_one(returns) / growth_of_one(returns).cummax() - 1.0).min())
    return {
        "total_return": total_return,
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": drawdown,
        "wiped_out": wiped_out,
    }


# -----------------------------------------------------------------------------
# The out-of-sample backtest engine
# -----------------------------------------------------------------------------

def run_oos_backtest(returns, weight_funcs, init_days=TRADING_DAYS):
    """Walk forward month by month with no look-ahead.

    weight_funcs is a dict {strategy name -> function(mu, cov) -> weights}, e.g.
    WEIGHT_FUNCS_LONG_SHORT or WEIGHT_FUNCS_LONG_ONLY. At each month, if at least
    init_days of history exist, estimate mu and Sigma from ALL past returns (an
    expanding window that ends the day before the month starts), solve each
    strategy's weights, and apply them to that month's daily returns. Returns a
    DataFrame of daily out-of-sample returns (one column per strategy), a dict of
    formation-date target weights, and a rebalance audit.

    The training window ends strictly before the held month, so the weights use
    only information available on the decision date -- the honest test.
    """
    strategies = list(weight_funcs)
    period = returns.index.to_period("M")
    months = sorted(period.unique())
    oos = {s: [] for s in strategies}
    weights_log = {s: [] for s in strategies}
    audit = []
    train_blocks = []
    train_days = 0

    for month in months:
        block = returns[period == month]
        if train_days >= init_days:
            train = pd.concat(train_blocks)
            mu = train.mean().to_numpy()
            cov = np.cov(train.to_numpy(), rowvar=False, ddof=1)
            for strategy in strategies:
                w = weight_funcs[strategy](mu, cov)
                oos[strategy].append(pd.Series(block.to_numpy() @ w, index=block.index))
                weights_log[strategy].append(pd.Series(w, index=returns.columns, name=block.index[0]))
            audit.append({
                "decision_month": str(month),
                "train_days": int(len(train)),
                "first_holding_date": block.index[0],
                "n_assets": int(returns.shape[1]),
            })
        train_blocks.append(block)
        train_days += len(block)

    oos_returns = pd.DataFrame({s: pd.concat(v) for s, v in oos.items()})
    weights = {s: pd.DataFrame(v) for s, v in weights_log.items()}
    return oos_returns, weights, pd.DataFrame(audit)
