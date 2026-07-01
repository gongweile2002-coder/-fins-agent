# ruff: noqa
"""Out-of-sample Data Factory Floor, Stages 1-2: the data and the inputs.

A strict extension of the Week 4 walkthrough, on the SAME 50-stock equity bundle.
Stage 1 loads the prices and turns them into daily returns. Stage 2 forms the two
inputs every portfolio model needs: the average return of each stock (mu) and the
covariance matrix of the returns (Sigma).

This script saves the wide daily-return table that Stage 3 (02_stage3_oos.py) reads.
Run it whole, or one numbered stage at a time in the PyCharm Python Console.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:  # running a highlighted selection in the console
    _candidates = [Path.cwd(), Path.cwd() / "fins2026" / "week5" / "scratch" / "dff_oos_walkthrough"]
    BASE_DIR = next((p for p in _candidates if (p / "dff_oos_helpers.py").is_file()), Path.cwd())
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import data_access as da
from dff_oos_helpers import (apply_ft_style, ft_header, growth_of_one, plain_log_yaxis,
                             save_both, TRADING_DAYS, FT_GREY)

OUTPUT_DIR = BASE_DIR / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
for folder in (FIGURE_DIR, TABLE_DIR):
    folder.mkdir(parents=True, exist_ok=True)
print(f"Saving outputs to: {OUTPUT_DIR}")


# -----------------------------------------------------------------------------
# 1. Stage 1 -- load the 50-stock equity bundle and compute daily returns
# -----------------------------------------------------------------------------

# load_equity_prices downloads ONE cached public ZIP and returns 50 US stocks with
# daily adjusted close and a sector label, 2020-2023 (the same data as Week 4).
prices_long = da.load_equity_prices()
prices_wide = prices_long.pivot(index="date", columns="ticker", values="adjClose").sort_index()
returns = prices_wide.pct_change().dropna()  # simple daily returns, drop the first empty row

print("\nStage 1: equity data")
print(f"  {returns.shape[1]} stocks, {len(returns):,} trading days, "
      f"{returns.index.min():%Y-%m-%d} to {returns.index.max():%Y-%m-%d}")
save_both(returns, OUTPUT_DIR / "returns_wide.csv", OUTPUT_DIR / "returns_wide.parquet")


# -----------------------------------------------------------------------------
# 2. Stage 2 -- the two model inputs: mu and Sigma
# -----------------------------------------------------------------------------

# Every portfolio model takes the same two inputs, estimated from the returns:
#   mu     = average daily return of each stock   (a 50-vector)
#   Sigma  = covariance matrix of the returns     (a 50 x 50 matrix)
# Sigma holds each stock's own variance (its risk) and every pair's co-movement
# (how they diversify). These two objects are all the optimiser ever sees.
mean_vector = returns.mean().to_numpy()
covariance = np.cov(returns.to_numpy(), rowvar=False, ddof=1)

print("\nStage 2: inputs")
print(f"  mu is a {mean_vector.shape[0]}-vector; Sigma is a {covariance.shape[0]} x {covariance.shape[1]} matrix")
print(f"  average stock: return {mean_vector.mean()*TRADING_DAYS*100:.1f}% a year, "
      f"volatility {np.sqrt(np.diag(covariance)).mean()*np.sqrt(TRADING_DAYS)*100:.1f}% a year")


# -----------------------------------------------------------------------------
# 3. Figure: the growth of $1 for all 50 stocks (the raw material)
# -----------------------------------------------------------------------------

apply_ft_style()
fig, ax = plt.subplots(figsize=(10, 6))
for ticker in returns.columns:
    ax.plot(growth_of_one(returns[ticker]).index, growth_of_one(returns[ticker]).values,
            color=FT_GREY, linewidth=0.6, alpha=0.5)
ax.set_yscale("log")
plain_log_yaxis(ax)
ax.grid(axis="x", visible=False)
ax.set_ylabel("Value of $1 (log scale)")
ft_header(fig, "Fifty stocks, fifty different paths",
          "Growth of $1 in each stock, 2020-2023. The job is to combine them into one portfolio.",
          "Source: course equity bundle | 2020-2023")
fig.savefig(FIGURE_DIR / "stage1_01_fifty_stocks_growth.png", dpi=150)
plt.close()
plt.rcParams.update(plt.rcParamsDefault)


# -----------------------------------------------------------------------------
# 3b. Figure: the growth of $1 for the cryptocurrencies in the bundle
# -----------------------------------------------------------------------------

# The crypto bundle is the second universe used later (03_universes.py). We show
# the same growth-of-$1 view here so the two asset classes sit side by side.
crypto_px = da.load_crypto_prices().pivot(index="date", columns="ticker", values="adjClose").sort_index()
crypto_returns = crypto_px.pct_change().dropna().dropna(axis=1, how="any")
print(f"\nCrypto bundle: {crypto_returns.shape[1]} cryptos, {len(crypto_returns):,} days, "
      f"{crypto_returns.index.min():%Y-%m-%d} to {crypto_returns.index.max():%Y-%m-%d}")

apply_ft_style()
fig, ax = plt.subplots(figsize=(10, 6))
palette = plt.get_cmap("tab10")
for i, ticker in enumerate(crypto_returns.columns):
    g = growth_of_one(crypto_returns[ticker])
    ax.plot(g.index, g.values, color=palette(i % 10), linewidth=1.3, alpha=0.9, label=ticker)
ax.set_yscale("log")
plain_log_yaxis(ax)
ax.grid(axis="x", visible=False)
ax.set_ylabel("Value of $1 (log scale)")
ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
ft_header(fig, "The cryptos in the sample, even wider dispersion",
          "Growth of $1 in each cryptocurrency, 2020-2023. Far more spread than the stocks.",
          "Source: course crypto bundle | 2020-2023")
fig.savefig(FIGURE_DIR / "stage1_02_cryptos_growth.png", dpi=150)
plt.close()
plt.rcParams.update(plt.rcParamsDefault)


# -----------------------------------------------------------------------------
# 4. Figure: each stock's risk and return (what the inputs look like)
# -----------------------------------------------------------------------------

apply_ft_style()
fig, ax = plt.subplots(figsize=(10, 6))
asset_vol = returns.std().to_numpy() * np.sqrt(TRADING_DAYS) * 100
asset_ret = mean_vector * TRADING_DAYS * 100
ax.scatter(asset_vol, asset_ret, s=22, color=FT_GREY, alpha=0.7)
ax.axhline(0.0, color="#66605C", linewidth=0.8)
ax.grid(axis="x", visible=False)
ax.set_xlabel("Annualized volatility (%)")
ax.set_ylabel("Annualized average return (%)")
ft_header(fig, "Each stock has its own risk and return",
          "Each dot is one of the 50 stocks, in-sample 2020-2023",
          "Source: course equity bundle | average (arithmetic) returns")
fig.savefig(FIGURE_DIR / "stage2_01_stock_risk_return.png", dpi=150)
plt.close()
plt.rcParams.update(plt.rcParamsDefault)

print("\nSaved returns_wide and figures stage1_01, stage1_02, stage2_01.")
