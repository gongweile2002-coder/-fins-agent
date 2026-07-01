# ruff: noqa
"""Out-of-sample DFF, extension: blending the naive and the optimal portfolio.

1/N is robust but takes whatever risk the market hands it; minimum-variance takes
much less risk but leans on the estimated covariance. You do not have to choose --
hold part of each. The blend is a dial,

    w = lambda * (1/N) + (1 - lambda) * MinVar(long-only),   0 <= lambda <= 1,

with no new optimiser to run (it is an average of two weight vectors we already
have). On the combined book, tilting toward minimum-variance sheds a lot of risk
for very little Sharpe -- the optimised piece earns its place by controlling risk.

Self-contained: loads the bundle directly. Run whole or one stage at a time.
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
from dff_oos_helpers import (apply_ft_style, blend_weights, equal_weights, ft_header,
                             minimum_variance_weights_long_only, run_oos_backtest, save_both,
                             scorecard, CRYPTO_DAYS, TRADING_DAYS, FT_BLUE, FT_GREY, FT_MAROON)

OUTPUT_DIR = BASE_DIR / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
for folder in (FIGURE_DIR, TABLE_DIR):
    folder.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 1. Build the three universes (identical to 03_universes.py)
# -----------------------------------------------------------------------------
eq_px = da.load_equity_prices().pivot(index="date", columns="ticker", values="adjClose").sort_index()
cr_px = da.load_crypto_prices().pivot(index="date", columns="ticker", values="adjClose").sort_index()
equity = eq_px.pct_change().dropna().dropna(axis=1, how="any")
crypto = cr_px.pct_change().dropna().dropna(axis=1, how="any")
combined = pd.concat([eq_px, cr_px.reindex(eq_px.index)], axis=1).pct_change().dropna().dropna(axis=1, how="any")
UNIVERSES = {"Equity (50)": (equity, TRADING_DAYS),
             "Crypto (10)": (crypto, CRYPTO_DAYS),
             "Combined (60)": (combined, TRADING_DAYS)}


def blend_fn(lmbda):
    """A (mu, cov) -> weights function for the lambda-blend, for run_oos_backtest."""
    return lambda mu, cov: blend_weights(lmbda, minimum_variance_weights_long_only(cov))


# -----------------------------------------------------------------------------
# 2. The lambda sweep on each universe (lambda = weight on 1/N)
# -----------------------------------------------------------------------------
TABLE_LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
rows = []
for uni, (R, A) in UNIVERSES.items():
    oos, _, _ = run_oos_backtest(R, {f"{l:.2f}": blend_fn(l) for l in TABLE_LAMBDAS})
    print(f"\n{uni}:  lambda=1 is pure 1/N, lambda=0 is pure minimum-variance")
    for l in TABLE_LAMBDAS:
        c = scorecard(oos[f"{l:.2f}"], A)
        rows.append({"universe": uni, "lambda": l, "sharpe": round(c["sharpe"], 2),
                     "ann_return": round(c["ann_return"], 3), "ann_vol": round(c["ann_vol"], 3),
                     "max_drawdown": round(c["max_drawdown"], 3)})
        print(f"  lambda={l:.2f}  Sharpe {c['sharpe']:5.2f}  return {c['ann_return']*100:6.1f}%  "
              f"vol {c['ann_vol']*100:5.1f}%  drawdown {c['max_drawdown']*100:6.1f}%")

table = pd.DataFrame(rows)
save_both(table.set_index(["universe", "lambda"]),
          TABLE_DIR / "blend_lambda_sweep.csv", TABLE_DIR / "blend_lambda_sweep.parquet")


# -----------------------------------------------------------------------------
# 3. Figure: the risk dial on the combined book
# -----------------------------------------------------------------------------
grid = np.round(np.arange(0.0, 1.0001, 0.05), 2)
oos_c, _, _ = run_oos_backtest(combined, {f"{l:.2f}": blend_fn(l) for l in grid})
vol = {l: scorecard(oos_c[f"{l:.2f}"], TRADING_DAYS)["ann_vol"] * 100 for l in grid}
dd = {l: abs(scorecard(oos_c[f"{l:.2f}"], TRADING_DAYS)["max_drawdown"]) * 100 for l in grid}
sr = {l: scorecard(oos_c[f"{l:.2f}"], TRADING_DAYS)["sharpe"] for l in grid}
vol_y = np.array([vol[l] for l in grid])
dd_y = np.array([dd[l] for l in grid])

apply_ft_style()
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(grid, vol_y, color=FT_MAROON, lw=2.4)
ax.plot(grid, dd_y, color=FT_BLUE, lw=2.4)
ax.annotate("Annualized\nvolatility", xy=(grid[-1], vol_y[-1]), xytext=(6, 0), textcoords="offset points",
            va="center", color=FT_MAROON, fontsize=10, fontweight="bold")
ax.annotate("Maximum\ndrawdown (depth)", xy=(grid[-1], dd_y[-1]), xytext=(6, 0), textcoords="offset points",
            va="center", color=FT_BLUE, fontsize=10, fontweight="bold")
for l, lab in {1.0: "1/N\nSharpe 0.97", 0.5: "50/50 blend\nSharpe 0.92", 0.0: "Min-var\nSharpe 0.60"}.items():
    ax.scatter([l, l], [vol[l], dd[l]], color=FT_GREY, zorder=5, s=28)
    ax.annotate(lab, xy=(l, dd[l]), xytext=(0, 12), textcoords="offset points",
                ha="center", va="bottom", fontsize=9, color="#2D3748", fontweight="bold")
ax.set_xlim(-0.04, 1.18)
ax.set_ylim(10, 30)
ax.set_xlabel(r"$\lambda$  (weight on 1/N):   0 = pure minimum-variance   $\rightarrow$   1 = pure 1/N")
ax.set_ylabel("Out-of-sample risk (%)")
ax.grid(axis="x", visible=False)
ft_header(fig, "Tilting the combined book toward minimum-variance sheds risk",
          "A 50/50 blend cuts volatility from 21.7% to 15.7% and the drawdown from 27% to 20%, giving up only 0.05 of Sharpe.",
          "Source: course bundle | out-of-sample, long-only, monthly, expanding window")
fig.savefig(FIGURE_DIR / "blend_combined.png", dpi=150, bbox_inches="tight")
plt.close()
plt.rcParams.update(plt.rcParamsDefault)

print(f"\nSaved blend_combined.png and blend_lambda_sweep table.")
print(f"check: vol 1/N={vol[1.0]:.1f}% blend={vol[0.5]:.1f}% minvar={vol[0.0]:.1f}%  |  "
      f"Sharpe 1/N={sr[1.0]:.2f} blend={sr[0.5]:.2f} minvar={sr[0.0]:.2f}")
