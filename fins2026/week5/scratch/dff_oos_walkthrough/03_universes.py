# ruff: noqa
"""Out-of-sample Data Factory Floor, Stage 3 extended: equity, crypto, and combined.

The same honest out-of-sample machine, run on three universes from the Week 4
bundle: 50 US stocks, 10 cryptocurrencies, and the two together (60 assets). The
combined book aligns crypto prices to the stock-market calendar first, exactly as
in the Week 5 practise. We use long-only weights here (the fair test) and ask: in
which universe can the optimiser beat 1/N out-of-sample?

Run 01 and 02 first if you want their outputs; this script is self-contained and
loads the bundle directly. Run it whole, or one stage at a time.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    _candidates = [Path.cwd(), Path.cwd() / "fins2026" / "week5" / "scratch" / "dff_oos_walkthrough"]
    BASE_DIR = next((p for p in _candidates if (p / "dff_oos_helpers.py").is_file()), Path.cwd())
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import data_access as da
from dff_oos_helpers import (apply_ft_style, ft_header, growth_of_one, run_oos_backtest, save_both,
                             scorecard, WEIGHT_FUNCS_LONG_ONLY, STRATEGY_COLORS, TRADING_DAYS,
                             CRYPTO_DAYS, FT_GREY)

OUTPUT_DIR = BASE_DIR / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
for folder in (FIGURE_DIR, TABLE_DIR):
    folder.mkdir(parents=True, exist_ok=True)
SHORT = {"Equal-weight (1/N)": "1/N", "Minimum-variance": "Min-var", "Mean-variance (tangency)": "Tangency"}


# -----------------------------------------------------------------------------
# 1. Build the three universes of daily returns
# -----------------------------------------------------------------------------

eq_px = da.load_equity_prices().pivot(index="date", columns="ticker", values="adjClose").sort_index()
cr_px = da.load_crypto_prices().pivot(index="date", columns="ticker", values="adjClose").sort_index()

equity = eq_px.pct_change().dropna().dropna(axis=1, how="any")
crypto = cr_px.pct_change().dropna().dropna(axis=1, how="any")
# Combined: align crypto prices to the stock-market calendar, THEN take returns
# (the Week 5 practise rule), so both asset classes share one calendar.
combined = pd.concat([eq_px, cr_px.reindex(eq_px.index)], axis=1).pct_change().dropna().dropna(axis=1, how="any")

universes = {
    "Equity (50)": (equity, TRADING_DAYS),
    "Crypto (10)": (crypto, CRYPTO_DAYS),
    "Combined (60)": (combined, TRADING_DAYS),
}


# -----------------------------------------------------------------------------
# 2. Run the long-only out-of-sample backtest on each universe
# -----------------------------------------------------------------------------

oos_by_universe = {}
rows = []
for uni_name, (R, A) in universes.items():
    oos_r, _, audit = run_oos_backtest(R, WEIGHT_FUNCS_LONG_ONLY)
    oos_by_universe[uni_name] = (oos_r, A)
    print(f"\n{uni_name}: {R.shape[1]} assets, {len(audit)} monthly rebalances, OOS from {oos_r.index.min():%Y-%m-%d}")
    for name in oos_r.columns:
        card = scorecard(oos_r[name], A)
        rows.append({"universe": uni_name, "strategy": SHORT[name], "sharpe": round(card["sharpe"], 2),
                     "ann_return": round(card["ann_return"], 3), "ann_vol": round(card["ann_vol"], 3),
                     "max_drawdown": round(card["max_drawdown"], 3)})
        print(f"  {SHORT[name]:9s}  OOS Sharpe {card['sharpe']:5.2f}  return {card['ann_return']*100:6.1f}%  vol {card['ann_vol']*100:5.1f}%")

table = pd.DataFrame(rows)
save_both(table.set_index(["universe", "strategy"]),
          TABLE_DIR / "oos_universes_long_only.csv", TABLE_DIR / "oos_universes_long_only.parquet")


# -----------------------------------------------------------------------------
# 3. Figure: out-of-sample Sharpe by universe and strategy
# -----------------------------------------------------------------------------

apply_ft_style()
fig, ax = plt.subplots(figsize=(10, 6))
uni_names = list(universes)
strat_names = list(WEIGHT_FUNCS_LONG_ONLY)
x = np.arange(len(uni_names))
width = 0.26
for j, name in enumerate(strat_names):
    vals = [table[(table.universe == u) & (table.strategy == SHORT[name])]["sharpe"].iloc[0] for u in uni_names]
    ax.bar(x + (j - 1) * width, vals, width, color=STRATEGY_COLORS[name], label=SHORT[name])
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.2f}", xy=(i + (j - 1) * width, v), ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(uni_names)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", frameon=False, fontsize=10)
ax.set_ylabel("Out-of-sample Sharpe ratio")
ft_header(fig, "1/N wins in equities and the combined book; min-variance edges it in crypto",
          "Long-only, out-of-sample. Crypto annualized with 365 days, equity and combined with 252.",
          "Source: course bundle | mean-variance (tangency) trails everywhere out-of-sample")
fig.savefig(FIGURE_DIR / "stage3_06_oos_sharpe_by_universe.png", dpi=150)
plt.close()
plt.rcParams.update(plt.rcParamsDefault)


# -----------------------------------------------------------------------------
# 4. Figure: combined book, growth of $1 out-of-sample
# -----------------------------------------------------------------------------

apply_ft_style()
fig, ax = plt.subplots(figsize=(10, 6))
comb_oos, _ = oos_by_universe["Combined (60)"]
for name in comb_oos.columns:
    g = growth_of_one(comb_oos[name])
    ax.plot(g.index, g.values, color=STRATEGY_COLORS[name], linewidth=1.9, label=SHORT[name])
ax.axhline(1.0, color="#66605C", linewidth=0.8)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", frameon=False, fontsize=11)
ax.set_ylabel("Value of $1")
ft_header(fig, "Combined book: the naive 1/N portfolio leads out-of-sample",
          "Growth of $1, 50 stocks plus 10 cryptos, long-only, monthly rebalanced",
          "Source: course bundle | crypto aligned to the stock-market calendar, out-of-sample")
fig.savefig(FIGURE_DIR / "stage3_07_combined_growth.png", dpi=150)
plt.close()
plt.rcParams.update(plt.rcParamsDefault)

print("\nSaved universe comparison figures (stage3_06, stage3_07) and the table.")
