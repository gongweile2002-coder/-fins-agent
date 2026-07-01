# ruff: noqa
"""Out-of-sample Data Factory Floor, Stage 3: the honest test on 50 stocks.

Week 4 chose the weights and judged them on the SAME data, so the optimised
portfolios looked spectacular (the tangency Sharpe was 2.82). Here we do the
honest test. Every month we estimate mu and Sigma from past data only, solve the
weights, and earn the next month's returns. We do this twice: once long-short
(short positions allowed, exactly like Week 4) and once long-only (no shorting),
to separate two questions -- is the blow-up just leverage, or does optimisation
fail even when constrained?

Run 01_stage1_2_inputs.py first. Then run this file whole, or one stage at a time.
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

from dff_oos_helpers import (apply_ft_style, efficient_frontier, ft_header, growth_of_one,
                             run_oos_backtest, save_both, scorecard, annualized_stats,
                             WEIGHT_FUNCS_LONG_SHORT, WEIGHT_FUNCS_LONG_ONLY,
                             STRATEGY_COLORS, TRADING_DAYS, FT_GREY, FT_MAROON)

OUTPUT_DIR = BASE_DIR / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
for folder in (FIGURE_DIR, TABLE_DIR):
    folder.mkdir(parents=True, exist_ok=True)

returns = pd.read_csv(OUTPUT_DIR / "returns_wide.csv", index_col=0, parse_dates=True).dropna()
mean_vector = returns.mean().to_numpy()
covariance = np.cov(returns.to_numpy(), rowvar=False, ddof=1)
SHORT = {"Equal-weight (1/N)": "1/N", "Minimum-variance": "Min-var", "Mean-variance (tangency)": "Tangency"}
print(f"Loaded {returns.shape[1]} stocks x {len(returns):,} days, {returns.index.min():%Y-%m-%d} to {returns.index.max():%Y-%m-%d}")


# -----------------------------------------------------------------------------
# 1. In-sample and out-of-sample scorecards, in both constraint modes
# -----------------------------------------------------------------------------

def insample_card(weight_funcs):
    rows = {}
    for name, func in weight_funcs.items():
        w = func(mean_vector, covariance)
        rows[name] = scorecard(returns.to_numpy() @ w, TRADING_DAYS)
    return pd.DataFrame(rows).T


modes = {"long_short": WEIGHT_FUNCS_LONG_SHORT, "long_only": WEIGHT_FUNCS_LONG_ONLY}
oos_returns = {}
cards = {}
for mode, funcs in modes.items():
    is_card = insample_card(funcs)
    oos_r, _, audit = run_oos_backtest(returns, funcs)
    oos_returns[mode] = oos_r
    oos_card = pd.DataFrame({name: scorecard(oos_r[name], TRADING_DAYS) for name in oos_r.columns}).T
    cards[(mode, "in_sample")] = is_card
    cards[(mode, "out_sample")] = oos_card
    print(f"\n=== {mode.upper().replace('_',' ')} ===  (OOS from {oos_r.index.min():%Y-%m-%d}, {len(audit)} monthly rebalances)")
    for name in funcs:
        print(f"  {SHORT[name]:9s}  in-sample Sharpe {is_card.loc[name,'sharpe']:5.2f}   "
              f"out-of-sample Sharpe {oos_card.loc[name,'sharpe']:6.2f}   "
              f"OOS return {oos_card.loc[name,'total_return']*100:6.0f}%   OOS vol {oos_card.loc[name,'ann_vol']*100:5.0f}%")

# Save a tidy scorecard table for the slides and report.
tidy = pd.concat({f"{m}_{s}": c for (m, s), c in cards.items()}, names=["panel"]).round(4)
save_both(tidy, TABLE_DIR / "oos_scorecards_equity.csv", TABLE_DIR / "oos_scorecards_equity.parquet")


# -----------------------------------------------------------------------------
# 2. Figure: out-of-sample growth of $1, LONG-SHORT (the leverage disaster)
# -----------------------------------------------------------------------------

def growth_figure(oos_r, filename, title, subtitle):
    apply_ft_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in oos_r.columns:
        g = growth_of_one(oos_r[name])
        ax.plot(g.index, g.values, color=STRATEGY_COLORS[name], linewidth=1.9, label=SHORT[name])
    ax.axhline(1.0, color="#66605C", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    ax.set_ylabel("Value of $1")
    ft_header(fig, title, subtitle, "Source: course equity bundle | out-of-sample, expanding window, monthly rebalanced")
    fig.savefig(FIGURE_DIR / filename, dpi=150)
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)


growth_figure(oos_returns["long_short"], "stage3_02_oos_growth_long_short.png",
              "Long-short: the optimised portfolio is wiped out",
              "Growth of $1 out-of-sample. The tangency portfolio's leverage destroys it.")
growth_figure(oos_returns["long_only"], "stage3_04_oos_growth_long_only.png",
              "Long-only: 1/N is still hard to beat",
              "Growth of $1 out-of-sample, no short selling allowed.")


# -----------------------------------------------------------------------------
# 3. Figure: same test window, only the weights differ (apples-to-apples)
# -----------------------------------------------------------------------------

# Every point is measured on the SAME out-of-sample window, so 1/N -- whose
# weights never change -- is a single fixed point. The circle uses full-sample
# (hindsight) weights; the star uses the honest rolling weights. Only the
# optimised portfolios move, and the gap between them is pure estimation error.
def point(card, name):
    return card.loc[name, "ann_vol"] * 100, card.loc[name, "ann_return"] * 100


oos_idx = oos_returns["long_only"].index          # the 2021-onward test window
oos_wide = returns.loc[oos_idx]                    # daily asset returns, OOS window only
oos_mean = oos_wide.mean().to_numpy()
oos_cov = np.cov(oos_wide.to_numpy(), rowvar=False, ddof=1)

# Circle = full-sample (hindsight) weights scored on the OOS window.
# Star   = out-of-sample rolling weights on the OOS window (cards[('long_only','out_sample')]).
hindsight_card = pd.DataFrame(
    {name: scorecard(oos_wide.to_numpy() @ func(mean_vector, covariance), TRADING_DAYS)
     for name, func in WEIGHT_FUNCS_LONG_ONLY.items()}).T
oos_card = cards[("long_only", "out_sample")]

apply_ft_style()
fig, ax = plt.subplots(figsize=(10, 6.5))
target_d, vol_d = efficient_frontier(oos_mean, oos_cov, max_return=0.42 / TRADING_DAYS)
ax.plot(vol_d * np.sqrt(TRADING_DAYS) * 100, target_d * TRADING_DAYS * 100,
        color="#262A33", linewidth=1.8, label="out-of-sample frontier")
for name in WEIGHT_FUNCS_LONG_ONLY:
    hx, hy = point(hindsight_card, name)
    osx, osy = point(oos_card, name)
    color = STRATEGY_COLORS[name]
    fixed = abs(hx - osx) < 0.1 and abs(hy - osy) < 0.1
    if not fixed:
        ax.annotate("", xy=(osx, osy), xytext=(hx, hy),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.4, alpha=0.7))
    ax.scatter([hx], [hy], s=90, color=color, marker="o", zorder=5)
    ax.scatter([osx], [osy], s=150, color=color, marker="*", zorder=6)
    tag = f" {SHORT[name]} (fixed)" if fixed else f" {SHORT[name]}"
    ax.annotate(tag, xy=(osx, osy), color=color, fontweight="bold", fontsize=10, va="center")
ax.scatter([], [], color=FT_GREY, marker="o", label="hindsight (full-sample) weights")
ax.scatter([], [], color=FT_GREY, marker="*", s=120, label="out-of-sample (rolling) weights")
ax.set_xlim(0, 30)
ax.set_ylim(0, 44)
ax.grid(axis="x", visible=False)
ax.legend(loc="lower right", frameon=False, fontsize=9)
ax.set_xlabel("Annualized volatility (%)")
ax.set_ylabel("Annualized return (%)")
ft_header(fig, "Same test window, only the weights differ",
          "Circle: hindsight (full-sample) weights. Star: out-of-sample (rolling) weights. Both on the 2021-onward window. 1/N never moves.",
          "Source: course equity bundle | long-short tangency omitted (lands far off-chart)")
fig.savefig(FIGURE_DIR / "stage3_03_frontier_is_vs_oos.png", dpi=150)
plt.close()
plt.rcParams.update(plt.rcParamsDefault)


# -----------------------------------------------------------------------------
# 4. Figure: in-sample vs out-of-sample Sharpe ratio (the punchline)
# -----------------------------------------------------------------------------

apply_ft_style()
fig, ax = plt.subplots(figsize=(10, 6))
names = list(WEIGHT_FUNCS_LONG_ONLY)
x = np.arange(len(names))
width = 0.38
is_sr = [cards[("long_only", "in_sample")].loc[n, "sharpe"] for n in names]
oos_sr = [cards[("long_only", "out_sample")].loc[n, "sharpe"] for n in names]
ax.bar(x - width / 2, is_sr, width, color="#C9BBA8", label="in-sample")
ax.bar(x + width / 2, oos_sr, width, color=[STRATEGY_COLORS[n] for n in names], label="out-of-sample")
for i, (a, b) in enumerate(zip(is_sr, oos_sr)):
    ax.annotate(f"{a:.2f}", xy=(i - width / 2, a), ha="center", va="bottom", fontsize=9, color=FT_GREY)
    ax.annotate(f"{b:.2f}", xy=(i + width / 2, b), ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.axhline(oos_sr[0], color="#1A1A1A", linewidth=0.9, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels([SHORT[n] for n in names])
ax.grid(axis="x", visible=False)
ax.legend(loc="upper right", frameon=False, fontsize=10)
ax.set_ylabel("Sharpe ratio")
ft_header(fig, "Out-of-sample, nothing beats 1/N",
          "Long-only Sharpe ratio. The dashed line is 1/N out-of-sample, the bar to beat.",
          "Source: course equity bundle | in-sample flatters the optimised portfolios")
fig.savefig(FIGURE_DIR / "stage3_05_sharpe_is_vs_oos.png", dpi=150)
plt.close()
plt.rcParams.update(plt.rcParamsDefault)

print("\nSaved Stage 3 figures (stage3_02 ... stage3_05) and the equity scorecard table.")
