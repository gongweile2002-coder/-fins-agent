# ruff: noqa
"""Out-of-sample DFF, extension: smarter risk (risk parity) and better timing (vol targeting).

1/N is hard to beat on the Sharpe ratio, but we can still improve the RISK of a
portfolio. Two levers, both honest:

  - Risk parity equalises each asset's RISK contribution, not its dollars. 1/N is
    naive about risk -- in the combined book the cryptos are a small share of the
    money but a large share of the risk. Risk parity fixes that.
  - Volatility targeting scales total exposure by recent volatility, taking less
    risk when markets are turbulent. It helps the crypto-heavy combined book but
    not equities -- a real but sample-dependent edge.

Self-contained: loads the bundle directly. Run whole or one stage at a time.
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
from dff_oos_helpers import (apply_ft_style, ft_header, equal_weights, inverse_vol_weights,
                             risk_parity_weights, minimum_variance_weights_long_only,
                             risk_contributions, run_oos_backtest, scorecard, save_both,
                             volatility_target, TRADING_DAYS, FT_BLUE, FT_TEAL, FT_MAROON, FT_GREY)

OUTPUT_DIR = BASE_DIR / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
for folder in (FIGURE_DIR, TABLE_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# Build equity and combined return panels (crypto aligned to the stock calendar).
eq_px = da.load_equity_prices().pivot(index="date", columns="ticker", values="adjClose").sort_index()
cr_px = da.load_crypto_prices().pivot(index="date", columns="ticker", values="adjClose").sort_index()
crypto_tickers = set(cr_px.columns)
equity = eq_px.pct_change().dropna().dropna(axis=1, how="any")
combined = pd.concat([eq_px, cr_px.reindex(eq_px.index)], axis=1).pct_change().dropna().dropna(axis=1, how="any")

RISK_BASED = {
    "Equal-weight (1/N)": lambda mu, cov: equal_weights(len(mu)),
    "Inverse-vol": lambda mu, cov: inverse_vol_weights(cov),
    "Risk parity": lambda mu, cov: risk_parity_weights(cov),
    "Minimum-variance": lambda mu, cov: minimum_variance_weights_long_only(cov),
}
COLORS = {"Equal-weight (1/N)": "#1A1A1A", "Inverse-vol": FT_BLUE, "Risk parity": FT_TEAL,
          "Minimum-variance": "#B36A00"}
SHORT = {"Equal-weight (1/N)": "1/N", "Inverse-vol": "Inverse-vol", "Risk parity": "Risk parity",
         "Minimum-variance": "Min-var"}


# -----------------------------------------------------------------------------
# 1. 1/N is naive about risk: the combined book's risk is dominated by crypto
# -----------------------------------------------------------------------------

cov_comb = np.cov(combined.to_numpy(), rowvar=False, ddof=1)
is_crypto = np.array([t in crypto_tickers for t in combined.columns])
n = combined.shape[1]
w_ew = equal_weights(n)
w_rp = risk_parity_weights(cov_comb)
rc_ew = risk_contributions(w_ew, cov_comb)
rc_rp = risk_contributions(w_rp, cov_comb)
crypto_money = w_ew[is_crypto].sum()
crypto_risk_ew = rc_ew[is_crypto].sum()
crypto_risk_rp = rc_rp[is_crypto].sum()
print(f"Combined book ({n} assets): crypto is {crypto_money*100:.0f}% of the money, "
      f"{crypto_risk_ew*100:.0f}% of the risk under 1/N, {crypto_risk_rp*100:.0f}% under risk parity.")

apply_ft_style()
fig, ax = plt.subplots(figsize=(10, 6))
labels = ["Share of money\n(1/N)", "Share of risk\n(1/N)", "Share of risk\n(risk parity)"]
vals = [crypto_money * 100, crypto_risk_ew * 100, crypto_risk_rp * 100]
bars = ax.bar(labels, vals, color=[FT_GREY, FT_MAROON, FT_TEAL], width=0.6)
for b, v in zip(bars, vals):
    ax.annotate(f"{v:.0f}%", xy=(b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontweight="bold")
ax.grid(axis="x", visible=False)
ax.set_ylabel("Crypto share of the combined book (%)")
ft_header(fig, "1/N is naive about risk; risk parity fixes it",
          "The 10 cryptos are a small share of the money but a large share of the risk under 1/N",
          "Source: course bundle | combined 50 stocks + 10 cryptos, full-sample covariance")
fig.savefig(FIGURE_DIR / "ext_01_risk_shares.png", dpi=150)
plt.close()
plt.rcParams.update(plt.rcParamsDefault)


# -----------------------------------------------------------------------------
# 2. The risk-based out-of-sample race (long-only, expanding, monthly)
# -----------------------------------------------------------------------------

universes = {"Equity (50)": equity, "Combined (60)": combined}
rows = []
oos_store = {}
for uni, R in universes.items():
    oos_r, _, audit = run_oos_backtest(R, RISK_BASED)
    oos_store[uni] = oos_r
    print(f"\n{uni}: {len(audit)} monthly rebalances, OOS from {oos_r.index.min():%Y-%m-%d}")
    for name in oos_r.columns:
        card = scorecard(oos_r[name], TRADING_DAYS)
        rows.append({"universe": uni, "strategy": SHORT[name], "sharpe": round(card["sharpe"], 2),
                     "ann_vol": round(card["ann_vol"], 3), "max_drawdown": round(card["max_drawdown"], 3)})
        print(f"  {SHORT[name]:11s} Sharpe {card['sharpe']:5.2f}  vol {card['ann_vol']*100:5.1f}%  DD {card['max_drawdown']*100:5.0f}%")
table = pd.DataFrame(rows)
save_both(table.set_index(["universe", "strategy"]),
          TABLE_DIR / "ext_risk_based.csv", TABLE_DIR / "ext_risk_based.parquet")

apply_ft_style()
fig, ax = plt.subplots(figsize=(10, 6))
unis = list(universes)
strats = list(RISK_BASED)
x = np.arange(len(unis))
width = 0.2
for j, name in enumerate(strats):
    vals = [table[(table.universe == u) & (table.strategy == SHORT[name])]["sharpe"].iloc[0] for u in unis]
    ax.bar(x + (j - 1.5) * width, vals, width, color=COLORS[name], label=SHORT[name])
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.2f}", xy=(i + (j - 1.5) * width, v), ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(unis)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)
ax.set_ylabel("Out-of-sample Sharpe ratio")
ft_header(fig, "Risk parity matches 1/N at lower risk, but does not beat it",
          "Long-only, out-of-sample. Risk parity and inverse-vol are risk-based and ignore expected returns.",
          "Source: course bundle | see the volatility column in the printout")
fig.savefig(FIGURE_DIR / "ext_02_risk_based_sharpe.png", dpi=150)
plt.close()
plt.rcParams.update(plt.rcParamsDefault)


# -----------------------------------------------------------------------------
# 3. Volatility targeting: a real but sample-dependent edge
# -----------------------------------------------------------------------------

vt_rows = []
managed = {}
for uni, R in universes.items():
    base = oos_store[uni]["Equal-weight (1/N)"]
    man, lev = volatility_target(base)
    managed[uni] = (base, man, lev)
    base_sr = scorecard(base, TRADING_DAYS)["sharpe"]
    man_sr = scorecard(man, TRADING_DAYS)["sharpe"]
    vt_rows.append({"universe": uni, "base_1N_sharpe": round(base_sr, 2), "vol_targeted_sharpe": round(man_sr, 2)})
    print(f"\nVol targeting on 1/N -- {uni}: base Sharpe {base_sr:.2f} -> managed {man_sr:.2f}")
vt = pd.DataFrame(vt_rows)
save_both(vt.set_index("universe"), TABLE_DIR / "ext_vol_target.csv", TABLE_DIR / "ext_vol_target.parquet")

apply_ft_style()
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(unis))
width = 0.36
base_vals = [vt[vt.universe == u]["base_1N_sharpe"].iloc[0] for u in unis]
man_vals = [vt[vt.universe == u]["vol_targeted_sharpe"].iloc[0] for u in unis]
ax.bar(x - width / 2, base_vals, width, color=FT_GREY, label="1/N")
ax.bar(x + width / 2, man_vals, width, color=FT_BLUE, label="1/N, volatility-targeted")
for i, (a, b) in enumerate(zip(base_vals, man_vals)):
    ax.annotate(f"{a:.2f}", xy=(i - width / 2, a), ha="center", va="bottom", fontsize=9)
    ax.annotate(f"{b:.2f}", xy=(i + width / 2, b), ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(unis)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", frameon=False, fontsize=10)
ax.set_ylabel("Out-of-sample Sharpe ratio")
ft_header(fig, "Volatility targeting helps the combined book, not equities",
          "Scaling 1/N exposure by recent volatility. A real edge, but sample-dependent -- no free lunch.",
          "Source: course bundle | exposure scaled by past 21-day volatility, capped at 2x")
fig.savefig(FIGURE_DIR / "ext_03_voltarget_sharpe.png", dpi=150)
plt.close()
plt.rcParams.update(plt.rcParamsDefault)

# Leverage path on the combined book, to show what vol targeting actually does.
apply_ft_style()
fig, ax = plt.subplots(figsize=(10, 6))
_, _, lev = managed["Combined (60)"]
ax.plot(lev.index, lev.values, color=FT_BLUE, linewidth=1.3)
ax.axhline(1.0, color="#66605C", linewidth=0.8)
ax.grid(axis="x", visible=False)
ax.set_ylabel("Exposure (multiple of fully invested)")
ft_header(fig, "Less exposure when markets are turbulent",
          "Volatility-targeting leverage on the combined 1/N book, capped at 2x",
          "Source: course bundle | exposure = past-21-day target volatility / recent volatility")
fig.savefig(FIGURE_DIR / "ext_04_voltarget_leverage.png", dpi=150)
plt.close()
plt.rcParams.update(plt.rcParamsDefault)

print("\nSaved extension figures (ext_01 ... ext_04) and tables.")
