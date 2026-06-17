# ruff: noqa


"""Week 2 end-of-class exercise: sector ETF narrative.

This script loads a sector ETF panel, cleans it, builds a performance summary,
creates professional figures, and writes a short narrative.
"""

from io import BytesIO
from pathlib import Path
import ssl
import urllib.request

import matplotlib.pyplot as plt
import pandas as pd

DATA_URL = "https://openbondassetpricing.com/wp-content/uploads/2026/06/sector_etf_panel.csv"

OUTPUT_DIR = Path("fins2026") / "week2" / "scratch" / "sector_etf_narrative"
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SECTOR_NAMES = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
}


def read_csv_from_url(url: str) -> pd.DataFrame:
    """Read the course CSV while avoiding macOS SSL certificate errors."""
    ssl_context = ssl._create_unverified_context()
    with urllib.request.urlopen(url, context=ssl_context) as response:
        csv_bytes = response.read()
    return pd.read_csv(BytesIO(csv_bytes))


# -----------------------------------------------------------------------------
# 1. Load and inspect
# -----------------------------------------------------------------------------
etf_panel = read_csv_from_url(DATA_URL)

print("Raw sector ETF data loaded")
print(etf_panel.head())
print(etf_panel.info())

etf_panel.columns = [col.lower() for col in etf_panel.columns]
etf_panel["date"] = pd.to_datetime(etf_panel["date"])

numeric_columns = [col for col in etf_panel.columns if col not in ["date", "ticker", "sector"]]
etf_panel[numeric_columns] = etf_panel[numeric_columns].apply(pd.to_numeric, errors="coerce")
etf_panel = etf_panel.sort_values(["ticker", "date"]).reset_index(drop=True)

print("\nStage 1 checks")
print(f"Rows: {len(etf_panel):,}")
print(f"Columns: {len(etf_panel.columns):,}")
print("Duplicate date-ticker rows:", etf_panel.duplicated(subset=["date", "ticker"]).sum())
print("Missing values by column:")
print(etf_panel.isna().sum())
print("Rows per ticker:")
print(etf_panel.groupby("ticker").size())

# -----------------------------------------------------------------------------
# 2. Feature engineering
# -----------------------------------------------------------------------------
if "close" not in etf_panel.columns:
    raise ValueError(f"Expected a close column. Available columns: {list(etf_panel.columns)}")

etf_panel["daily_return"] = etf_panel.groupby("ticker")["close"].pct_change()

if "usd_volume" in etf_panel.columns:
    etf_panel["dollar_volume"] = etf_panel["usd_volume"]
elif "volume" in etf_panel.columns:
    etf_panel["dollar_volume"] = etf_panel["close"] * etf_panel["volume"]
elif "share_volume" in etf_panel.columns:
    etf_panel["dollar_volume"] = etf_panel["close"] * etf_panel["share_volume"]
else:
    etf_panel["dollar_volume"] = pd.NA

close_wide = etf_panel.pivot_table(index="date", columns="ticker", values="close")
return_wide = close_wide.pct_change()
wealth = (1 + return_wide.fillna(0)).cumprod()
drawdown = wealth / wealth.cummax() - 1
correlation = return_wide.corr()

TRADING_DAYS_PER_YEAR = 252

summary = etf_panel.groupby("ticker").agg(
    first_date=("date", "min"),
    last_date=("date", "max"),
    observations=("daily_return", "count"),
    mean_daily_return=("daily_return", "mean"),
    daily_volatility=("daily_return", "std"),
    average_dollar_volume=("dollar_volume", "mean"),
)
summary = summary.join(wealth.tail(1).T.iloc[:, 0].rename("final_value_of_1"))
summary = summary.join(drawdown.min().rename("max_drawdown"))
summary["annualized_return_pct"] = summary["mean_daily_return"] * TRADING_DAYS_PER_YEAR * 100
summary["annualized_volatility_pct"] = summary["daily_volatility"] * (TRADING_DAYS_PER_YEAR ** 0.5) * 100
summary["sharpe_ratio_rf_0"] = (
    summary["mean_daily_return"] / summary["daily_volatility"] * (TRADING_DAYS_PER_YEAR ** 0.5)
)
summary["max_drawdown_pct"] = summary["max_drawdown"] * 100
summary["average_dollar_volume_millions"] = summary["average_dollar_volume"] / 1_000_000
summary["sector"] = summary.index.map(SECTOR_NAMES)

summary_out = summary[
    [
        "sector",
        "first_date",
        "last_date",
        "observations",
        "final_value_of_1",
        "annualized_return_pct",
        "annualized_volatility_pct",
        "sharpe_ratio_rf_0",
        "max_drawdown_pct",
        "average_dollar_volume_millions",
    ]
].sort_values("final_value_of_1", ascending=False).round(3)

summary_path = OUTPUT_DIR / "sector_etf_performance_summary.csv"
summary_out.to_csv(summary_path)

print("\nSector ETF performance summary")
print(summary_out)
print("\nSaved summary table")
print(summary_path)

# -----------------------------------------------------------------------------
# 3. Figure 1: growth of $1 by sector
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
wealth.plot(ax=plt.gca(), logy=True)
plt.title("Technology led sector ETF wealth growth in the sample")
plt.xlabel("Date")
plt.ylabel("Value of $1 invested, log scale")
plt.legend(title="Ticker")
plt.figtext(
    0.01,
    0.01,
    "Source: course sector ETF panel. Note: daily close-to-close returns; log scale.",
    ha="left",
    fontsize=8,
)
plt.tight_layout(rect=(0, 0.04, 1, 1))
fig1_path = FIGURE_DIR / "sector_01_growth_of_1.png"
plt.savefig(fig1_path, dpi=150)
plt.show()
plt.close()

# -----------------------------------------------------------------------------
# 4. Figure 2: drawdowns by sector
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
(drawdown * 100).plot(ax=plt.gca())
plt.title("Sector leadership came with repeated drawdown risk")
plt.xlabel("Date")
plt.ylabel("Drawdown from previous peak (%)")
plt.legend(title="Ticker")
plt.figtext(
    0.01,
    0.01,
    "Source: course sector ETF panel. Note: drawdown is measured from each ticker's prior wealth peak.",
    ha="left",
    fontsize=8,
)
plt.tight_layout(rect=(0, 0.04, 1, 1))
fig2_path = FIGURE_DIR / "sector_02_drawdowns.png"
plt.savefig(fig2_path, dpi=150)
plt.show()
plt.close()

# -----------------------------------------------------------------------------
# 5. Figure 3: risk-return scatter
# -----------------------------------------------------------------------------
plot_data = summary.reset_index()
plt.figure(figsize=(8, 6))
plt.scatter(plot_data["annualized_volatility_pct"], plot_data["annualized_return_pct"])
for _, row in plot_data.iterrows():
    plt.annotate(
        row["ticker"],
        (row["annualized_volatility_pct"], row["annualized_return_pct"]),
        textcoords="offset points",
        xytext=(5, 5),
    )
plt.title("Sector returns and risk were not evenly distributed")
plt.xlabel("Annualized volatility (%)")
plt.ylabel("Annualized mean return (%)")
plt.figtext(
    0.01,
    0.01,
    "Source: course sector ETF panel. Note: annualized from daily returns; risk-free rate not subtracted.",
    ha="left",
    fontsize=8,
)
plt.tight_layout(rect=(0, 0.04, 1, 1))
fig3_path = FIGURE_DIR / "sector_03_risk_return_scatter.png"
plt.savefig(fig3_path, dpi=150)
plt.show()
plt.close()

correlation_path = OUTPUT_DIR / "sector_return_correlation.csv"
correlation.round(3).to_csv(correlation_path)

print("\nSaved figures")
print(fig1_path)
print(fig2_path)
print(fig3_path)
print("Saved correlation table")
print(correlation_path)

# -----------------------------------------------------------------------------
# 6. Narrative
# -----------------------------------------------------------------------------
winner = summary["final_value_of_1"].idxmax()
worst_drawdown = summary["max_drawdown_pct"].idxmin()
most_volatile = summary["annualized_volatility_pct"].idxmax()
best_final_value = summary.loc[winner, "final_value_of_1"]

mask = pd.DataFrame(True, index=correlation.index, columns=correlation.columns)
for ticker in correlation.index:
    mask.loc[ticker, ticker] = False
average_pairwise_corr = correlation.where(mask).stack().mean()

narrative = f"""Headline claim: Sector exposure mattered because {winner} ({SECTOR_NAMES.get(winner, winner)}) produced the strongest wealth growth, ending at about ${best_final_value:.2f} for every $1 invested.
The figures show that the return leader was not risk-free: {most_volatile} had the highest annualized volatility and {worst_drawdown} had the deepest peak-to-trough drawdown.
Co-movement was meaningful but not perfect, with average pairwise return correlation of about {average_pairwise_corr:.2f}, so diversification across sectors still had some value.
Risk caveat: the sample is historical and does not prove that the winning sector will continue to lead in a different inflation, rate, or earnings environment.
Portfolio implication: a sector portfolio can reduce single-sector dependence, but it should still monitor drawdowns because sector ETFs can fall together during broad market stress.
"""

narrative_path = OUTPUT_DIR / "sector_etf_narrative.txt"
narrative_path.write_text(narrative)

print("\nShort narrative")
print(narrative)
print("Saved narrative")
print(narrative_path)