# ruff: noqa
"""Week 2 exercise: crypto OHLCV data narrative."""

from io import BytesIO
from pathlib import Path
import ssl
import urllib.request

import matplotlib.pyplot as plt
import pandas as pd

DATA_URL = "https://openbondassetpricing.com/wp-content/uploads/2026/06/crypto_panel.csv"

OUTPUT_DIR = Path("fins2026") / "week2" / "scratch" / "week1_recap"
FIGURE_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_from_url(url: str) -> pd.DataFrame:
    ssl_context = ssl._create_unverified_context()
    with urllib.request.urlopen(url, context=ssl_context) as response:
        csv_bytes = response.read()
    return pd.read_csv(BytesIO(csv_bytes))


crypto_panel = read_csv_from_url(DATA_URL)

print("Raw crypto data loaded")
print(crypto_panel.head())
print(crypto_panel.info())

crypto_panel.columns = [col.lower() for col in crypto_panel.columns]
crypto_panel["date"] = pd.to_datetime(crypto_panel["date"])

numeric_columns = [col for col in crypto_panel.columns if col not in ["date", "ticker"]]
crypto_panel[numeric_columns] = crypto_panel[numeric_columns].apply(
    pd.to_numeric,
    errors="coerce",
)

crypto_panel = crypto_panel.sort_values(["ticker", "date"]).reset_index(drop=True)

print("\nStage 1 checks")
print(f"Rows: {len(crypto_panel):,}")
print(f"Columns: {len(crypto_panel.columns):,}")
print(
    "Duplicate date-ticker rows:",
    crypto_panel.duplicated(subset=["date", "ticker"]).sum(),
)
print("Missing values by column:")
print(crypto_panel.isna().sum())
print("Rows per ticker:")
print(crypto_panel.groupby("ticker").size())

crypto_panel["daily_return"] = crypto_panel.groupby("ticker")["close"].pct_change()
crypto_panel["dollar_volume"] = crypto_panel["usd_volume"]

close_wide = crypto_panel.pivot_table(
    index="date",
    columns="ticker",
    values="close",
)

return_wide = close_wide.pct_change()
wealth = (1 + return_wide.fillna(0)).cumprod()

running_max = wealth.cummax()
drawdown = wealth / running_max - 1

TRADING_DAYS_PER_YEAR = 365

summary = crypto_panel.groupby("ticker").agg(
    first_date=("date", "min"),
    last_date=("date", "max"),
    observations=("daily_return", "count"),
    mean_daily_return=("daily_return", "mean"),
    daily_volatility=("daily_return", "std"),
    average_dollar_volume=("dollar_volume", "mean"),
)

summary = summary.join(wealth.tail(1).T.iloc[:, 0].rename("final_value_of_1"))
summary = summary.join(drawdown.min().rename("max_drawdown"))

summary["annualized_return_pct"] = (
    summary["mean_daily_return"] * TRADING_DAYS_PER_YEAR * 100
)
summary["annualized_volatility_pct"] = (
    summary["daily_volatility"] * (TRADING_DAYS_PER_YEAR ** 0.5) * 100
)
summary["max_drawdown_pct"] = summary["max_drawdown"] * 100
summary["average_dollar_volume_millions"] = (
    summary["average_dollar_volume"] / 1_000_000
)

summary_out = summary[
    [
        "first_date",
        "last_date",
        "observations",
        "final_value_of_1",
        "annualized_return_pct",
        "annualized_volatility_pct",
        "max_drawdown_pct",
        "average_dollar_volume_millions",
    ]
].round(3)

summary_path = OUTPUT_DIR / "crypto_summary_table.csv"
summary_out.to_csv(summary_path)

print("\nCrypto summary table")
print(summary_out)

print("\nSaved summary table")
print(summary_path)

plt.figure(figsize=(10, 6))
wealth.plot(ax=plt.gca(), logy=True)
plt.title("Growth of $1 by crypto asset")
plt.xlabel("Date")
plt.ylabel("Value of $1 invested, log scale")
plt.legend(title="Ticker")
plt.tight_layout()

fig1_path = FIGURE_DIR / "crypto_01_growth_of_1.png"
plt.savefig(fig1_path, dpi=150)
plt.show()
plt.close()

plot_data = summary.reset_index()

plt.figure(figsize=(8, 6))
plt.scatter(
    plot_data["annualized_volatility_pct"],
    plot_data["annualized_return_pct"],
)

for _, row in plot_data.iterrows():
    plt.annotate(
        row["ticker"],
        (row["annualized_volatility_pct"], row["annualized_return_pct"]),
        textcoords="offset points",
        xytext=(5, 5),
    )

plt.title("Crypto risk-return comparison")
plt.xlabel("Annualized volatility (%)")
plt.ylabel("Annualized mean return (%)")
plt.tight_layout()

fig2_path = FIGURE_DIR / "crypto_02_risk_return_scatter.png"
plt.savefig(fig2_path, dpi=150)
plt.show()
plt.close()

print("\nSaved figures")
print(fig1_path)
print(fig2_path)

winner = summary["final_value_of_1"].idxmax()
most_volatile = summary["annualized_volatility_pct"].idxmax()
largest_drawdown = summary["max_drawdown_pct"].idxmin()
final_value = summary.loc[winner, "final_value_of_1"]

narrative = f"""
Headline claim: The crypto panel shows that return leadership and risk are tightly linked, but not identical.
{winner} delivered the strongest growth of $1 over the sample, reaching about ${final_value:.2f} for every $1 invested at the start of the sample.
The risk-return scatter shows that {most_volatile} had the highest annualized volatility, while the drawdown statistics show that {largest_drawdown} experienced the deepest peak-to-trough loss.
This means the strongest narrative is not simply 'crypto went up'; the evidence shows large differences across coins in return, volatility, and downside risk.
A limitation is that this analysis uses historical daily OHLCV data only, so it does not explain the news, liquidity conditions, or crypto-specific events that may have driven the moves.
"""

narrative_path = OUTPUT_DIR / "crypto_narrative.txt"
narrative_path.write_text(narrative)

print("\nShort narrative")
print(narrative)

print("Saved narrative")
print(narrative_path)
