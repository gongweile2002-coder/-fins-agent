# ruff: noqa: E501
"""Week 5 practice: a risk-adjusted Bitcoin versus S&P 500 narrative."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "provided_data"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BTC_COLOR = "#E85D04"
SP500_COLOR = "#0F5499"
TEAL = "#2F7F73"
INK = "#262A33"
GREY = "#6B625C"
CREAM = "#FDF1E6"


def apply_ft_style() -> None:
    """Apply a restrained Financial Times-style chart theme."""
    plt.rcParams.update(
        {
            "figure.facecolor": CREAM,
            "axes.facecolor": CREAM,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.edgecolor": GREY,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#DED4CB",
            "font.family": "DejaVu Sans",
            "font.size": 11,
        }
    )


def add_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.01, 0.97, title, fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.92, subtitle, fontsize=10.5, color=GREY, va="top")
    fig.text(
        0.01,
        0.015,
        "Source: course data | Common sample: 2010-07-18 to 2025-12-31",
        fontsize=8,
        color=GREY,
    )
    fig.subplots_adjust(top=0.84, bottom=0.13)


def load_series(filename: str, value_column: str) -> pd.Series:
    frame = pd.read_csv(DATA_DIR / filename, parse_dates=["date"])
    frame = frame.sort_values("date").drop_duplicates("date")
    return frame.set_index("date")[value_column].astype(float)


def attach_risk_free(rfr: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill the business-day rate onto the requested asset calendar."""
    expanded = rfr.reindex(rfr.index.union(dates)).sort_index().ffill()
    return expanded.reindex(dates)


def wealth_with_start(returns: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    selected = returns.loc[(returns.index > start) & (returns.index <= end)]
    wealth = (1.0 + selected).cumprod()
    return pd.concat([pd.Series([1.0], index=[start]), wealth])


def calculate_metrics(
    returns: pd.Series,
    risk_free: pd.Series,
    periods_per_year: int,
) -> dict[str, float]:
    joined = pd.concat([returns.rename("ret"), risk_free.rename("rfr")], axis=1).dropna()
    ret = joined["ret"]
    excess = ret - joined["rfr"]
    wealth = (1.0 + ret).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    total_return = float(wealth.iloc[-1] - 1.0)
    years = (ret.index.max() - ret.index.min()).days / 365.25
    cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0)
    volatility = float(ret.std() * np.sqrt(periods_per_year))
    sharpe = float(excess.mean() / ret.std() * np.sqrt(periods_per_year))
    downside_deviation = float(np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)))
    sortino = float(excess.mean() / downside_deviation * np.sqrt(periods_per_year))
    var_cutoff = float(ret.quantile(0.05))
    expected_shortfall = float(ret.loc[ret <= var_cutoff].mean())
    return {
        "Total return": total_return,
        "Annualized return (CAGR)": cagr,
        "Annualized volatility": volatility,
        "Sharpe ratio": sharpe,
        "Sortino ratio": sortino,
        "Maximum drawdown": float(drawdown.min()),
        "Daily VaR (95%)": -var_cutoff,
        "Daily expected shortfall (95%)": -expected_shortfall,
        "Worst daily return": float(ret.min()),
        "Daily return skewness": float(ret.skew()),
        "Daily excess kurtosis": float(ret.kurt()),
    }


def save_scorecard(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    scorecard = pd.DataFrame(metrics)
    scorecard.index.name = "metric"
    scorecard.to_csv(OUTPUT_DIR / "scorecard.csv")

    display = scorecard.astype(object)
    percentage_rows = [
        "Total return",
        "Annualized return (CAGR)",
        "Annualized volatility",
        "Maximum drawdown",
        "Daily VaR (95%)",
        "Daily expected shortfall (95%)",
        "Worst daily return",
    ]
    for row in percentage_rows:
        display.loc[row] = display.loc[row].map(lambda value: f"{value:.2%}")
    for row in display.index.difference(percentage_rows):
        display.loc[row] = display.loc[row].map(lambda value: f"{value:.2f}")
    display.to_csv(OUTPUT_DIR / "scorecard_formatted.csv")
    return scorecard


def plot_growth(common_wealth: pd.DataFrame) -> None:
    apply_ft_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"Bitcoin": BTC_COLOR, "S&P 500": SP500_COLOR}
    for asset in common_wealth:
        series = common_wealth[asset]
        ax.plot(series.index, series, color=colors[asset], linewidth=2.0)
        ax.annotate(
            f"{asset}  ${series.iloc[-1]:,.0f}",
            xy=(series.index[-1], series.iloc[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            color=colors[asset],
            va="center",
            fontweight="bold",
        )
    ax.set_yscale("log")
    ax.set_ylabel("Value of $1 (log scale)")
    ax.grid(axis="x", visible=False)
    add_header(
        fig,
        "Bitcoin created far more wealth, but the path was much rougher",
        "Growth of $1 on a shared daily timeline; the S&P 500 is carried flat on market-closed days.",
    )
    fig.savefig(OUTPUT_DIR / "01_growth_of_one_log.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_drawdowns(common_drawdown: pd.DataFrame) -> None:
    apply_ft_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"Bitcoin": BTC_COLOR, "S&P 500": SP500_COLOR}
    for asset in common_drawdown:
        ax.plot(
            common_drawdown.index,
            common_drawdown[asset] * 100,
            color=colors[asset],
            linewidth=1.8,
            label=asset,
        )
    ax.axhline(0, color=GREY, linewidth=0.8)
    ax.set_ylabel("Drawdown from previous peak (%)")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(axis="x", visible=False)
    add_header(
        fig,
        "Bitcoin's reward came with repeated, exceptionally deep drawdowns",
        "Percentage decline from each asset's previous wealth peak on the common sample.",
    )
    fig.savefig(OUTPUT_DIR / "02_drawdowns.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_rolling_correlation(common_returns: pd.DataFrame) -> None:
    rolling = common_returns["Bitcoin"].rolling(252).corr(common_returns["S&P 500"])
    apply_ft_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(rolling.index, rolling, color=TEAL, linewidth=1.8)
    ax.axhline(0, color=GREY, linewidth=0.8)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Rolling 252-observation correlation")
    ax.grid(axis="x", visible=False)
    add_header(
        fig,
        "Bitcoin-stock correlation is unstable rather than permanently high",
        "Rolling correlation uses only dates on which both return series are observed.",
    )
    fig.savefig(OUTPUT_DIR / "03_rolling_correlation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_distributions(btc_returns: pd.Series, sp_returns: pd.Series) -> None:
    apply_ft_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.8))
    for ax, asset, returns, color in [
        (axes[0], "Bitcoin", btc_returns, BTC_COLOR),
        (axes[1], "S&P 500", sp_returns, SP500_COLOR),
    ]:
        lower, upper = returns.quantile([0.005, 0.995])
        ax.hist(returns.clip(lower, upper) * 100, bins=60, color=color, alpha=0.82)
        ax.axvline(0, color=INK, linewidth=0.8)
        ax.set_title(asset, fontweight="bold")
        ax.set_xlabel("Daily return (%)")
        ax.set_ylabel("Number of observations")
        ax.grid(axis="x", visible=False)
    add_header(
        fig,
        "Bitcoin's daily return distribution is wider and more heavy-tailed",
        "The outer 0.5% on each side is clipped only for readability; tail metrics use all observations.",
    )
    fig.savefig(OUTPUT_DIR / "04_return_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_narrative(scorecard: pd.DataFrame, correlation: float) -> None:
    btc = scorecard["Bitcoin"]
    stocks = scorecard["S&P 500"]
    better = "Bitcoin" if btc["Sharpe ratio"] > stocks["Sharpe ratio"] else "S&P 500"
    content = f"""# Week 5 Practice: Bitcoin versus the S&P 500

## Executive conclusion

On the shared 2010-07-18 to 2025-12-31 sample, Bitcoin produced vastly more total wealth, but it also exposed investors to much higher volatility, heavier daily tail losses, and a far deeper maximum drawdown. On the full-sample Sharpe-ratio comparison, **{better}** delivered the stronger risk-adjusted result. This is a historical result, not a forecast.

## Evidence

- Bitcoin CAGR: **{btc['Annualized return (CAGR)']:.1%}**; S&P 500 CAGR: **{stocks['Annualized return (CAGR)']:.1%}**.
- Bitcoin annualized volatility: **{btc['Annualized volatility']:.1%}**; S&P 500: **{stocks['Annualized volatility']:.1%}**.
- Sharpe ratios: **{btc['Sharpe ratio']:.2f}** for Bitcoin and **{stocks['Sharpe ratio']:.2f}** for the S&P 500.
- Sortino ratios: **{btc['Sortino ratio']:.2f}** for Bitcoin and **{stocks['Sortino ratio']:.2f}** for the S&P 500.
- Maximum drawdowns: **{btc['Maximum drawdown']:.1%}** for Bitcoin and **{stocks['Maximum drawdown']:.1%}** for the S&P 500.
- Daily 95% Expected Shortfall: **{btc['Daily expected shortfall (95%)']:.1%}** for Bitcoin and **{stocks['Daily expected shortfall (95%)']:.1%}** for the S&P 500.
- Same-date daily-return correlation: **{correlation:.2f}**. The low-to-moderate average suggests diversification potential, but the rolling chart shows that correlation changes materially over time.

## Interpretation

Bitcoin won decisively on raw wealth creation, yet that statement alone hides the cost of the journey. Its much larger volatility and drawdowns mean that an investor needed both a high risk tolerance and the ability to remain invested through repeated crashes. The S&P 500 offered a smoother path and shallower tail losses. Which investment was 'better' therefore depends on the objective: maximum historical growth favoured Bitcoin; capital stability favoured the S&P 500; the Sharpe and Sortino ratios provide the explicit risk-adjusted comparison.

## Method and caveats

Returns are computed on each asset's native calendar before comparison. Bitcoin is annualized with 365 observations and the S&P 500 with 252. The business-day risk-free rate is forward-filled across Bitcoin weekends and holidays. For shared-timeline figures, S&P 500 wealth is carried flat on market-closed days. Bitcoin's 2010-2014 market was small and illiquid, so early returns may not represent an investable institutional experience. Results are sensitive to the starting date and should not be treated as evidence that past performance will repeat.
"""
    (OUTPUT_DIR / "ANALYSIS.md").write_text(content, encoding="utf-8")


def main() -> None:
    btc_price = load_series("btc_usd_daily.csv", "adjClose")
    sp_price = load_series("sp500_tr_index_daily.csv", "sp500_tr_index")
    rfr = load_series("french_daily_rfr.csv", "rfr")

    # Returns are formed on each native calendar before any alignment.
    btc_returns_full = btc_price.pct_change(fill_method=None).dropna()
    sp_returns_full = sp_price.pct_change(fill_method=None).dropna()
    sample_start = max(btc_price.index.min(), sp_price.index.min())
    sample_end = min(btc_price.index.max(), sp_price.index.max())
    btc_returns = btc_returns_full.loc[sample_start:sample_end]
    sp_returns = sp_returns_full.loc[sample_start:sample_end]

    btc_rfr = attach_risk_free(rfr, btc_returns.index)
    sp_rfr = attach_risk_free(rfr, sp_returns.index)
    metrics = {
        "Bitcoin": calculate_metrics(btc_returns, btc_rfr, 365),
        "S&P 500": calculate_metrics(sp_returns, sp_rfr, 252),
    }
    scorecard = save_scorecard(metrics)

    shared_returns = pd.concat(
        [btc_returns.rename("Bitcoin"), sp_returns.rename("S&P 500")],
        axis=1,
        join="inner",
    ).dropna()
    correlation = float(shared_returns.corr().iloc[0, 1])
    pd.DataFrame(
        {
            "value": [sample_start.date(), sample_end.date(), len(btc_returns), len(sp_returns), correlation]
        },
        index=["sample_start", "sample_end", "bitcoin_observations", "sp500_observations", "same_date_correlation"],
    ).to_csv(OUTPUT_DIR / "calendar_alignment.csv")

    annual_returns = pd.DataFrame(
        {
            "Bitcoin": btc_returns.groupby(btc_returns.index.year).apply(lambda x: (1.0 + x).prod() - 1.0),
            "S&P 500": sp_returns.groupby(sp_returns.index.year).apply(lambda x: (1.0 + x).prod() - 1.0),
        }
    )
    annual_returns.index.name = "year"
    annual_returns.to_csv(OUTPUT_DIR / "annual_returns.csv")

    daily_index = pd.date_range(sample_start, sample_end, freq="D")
    common_wealth = pd.DataFrame(
        {
            "Bitcoin": wealth_with_start(btc_returns, sample_start, sample_end).reindex(daily_index).ffill(),
            "S&P 500": wealth_with_start(sp_returns, sample_start, sample_end).reindex(daily_index).ffill(),
        }
    )
    common_drawdown = common_wealth / common_wealth.cummax() - 1.0
    common_wealth.to_csv(OUTPUT_DIR / "common_timeline_wealth.csv", index_label="date")
    common_drawdown.to_csv(OUTPUT_DIR / "common_timeline_drawdown.csv", index_label="date")

    plot_growth(common_wealth)
    plot_drawdowns(common_drawdown)
    plot_rolling_correlation(shared_returns)
    plot_distributions(btc_returns, sp_returns)
    write_narrative(scorecard, correlation)

    print("Week 5 crypto-versus-stocks analysis complete.")
    print(scorecard.loc[["Annualized return (CAGR)", "Annualized volatility", "Sharpe ratio", "Maximum drawdown"]].round(3))
    print(f"Same-date daily-return correlation: {correlation:.3f}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
