# ruff: noqa
"""Reproduce Part A results. Run from the project root:

    python scripts/run_part_a.py
"""
from __future__ import annotations

import pathlib
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import data_access, etl, features  # noqa: E402


ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
DATA = RESULTS / "data"
REPORT = ROOT / "report"

SIGNAL_WORDS = {
    "beat", "beats", "bullish", "buyback", "decline", "downgrade", "drop",
    "fall", "gain", "growth", "higher", "jump", "lawsuit", "loss", "miss",
    "profit", "rally", "recession", "record", "risk", "rise", "strong",
    "surge", "upgrade", "weak",
}
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")


def _ensure_dirs() -> None:
    for path in [FIGURES, TABLES, DATA, REPORT]:
        path.mkdir(parents=True, exist_ok=True)


def _savefig(path: pathlib.Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def descriptive_stats(long_returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for asset_class, group in long_returns.groupby("asset_class"):
        r = group["return"].dropna()
        rows.append(
            {
                "asset_class": asset_class,
                "observations": int(len(r)),
                "tickers": int(group["ticker"].nunique()),
                "mean_daily_return": float(r.mean()),
                "median_daily_return": float(r.median()),
                "std_daily_return": float(r.std()),
                "min_daily_return": float(r.min()),
                "p01_daily_return": float(r.quantile(0.01)),
                "p99_daily_return": float(r.quantile(0.99)),
                "max_daily_return": float(r.max()),
            }
        )
    return pd.DataFrame(rows)


def text_summary(panel: pd.DataFrame) -> pd.DataFrame:
    def count_signal_words(texts: pd.Series) -> int:
        count = 0
        for text in texts.astype(str):
            count += sum(1 for token in TOKEN_RE.findall(text.lower()) if token in SIGNAL_WORDS)
        return count

    return (
        panel.groupby("sector", as_index=False)
        .agg(
            ticker_days=("ticker", "size"),
            articles=("article_count", "sum"),
            total_words=("word_count", "sum"),
            signal_word_count=("text", count_signal_words),
        )
        .sort_values("articles", ascending=False)
    )


def _plot_cumulative(eq_ret: pd.DataFrame, cr_ret: pd.DataFrame) -> None:
    plt.figure(figsize=(9.5, 5.3))
    eq_index = features.cumulative_equal_weight(eq_ret)
    cr_index = features.cumulative_equal_weight(cr_ret)
    plt.plot(eq_index.index, eq_index, label="Equity equal-weight", linewidth=2, color="#1f77b4")
    plt.plot(cr_index.index, cr_index, label="Crypto equal-weight", linewidth=2, color="#d62728")
    plt.title("Cumulative equal-weight return index")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    _savefig(FIGURES / "cumulative_returns.png")


def _plot_distribution(long_returns: pd.DataFrame) -> None:
    plt.figure(figsize=(9.5, 5.3))
    for asset_class, color in [("Equity", "#1f77b4"), ("Crypto", "#d62728")]:
        r = long_returns.loc[long_returns["asset_class"].eq(asset_class), "return"]
        plt.hist(r.clip(r.quantile(0.005), r.quantile(0.995)), bins=80, alpha=0.55, label=asset_class, color=color)
    plt.title("Daily return distributions, 0.5%-99.5% clipped for display")
    plt.xlabel("Daily return")
    plt.ylabel("Frequency")
    plt.legend()
    _savefig(FIGURES / "returns_distribution_outliers.png")


def _plot_text(panel: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    summary.head(10).plot.barh(x="sector", y="articles", ax=axes[0], color="#2ca02c", legend=False)
    axes[0].invert_yaxis()
    axes[0].set_title("Articles by sector")
    axes[0].set_xlabel("Articles")
    daily = panel.groupby("trading_date")["article_count"].sum()
    axes[1].plot(daily.index, daily.rolling(20, min_periods=1).mean(), color="#9467bd")
    axes[1].set_title("20-day average article count")
    axes[1].set_xlabel("Trading date")
    axes[1].set_ylabel("Articles")
    _savefig(FIGURES / "text_exploration.png")


def _write_report(inventory: pd.DataFrame, stats: pd.DataFrame, text_stats: pd.DataFrame) -> None:
    doc = SimpleDocTemplate(
        str(REPORT / "report.pdf"),
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
    )
    styles = getSampleStyleSheet()
    story = [Paragraph("FINS5545 Project Part A: Data Foundation", styles["Title"])]

    def h(text: str) -> None:
        story.append(Paragraph(text, styles["Heading2"]))

    def p(text: str) -> None:
        story.append(Paragraph(text, styles["BodyText"]))
        story.append(Spacer(1, 0.18 * cm))

    p("This Part A draft builds the data foundation for a systematic multi-asset fund app. The target user is an investor or analyst who wants transparent equity, crypto and news inputs before moving to portfolio construction and sentiment analytics in Part B.")
    h("Data sources and cleaning")
    p("The project loads all raw data through the provided data_access helper. Equity and crypto prices are sorted by ticker and date, duplicate ticker-date rows are removed, and only positive adjusted close observations through 2023-12-31 are retained. News is de-duplicated by ticker, date and title.")
    inv_rows = [["Dataset", "Rows", "Tickers", "Dates", "Duplicate keys"]]
    for _, row in inventory.iterrows():
        inv_rows.append([row["dataset"], f"{row['clean_rows']:,}", str(row["tickers"]), f"{row['start_date']} to {row['end_date']}", str(row["duplicate_keys_after_clean"])])
    inv_table = Table(inv_rows, repeatRows=1)
    inv_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
    story.append(inv_table)
    story.append(Spacer(1, 0.25 * cm))

    h("Return features")
    p("Returns are simple daily percentage returns computed within each ticker before cross-asset alignment. This prevents crypto weekend observations from contaminating equity trading-day calculations. Outliers are retained because they are real market moves and are important for risk measurement.")
    stat_rows = [["Asset", "Obs.", "Mean", "Std", "Min", "P99", "Max"]]
    for _, row in stats.iterrows():
        stat_rows.append([row["asset_class"], f"{row['observations']:,}", f"{row['mean_daily_return']:.3%}", f"{row['std_daily_return']:.3%}", f"{row['min_daily_return']:.2%}", f"{row['p99_daily_return']:.2%}", f"{row['max_daily_return']:.2%}"])
    stat_table = Table(stat_rows, repeatRows=1)
    stat_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#374151")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
    story.append(stat_table)
    story.append(Spacer(1, 0.25 * cm))
    for image, caption in [
        ("cumulative_returns.png", "Figure 1. Cumulative equal-weight return indices for equities and crypto."),
        ("returns_distribution_outliers.png", "Figure 2. Daily return distribution with display clipping; original outliers are retained in the data."),
    ]:
        story.append(Image(str(FIGURES / image), width=16.2 * cm, height=7.8 * cm))
        p(caption)

    h("Text panel")
    p("Headlines are assembled into a ticker-sector daily text panel. Weekend and non-trading-day headlines are mapped to the next equity trading day. Part A deliberately does not score sentiment; it only reports article counts, word counts and signal-word counts to describe coverage.")
    top_sector = text_stats.iloc[0]
    p(f"The largest news sector by article count is {top_sector['sector']} with {int(top_sector['articles']):,} articles. These coverage differences matter because Part B sentiment signals must account for sparse or uneven news flow.")
    story.append(Image(str(FIGURES / "text_exploration.png"), width=16.2 * cm, height=7.4 * cm))
    p("Figure 3. Sector article coverage and time-series news volume.")
    h("Next step")
    p("This foundation enables Part B to estimate walk-forward funds, compute lagged sentiment indices, and serve the precomputed outputs in a Streamlit dashboard without committing raw data.")
    doc.build(story)


def main() -> None:
    _ensure_dirs()
    raw_eq = data_access.load_equity_prices()
    raw_cr = data_access.load_crypto_prices()
    raw_news = data_access.load_news_headlines()
    eq = etl.load_clean_equities()
    cr = etl.load_clean_crypto()
    news = etl.load_clean_headlines()
    print("clean equities:", eq.shape, "crypto:", cr.shape, "headlines:", news.shape)

    inventory = pd.DataFrame(
        [
            etl.inventory_row("equity_prices", raw_eq, eq, etl.PRICE_KEY, "Equity"),
            etl.inventory_row("crypto_prices", raw_cr, cr, etl.PRICE_KEY, "Crypto"),
            etl.inventory_row("news_headlines", raw_news, news, etl.NEWS_KEY, "Equity news"),
        ]
    )
    inventory.to_csv(TABLES / "dataset_inventory.csv", index=False)

    eq_long = features.daily_returns(eq, wide=False).assign(asset_class="Equity")
    cr_long = features.daily_returns(cr, wide=False).assign(asset_class="Crypto")
    long_returns = pd.concat([eq_long, cr_long], ignore_index=True)
    stats = descriptive_stats(long_returns)
    stats.to_csv(TABLES / "descriptive_stats_returns.csv", index=False)

    integrity = inventory[["dataset", "raw_rows", "clean_rows", "rows_removed", "duplicate_keys_after_clean"]].copy()
    integrity.to_csv(TABLES / "data_integrity_summary.csv", index=False)

    eq_ret = features.daily_returns(eq)
    cr_ret = features.daily_returns(cr)
    panel = features.assemble_headline_panel(news, trading_calendar=eq_ret.index)
    tsummary = text_summary(panel)
    tsummary.to_csv(TABLES / "text_data_summary.csv", index=False)
    panel.head(5000).to_csv(DATA / "headline_panel_sample.csv", index=False)
    eq_long.head(5000).to_csv(DATA / "equity_returns_sample.csv", index=False)
    cr_long.head(5000).to_csv(DATA / "crypto_returns_sample.csv", index=False)

    _plot_cumulative(eq_ret, cr_ret)
    _plot_distribution(long_returns)
    _plot_text(panel, tsummary)
    _write_report(inventory, stats, tsummary)
    print("wrote Part A results to", RESULTS)


if __name__ == "__main__":
    main()
