# ruff: noqa
"""Run the Week 1 KO vs PEP practice workflow end to end."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent

if SCRIPT_DIR.name == "scripts":
    DATA_DIR = SCRIPT_DIR.parent / "data"
    RESULTS_ROOT = SCRIPT_DIR.parent / "results"
else:
    DATA_DIR = SCRIPT_DIR.parent / "data"
    RESULTS_ROOT = SCRIPT_DIR / "results"

SOURCE_TSV = DATA_DIR / "week1_assignment_data.txt"
RESULTS_DATA_DIR = RESULTS_ROOT / "data"
RESULTS_FIGURES_DIR = RESULTS_ROOT / "figures"
PARQUET_OUTPUT = RESULTS_DATA_DIR / "week1_assignment_data.parquet"
FIGURE_OUTPUT = RESULTS_FIGURES_DIR / "ko_vs_pep_growth.pdf"
COVID_CRASH_DATE = pd.Timestamp("2020-03-16")


def require_duckdb():
    """Import duckdb with a student-facing error if it is unavailable."""

    try:
        import duckdb
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "DuckDB is required for this script. Install the repo requirements and rerun."
        ) from exc
    return duckdb


def load_assignment_panel() -> pd.DataFrame:
    """Load the KO/PEP TSV and add a typed `Date` column."""

    panel = pd.read_csv(SOURCE_TSV, sep="\t")
    panel["Date"] = pd.to_datetime(panel["DlyCalDt"], format="%Y%m%d")
    return panel


def task_1_load_and_inspect(panel: pd.DataFrame) -> None:
    """Print the Week 1 load checks and save the typed parquet copy."""

    RESULTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PARQUET_OUTPUT, index=False)

    print("Task 1 - Load and inspect")
    print(f"Shape: {panel.shape}")
    print(f"Date range: {panel['Date'].min().date()} to {panel['Date'].max().date()}")
    print("Tickers and security names:")
    print(panel[["Ticker", "SecurityNm"]].drop_duplicates().to_string(index=False))
    print()
    print("Dtypes:")
    print(panel.dtypes.to_string())
    print()
    print(f"Wrote typed copy: {PARQUET_OUTPUT}")


def task_2_stage_1(panel: pd.DataFrame) -> None:
    """Run the three standard Stage 1 checks."""

    duplicates = int(panel.duplicated(["Date", "Ticker"]).sum())
    nulls = panel.isna().sum()
    counts = panel.groupby("Ticker").size()

    print()
    print("Task 2 - Stage 1 sanity check")
    print(f"Duplicate (Date, Ticker) rows: {duplicates}")
    print("Nulls per column:")
    print(nulls.to_string())
    print()
    print("Rows per ticker:")
    print(counts.to_string())
    print()
    print("Closing sentence:")
    if duplicates == 0 and int(nulls.sum()) == 0 and (counts == counts.iloc[0]).all():
        print("The panel ships ready to use after date conversion.")
    else:
        print("The panel needs cleaning before downstream analysis.")


def task_3_summaries(panel: pd.DataFrame) -> None:
    """Compare the DuckDB and pandas per-ticker summaries."""

    duckdb = require_duckdb()

    duck = duckdb.sql(f"""
        SELECT Ticker,
               AVG(DlyPrc)          AS mean_price,
               STDDEV_POP(DlyPrc)   AS sd_price,
               AVG(DlyCap) / 1000.0 AS mean_cap_million_usd,
               COUNT(*)             AS n_days
        FROM '{PARQUET_OUTPUT.as_posix()}'
        GROUP BY Ticker
        ORDER BY mean_price DESC
    """).df()

    pandas_summary = (
        panel.assign(cap_million_usd=panel["DlyCap"] / 1000.0)
        .groupby("Ticker")
        .agg(
            mean_price=("DlyPrc", "mean"),
            sd_price=("DlyPrc", lambda s: s.std(ddof=0)),
            mean_cap_million_usd=("cap_million_usd", "mean"),
            n_days=("DlyPrc", "size"),
        )
        .sort_values("mean_price", ascending=False)
        .reset_index()
    )

    max_diff = (
        duck.set_index("Ticker")[
            ["mean_price", "sd_price", "mean_cap_million_usd", "n_days"]
        ]
        - pandas_summary.set_index("Ticker")[
            ["mean_price", "sd_price", "mean_cap_million_usd", "n_days"]
        ]
    ).abs().max().max()

    print()
    print("Task 3 - Per-ticker summary, two ways")
    print("DuckDB:")
    print(duck.to_string(index=False))
    print()
    print("pandas:")
    print(pandas_summary.to_string(index=False))
    print()
    print(f"Max absolute difference: {max_diff:.2e}")


def task_4_slices(panel: pd.DataFrame) -> None:
    """Print the KO/PEP cross-section and KO time-series slice."""

    crash = panel[panel["Date"] == COVID_CRASH_DATE][
        ["Date", "Ticker", "DlyPrc", "DlyRet"]
    ].sort_values("DlyRet")
    ko = panel[panel["Ticker"] == "KO"].sort_values("Date")
    winner = crash.loc[crash["DlyRet"].idxmax()]
    loser = crash.loc[crash["DlyRet"].idxmin()]
    gap_pct_points = 100.0 * (winner["DlyRet"] - loser["DlyRet"])

    print()
    print("Task 4 - Cross-section and time-series slices")
    print("Cross-section on 2020-03-16:")
    print(crash.to_string(index=False))
    print()
    print(
        f"{winner['Ticker']} held up better by {gap_pct_points:.2f} percentage points "
        f"relative to {loser['Ticker']}."
    )
    print()
    print(f"KO time-series shape: {ko.shape}")
    print("First two KO rows:")
    print(ko.head(2).to_string(index=False))
    print()
    print("Last two KO rows:")
    print(ko.tail(2).to_string(index=False))


def task_5_growth_of_one_dollar(panel: pd.DataFrame) -> None:
    """Compute and plot the KO vs PEP growth-of-$1 paths."""

    import matplotlib.pyplot as plt

    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    wide = panel.pivot(index="Date", columns="Ticker", values="DlyRet").dropna()
    growth = (1 + wide).cumprod()
    ending_values = growth.tail(1).round(4)

    winner = ending_values.iloc[0].idxmax()
    factor = float(ending_values.iloc[0].max() / ending_values.iloc[0].min())

    ax = growth.plot(
        title="KO vs PEP growth of $1, 2000-2025",
        logy=True,
        ylabel="value of $1 invested (log scale, USD)",
        figsize=(8, 4.5),
    )
    ax.figure.tight_layout()
    ax.figure.savefig(FIGURE_OUTPUT)
    plt.close(ax.figure)

    print()
    print("Task 5 - KO vs PEP, growth of $1")
    print("Final dollar values:")
    print(ending_values.to_string())
    print()
    print(f"Winner: {winner}")
    print(f"Winner factor: {factor:.2f}x")
    print(
        "Finance interpretation: KO and PEP are a sensible comparison because they are "
        "both mature consumer-staples firms with similar investor use cases, unlike a "
        "cross-sector comparison such as KO versus AAPL."
    )
    print(f"Wrote figure: {FIGURE_OUTPUT}")


def main() -> None:
    """Run the five Week 1 practice tasks in order."""

    panel = load_assignment_panel()
    task_1_load_and_inspect(panel)
    task_2_stage_1(panel)
    task_3_summaries(panel)
    task_4_slices(panel)
    task_5_growth_of_one_dollar(panel)


if __name__ == "__main__":
    main()
