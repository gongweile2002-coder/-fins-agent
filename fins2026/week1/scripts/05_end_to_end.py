# ruff: noqa
"""Run the full Week 1 Tech 4 pipeline in one file."""

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

RAW_CSV = DATA_DIR / "week1_workshop_panel.csv"
RESULTS_DATA_DIR = RESULTS_ROOT / "data"
RESULTS_FIGURES_DIR = RESULTS_ROOT / "figures"
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


def step_1_load(csv_path: Path) -> pd.DataFrame:
    """Load the raw CSV with the correct day-first date parser."""

    panel = pd.read_csv(csv_path, parse_dates=["Date"], dayfirst=True)
    print(f"Step 1 LOAD : shape={panel.shape}  Date dtype={panel['Date'].dtype}")
    return panel


def step_2_clean_and_save(panel: pd.DataFrame, parquet_path: Path) -> pd.DataFrame:
    """Drop duplicate rows and save the clean parquet."""

    n_dup = int(panel.duplicated(subset=["Date", "Ticker"]).sum())
    clean = panel.drop_duplicates(subset=["Date", "Ticker"])
    clean = clean.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(parquet_path, index=False)
    print(
        "Step 2 CLEAN: "
        f"dropped {n_dup} duplicate rows -> shape={clean.shape}; "
        f"wrote {parquet_path.name}"
    )
    return clean


def step_3_duckdb_summary(parquet_path: Path) -> pd.DataFrame:
    """Summarize the clean parquet with DuckDB."""

    duckdb = require_duckdb()
    ranks = duckdb.sql(f"""
        SELECT Ticker,
               AVG(Price)        AS mean_price,
               STDDEV_POP(Price) AS sd_price,
               COUNT(*)          AS n_days
        FROM   '{parquet_path.as_posix()}'
        GROUP  BY Ticker
        ORDER  BY mean_price DESC
    """).df()
    print("Step 3 QUERY: per-ticker price summary")
    print(ranks.to_string(index=False))
    return ranks


def step_4_slice(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice the panel into time-series and cross-section views."""

    aapl_timeseries = panel[panel["Ticker"] == "AAPL"].sort_values("Date")
    cross_section = panel[panel["Date"] == COVID_CRASH_DATE]
    print(
        "Step 4 SLICE: "
        f"AAPL TS shape={aapl_timeseries.shape}, "
        f"2020-03-16 XS shape={cross_section.shape}"
    )
    return aapl_timeseries, cross_section


def step_5_compound(panel: pd.DataFrame) -> pd.DataFrame:
    """Compound total returns into growth of $1."""

    wide_return = panel.pivot(index="Date", columns="Ticker", values="TotalReturn").dropna()
    growth_of_one_dollar = (1 + wide_return).cumprod()
    print("Step 5 COMPOUND: value of $1 at end-of-sample:")
    print(growth_of_one_dollar.tail(1).round(2).to_string())
    return growth_of_one_dollar


def step_6_plot(growth_of_one_dollar: pd.DataFrame, output_path: Path) -> None:
    """Write the growth-of-$1 chart to the Week 1 figures directory."""

    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ax = growth_of_one_dollar.plot(
        title="Tech 4 growth of $1, 2000-2025",
        logy=True,
        ylabel="value of $1 invested (log scale, USD)",
        figsize=(8, 4.5),
    )
    ax.figure.tight_layout()
    ax.figure.savefig(output_path)
    plt.close(ax.figure)
    print(f"Step 6 PLOT : wrote {output_path.name}")


def main() -> None:
    parquet_path = RESULTS_DATA_DIR / "week1_workshop_panel_clean.parquet"

    panel = step_1_load(RAW_CSV)
    clean_panel = step_2_clean_and_save(panel, parquet_path)
    step_3_duckdb_summary(parquet_path)
    step_4_slice(clean_panel)
    growth_of_one_dollar = step_5_compound(clean_panel)
    step_6_plot(growth_of_one_dollar, RESULTS_FIGURES_DIR / "tech4_growth.pdf")


if __name__ == "__main__":
    main()
