# ruff: noqa
"""Query the clean Tech 4 panel with DuckDB and pandas."""

from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

if SCRIPT_DIR.name == "scripts":
    DATA_DIR = SCRIPT_DIR.parent / "data"
    RESULTS_ROOT = SCRIPT_DIR.parent / "results"
else:
    DATA_DIR = SCRIPT_DIR.parent / "data"
    RESULTS_ROOT = SCRIPT_DIR / "results"

CANDIDATE_PARQUET_PATHS = [
    RESULTS_ROOT / "data" / "week1_workshop_panel_clean.parquet",
    DATA_DIR / "week1_workshop_panel.parquet",
]


def find_parquet() -> Path:
    """Prefer the generated clean parquet, then fall back to the committed one."""

    for candidate in CANDIDATE_PARQUET_PATHS:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find a clean Week 1 workshop parquet.")


def require_duckdb():
    """Import duckdb with a student-facing error if it is unavailable."""

    try:
        import duckdb
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "DuckDB is required for this script. Install the repo requirements and rerun."
        ) from exc
    return duckdb


def per_ticker_summary_duckdb(parquet_path: Path) -> pd.DataFrame:
    """Run SQL on the Parquet file directly."""

    duckdb = require_duckdb()
    query = f"""
        SELECT Ticker,
               AVG(Price)        AS mean_price,
               STDDEV_POP(Price) AS sd_price,
               MIN(Price)        AS min_price,
               MAX(Price)        AS max_price,
               COUNT(*)          AS n_days
        FROM   '{parquet_path.as_posix()}'
        GROUP  BY Ticker
        ORDER  BY mean_price DESC
    """
    return duckdb.sql(query).df()


def per_ticker_summary_pandas(panel: pd.DataFrame) -> pd.DataFrame:
    """Build the same summary using pandas groupby."""

    return (
        panel.groupby("Ticker")["Price"]
        .agg(
            mean_price="mean",
            sd_price=lambda s: s.std(ddof=0),
            min_price="min",
            max_price="max",
            n_days="size",
        )
        .sort_values("mean_price", ascending=False)
        .reset_index()
    )


def covid_crash_cross_section(parquet_path: Path) -> pd.DataFrame:
    """Return the Tech 4 cross-section for the COVID-crash Monday."""

    duckdb = require_duckdb()
    query = f"""
        SELECT Ticker, Price, TotalReturn
        FROM   '{parquet_path.as_posix()}'
        WHERE  Date = '2020-03-16'
        ORDER  BY TotalReturn ASC
    """
    return duckdb.sql(query).df()


def main() -> None:
    parquet_path = find_parquet()
    print(f"Querying {parquet_path}")
    print()

    ranks_duckdb = per_ticker_summary_duckdb(parquet_path)
    print("Per-ticker summary via DuckDB:")
    print(ranks_duckdb.to_string(index=False))
    print()

    panel = pd.read_parquet(parquet_path)
    ranks_pandas = per_ticker_summary_pandas(panel)
    print("Per-ticker summary via pandas:")
    print(ranks_pandas.to_string(index=False))

    diff = (
        ranks_duckdb.set_index("Ticker")[["mean_price", "sd_price", "n_days"]]
        - ranks_pandas.set_index("Ticker")[["mean_price", "sd_price", "n_days"]]
    ).abs().max().max()
    print()
    print(f"Max absolute diff between DuckDB and pandas results: {diff:.2e}")

    if not (ranks_duckdb["n_days"] == ranks_duckdb["n_days"].iloc[0]).all():
        print()
        print("WARNING: n_days uneven across tickers. Re-check the Stage 1 dedup step.")

    print()
    print("Cross-section on the COVID-crash Monday (2020-03-16):")
    crash = covid_crash_cross_section(parquet_path)
    print(crash.to_string(index=False))


if __name__ == "__main__":
    main()
