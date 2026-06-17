# ruff: noqa
"""Slice the clean Tech 4 panel into multiple Week 1 data shapes."""

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
COVID_CRASH_DATE = pd.Timestamp("2020-03-16")


def find_parquet() -> Path:
    """Prefer the generated clean parquet, then fall back to the committed one."""

    for candidate in CANDIDATE_PARQUET_PATHS:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find a clean Week 1 workshop parquet.")


def panel_to_timeseries(panel: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Filter the panel to one ticker: the time-series shape."""

    return panel[panel["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)


def panel_to_cross_section(panel: pd.DataFrame, trading_date: pd.Timestamp) -> pd.DataFrame:
    """Filter the panel to one date: the cross-section shape."""

    return panel[panel["Date"] == trading_date].reset_index(drop=True)


def panel_to_wide_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Pivot to one column per ticker of total return: the wide shape."""

    return panel.pivot(index="Date", columns="Ticker", values="TotalReturn").dropna()


def classify_shape(frame: pd.DataFrame, label: str) -> None:
    """Print the structural shape of a frame."""

    n_rows, n_cols = frame.shape
    if "Ticker" in frame.columns and "Date" in frame.columns:
        n_tickers = frame["Ticker"].nunique()
        n_dates = frame["Date"].nunique()
        shape = (
            "panel"
            if n_tickers > 1 and n_dates > 1
            else "time-series"
            if n_tickers == 1 and n_dates > 1
            else "cross-section"
            if n_dates == 1 and n_tickers > 1
            else "single observation"
        )
        print(f"  {label:14s} rows={n_rows:>5}  cols={n_cols}  -> {shape}")
    else:
        print(f"  {label:14s} rows={n_rows:>5}  cols={n_cols}  -> wide (one col per ticker)")


def main() -> None:
    parquet_path = find_parquet()
    print(f"Loading {parquet_path}")
    panel = pd.read_parquet(parquet_path)
    print()
    print("Original panel:")
    classify_shape(panel, "panel")

    aapl_timeseries = panel_to_timeseries(panel, "AAPL")
    cross_section = panel_to_cross_section(panel, COVID_CRASH_DATE)
    wide_return = panel_to_wide_returns(panel)

    print()
    print("Slices:")
    classify_shape(aapl_timeseries, "AAPL TS")
    classify_shape(cross_section, "2020-03-16 XS")
    classify_shape(wide_return, "Wide returns")

    print()
    print("Cross-section preview (2020-03-16):")
    print(cross_section[["Ticker", "Price", "TotalReturn"]].to_string(index=False))


if __name__ == "__main__":
    main()
