# ruff: noqa
"""Load the raw Week 1 Tech 4 CSV and verify the date parse."""

from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

if SCRIPT_DIR.name == "scripts":
    DATA_DIR = SCRIPT_DIR.parent / "data"
else:
    DATA_DIR = SCRIPT_DIR.parent / "data"

RAW_CSV = DATA_DIR / "week1_workshop_panel.csv"
EXPECTED_TICKERS = {"AAPL", "MSFT", "NVDA", "ORCL"}
EXPECTED_COLUMNS = [
    "Date",
    "CompanyName",
    "Ticker",
    "Price",
    "TotalReturn",
    "PriceReturn",
]


def load_panel(csv_path: Path) -> pd.DataFrame:
    """Read the raw CSV with the correct day-first date parser."""

    panel = pd.read_csv(csv_path, parse_dates=["Date"], dayfirst=True)
    panel["Ticker"] = panel["Ticker"].astype("string")
    panel["CompanyName"] = panel["CompanyName"].astype("string")
    return panel


def report_panel_health(panel: pd.DataFrame) -> None:
    """Print the key Week 1 load checks."""

    print(f"Loaded panel: {panel.shape[0]:,} rows x {panel.shape[1]} columns")
    print()
    print("Dtypes:")
    print(panel.dtypes.to_string())
    print()

    tickers_present = set(panel["Ticker"].unique())
    missing = EXPECTED_TICKERS - tickers_present
    extra = tickers_present - EXPECTED_TICKERS
    if missing or extra:
        print(f"WARNING: ticker mismatch (missing={missing}, extra={extra})")
    else:
        print(f"All expected tickers present: {sorted(tickers_present)}")

    rows_per_ticker = panel.groupby("Ticker").size().sort_values()
    print()
    print("Rows per ticker:")
    print(rows_per_ticker.to_string())
    print()
    print(f"Date range: {panel['Date'].min().date()} to {panel['Date'].max().date()}")


def main() -> None:
    print(f"Loading {RAW_CSV}")
    panel = load_panel(RAW_CSV)
    expected_set = set(EXPECTED_COLUMNS)
    actual_set = set(panel.columns)
    if expected_set != actual_set:
        raise ValueError(
            f"Column mismatch: expected {sorted(expected_set)}, got {sorted(actual_set)}"
        )
    report_panel_health(panel)


if __name__ == "__main__":
    main()
