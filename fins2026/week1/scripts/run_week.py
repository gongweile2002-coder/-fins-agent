# ruff: noqa
"""Print the Week 1 run order and confirm output directories."""

from __future__ import annotations

from pathlib import Path

from describe_data import describe_week_data

WEEK_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIRS = [
    WEEK_ROOT / "results" / "data",
    WEEK_ROOT / "results" / "figures",
    WEEK_ROOT / "results" / "tables",
    WEEK_ROOT / "results" / "app",
]
SCRIPT_ORDER = [
    "01_load_panel.py",
    "02_save_formats.py",
    "03_duckdb_query.py",
    "04_panel_slices.py",
    "05_end_to_end.py",
    "06_coke_pepsi_practice.py",
]


def main() -> None:
    """Print the Week 1 inventory and canonical script order."""

    for directory in RESULTS_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
    print("Week 1: Structured Data Foundations")
    print()
    print(describe_week_data())
    print()
    print("Canonical run order:")
    for script_name in SCRIPT_ORDER:
        print(f"- python fins2026/week1/scripts/{script_name}")
    print()
    print("Key checkpoints:")
    print("- Tech 4 raw CSV has 3 duplicate (Date, Ticker) rows")
    print("- Tech 4 clean panel is balanced at 6,539 rows per ticker")
    print("- KO/PEP practice panel has 0 duplicates, 0 nulls, and 6,539 rows each")
    print("- Write derived datasets to results/data and figures to results/figures")
    print("- Refresh guidance after major Week 1 changes")


if __name__ == "__main__":
    main()
