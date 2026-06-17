# ruff: noqa
"""Describe the Week 1 workshop and practice datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

WEEK_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = WEEK_ROOT / "data"
RESULTS_DATA_DIR = WEEK_ROOT / "results" / "data"
WORKSHOP_CSV = DATA_DIR / "week1_workshop_panel.csv"
WORKSHOP_PARQUET = DATA_DIR / "week1_workshop_panel.parquet"
ASSIGNMENT_TSV = DATA_DIR / "week1_assignment_data.txt"


def visible_files(directory: Path) -> list[Path]:
    """Return non-placeholder files inside a directory tree."""

    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.name != ".gitkeep"
    )


def describe_directory(label: str, directory: Path) -> list[str]:
    """Return a short inventory for one Week 1 data directory."""

    files = visible_files(directory)
    lines = [f"{label}: {directory.relative_to(WEEK_ROOT).as_posix()}"]
    if not files:
        lines.append("- no files yet")
        return lines
    for path in files:
        rel = path.relative_to(WEEK_ROOT).as_posix()
        lines.append(f"- {rel} ({path.stat().st_size} bytes)")
    return lines


def summarize_workshop_panel() -> list[str]:
    """Return key facts for the Tech 4 workshop panel."""

    panel = pd.read_csv(WORKSHOP_CSV, parse_dates=["Date"], dayfirst=True)
    duplicates = int(panel.duplicated(["Date", "Ticker"]).sum())
    return [
        "Tech 4 workshop panel",
        f"- file: {WORKSHOP_CSV.relative_to(WEEK_ROOT).as_posix()}",
        f"- raw shape: {panel.shape[0]:,} rows x {panel.shape[1]} columns",
        f"- tickers: {', '.join(sorted(panel['Ticker'].unique()))}",
        f"- date range: {panel['Date'].min().date()} to {panel['Date'].max().date()}",
        f"- duplicate (Date, Ticker) rows in raw CSV: {duplicates}",
        (
            "- committed clean companion: "
            f"{WORKSHOP_PARQUET.relative_to(WEEK_ROOT).as_posix()}"
        ),
    ]


def summarize_assignment_panel() -> list[str]:
    """Return key facts for the KO/PEP practice panel."""

    panel = pd.read_csv(ASSIGNMENT_TSV, sep="\t")
    panel["Date"] = pd.to_datetime(panel["DlyCalDt"], format="%Y%m%d")
    duplicates = int(panel.duplicated(["Date", "Ticker"]).sum())
    counts = panel.groupby("Ticker").size().to_dict()
    return [
        "KO vs PEP practice panel",
        f"- file: {ASSIGNMENT_TSV.relative_to(WEEK_ROOT).as_posix()}",
        f"- raw shape: {panel.shape[0]:,} rows x 10 columns before adding Date",
        f"- tickers: {', '.join(sorted(panel['Ticker'].unique()))}",
        f"- date range: {panel['Date'].min().date()} to {panel['Date'].max().date()}",
        f"- duplicate (Date, Ticker) rows: {duplicates}",
        f"- null cells: {int(panel.isna().sum().sum())}",
        f"- rows per ticker: {counts}",
    ]


def describe_week_data() -> str:
    """Return a plain-text summary of source and generated datasets."""

    lines = ["Week data inventory", ""]
    lines.extend(summarize_workshop_panel())
    lines.append("")
    lines.extend(summarize_assignment_panel())
    lines.append("")
    lines.extend(describe_directory("Source data", DATA_DIR))
    lines.append("")
    lines.extend(describe_directory("Generated data", RESULTS_DATA_DIR))
    return "\n".join(lines)


def main() -> None:
    print(describe_week_data())


if __name__ == "__main__":
    main()
