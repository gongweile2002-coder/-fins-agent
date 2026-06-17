# ruff: noqa
"""Clean the Tech 4 workshop CSV and save typed copies under results/data."""

from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

if SCRIPT_DIR.name == "scripts":
    DATA_DIR = SCRIPT_DIR.parent / "data"
    RESULTS_ROOT = SCRIPT_DIR.parent / "results"
else:
    DATA_DIR = SCRIPT_DIR.parent / "data"
    RESULTS_ROOT = SCRIPT_DIR / "results"

RAW_CSV = DATA_DIR / "week1_workshop_panel.csv"
RESULTS_DATA_DIR = RESULTS_ROOT / "data"


def load_raw(csv_path: Path) -> pd.DataFrame:
    """Load the raw workshop CSV."""

    return pd.read_csv(csv_path, parse_dates=["Date"], dayfirst=True)


def dedupe_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate `(Date, Ticker)` rows before downstream work."""

    n_dup = int(panel.duplicated(subset=["Date", "Ticker"]).sum())
    print(f"Duplicate (Date, Ticker) rows in raw CSV: {n_dup}")
    if n_dup:
        offenders = panel[panel.duplicated(["Date", "Ticker"], keep=False)]
        offenders = offenders.sort_values(["Ticker", "Date"])
        print("Offending rows:")
        print(offenders.to_string(index=False))
        clean = panel.drop_duplicates(subset=["Date", "Ticker"])
        clean = clean.sort_values(["Ticker", "Date"]).reset_index(drop=True)
        print(f"Dropped {n_dup} rows. New shape: {clean.shape}")
        return clean
    return panel


def write_roundtrip(clean_panel: pd.DataFrame) -> tuple[Path, Path]:
    """Write clean CSV and Parquet copies for analytical reuse."""

    RESULTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = RESULTS_DATA_DIR / "week1_workshop_panel_clean.parquet"
    csv_clean_path = RESULTS_DATA_DIR / "week1_workshop_panel_clean.csv"
    clean_panel.to_parquet(parquet_path, index=False)
    clean_panel.to_csv(csv_clean_path, index=False)
    return csv_clean_path, parquet_path


def report_sizes(csv_path: Path, parquet_path: Path) -> None:
    """Compare file sizes for the clean CSV and Parquet copies."""

    csv_bytes = csv_path.stat().st_size
    parquet_bytes = parquet_path.stat().st_size
    print()
    print(f"CSV    : {csv_bytes:>10,} bytes  ({csv_path.name})")
    print(f"Parquet: {parquet_bytes:>10,} bytes  ({parquet_path.name})")
    print(f"Parquet is {csv_bytes / parquet_bytes:.1f}x smaller for this panel")


def report_dtype_preservation(csv_path: Path, parquet_path: Path) -> None:
    """Show why Parquet is the better analytical storage format."""

    csv_reread = pd.read_csv(csv_path)
    parquet_reread = pd.read_parquet(parquet_path)
    print()
    print("Dtype preservation through round-trip (Date column):")
    print(f"  CSV    : {csv_reread['Date'].dtype}   (becomes object / string)")
    print(f"  Parquet: {parquet_reread['Date'].dtype}    (stays datetime64[ns])")


def main() -> None:
    print(f"Loading {RAW_CSV}")
    panel = load_raw(RAW_CSV)
    clean_panel = dedupe_panel(panel)
    csv_clean_path, parquet_path = write_roundtrip(clean_panel)
    print(f"Wrote clean parquet: {parquet_path}")
    report_sizes(csv_clean_path, parquet_path)
    report_dtype_preservation(csv_clean_path, parquet_path)


if __name__ == "__main__":
    main()
