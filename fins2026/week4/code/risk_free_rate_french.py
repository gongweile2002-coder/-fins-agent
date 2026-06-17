# ruff: noqa
"""Helpers for the Week 4 Kenneth French daily risk-free download."""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

WEEK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRENCH_RFR_OUTPUT_PATH = (
    WEEK_ROOT / "results" / "data" / "french_daily_rfr" / "french_daily_rfr.parquet"
)
FRENCH_RFR_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)


def fetch_french_rfr_zip(url: str = FRENCH_RFR_URL, *, timeout_seconds: int = 60) -> bytes:
    """Download the zipped Kenneth French daily factors file."""

    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    return response.content


def extract_first_csv_text(zip_bytes: bytes) -> str:
    """Read the first CSV member from the downloaded zip archive."""

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_members:
            raise ValueError("Kenneth French zip file did not contain a CSV member.")
        with archive.open(csv_members[0]) as handle:
            return handle.read().decode("utf-8-sig")


def parse_french_daily_rfr(csv_text: str) -> pd.DataFrame:
    """Keep only the daily RF series and scale it from percent to decimal."""

    rows: list[tuple[pd.Timestamp, float]] = []
    reader = csv.reader(io.StringIO(csv_text))
    for row in reader:
        if not row:
            continue
        date_token = row[0].strip()
        if len(date_token) != 8 or not date_token.isdigit():
            continue
        if len(row) < 5:
            continue
        rf_text = row[-1].strip()
        if not rf_text:
            continue
        rows.append(
            (
                pd.to_datetime(date_token, format="%Y%m%d"),
                float(rf_text) / 100.0,
            )
        )

    if not rows:
        raise ValueError("No daily Kenneth French RF rows were found in the downloaded file.")

    frame = pd.DataFrame(rows, columns=["date", "rfr"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame["rfr"] = frame["rfr"].astype(float)
    return frame.sort_values("date").reset_index(drop=True)


def write_french_rfr_parquet(frame: pd.DataFrame, output_path: Path) -> Path:
    """Write only the final Parquet file required for Week 4."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return output_path
