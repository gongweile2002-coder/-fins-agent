# ruff: noqa
"""Beginner helpers for the free ECB API example."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

WEEK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ECB_OUTPUT_DIR = WEEK_ROOT / "results" / "data" / "ecb_api_intro"
ECB_BASE_URL = "https://data-api.ecb.europa.eu/service/data/EXR"


def build_ecb_url(
    *,
    base_currency: str = "USD",
    quote_currency: str = "EUR",
    frequency: str = "D",
    start_date: str = "2026-05-01",
    end_date: str = "2026-05-15",
) -> str:
    """Build the simple ECB exchange-rate endpoint used in class."""

    series_key = f"{frequency}.{base_currency.upper()}.{quote_currency.upper()}.SP00.A"
    query = f"startPeriod={start_date}&endPeriod={end_date}&format=csvdata"
    return f"{ECB_BASE_URL}/{series_key}?{query}"


def fetch_ecb_csv(url: str, *, timeout_seconds: int = 30) -> str:
    """Perform a single GET request and return the raw CSV response body."""

    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    return response.text


def tidy_ecb_exchange_rates(csv_text: str) -> pd.DataFrame:
    """Keep the teaching example focused on the handful of useful fields."""

    raw = pd.read_csv(StringIO(csv_text))
    tidy = raw.loc[
        :,
        ["TIME_PERIOD", "OBS_VALUE", "CURRENCY", "CURRENCY_DENOM", "TITLE"],
    ].copy()
    tidy = tidy.rename(
        columns={
            "TIME_PERIOD": "date",
            "OBS_VALUE": "exchange_rate",
            "CURRENCY": "base_currency",
            "CURRENCY_DENOM": "quote_currency",
            "TITLE": "series_title",
        }
    )
    tidy["date"] = pd.to_datetime(tidy["date"])
    return tidy.sort_values("date").reset_index(drop=True)


def write_ecb_outputs(output_dir: Path, csv_text: str, tidy: pd.DataFrame) -> dict[str, Path]:
    """Write both the raw response and the cleaned teaching table."""

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv_path = output_dir / "ecb_exchange_rates_raw.csv"
    tidy_csv_path = output_dir / "ecb_exchange_rates_tidy.csv"
    tidy_parquet_path = output_dir / "ecb_exchange_rates_tidy.parquet"

    raw_csv_path.write_text(csv_text, encoding="utf-8")
    tidy.to_csv(tidy_csv_path, index=False)
    tidy.to_parquet(tidy_parquet_path, index=False)

    return {
        "raw_csv": raw_csv_path,
        "tidy_csv": tidy_csv_path,
        "tidy_parquet": tidy_parquet_path,
    }
