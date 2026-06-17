# ruff: noqa
"""Beginner helpers for authenticated Tiingo equity pulls."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests import Response, Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

WEEK_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = WEEK_ROOT / "data"
DEFAULT_TIINGO_SMALL_FILE = DATA_DIR / "tiingo_intro_3.txt"
DEFAULT_TIINGO_FAMOUS_50_FILE = DATA_DIR / "tiingo_famous_50_pre2000.txt"
DEFAULT_TIINGO_SMALL_OUTPUT_DIR = WEEK_ROOT / "results" / "data" / "tiingo_small_panel"
DEFAULT_TIINGO_FULL_OUTPUT_DIR = WEEK_ROOT / "results" / "data" / "tiingo_famous_50"
TIINGO_BASE_URL = "https://api.tiingo.com/tiingo/daily"
PANEL_COLUMN_ORDER = [
    "ticker",
    "date",
    "close",
    "high",
    "low",
    "open",
    "volume",
    "adjClose",
    "adjHigh",
    "adjLow",
    "adjOpen",
    "adjVolume",
    "divCash",
    "splitFactor",
]


@dataclass(frozen=True)
class TiingoPullConfig:
    """Simple config used by the beginner scripts."""

    api_key: str
    tickers: tuple[str, ...]
    start_date: str
    output_dir: Path
    pause_seconds: float
    timeout_seconds: int
    skip_metadata: bool = False


def resolve_api_key(explicit_api_key: str | None = None) -> str:
    """Read the API key from the flag or the standard environment variable."""

    api_key = (explicit_api_key or os.environ.get("TIINGO_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit(
            "Missing Tiingo API key. Set TIINGO_API_KEY or pass --api-key "
            "before running this script."
        )
    return api_key


def load_tickers_from_file(path: Path) -> tuple[str, ...]:
    """Load newline-delimited tickers, skipping comments and blank lines."""

    tickers = tuple(
        line.strip().upper()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not tickers:
        raise SystemExit(f"No tickers found in {path}.")
    return tickers


def build_tiingo_session(api_key: str) -> Session:
    """Create one session with retries so students do not need to manage this manually."""

    session = requests.Session()
    session.headers.update({"Authorization": f"Token {api_key}"})

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _request_json(
    session: Session,
    url: str,
    *,
    timeout_seconds: int,
    params: dict[str, Any] | None = None,
) -> Any:
    response = session.get(url, params=params, timeout=timeout_seconds)
    _raise_for_status(response)
    return response.json()


def _raise_for_status(response: Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text.strip()
        if body:
            raise requests.HTTPError(f"{exc}. Response body: {body}") from exc
        raise


def fetch_tiingo_metadata(session: Session, ticker: str, *, timeout_seconds: int) -> dict[str, Any]:
    """Fetch the lightweight per-ticker metadata record."""

    payload = _request_json(
        session,
        f"{TIINGO_BASE_URL}/{ticker}",
        timeout_seconds=timeout_seconds,
    )
    return {
        "ticker": payload["ticker"],
        "name": payload.get("name"),
        "exchange": payload.get("exchangeCode"),
        "description": payload.get("description"),
        "metadata_start_date": payload.get("startDate"),
        "metadata_end_date": payload.get("endDate"),
    }


def parse_tiingo_prices(payload: list[dict[str, Any]], ticker: str) -> pd.DataFrame:
    """Convert one ticker's JSON payload into a clean long frame."""

    frame = pd.DataFrame(payload)
    if frame.empty:
        raise ValueError(f"Tiingo returned no rows for ticker {ticker}.")
    frame.insert(0, "ticker", ticker)
    frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.tz_localize(None)
    return frame.sort_values(["ticker", "date"]).reset_index(drop=True)


def fetch_tiingo_prices(
    session: Session,
    ticker: str,
    *,
    start_date: str,
    timeout_seconds: int,
) -> pd.DataFrame:
    """Fetch one ticker's daily price history from the requested start date."""

    payload = _request_json(
        session,
        f"{TIINGO_BASE_URL}/{ticker}/prices",
        timeout_seconds=timeout_seconds,
        params={"startDate": start_date, "resampleFreq": "daily"},
    )
    return parse_tiingo_prices(payload, ticker)


def pull_tiingo_panel(config: TiingoPullConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull a panel ticker by ticker and keep the output validated and sorted."""

    session = build_tiingo_session(config.api_key)
    price_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []

    for index, ticker in enumerate(config.tickers, start=1):
        action = "prices only" if config.skip_metadata else "metadata and prices"
        print(f"[{index}/{len(config.tickers)}] Pulling {action} for {ticker}...")
        if not config.skip_metadata:
            metadata_rows.append(
                fetch_tiingo_metadata(session, ticker, timeout_seconds=config.timeout_seconds)
            )
        price_frames.append(
            fetch_tiingo_prices(
                session,
                ticker,
                start_date=config.start_date,
                timeout_seconds=config.timeout_seconds,
            )
        )
        time.sleep(config.pause_seconds)

    panel = pd.concat(price_frames, ignore_index=True)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel = panel[PANEL_COLUMN_ORDER]
    if panel[["ticker", "date"]].duplicated().any():
        duplicates = panel.loc[panel[["ticker", "date"]].duplicated(), ["ticker", "date"]]
        raise ValueError(f"Duplicate panel keys detected:\n{duplicates.head()}")

    if metadata_rows:
        metadata = pd.DataFrame(metadata_rows).sort_values("ticker").reset_index(drop=True)
    else:
        metadata = pd.DataFrame(
            {
                "ticker": list(config.tickers),
                "name": [None] * len(config.tickers),
                "exchange": [None] * len(config.tickers),
                "description": [None] * len(config.tickers),
                "metadata_start_date": [None] * len(config.tickers),
                "metadata_end_date": [None] * len(config.tickers),
            }
        )
    return panel, metadata


def build_coverage_summary(
    panel: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    requested_start_date: str,
) -> pd.DataFrame:
    """Summarize how much history each ticker actually delivered."""

    coverage = (
        panel.groupby("ticker", as_index=False)
        .agg(
            panel_start_date=("date", "min"),
            panel_end_date=("date", "max"),
            row_count=("date", "size"),
        )
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    coverage["panel_start_date"] = coverage["panel_start_date"].dt.date
    coverage["panel_end_date"] = coverage["panel_end_date"].dt.date
    requested_start = pd.to_datetime(requested_start_date).date()
    effective_start = pd.bdate_range(requested_start, periods=1)[0].date()
    coverage["requested_start_date"] = requested_start
    coverage["effective_start_date"] = effective_start
    coverage["covers_requested_start"] = (
        coverage["panel_start_date"] <= coverage["effective_start_date"]
    )

    metadata_copy = metadata.copy()
    metadata_copy["metadata_start_date"] = pd.to_datetime(
        metadata_copy["metadata_start_date"], errors="coerce"
    ).dt.date
    metadata_copy["metadata_end_date"] = pd.to_datetime(
        metadata_copy["metadata_end_date"], errors="coerce"
    ).dt.date

    return coverage.merge(
        metadata_copy[["ticker", "name", "exchange", "metadata_start_date", "metadata_end_date"]],
        on="ticker",
        how="left",
    )


def build_wide_price_table(panel: pd.DataFrame, *, value_column: str = "close") -> pd.DataFrame:
    """Pivot a long panel into the wide table students usually expect in spreadsheets."""

    if value_column not in panel.columns:
        raise ValueError(f"{value_column!r} is not a column in the long panel.")
    wide = panel.pivot(index="date", columns="ticker", values=value_column).sort_index()
    wide.columns.name = None
    return wide.reset_index()


def write_tiingo_outputs(
    *,
    panel: pd.DataFrame,
    metadata: pd.DataFrame,
    output_dir: Path,
    requested_start_date: str,
) -> dict[str, Path]:
    """Write the long panel, metadata, and coverage summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    panel_csv = output_dir / "tiingo_eod_panel_long.csv"
    panel_parquet = output_dir / "tiingo_eod_panel_long.parquet"
    metadata_csv = output_dir / "tiingo_eod_metadata.csv"
    coverage_csv = output_dir / "tiingo_eod_coverage_summary.csv"

    coverage = build_coverage_summary(
        panel,
        metadata,
        requested_start_date=requested_start_date,
    )

    panel.to_csv(panel_csv, index=False)
    panel.to_parquet(panel_parquet, index=False)
    metadata.to_csv(metadata_csv, index=False)
    coverage.to_csv(coverage_csv, index=False)

    return {
        "panel_csv": panel_csv,
        "panel_parquet": panel_parquet,
        "metadata_csv": metadata_csv,
        "coverage_csv": coverage_csv,
    }


def write_tiingo_wide_outputs(
    wide: pd.DataFrame,
    *,
    output_dir: Path,
    stem: str = "tiingo_close_wide",
) -> dict[str, Path]:
    """Write the derived wide-format price table."""

    output_dir.mkdir(parents=True, exist_ok=True)
    wide_csv = output_dir / f"{stem}.csv"
    wide_parquet = output_dir / f"{stem}.parquet"
    wide.to_csv(wide_csv, index=False)
    wide.to_parquet(wide_parquet, index=False)
    return {"wide_csv": wide_csv, "wide_parquet": wide_parquet}
