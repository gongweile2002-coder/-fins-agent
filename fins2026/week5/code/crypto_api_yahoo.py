# ruff: noqa
"""Week 5 Yahoo Finance crypto helpers."""

from __future__ import annotations

import datetime as dt
import json
import random
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
DEFAULT_YAHOO_CRYPTO_SMALL_FILE = DATA_DIR / "yahoo_crypto_intro_5.txt"
DEFAULT_YAHOO_CRYPTO_PANEL_FILE = DATA_DIR / "yahoo_crypto_20_since_2019.txt"
DEFAULT_YAHOO_CRYPTO_INTRO_OUTPUT_DIR = WEEK_ROOT / "results" / "data" / "yahoo_crypto_intro_5"
DEFAULT_YAHOO_CRYPTO_FULL_OUTPUT_DIR = WEEK_ROOT / "results" / "data" / "yahoo_crypto_20_since_2019"
YAHOO_BASE_URLS = (
    "https://query2.finance.yahoo.com/v8/finance/chart",
    "https://query1.finance.yahoo.com/v8/finance/chart",
)
YAHOO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}
YAHOO_TICKER_CACHE_DIRNAME = "_ticker_cache"
YAHOO_PANEL_COLUMN_ORDER = [
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "adjClose",
    "volume",
    "dividend",
    "splitFactor",
]


@dataclass(frozen=True)
class YahooCryptoPullConfig:
    """Simple config used by the Week 5 Yahoo crypto scripts."""

    tickers: tuple[str, ...]
    start_date: str
    end_date: str
    output_dir: Path
    pause_seconds: float
    timeout_seconds: int
    max_attempts: int
    backoff_seconds: float
    resume: bool = True


def load_yahoo_crypto_tickers_from_file(path: Path) -> tuple[str, ...]:
    """Load newline-delimited crypto tickers, skipping comments and blank lines."""

    tickers = tuple(
        line.strip().upper()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not tickers:
        raise SystemExit(f"No tickers found in {path}.")
    return tickers


def build_yahoo_session() -> Session:
    """Create a session with browser-style headers and light transport retries."""

    session = requests.Session()
    session.headers.update(YAHOO_HEADERS)
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=0,
        backoff_factor=0.5,
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def to_epoch_seconds(date_text: str) -> int:
    """Convert an ISO date string into the Unix timestamp Yahoo expects."""

    return int(dt.datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=dt.UTC).timestamp())


def compute_wait_seconds(
    retry_after: str | None,
    *,
    attempt: int,
    backoff_seconds: float,
) -> float:
    """Pick the next wait time for 429/5xx retries."""

    if retry_after:
        try:
            return max(float(retry_after), 0.0)
        except ValueError:
            pass

    jitter = random.uniform(0.0, 0.5)
    return backoff_seconds * (2 ** (attempt - 1)) + jitter


def raise_for_status(response: Response) -> None:
    """Raise a requests error with the response body when available."""

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        message = response.text.strip()
        if message:
            raise requests.HTTPError(f"{exc}. Response body: {message}") from exc
        raise


def request_yahoo_chart_json(
    session: Session,
    ticker: str,
    *,
    start_date: str,
    end_date: str,
    timeout_seconds: int,
    max_attempts: int,
    backoff_seconds: float,
) -> Any:
    """Pull one ticker from Yahoo's chart endpoint with host fallback and retries."""

    params = {
        "period1": to_epoch_seconds(start_date),
        "period2": to_epoch_seconds(end_date),
        "interval": "1d",
        "includeAdjustedClose": "true",
        "events": "div,splits",
    }
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        base_url = YAHOO_BASE_URLS[(attempt - 1) % len(YAHOO_BASE_URLS)]
        try:
            response = session.get(
                f"{base_url}/{ticker}",
                params=params,
                timeout=timeout_seconds,
            )
            if response.status_code in {429, 500, 502, 503, 504} and attempt < max_attempts:
                wait_seconds = compute_wait_seconds(
                    response.headers.get("Retry-After"),
                    attempt=attempt,
                    backoff_seconds=backoff_seconds,
                )
                print(
                    f"  Yahoo retry for {ticker}: status {response.status_code} on "
                    f"attempt {attempt}/{max_attempts}; sleeping {wait_seconds:.1f}s."
                )
                time.sleep(wait_seconds)
                continue

            raise_for_status(response)
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt >= max_attempts:
                break

            wait_seconds = compute_wait_seconds(
                None,
                attempt=attempt,
                backoff_seconds=backoff_seconds,
            )
            print(
                f"  Yahoo retry for {ticker}: {exc.__class__.__name__} on "
                f"attempt {attempt}/{max_attempts}; sleeping {wait_seconds:.1f}s."
            )
            time.sleep(wait_seconds)

    if last_error is not None:
        raise RuntimeError(
            f"Yahoo request failed for {ticker} after {max_attempts} attempts: {last_error}"
        )
    raise RuntimeError(f"Yahoo request failed for {ticker} after {max_attempts} attempts.")


def normalize_event_map(
    events: dict[str, Any] | None,
    *,
    timezone_name: str,
    field: str,
) -> dict[dt.date, Any]:
    """Convert Yahoo dividend events into a simple date-to-value mapping."""

    if not events:
        return {}

    timestamps = pd.to_datetime(
        [event["date"] for event in events.values()],
        unit="s",
        utc=True,
    ).tz_convert(timezone_name)
    values = [event[field] for event in events.values()]
    return dict(zip(timestamps.date, values))


def normalize_split_factors(
    events: dict[str, Any] | None,
    *,
    timezone_name: str,
) -> dict[dt.date, float]:
    """Convert Yahoo split events into date-to-split-factor mapping."""

    if not events:
        return {}

    timestamps = pd.to_datetime(
        [event["date"] for event in events.values()],
        unit="s",
        utc=True,
    ).tz_convert(timezone_name)
    values = [event["numerator"] / event["denominator"] for event in events.values()]
    return dict(zip(timestamps.date, values))


def normalize_yahoo_crypto_chart_payload(
    ticker: str,
    payload: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize one raw Yahoo crypto payload into a long panel and metadata row."""

    chart = payload.get("chart") or {}
    error = chart.get("error")
    if error:
        raise ValueError(f"Yahoo returned chart error for {ticker}: {error}")

    result = ((chart.get("result") or [None])[0]) or {}
    meta = result.get("meta") or {}
    quote = (((result.get("indicators") or {}).get("quote") or [None])[0]) or {}
    adjclose = (((result.get("indicators") or {}).get("adjclose") or [None])[0]) or {}
    timezone_name = meta.get("exchangeTimezoneName") or "UTC"

    timestamps = pd.to_datetime(
        result.get("timestamp") or [],
        unit="s",
        utc=True,
    ).tz_convert(timezone_name)
    if len(timestamps) == 0:
        raise ValueError(f"Yahoo returned no timestamps for {ticker}.")

    frame = pd.DataFrame(
        {
            "ticker": ticker,
            "date": timestamps.date,
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "adjClose": adjclose.get("adjclose"),
            "volume": quote.get("volume"),
        }
    )

    events = result.get("events") or {}
    dividend_map = normalize_event_map(
        events.get("dividends"),
        timezone_name=timezone_name,
        field="amount",
    )
    split_factor_map = normalize_split_factors(
        events.get("splits"),
        timezone_name=timezone_name,
    )

    frame["dividend"] = frame["date"].map(dividend_map).fillna(0.0)
    frame["splitFactor"] = frame["date"].map(split_factor_map).fillna(1.0)
    core_value_columns = ["open", "high", "low", "close", "adjClose", "volume"]
    frame = frame.loc[~frame[core_value_columns].isna().all(axis=1)].copy()
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    frame = frame[YAHOO_PANEL_COLUMN_ORDER]

    metadata = {
        "ticker": ticker,
        "currency": meta.get("currency"),
        "exchangeName": meta.get("exchangeName"),
        "instrumentType": meta.get("instrumentType"),
        "firstTradeDate": (
            pd.to_datetime(meta["firstTradeDate"], unit="s", utc=True)
            .tz_convert(timezone_name)
            .date()
            if meta.get("firstTradeDate") is not None
            else None
        ),
        "exchangeTimezoneName": timezone_name,
        "dataGranularity": meta.get("dataGranularity"),
    }
    return frame, metadata


def yahoo_cache_dir(output_dir: Path) -> Path:
    """Return the per-output cache folder."""

    return output_dir / YAHOO_TICKER_CACHE_DIRNAME


def yahoo_cache_paths(output_dir: Path, ticker: str) -> tuple[Path, Path]:
    """Return the cached parquet and metadata-json paths for one ticker."""

    cache_root = yahoo_cache_dir(output_dir)
    return cache_root / f"{ticker}.parquet", cache_root / f"{ticker}.metadata.json"


def load_cached_yahoo_ticker(
    output_dir: Path,
    ticker: str,
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    """Load a cached ticker result when resume mode is active."""

    panel_path, metadata_path = yahoo_cache_paths(output_dir, ticker)
    if not panel_path.exists() or not metadata_path.exists():
        return None

    frame = pd.read_parquet(panel_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return frame, metadata


def write_cached_yahoo_ticker(
    output_dir: Path,
    ticker: str,
    frame: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    """Cache one successful ticker pull so larger runs can resume."""

    panel_path, metadata_path = yahoo_cache_paths(output_dir, ticker)
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(panel_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


def pull_yahoo_crypto_panel(
    config: YahooCryptoPullConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pull the Yahoo crypto panel, optionally resuming from cached tickers."""

    session = build_yahoo_session()
    price_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for index, ticker in enumerate(config.tickers, start=1):
        cached = load_cached_yahoo_ticker(config.output_dir, ticker) if config.resume else None
        if cached is not None:
            print(f"[{index}/{len(config.tickers)}] Using cached Yahoo history for {ticker}...")
            frame, metadata = cached
            price_frames.append(frame)
            metadata_rows.append(metadata)
            continue

        print(f"[{index}/{len(config.tickers)}] Pulling Yahoo chart history for {ticker}...")
        try:
            payload = request_yahoo_chart_json(
                session,
                ticker,
                start_date=config.start_date,
                end_date=config.end_date,
                timeout_seconds=config.timeout_seconds,
                max_attempts=config.max_attempts,
                backoff_seconds=config.backoff_seconds,
            )
            frame, metadata = normalize_yahoo_crypto_chart_payload(ticker, payload)
            write_cached_yahoo_ticker(config.output_dir, ticker, frame, metadata)
            price_frames.append(frame)
            metadata_rows.append(metadata)
        except Exception as exc:
            print(f"  Failed Yahoo pull for {ticker}: {exc}")
            failures.append({"ticker": ticker, "error": str(exc)})
        time.sleep(config.pause_seconds)

    if not price_frames:
        raise RuntimeError("Yahoo crypto pull produced no successful ticker results.")

    panel = pd.concat(price_frames, ignore_index=True)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    if panel[["ticker", "date"]].duplicated().any():
        duplicates = panel.loc[panel[["ticker", "date"]].duplicated(), ["ticker", "date"]]
        raise ValueError(f"Duplicate panel keys detected:\n{duplicates.head()}")

    metadata = pd.DataFrame(metadata_rows).sort_values("ticker").reset_index(drop=True)
    failures_frame = pd.DataFrame(failures)
    return panel, metadata, failures_frame


def build_yahoo_crypto_coverage_summary(
    panel: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    requested_start_date: str,
) -> pd.DataFrame:
    """Summarize how much history Yahoo actually returned for each ticker."""

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
    requested_start_ts = pd.Timestamp(requested_start_date).normalize()
    requested_start = requested_start_ts.date()
    coverage["requested_start_date"] = requested_start
    coverage["effective_start_date"] = requested_start
    coverage["covers_requested_start"] = (
        pd.to_datetime(coverage["panel_start_date"]) <= requested_start_ts
    )
    return coverage.merge(metadata, on="ticker", how="left")


def write_yahoo_crypto_outputs(
    *,
    panel: pd.DataFrame,
    metadata: pd.DataFrame,
    failures: pd.DataFrame,
    output_dir: Path,
    requested_start_date: str,
) -> dict[str, Path]:
    """Write the Yahoo crypto panel, metadata, coverage summary, and failures file."""

    output_dir.mkdir(parents=True, exist_ok=True)
    panel_csv = output_dir / "yahoo_chart_panel_long.csv"
    panel_parquet = output_dir / "yahoo_chart_panel_long.parquet"
    metadata_csv = output_dir / "yahoo_chart_metadata.csv"
    coverage_csv = output_dir / "yahoo_chart_coverage_summary.csv"
    failures_csv = output_dir / "yahoo_chart_failures.csv"

    coverage = build_yahoo_crypto_coverage_summary(
        panel,
        metadata,
        requested_start_date=requested_start_date,
    )

    panel.to_csv(panel_csv, index=False)
    panel.to_parquet(panel_parquet, index=False)
    metadata.to_csv(metadata_csv, index=False)
    coverage.to_csv(coverage_csv, index=False)
    outputs: dict[str, Path] = {
        "panel_csv": panel_csv,
        "panel_parquet": panel_parquet,
        "metadata_csv": metadata_csv,
        "coverage_csv": coverage_csv,
    }
    if not failures.empty:
        failures.to_csv(failures_csv, index=False)
        outputs["failures_csv"] = failures_csv
    return outputs
