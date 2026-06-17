"""Station 1 ETL: load and clean the project data."""
from __future__ import annotations

import pandas as pd

from src import data_access


PRICE_KEY = ["ticker", "date"]
NEWS_KEY = ["ticker", "date", "title"]


def _clean_price_frame(df: pd.DataFrame, asset_class: str) -> pd.DataFrame:
    clean = df.copy()
    clean["date"] = pd.to_datetime(clean["date"]).dt.tz_localize(None)
    clean = clean.sort_values(PRICE_KEY)
    clean = clean.drop_duplicates(PRICE_KEY, keep="last")
    clean = clean[clean["date"] <= pd.Timestamp("2023-12-31")]
    clean = clean[clean["adjClose"].notna() & (clean["adjClose"] > 0)]
    clean["asset_class"] = asset_class
    if "sector" not in clean.columns:
        clean["sector"] = "Crypto"
    return clean.reset_index(drop=True)


def load_clean_equities() -> pd.DataFrame:
    """Load equity prices and remove duplicate ticker-date rows."""
    return _clean_price_frame(data_access.load_equity_prices(), "Equity")


def load_clean_crypto() -> pd.DataFrame:
    """Load crypto prices on the native 365-day calendar."""
    return _clean_price_frame(data_access.load_crypto_prices(), "Crypto")


def load_clean_headlines() -> pd.DataFrame:
    """Load headlines and remove exact duplicate ticker-date-title rows."""
    news = data_access.load_news_headlines().copy()
    news["date"] = pd.to_datetime(news["date"]).dt.tz_localize(None)
    news["title"] = news["title"].fillna("").astype(str).str.strip()
    news = news[news["title"].ne("")]
    news = news.drop_duplicates(NEWS_KEY, keep="first")
    return news.sort_values(["date", "ticker", "title"]).reset_index(drop=True)


def inventory_row(
    name: str,
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    key: list[str],
    asset_class: str,
) -> dict:
    """Dataset inventory row for the report."""
    return {
        "dataset": name,
        "asset_class": asset_class,
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(clean)),
        "rows_removed": int(len(raw) - len(clean)),
        "tickers": int(clean["ticker"].nunique()) if "ticker" in clean else "",
        "start_date": clean["date"].min().date().isoformat(),
        "end_date": clean["date"].max().date().isoformat(),
        "primary_key": "+".join(key),
        "duplicate_keys_after_clean": int(clean.duplicated(key).sum()),
    }
