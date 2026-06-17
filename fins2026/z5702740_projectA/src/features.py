# ruff: noqa
"""Station 2 features: returns and headline text assembly."""
from __future__ import annotations

import numpy as np
import pandas as pd


def daily_returns(
    prices: pd.DataFrame,
    price_col: str = "adjClose",
    wide: bool = True,
) -> pd.DataFrame:
    """Compute simple daily returns within each ticker."""
    df = prices[["date", "ticker", price_col]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values(["ticker", "date"])
    df["return"] = df.groupby("ticker", group_keys=False)[price_col].pct_change()
    df["return"] = df["return"].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["return"])
    if wide:
        return df.pivot(index="date", columns="ticker", values="return").sort_index()
    return df[["date", "ticker", "return"]].reset_index(drop=True)


def align_dates_to_trading_calendar(
    dates: pd.Series,
    trading_calendar: pd.DatetimeIndex,
) -> pd.Series:
    """Map each headline date to the same or next equity trading day."""
    calendar = pd.DatetimeIndex(pd.to_datetime(trading_calendar)).sort_values()
    raw = pd.to_datetime(dates).dt.tz_localize(None)
    positions = np.searchsorted(calendar.values, raw.values, side="left")
    aligned = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    valid = positions < len(calendar)
    aligned.loc[valid] = calendar.values[positions[valid]]
    return aligned


def assemble_headline_panel(
    headlines: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Assemble the news into a ticker-day text panel without scoring sentiment."""
    news = headlines.copy()
    news["date"] = pd.to_datetime(news["date"]).dt.tz_localize(None)
    news["title"] = news["title"].fillna("").astype(str).str.strip()
    news = news[news["title"].ne("")]
    if trading_calendar is not None:
        news["trading_date"] = align_dates_to_trading_calendar(news["date"], trading_calendar)
        news = news.dropna(subset=["trading_date"])
    else:
        news["trading_date"] = news["date"]

    panel = (
        news.groupby(["trading_date", "ticker", "sector"], as_index=False)
        .agg(
            article_count=("title", "size"),
            text=("title", lambda x: " | ".join(x.astype(str))),
        )
        .sort_values(["trading_date", "sector", "ticker"])
    )
    panel["word_count"] = panel["text"].str.split().str.len()
    return panel.reset_index(drop=True)


def cumulative_equal_weight(returns: pd.DataFrame) -> pd.Series:
    """Equal-weight cumulative return index from a wide return frame."""
    ew = returns.fillna(0).mean(axis=1)
    return (1 + ew).cumprod()
