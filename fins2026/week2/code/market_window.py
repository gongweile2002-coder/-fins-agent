# ruff: noqa
"""Shared sample-window rules for the Week 2 FRED workflow."""

from __future__ import annotations

import pandas as pd

WEEK2_FRED_WINDOW_YEARS = 10
WEEK2_FRED_LECTURE_START = pd.Timestamp("2015-01-01")
WEEK2_FRED_LECTURE_END = pd.Timestamp("2025-12-31")


def _window_bounds(
    *,
    latest: pd.Timestamp,
    years: int = WEEK2_FRED_WINDOW_YEARS,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Resolve the active sample window."""

    resolved_end = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp(latest)
    if start_date is not None:
        resolved_start = pd.Timestamp(start_date)
    else:
        resolved_start = resolved_end - pd.DateOffset(years=years)
    return resolved_start, resolved_end


def clip_date_indexed_frame(
    frame: pd.DataFrame,
    *,
    years: int = WEEK2_FRED_WINDOW_YEARS,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Keep only the latest ``years`` of observations from a date index."""

    result = frame.copy().sort_index()
    if result.empty:
        return result
    start, end = _window_bounds(
        latest=pd.Timestamp(result.index.max()),
        years=years,
        start_date=start_date,
        end_date=end_date,
    )
    return result.loc[(result.index >= start) & (result.index <= end)].copy()


def clip_dated_column_frame(
    frame: pd.DataFrame,
    *,
    date_column: str = "date",
    years: int = WEEK2_FRED_WINDOW_YEARS,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Keep only the latest ``years`` of observations from a date column."""

    result = frame.copy()
    if result.empty:
        return result
    result[date_column] = pd.to_datetime(result[date_column], errors="coerce")
    result = result.dropna(subset=[date_column]).sort_values(date_column).reset_index(drop=True)
    if result.empty:
        return result
    start, end = _window_bounds(
        latest=pd.Timestamp(result[date_column].max()),
        years=years,
        start_date=start_date,
        end_date=end_date,
    )
    return result.loc[
        (result[date_column] >= start) & (result[date_column] <= end)
    ].reset_index(drop=True)
