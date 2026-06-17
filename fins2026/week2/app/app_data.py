# ruff: noqa
"""Data loading and source-status helpers for the Week 2 Streamlit intro app."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fins2026.week2.app.app_config import FRED_SERIES, SAMPLE_PERIODS
from fins2026.week2.code.market_panel import (
    DAILY_MARKET_COLUMNS,
    MONTHLY_MACRO_COLUMNS,
    build_month_end_panel,
)
from fins2026.week2.code.market_window import clip_date_indexed_frame
from fintools.apps import clean_fred_graph_csv, fred_graph_url, read_fred_graph_csv
from fintools.datasets import load_validation_dataset


@st.cache_data(ttl=86400)
def load_live_market_data() -> tuple[pd.DataFrame, pd.Timestamp]:
    """Load current no-key FRED graph CSV data and build the month-end panel."""

    raw = clean_fred_graph_csv(read_fred_graph_csv(fred_graph_url(FRED_SERIES))).sort_index()
    frame = clip_date_indexed_frame(build_month_end_panel(
        raw[DAILY_MARKET_COLUMNS],
        raw[MONTHLY_MACRO_COLUMNS],
    ))
    return frame, pd.Timestamp.now(tz="UTC")


@st.cache_data(ttl=86400)
def load_fixture_market_data() -> pd.DataFrame:
    """Load frozen validation fixtures and build the month-end panel."""

    rates = load_validation_dataset("fred_rates_daily").data[["DGS10", "DGS2", "DTB3", "T10Y2Y"]]
    stress = load_validation_dataset("fred_financial_stress_daily").data[["VIXCLS"]]
    sp500 = load_validation_dataset("fred_sp500_daily").data[["SP500"]]
    daily_market = rates.join(stress, how="outer").join(sp500, how="outer")
    macro = load_validation_dataset("fred_macro_monthly").data[MONTHLY_MACRO_COLUMNS]
    return clip_date_indexed_frame(build_month_end_panel(daily_market, macro))


def apply_sample_period(frame: pd.DataFrame, sample_period: str) -> pd.DataFrame:
    """Restrict a date-indexed dataframe to the selected analysis sample."""

    years = SAMPLE_PERIODS[sample_period]
    if years is None:
        return frame.copy()
    cutoff = frame.index.max() - pd.DateOffset(years=years)
    return frame.loc[frame.index >= cutoff].copy()


def load_market_data(
    data_mode: str,
) -> tuple[pd.DataFrame, str, str | None, pd.Timestamp | None]:
    """Load the requested data source, falling back to fixtures when needed."""

    if data_mode == "Fixture":
        return load_fixture_market_data(), "Fixture", None, None
    try:
        frame, loaded_at_utc = load_live_market_data()
        return frame, "Live FRED", None, loaded_at_utc
    except Exception:
        message = (
            "Live FRED is temporarily unavailable, so the app is showing "
            "the frozen validation fixture instead."
        )
        return load_fixture_market_data(), "Fixture", message, None


def source_status_text(
    frame: pd.DataFrame,
    *,
    series: pd.Series,
    series_label: str,
    active_data_mode: str,
    loaded_at_utc: pd.Timestamp | None = None,
    warning: str | None = None,
) -> str:
    """Return app-facing source freshness text."""

    snapshot_latest = "n/a" if frame.empty else f"{pd.Timestamp(frame.index.max()):%Y-%m-%d}"
    observed = series.dropna()
    selected_latest = (
        "n/a" if observed.empty else f"{pd.Timestamp(observed.index.max()):%Y-%m-%d}"
    )
    if active_data_mode == "Live FRED":
        loaded = "n/a" if loaded_at_utc is None else f"{loaded_at_utc:%Y-%m-%d %H:%M} UTC"
        return (
            f"Live FRED cache loaded at {loaded}; dataset snapshot through {snapshot_latest}; "
            f"latest observation for {series_label} is {selected_latest}. "
            "Cached data refreshes at most once every 24 hours."
        )
    if warning:
        return (
            f"Fixture fallback snapshot through {snapshot_latest}; latest observation for "
            f"{series_label} is {selected_latest}."
        )
    return (
        f"Fixture snapshot through {snapshot_latest}; latest observation for "
        f"{series_label} is {selected_latest}."
    )
