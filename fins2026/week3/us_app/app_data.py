# ruff: noqa
"""Data loading and source-status helpers for the Week 3 companion U.S. app."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from fins2026.week3.us_app.app_config import FRED_SERIES, SAMPLE_PERIODS
from fintools.apps import clean_fred_graph_csv, fred_graph_url, read_fred_graph_csv
from fintools.datasets import load_validation_dataset


@st.cache_data(ttl=86400)
def load_live_market_data() -> tuple[pd.DataFrame, pd.Timestamp]:
    """Load current no-key FRED graph CSV data with cache timestamp."""

    frame = clean_fred_graph_csv(read_fred_graph_csv(fred_graph_url(FRED_SERIES)))
    return frame, pd.Timestamp.now(tz="UTC")


def _synthetic_gdp_fixture() -> pd.DataFrame:
    """Create a stable quarterly GDP fixture for offline validation."""

    dates = pd.date_range("1985-03-31", "2026-03-31", freq="QE-DEC")
    cycle = np.sin(np.linspace(0.0, 11.5 * np.pi, len(dates)))
    annualized_growth = 2.2 + 1.4 * cycle
    annualized_growth[(dates >= "2008-09-30") & (dates <= "2009-06-30")] -= 5.5
    annualized_growth[(dates >= "2020-03-31") & (dates <= "2020-06-30")] -= 16.0
    annualized_growth[(dates >= "2020-09-30") & (dates <= "2021-03-31")] += 9.0
    level = [13_000.0]
    for growth in annualized_growth[1:]:
        quarter_growth = (1.0 + growth / 100.0) ** 0.25 - 1.0
        level.append(level[-1] * (1.0 + quarter_growth))
    return pd.DataFrame({"GDPC1": level}, index=dates)


@st.cache_data(ttl=86400)
def load_fixture_market_data() -> pd.DataFrame:
    """Load frozen validation fixtures so the app works offline."""

    rates = load_validation_dataset("fred_rates_daily").data[["DGS10", "DGS2", "DTB3", "T10Y2Y"]]
    stress = load_validation_dataset("fred_financial_stress_daily").data[
        ["VIXCLS", "BAMLH0A0HYM2"]
    ]
    return rates.join(stress, how="outer").join(_synthetic_gdp_fixture(), how="outer").sort_index()


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
    except Exception as exc:
        message = (
            "Live FRED is temporarily unavailable, so the app is showing the "
            f"frozen validation fixture instead. Technical detail: {exc}"
        )
        return load_fixture_market_data(), "Fixture", message, None


def source_status_text(
    frame: pd.DataFrame,
    *,
    active_data_mode: str,
    loaded_at_utc: pd.Timestamp | None = None,
    warning: str | None = None,
) -> str:
    """Return app-facing source freshness text."""

    latest = "n/a" if frame.empty else f"{pd.Timestamp(frame.index.max()):%Y-%m-%d}"
    if active_data_mode == "Live FRED":
        loaded = "n/a" if loaded_at_utc is None else f"{loaded_at_utc:%Y-%m-%d %H:%M} UTC"
        return (
            f"Live FRED cache loaded at {loaded}; latest observation in the "
            f"selected source is {latest}. Cached data refreshes at most once every 24 hours."
        )
    if warning:
        return f"Fixture fallback snapshot through {latest}; live FRED could not be loaded."
    return f"Fixture snapshot through {latest}; no live refresh attempted."

