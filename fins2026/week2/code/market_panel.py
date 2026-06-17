# ruff: noqa
"""Shared month-end panel helpers for the Week 2 FRED workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd

DAILY_MARKET_COLUMNS = [
    "DGS10",
    "DGS2",
    "DTB3",
    "T10Y2Y",
    "VIXCLS",
    "SP500",
]
MONTHLY_MACRO_COLUMNS = ["UNRATE", "INDPRO", "PAYEMS", "FEDFUNDS"]


def change_in_basis_points(series: pd.Series) -> pd.Series:
    """Return one-period changes in basis points for a percent rate series."""

    return series.astype(float).diff() * 100.0


def change_in_percentage_points(series: pd.Series) -> pd.Series:
    """Return one-period changes in percentage points."""

    return series.astype(float).diff()


def percent_change(series: pd.Series) -> pd.Series:
    """Return one-period simple percentage change."""

    return series.astype(float).pct_change() * 100.0


def monthly_equivalent_percent(series: pd.Series) -> pd.Series:
    """Convert an annualized percent-volatility series to a monthly equivalent."""

    return series.astype(float) / np.sqrt(12.0)


def log_change_percent(series: pd.Series) -> pd.Series:
    """Return one-period log change in percent."""

    clean = series.astype(float)
    prior = clean.shift(1)
    ratio = clean / prior
    return np.log(ratio.where((clean > 0) & (prior > 0))) * 100.0


def resample_to_month_end(frame: pd.DataFrame) -> pd.DataFrame:
    """Resample date-indexed level data to month-end using the last observation."""

    return frame.sort_index().resample("ME").last()


def align_observations_to_month_end(frame: pd.DataFrame) -> pd.DataFrame:
    """Move lower-frequency observations onto month-end timestamps."""

    result = frame.copy().sort_index()
    result.index = result.index.to_period("M").to_timestamp("M")
    return result.groupby(level=0).last()


def add_week2_series(frame: pd.DataFrame) -> pd.DataFrame:
    """Add Week 2 transformed series to a month-end panel."""

    result = frame.copy().sort_index()

    for column in ["DGS10", "DGS2", "DTB3", "T10Y2Y", "FEDFUNDS"]:
        if column in result:
            result[f"{column}_CHANGE_BP"] = change_in_basis_points(result[column])

    if "UNRATE" in result:
        result["UNRATE_CHANGE_PP"] = change_in_percentage_points(result["UNRATE"])

    for column in ["INDPRO", "PAYEMS"]:
        if column in result:
            result[f"{column}_LOG_GROWTH_PCT"] = log_change_percent(result[column])

    if "SP500" in result:
        sp500 = result["SP500"].dropna()
        result["SP500_RETURN_PCT"] = percent_change(result["SP500"])
        result["SP500_LOG_RETURN_PCT"] = log_change_percent(result["SP500"])
        if sp500.empty:
            result["SP500_CUMULATIVE_RETURN_PCT"] = pd.NA
        else:
            result["SP500_CUMULATIVE_RETURN_PCT"] = (
                (result["SP500"] / float(sp500.iloc[0]) - 1.0) * 100.0
            )

    if "VIXCLS" in result:
        result["VIX_MONTHLY_VOL_PCT"] = monthly_equivalent_percent(result["VIXCLS"])
        result["VIX_MONTHLY_VOL_CHANGE_PP"] = change_in_percentage_points(
            result["VIX_MONTHLY_VOL_PCT"]
        )
        result["VIX_CHANGE_PCT"] = percent_change(result["VIXCLS"])

    return result


def build_month_end_panel(
    daily_market: pd.DataFrame,
    monthly_macro: pd.DataFrame,
) -> pd.DataFrame:
    """Build the merged month-end panel used by the Week 2 app and figures."""

    market_month_end = resample_to_month_end(daily_market.reindex(columns=DAILY_MARKET_COLUMNS))
    macro_month_end = align_observations_to_month_end(
        monthly_macro.reindex(columns=MONTHLY_MACRO_COLUMNS)
    )
    panel = market_month_end.join(macro_month_end, how="outer")
    return add_week2_series(panel.sort_index())
