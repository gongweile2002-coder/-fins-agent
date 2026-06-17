# ruff: noqa
"""Week 3 self-contained panel helpers for deployable app bundles."""

from __future__ import annotations

import numpy as np
import pandas as pd

CLASSROOM_SAMPLE_START = pd.Timestamp("2000-01-01")
CLASSROOM_INFORMATION_SET_MONTH_END = pd.Timestamp("2026-03-31")

DAILY_MARKET_COLUMNS = [
    "DGS10",
    "DGS2",
    "DTB3",
    "T10Y2Y",
    "VIXCLS",
    "SP500",
]
MONTHLY_MACRO_COLUMNS = ["UNRATE", "INDPRO", "PAYEMS", "FEDFUNDS"]


def month_end_timestamp(value: object) -> pd.Timestamp:
    """Return the month-end timestamp for an observation date."""

    return pd.Timestamp(value).to_period("M").to_timestamp("M")


def log_change_percent(series: pd.Series) -> pd.Series:
    """Return one-period log change in percent."""

    clean = pd.to_numeric(series, errors="coerce").astype(float)
    prior = clean.shift(1)
    ratio = clean / prior
    return np.log(ratio.where((clean > 0) & (prior > 0))) * 100.0


def load_stage1_fixture_csv(path: str) -> pd.DataFrame:
    """Load a committed Australia Stage 1 fixture from CSV."""

    frame = pd.read_csv(path)
    for column in [
        "raw_date",
        "reference_date",
        "release_date",
        "observable_month_end",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    if "include_in_dec2025_core_pack" in frame.columns:
        frame["include_in_dec2025_core_pack"] = frame[
            "include_in_dec2025_core_pack"
        ].astype(bool)
    return frame


def build_observable_panel(
    stage1: pd.DataFrame,
    *,
    sample_start: pd.Timestamp = CLASSROOM_SAMPLE_START,
    information_set_month_end: pd.Timestamp = CLASSROOM_INFORMATION_SET_MONTH_END,
) -> pd.DataFrame:
    """Build the classroom observable-month-end panel from the Stage 1 long table."""

    month_index = pd.date_range(
        start=month_end_timestamp(sample_start),
        end=month_end_timestamp(information_set_month_end),
        freq="ME",
    )
    panel = pd.DataFrame(index=month_index)
    for label, subset in stage1.groupby("display_name"):
        series = (
            subset.dropna(subset=["observable_month_end", "value"])
            .groupby("observable_month_end")["value"]
            .last()
            .sort_index()
        )
        panel[label] = pd.to_numeric(series, errors="coerce").reindex(month_index).ffill()
    panel.index.name = "date"
    return panel


def resample_to_month_end(frame: pd.DataFrame) -> pd.DataFrame:
    """Resample date-indexed level data to month-end using the last observation."""

    return frame.sort_index().resample("ME").last()


def align_observations_to_month_end(frame: pd.DataFrame) -> pd.DataFrame:
    """Move lower-frequency observations onto month-end timestamps."""

    result = frame.copy().sort_index()
    result.index = result.index.to_period("M").to_timestamp("M")
    return result.groupby(level=0).last()


def _change_in_basis_points(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float).diff() * 100.0


def _change_in_percentage_points(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float).diff()


def _percent_change(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float).pct_change() * 100.0


def _monthly_equivalent_percent(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float) / np.sqrt(12.0)


def add_week3_context_series(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the transformed U.S. context columns used by Week 3."""

    result = frame.copy().sort_index()

    for column in ["DGS10", "DGS2", "DTB3", "T10Y2Y", "FEDFUNDS"]:
        if column in result:
            result[f"{column}_CHANGE_BP"] = _change_in_basis_points(result[column])

    if "UNRATE" in result:
        result["UNRATE_CHANGE_PP"] = _change_in_percentage_points(result["UNRATE"])

    for column in ["INDPRO", "PAYEMS"]:
        if column in result:
            result[f"{column}_LOG_GROWTH_PCT"] = log_change_percent(result[column])

    if "SP500" in result:
        sp500 = result["SP500"].dropna()
        result["SP500_RETURN_PCT"] = _percent_change(result["SP500"])
        result["SP500_LOG_RETURN_PCT"] = log_change_percent(result["SP500"])
        if sp500.empty:
            result["SP500_CUMULATIVE_RETURN_PCT"] = pd.NA
        else:
            result["SP500_CUMULATIVE_RETURN_PCT"] = (
                (result["SP500"] / float(sp500.iloc[0]) - 1.0) * 100.0
            )

    if "VIXCLS" in result:
        result["VIX_MONTHLY_VOL_PCT"] = _monthly_equivalent_percent(result["VIXCLS"])
        result["VIX_MONTHLY_VOL_CHANGE_PP"] = _change_in_percentage_points(
            result["VIX_MONTHLY_VOL_PCT"]
        )
        result["VIX_CHANGE_PCT"] = _percent_change(result["VIXCLS"])

    return result


def build_month_end_panel(
    daily_market: pd.DataFrame,
    monthly_macro: pd.DataFrame,
) -> pd.DataFrame:
    """Build the month-end U.S. panel used by the Week 3 app and scripts."""

    market_month_end = resample_to_month_end(
        daily_market.reindex(columns=DAILY_MARKET_COLUMNS)
    )
    macro_month_end = align_observations_to_month_end(
        monthly_macro.reindex(columns=MONTHLY_MACRO_COLUMNS)
    )
    panel = market_month_end.join(macro_month_end, how="outer")
    return add_week3_context_series(panel.sort_index())
