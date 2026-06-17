# ruff: noqa
"""Week 2 FRED Stage 1 timing contracts and long-table helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fins2026.week2.code.market_window import (
    WEEK2_FRED_LECTURE_END,
    WEEK2_FRED_LECTURE_START,
    WEEK2_FRED_WINDOW_YEARS,
    clip_dated_column_frame,
)
from fintools.datasets import load_validation_dataset

FRED_SOURCE_NAME = "Federal Reserve Economic Data (FRED)"
FRED_SOURCE_TABLE = "fredgraph.csv"
FRED_GRAPH_CSV_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="


@dataclass(frozen=True)
class FredSeriesSpec:
    """Metadata contract for one Week 2 FRED series."""

    series_id: str
    display_name: str
    native_frequency: str
    reference_date_rule: str
    observable_lag_months: int
    units: str
    default_transform: str
    category: str = ""
    note: str = ""


FRED_STAGE1_SPECS: dict[str, FredSeriesSpec] = {
    "DGS10": FredSeriesSpec(
        series_id="DGS10",
        display_name="10Y Treasury yield",
        native_frequency="daily",
        reference_date_rule="calendar_date",
        observable_lag_months=0,
        units="Per cent",
        default_transform="level_and_basis_points_change",
        category="rates",
    ),
    "DGS2": FredSeriesSpec(
        series_id="DGS2",
        display_name="2Y Treasury yield",
        native_frequency="daily",
        reference_date_rule="calendar_date",
        observable_lag_months=0,
        units="Per cent",
        default_transform="level_and_basis_points_change",
        category="rates",
    ),
    "DTB3": FredSeriesSpec(
        series_id="DTB3",
        display_name="3M Treasury bill rate",
        native_frequency="daily",
        reference_date_rule="calendar_date",
        observable_lag_months=0,
        units="Per cent",
        default_transform="level_and_basis_points_change",
        category="rates",
    ),
    "T10Y2Y": FredSeriesSpec(
        series_id="T10Y2Y",
        display_name="10Y minus 2Y Treasury spread",
        native_frequency="daily",
        reference_date_rule="calendar_date",
        observable_lag_months=0,
        units="Per cent",
        default_transform="level_and_basis_points_change",
        category="rates",
    ),
    "VIXCLS": FredSeriesSpec(
        series_id="VIXCLS",
        display_name="VIX",
        native_frequency="daily",
        reference_date_rule="calendar_date",
        observable_lag_months=0,
        units="Index",
        default_transform="level_and_change",
        category="stress",
    ),
    "UNRATE": FredSeriesSpec(
        series_id="UNRATE",
        display_name="Unemployment rate",
        native_frequency="monthly",
        reference_date_rule="month_end",
        observable_lag_months=1,
        units="Per cent",
        default_transform="level_and_percentage_points_change",
        category="labour",
        note=(
            "Week 2 uses a classroom month-end release proxy rather than the "
            "exact BLS publication day."
        ),
    ),
    "INDPRO": FredSeriesSpec(
        series_id="INDPRO",
        display_name="Industrial production index",
        native_frequency="monthly",
        reference_date_rule="month_end",
        observable_lag_months=1,
        units="Index",
        default_transform="log_growth",
        category="activity",
        note=(
            "Week 2 uses a classroom month-end release proxy rather than the "
            "exact Federal Reserve publication day."
        ),
    ),
    "PAYEMS": FredSeriesSpec(
        series_id="PAYEMS",
        display_name="Nonfarm payroll employment",
        native_frequency="monthly",
        reference_date_rule="month_end",
        observable_lag_months=1,
        units="Thousands of persons",
        default_transform="log_growth",
        category="activity",
        note=(
            "Week 2 uses a classroom month-end release proxy rather than the "
            "exact BLS publication day."
        ),
    ),
    "FEDFUNDS": FredSeriesSpec(
        series_id="FEDFUNDS",
        display_name="Federal funds rate",
        native_frequency="monthly",
        reference_date_rule="month_end",
        observable_lag_months=1,
        units="Per cent",
        default_transform="level_and_basis_points_change",
        category="policy",
        note=(
            "Week 2 uses a classroom month-end release proxy rather than the "
            "exact H.15 publication day."
        ),
    ),
    "SP500": FredSeriesSpec(
        series_id="SP500",
        display_name="S&P 500 index level",
        native_frequency="daily",
        reference_date_rule="calendar_date",
        observable_lag_months=0,
        units="Index",
        default_transform="return_and_growth_of_one_dollar",
        category="equity",
    ),
}

FRED_SERIES = tuple(FRED_STAGE1_SPECS)
MONTHLY_FRED_SERIES = tuple(
    series_id
    for series_id, spec in FRED_STAGE1_SPECS.items()
    if spec.native_frequency == "monthly"
)
DAILY_FRED_SERIES = tuple(
    series_id
    for series_id, spec in FRED_STAGE1_SPECS.items()
    if spec.native_frequency == "daily"
)


def month_end_timestamp(value: object) -> pd.Timestamp:
    """Return the month-end timestamp for an observation date."""

    return pd.Timestamp(value).to_period("M").to_timestamp("M")


def shift_month_end(value: object, months: int) -> pd.Timestamp:
    """Shift a month-end timestamp forward by whole month-end steps."""

    return month_end_timestamp(value) + pd.offsets.MonthEnd(months)


def fred_csv_url(series: tuple[str, ...] = FRED_SERIES) -> str:
    """Return the no-key FRED graph CSV URL for the selected series."""

    return FRED_GRAPH_CSV_BASE_URL + ",".join(series)


def clean_raw_fred_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw FRED graph CSV data to typed Week 2 columns."""

    frame = raw.copy()
    date_candidates = ["observation_date", "DATE", "date", "Date"]
    date_column = next((column for column in date_candidates if column in frame.columns), None)
    if date_column is None:
        raise ValueError("FRED data is missing a date column.")
    frame = frame.rename(columns={date_column: "date"})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

    for column in FRED_SERIES:
        if column not in frame.columns:
            raise ValueError(f"FRED data is missing required series: {column}")
        frame[column] = pd.to_numeric(frame[column].replace(".", pd.NA), errors="coerce")

    return (
        frame[["date", *FRED_SERIES]]
        .dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )


def build_fixture_raw_fred_frame() -> pd.DataFrame:
    """Return the frozen Week 2 FRED wide frame from repo validation fixtures."""

    rates = load_validation_dataset("fred_rates_daily").data
    stress = load_validation_dataset("fred_financial_stress_daily").data
    macro = load_validation_dataset("fred_macro_monthly").data
    sp500 = load_validation_dataset("fred_sp500_daily").data
    frame = (
        rates.join(stress, how="outer")
        .join(macro, how="outer")
        .join(sp500, how="outer")
        .reset_index()
    )
    return clean_raw_fred_frame(frame)


def _reference_date_for_spec(raw_date: pd.Series, spec: FredSeriesSpec) -> pd.Series:
    if spec.reference_date_rule == "month_end":
        return raw_date.dt.to_period("M").dt.to_timestamp("M")
    return raw_date


def build_fred_stage1_long_table(
    raw: pd.DataFrame,
    *,
    rolling_window: bool = False,
    years: int = WEEK2_FRED_WINDOW_YEARS,
) -> pd.DataFrame:
    """Build the Week 2 FRED Stage 1 long table."""

    cleaned = clean_raw_fred_frame(raw)
    frame = (
        clip_dated_column_frame(cleaned, years=years)
        if rolling_window
        else clip_dated_column_frame(
            cleaned,
            start_date=WEEK2_FRED_LECTURE_START,
            end_date=WEEK2_FRED_LECTURE_END,
        )
    )
    records: list[pd.DataFrame] = []
    for series_id, spec in FRED_STAGE1_SPECS.items():
        history = frame[["date", series_id]].rename(
            columns={"date": "raw_date", series_id: "value"}
        )
        history["raw_date"] = pd.to_datetime(history["raw_date"], errors="coerce")
        history["reference_date"] = _reference_date_for_spec(history["raw_date"], spec)
        history = history.dropna(subset=["reference_date", "value"]).copy()
        if history.empty:
            continue
        history["release_date"] = history["reference_date"].map(
            lambda value: shift_month_end(value, spec.observable_lag_months)
        )
        history["observable_month_end"] = history["release_date"]
        history["series_id"] = spec.series_id
        history["display_name"] = spec.display_name
        history["source_name"] = FRED_SOURCE_NAME
        history["source_table"] = FRED_SOURCE_TABLE
        history["native_frequency"] = spec.native_frequency
        history["reference_date_rule"] = spec.reference_date_rule
        history["observable_lag_months"] = spec.observable_lag_months
        history["units"] = spec.units
        history["default_transform"] = spec.default_transform
        history["category"] = spec.category
        history["note"] = spec.note
        records.append(
            history[
                [
                    "series_id",
                    "display_name",
                    "source_name",
                    "source_table",
                    "native_frequency",
                    "reference_date_rule",
                    "raw_date",
                    "reference_date",
                    "release_date",
                    "observable_month_end",
                    "observable_lag_months",
                    "value",
                    "units",
                    "default_transform",
                    "category",
                    "note",
                ]
            ]
        )
    if not records:
        return pd.DataFrame()
    return (
        pd.concat(records, ignore_index=True)
        .sort_values(["series_id", "reference_date", "raw_date"])
        .reset_index(drop=True)
    )


def stage1_assertion_report(stage1: pd.DataFrame) -> dict[str, bool]:
    """Return the Week 2 Stage 1 assertion outcomes for a long table."""

    expected_series = set(FRED_SERIES)
    actual_series = set(stage1["series_id"].dropna().unique())
    duplicate_pairs = stage1.duplicated(["series_id", "reference_date"]).any()
    return {
        "typed_dates": pd.api.types.is_datetime64_any_dtype(stage1["reference_date"])
        and pd.api.types.is_datetime64_any_dtype(stage1["release_date"])
        and pd.api.types.is_datetime64_any_dtype(stage1["observable_month_end"]),
        "typed_values": pd.api.types.is_numeric_dtype(stage1["value"]),
        "all_series_present": expected_series == actual_series,
        "ordered_dates": bool((stage1["reference_date"] <= stage1["release_date"]).all()),
        "unique_pairs": not bool(duplicate_pairs),
    }
