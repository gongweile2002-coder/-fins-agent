# ruff: noqa
"""Week 2 Australia macro data loaders, panels, and feature builders."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from io import BytesIO, StringIO
from typing import Any

import numpy as np
import pandas as pd
import requests

from fins2026.week2.code.australia_macro_specs import (
    AUSTRALIA_MACRO_SOURCE,
    AUSTRALIA_MACRO_SPECS,
    CLASSROOM_INFORMATION_SET_MONTH_END,
    CLASSROOM_REFERENCE_ENDPOINT,
    CLASSROOM_SAMPLE_START,
    MacroSeriesSpec,
    core_comparison_specs,
    required_source_tables,
)

RBA_CSV_URLS = {
    "F1.1": "https://www.rba.gov.au/statistics/tables/csv/f1.1-data.csv",
    "F2.1": "https://www.rba.gov.au/statistics/tables/csv/f2.1-data.csv",
    "F11": "https://www.rba.gov.au/statistics/tables/csv/f11-data.csv",
    "G1": "https://www.rba.gov.au/statistics/tables/csv/g1-data.csv",
    "G4": "https://www.rba.gov.au/statistics/tables/csv/g4-data.csv",
    "H1": "https://www.rba.gov.au/statistics/tables/csv/h1-data.csv",
    "H4": "https://www.rba.gov.au/statistics/tables/csv/h4-data.csv",
    "H5": "https://www.rba.gov.au/statistics/tables/csv/h5-data.csv",
    "I2": "https://www.rba.gov.au/statistics/tables/csv/i2-data.csv",
}
RBA_HISTORICAL_XLS_URLS = {
    "F11HIST": "https://www.rba.gov.au/statistics/tables/xls-hist/f11hist-1969-2009.xls",
}


@dataclass(frozen=True)
class RBATableBundle:
    """One parsed RBA table plus its per-series metadata."""

    table_code: str
    table_title: str
    frame: pd.DataFrame
    metadata: pd.DataFrame
    publication_date: pd.Timestamp | pd.NaT


def month_end_timestamp(value: object) -> pd.Timestamp:
    """Return the month-end timestamp for an observation date."""

    return pd.Timestamp(value).to_period("M").to_timestamp("M")


def shift_month_end(value: object, months: int) -> pd.Timestamp:
    """Shift a month-end timestamp forward by whole month-end steps."""

    return month_end_timestamp(value) + pd.offsets.MonthEnd(months)


def sparse_difference(series: pd.Series, *, scale: float = 1.0) -> pd.Series:
    """Difference sparse month-end or quarter-end data without filling gaps first."""

    clean = series.dropna().astype(float)
    return (clean.diff() * scale).reindex(series.index)


def sparse_log_change_percent(series: pd.Series) -> pd.Series:
    """Compute sparse log changes in percent without forcing dense fills first."""

    clean = series.dropna().astype(float)
    prior = clean.shift(1)
    ratio = clean / prior
    values = np.log(ratio.where((clean > 0) & (prior > 0))) * 100.0
    return values.reindex(series.index)


def quarterly_average_from_monthly(series: pd.Series) -> pd.Series:
    """Average monthly observations within each quarter and stamp to quarter-end."""

    clean = series.dropna().astype(float)
    quarterly = clean.resample("QE").mean()
    quarterly.index = quarterly.index.to_period("Q").to_timestamp("Q")
    return quarterly


def _parse_optional_date(value: object) -> pd.Timestamp | pd.NaT:
    if isinstance(value, pd.Timestamp):
        return value
    text = str(value).strip()
    if not text:
        return pd.NaT
    dayfirst = "/" in text
    parsed = pd.to_datetime(text, dayfirst=dayfirst, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    return pd.Timestamp(parsed)


def _normalize_frequency(value: object) -> str:
    text = str(value).strip().lower()
    if "day" in text:
        return "daily"
    if "month" in text:
        return "monthly"
    if "quarter" in text:
        return "quarterly"
    return text or "unknown"


def _rows_to_text_matrix(rows: list[list[Any]]) -> list[list[str]]:
    return [[("" if pd.isna(item) else str(item)).strip() for item in row] for row in rows]


def _safe_slice(row: list[str], start: int) -> list[str]:
    return row[start:] if len(row) > start else []


def _parse_rba_rows(rows: list[list[str]], table_code: str) -> RBATableBundle:
    """Parse the common RBA table layout shared by CSV and historical XLS files."""

    table_title = rows[0][0] if rows and rows[0] else table_code
    title_row = _safe_slice(rows[1], 1)
    description_row = _safe_slice(rows[2], 1)
    frequency_row = _safe_slice(rows[3], 1)
    type_row = _safe_slice(rows[4], 1)
    units_row = _safe_slice(rows[5], 1)
    source_row = _safe_slice(rows[8], 1)
    publication_row = _safe_slice(rows[9], 1)
    series_row = _safe_slice(rows[10], 1)
    series_count = len(series_row)

    records: list[dict[str, Any]] = []
    for row in rows[11:]:
        if not row:
            continue
        date_text = row[0].strip() if row[0] else ""
        date_value = _parse_optional_date(date_text)
        if pd.isna(date_value):
            continue
        padded = list(row[1:1 + series_count])
        if len(padded) < series_count:
            padded.extend([""] * (series_count - len(padded)))
        record: dict[str, Any] = {"date": pd.Timestamp(date_value)}
        for series_id, raw_value in zip(series_row, padded, strict=False):
            if not series_id:
                continue
            record[series_id] = pd.to_numeric(raw_value, errors="coerce")
        records.append(record)

    frame = pd.DataFrame.from_records(records).sort_values("date").reset_index(drop=True)
    metadata_records: list[dict[str, Any]] = []
    for index, series_id in enumerate(series_row):
        if not series_id:
            continue
        metadata_records.append(
            {
                "table_code": table_code,
                "table_title": table_title,
                "series_id": series_id,
                "series_title": title_row[index] if index < len(title_row) else "",
                "series_description": (
                    description_row[index] if index < len(description_row) else ""
                ),
                "native_frequency": (
                    _normalize_frequency(frequency_row[index])
                    if index < len(frequency_row)
                    else "unknown"
                ),
                "series_type": type_row[index] if index < len(type_row) else "",
                "units": units_row[index] if index < len(units_row) else "",
                "source_name": (
                    source_row[index]
                    if index < len(source_row)
                    else AUSTRALIA_MACRO_SOURCE
                ),
                "publication_date": (
                    _parse_optional_date(publication_row[index])
                    if index < len(publication_row)
                    else pd.NaT
                ),
            }
        )
    metadata = pd.DataFrame.from_records(metadata_records)
    publication_date = (
        metadata["publication_date"].dropna().max()
        if not metadata.empty
        else pd.NaT
    )
    return RBATableBundle(
        table_code=table_code,
        table_title=table_title,
        frame=frame,
        metadata=metadata,
        publication_date=publication_date,
    )


def parse_rba_csv_text(text: str, table_code: str) -> RBATableBundle:
    """Parse an RBA CSV statistical table into a typed dataframe plus metadata."""

    rows = list(csv.reader(StringIO(text)))
    return _parse_rba_rows(rows, table_code)


def parse_rba_historical_xls(content: bytes, table_code: str) -> RBATableBundle:
    """Parse an RBA historical XLS table using the shared metadata layout."""

    workbook = pd.read_excel(BytesIO(content), sheet_name="Data", header=None)
    rows = _rows_to_text_matrix(workbook.values.tolist())
    return _parse_rba_rows(rows, table_code)


def download_rba_table_bundle(table_code: str) -> RBATableBundle:
    """Download and parse one live RBA table."""

    if table_code in RBA_CSV_URLS:
        response = requests.get(RBA_CSV_URLS[table_code], timeout=60)
        response.raise_for_status()
        return parse_rba_csv_text(response.text, table_code)
    if table_code in RBA_HISTORICAL_XLS_URLS:
        response = requests.get(RBA_HISTORICAL_XLS_URLS[table_code], timeout=60)
        response.raise_for_status()
        return parse_rba_historical_xls(response.content, table_code)
    raise KeyError(f"unknown RBA table code: {table_code}")


def download_live_rba_bundles(table_codes: list[str] | None = None) -> dict[str, RBATableBundle]:
    """Download the Week 2 Australia source tables from official RBA URLs."""

    return {
        table_code: download_rba_table_bundle(table_code)
        for table_code in (table_codes or required_source_tables())
    }


def _series_from_bundle(bundle: RBATableBundle, series_id: str) -> pd.DataFrame:
    if series_id not in bundle.frame.columns:
        raise KeyError(f"{series_id} not found in RBA table {bundle.table_code}")
    frame = bundle.frame[["date", series_id]].rename(columns={series_id: "value"}).copy()
    return frame.dropna(subset=["value"]).reset_index(drop=True)


def resolve_series_history(
    series_id: str,
    bundles: dict[str, RBATableBundle],
) -> pd.DataFrame:
    """Resolve one spec series, including the historical/current TWI stitch."""

    if series_id == "FXRTWI":
        historical = _series_from_bundle(bundles["F11HIST"], series_id)
        current = _series_from_bundle(bundles["F11"], series_id)
        combined = (
            pd.concat([historical, current], ignore_index=True)
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )
        return combined

    spec = AUSTRALIA_MACRO_SPECS[series_id]
    return _series_from_bundle(bundles[spec.source_table], series_id)


def _reference_date_for_spec(raw_date: pd.Series, spec: MacroSeriesSpec) -> pd.Series:
    if spec.reference_date_rule in {"month_end", "survey_month_end"}:
        return raw_date.dt.to_period("M").dt.to_timestamp("M")
    if spec.reference_date_rule == "quarter_end":
        return raw_date.dt.to_period("Q").dt.to_timestamp("Q")
    return raw_date


def build_stage1_long_table(
    bundles: dict[str, RBATableBundle],
    *,
    sample_start: pd.Timestamp = CLASSROOM_SAMPLE_START,
    reference_endpoint: pd.Timestamp = CLASSROOM_REFERENCE_ENDPOINT,
) -> pd.DataFrame:
    """Build the Stage 1 long table with explicit timing and observability fields."""

    records: list[pd.DataFrame] = []
    for series_id, spec in AUSTRALIA_MACRO_SPECS.items():
        history = resolve_series_history(series_id, bundles).copy()
        history["raw_date"] = pd.to_datetime(history["date"], errors="coerce")
        history["reference_date"] = _reference_date_for_spec(history["raw_date"], spec)
        history = history.dropna(subset=["reference_date", "value"])
        history = history[
            history["reference_date"].between(sample_start, reference_endpoint)
        ].copy()
        if history.empty:
            continue
        history["release_date"] = history["reference_date"].map(
            lambda value: shift_month_end(value, spec.observable_lag_months)
        )
        history["observable_month_end"] = history["release_date"]
        history["series_id"] = spec.series_id
        history["display_name"] = spec.display_name
        history["source_name"] = AUSTRALIA_MACRO_SOURCE
        history["source_table"] = spec.source_table
        history["native_frequency"] = spec.native_frequency
        history["reference_date_rule"] = spec.reference_date_rule
        history["observable_lag_months"] = spec.observable_lag_months
        history["units"] = spec.units
        history["default_transform"] = spec.default_transform
        history["include_in_dec2025_core_pack"] = spec.include_in_dec2025_core_pack
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
                    "include_in_dec2025_core_pack",
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


def load_fixture_long_table(path: str | bytes | Any) -> pd.DataFrame:
    """Load the committed Stage 1 Australia fixture from CSV."""

    frame = pd.read_csv(path)
    for column in ["raw_date", "reference_date", "release_date", "observable_month_end"]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["include_in_dec2025_core_pack"] = frame["include_in_dec2025_core_pack"].astype(bool)
    return frame


def build_monthly_native_panel(stage1: pd.DataFrame) -> pd.DataFrame:
    """Return a native-frequency monthly panel for Stage 1 teaching."""

    monthly = stage1.loc[stage1["native_frequency"] == "monthly"].copy()
    panel = monthly.pivot_table(
        index="reference_date",
        columns="display_name",
        values="value",
        aggfunc="last",
    ).sort_index()
    panel.index.name = "date"
    return panel


def build_quarterly_native_panel(stage1: pd.DataFrame) -> pd.DataFrame:
    """Return a native-frequency quarterly panel for Stage 1 teaching."""

    quarterly = stage1.loc[
        stage1["native_frequency"].isin(["quarterly", "point_in_time_quarterly"])
    ].copy()
    panel = quarterly.pivot_table(
        index="reference_date",
        columns="display_name",
        values="value",
        aggfunc="last",
    ).sort_index()
    panel.index.name = "date"
    return panel


def build_reference_panel(
    stage1: pd.DataFrame,
    *,
    sample_start: pd.Timestamp = CLASSROOM_SAMPLE_START,
    reference_endpoint: pd.Timestamp = CLASSROOM_REFERENCE_ENDPOINT,
) -> pd.DataFrame:
    """Build the Stage 2 reference-date panel without forward-filling sparse series."""

    month_index = pd.date_range(
        start=month_end_timestamp(sample_start),
        end=month_end_timestamp(reference_endpoint),
        freq="ME",
    )
    panel = pd.DataFrame(index=month_index)
    for series_id, spec in AUSTRALIA_MACRO_SPECS.items():
        subset = stage1.loc[stage1["series_id"] == series_id, ["reference_date", "value"]].copy()
        if subset.empty:
            continue
        if spec.native_frequency == "daily":
            series = (
                subset.assign(date=subset["reference_date"].dt.to_period("M").dt.to_timestamp("M"))
                .groupby("date")["value"]
                .last()
            )
        else:
            series = (
                subset.assign(date=subset["reference_date"].dt.to_period("M").dt.to_timestamp("M"))
                .groupby("date")["value"]
                .last()
            )
        panel[spec.display_name] = series.reindex(month_index)
    panel.index.name = "date"
    return panel


def build_observable_panel(
    stage1: pd.DataFrame,
    *,
    sample_start: pd.Timestamp = CLASSROOM_SAMPLE_START,
    information_set_month_end: pd.Timestamp = CLASSROOM_INFORMATION_SET_MONTH_END,
) -> pd.DataFrame:
    """Build the classroom observable-month-end panel with post-release carries only."""

    month_index = pd.date_range(
        start=month_end_timestamp(sample_start),
        end=month_end_timestamp(information_set_month_end),
        freq="ME",
    )
    panel = pd.DataFrame(index=month_index)
    for series_id, spec in AUSTRALIA_MACRO_SPECS.items():
        subset = stage1.loc[
            stage1["series_id"] == series_id, ["observable_month_end", "value"]
        ].copy()
        if subset.empty:
            continue
        series = subset.groupby("observable_month_end")["value"].last().sort_index()
        panel[spec.display_name] = series.reindex(month_index).ffill()
    panel.index.name = "date"
    return panel


def build_feature_panel(reference_panel: pd.DataFrame) -> pd.DataFrame:
    """Build the Stage 2 communication-ready feature panel for the figures."""

    panel = reference_panel.copy()
    if "Cash rate target" in panel:
        panel["Cash rate target change (bp)"] = sparse_difference(
            panel["Cash rate target"],
            scale=100.0,
        )
    if "10Y government bond yield" in panel:
        panel["10Y government bond yield change (bp)"] = sparse_difference(
            panel["10Y government bond yield"],
            scale=100.0,
        )
    if "Unemployment rate" in panel:
        panel["Unemployment rate change (pp)"] = sparse_difference(
            panel["Unemployment rate"],
            scale=1.0,
        )
    if "Participation rate" in panel:
        panel["Participation rate change (pp)"] = sparse_difference(
            panel["Participation rate"],
            scale=1.0,
        )
    if "Trade-weighted index" in panel:
        panel["Trade-weighted index log change (%)"] = sparse_log_change_percent(
            panel["Trade-weighted index"]
        )
    if "Commodity price index (A$)" in panel:
        panel["Commodity price index (A$) log change (%)"] = sparse_log_change_percent(
            panel["Commodity price index (A$)"]
        )
    if "Real GDP" in panel:
        panel["Real GDP quarterly log growth (%)"] = sparse_log_change_percent(
            panel["Real GDP"]
        )
    return panel


def build_common_endpoint_snapshot(
    reference_panel: pd.DataFrame,
    *,
    start_date: str = "2000-12-31",
    end_date: str = "2025-12-31",
) -> pd.DataFrame:
    """Build the December 2000 versus December 2025 comparison frame."""

    records: list[dict[str, Any]] = []
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    for spec in core_comparison_specs():
        if spec.display_name not in reference_panel:
            continue
        start_value = reference_panel.at[start_ts, spec.display_name]
        end_value = reference_panel.at[end_ts, spec.display_name]
        if pd.isna(start_value) or pd.isna(end_value):
            continue
        records.append(
            {
                "Series": spec.display_name,
                "2000-12": float(start_value),
                "2025-12": float(end_value),
                "Units": spec.units,
            }
        )
    return pd.DataFrame.from_records(records)


def build_release_lag_frame() -> pd.DataFrame:
    """Return the frozen common-endpoint observability map used in the lecture pack."""

    records = []
    for spec in core_comparison_specs():
        records.append(
            {
                "Series": spec.display_name,
                "Reference month-end": 0.0,
                "First observable month-end": float(spec.observable_lag_months),
            }
        )
    return pd.DataFrame.from_records(records)


def build_wage_phillips_frame(feature_panel: pd.DataFrame) -> pd.DataFrame:
    """Pair quarterly wage growth with the same-quarter unemployment rate."""

    frame = feature_panel[
        ["Unemployment rate", "Wage Price Index growth"]
    ].dropna().copy()
    frame.index.name = "date"
    return frame


def build_okun_frame(feature_panel: pd.DataFrame) -> pd.DataFrame:
    """Build a quarterly Okun-style frame from GDP growth and unemployment changes."""

    unemployment = quarterly_average_from_monthly(feature_panel["Unemployment rate"])
    unemployment_change = unemployment.diff()
    gdp_growth = feature_panel["Real GDP quarterly log growth (%)"].dropna()
    frame = pd.concat(
        [
            gdp_growth.rename("Real GDP quarterly log growth (%)"),
            unemployment_change.rename("Change in quarterly average unemployment rate (pp)"),
        ],
        axis=1,
    ).dropna()
    frame.index.name = "date"
    return frame


def build_beveridge_frame(reference_panel: pd.DataFrame) -> pd.DataFrame:
    """Build the vacancies-rate versus unemployment-rate quarterly pairing."""

    frame = reference_panel[
        ["Vacancies to labour force ratio", "Unemployment rate"]
    ].dropna().copy()
    frame.index.name = "date"
    return frame


def build_external_relationship_frame(feature_panel: pd.DataFrame) -> pd.DataFrame:
    """Pair monthly commodity-price and TWI changes for the external-sector scatter."""

    frame = feature_panel[
        [
            "Commodity price index (A$) log change (%)",
            "Trade-weighted index log change (%)",
        ]
    ].dropna().copy()
    frame.index.name = "date"
    return frame
