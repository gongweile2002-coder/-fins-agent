# ruff: noqa
"""Week 2 Australia macro data contracts and frozen classroom timing."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

CLASSROOM_SAMPLE_START = pd.Timestamp("2000-01-01")
CLASSROOM_REFERENCE_ENDPOINT = pd.Timestamp("2025-12-31")
CLASSROOM_INFORMATION_SET_MONTH_END = pd.Timestamp("2026-03-31")
AUSTRALIA_MACRO_SOURCE = "Reserve Bank of Australia statistical tables"


@dataclass(frozen=True)
class MacroSeriesSpec:
    """Metadata contract for one Australian macro series."""

    series_id: str
    display_name: str
    source_table: str
    native_frequency: str
    reference_date_rule: str
    observable_lag_months: int
    units: str
    default_transform: str
    include_in_dec2025_core_pack: bool = True
    category: str = ""
    note: str = ""


AUSTRALIA_MACRO_SPECS: dict[str, MacroSeriesSpec] = {
    "FIRMMCRT": MacroSeriesSpec(
        series_id="FIRMMCRT",
        display_name="Cash rate target",
        source_table="F1.1",
        native_frequency="daily",
        reference_date_rule="calendar_date",
        observable_lag_months=0,
        units="Per cent",
        default_transform="level_and_basis_points_change",
        category="policy",
    ),
    "FCMYGBAG10": MacroSeriesSpec(
        series_id="FCMYGBAG10",
        display_name="10Y government bond yield",
        source_table="F2.1",
        native_frequency="daily",
        reference_date_rule="calendar_date",
        observable_lag_months=0,
        units="Per cent per annum",
        default_transform="level_and_basis_points_change",
        category="rates",
    ),
    "FXRTWI": MacroSeriesSpec(
        series_id="FXRTWI",
        display_name="Trade-weighted index",
        source_table="F11",
        native_frequency="monthly",
        reference_date_rule="month_end",
        observable_lag_months=0,
        units="Index",
        default_transform="level_and_log_change",
        category="external",
        note="Combined from historical and current RBA F11 files.",
    ),
    "GCPIAGYP": MacroSeriesSpec(
        series_id="GCPIAGYP",
        display_name="Headline CPI inflation",
        source_table="G1",
        native_frequency="quarterly",
        reference_date_rule="quarter_end",
        observable_lag_months=1,
        units="Per cent",
        default_transform="year_ended_rate",
        category="inflation",
    ),
    "GCPIOCPMTMYP": MacroSeriesSpec(
        series_id="GCPIOCPMTMYP",
        display_name="Trimmed mean inflation",
        source_table="G1",
        native_frequency="quarterly",
        reference_date_rule="quarter_end",
        observable_lag_months=1,
        units="Per cent",
        default_transform="year_ended_rate",
        include_in_dec2025_core_pack=False,
        category="inflation",
    ),
    "GCPIAGYPM": MacroSeriesSpec(
        series_id="GCPIAGYPM",
        display_name="Monthly headline CPI inflation",
        source_table="G4",
        native_frequency="monthly",
        reference_date_rule="month_end",
        observable_lag_months=1,
        units="Per cent",
        default_transform="year_ended_rate",
        include_in_dec2025_core_pack=False,
        category="inflation",
        note="Short monthly inflation history used for the post-2024 transition lesson only.",
    ),
    "GGDPCVGDP": MacroSeriesSpec(
        series_id="GGDPCVGDP",
        display_name="Real GDP",
        source_table="H1",
        native_frequency="quarterly",
        reference_date_rule="quarter_end",
        observable_lag_months=3,
        units="$ million",
        default_transform="quarterly_log_growth",
        include_in_dec2025_core_pack=False,
        category="activity",
    ),
    "GGDPCVGDPY": MacroSeriesSpec(
        series_id="GGDPCVGDPY",
        display_name="Real GDP growth",
        source_table="H1",
        native_frequency="quarterly",
        reference_date_rule="quarter_end",
        observable_lag_months=3,
        units="Per cent",
        default_transform="year_ended_rate",
        category="activity",
    ),
    "GWPIYP": MacroSeriesSpec(
        series_id="GWPIYP",
        display_name="Wage Price Index growth",
        source_table="H4",
        native_frequency="quarterly",
        reference_date_rule="quarter_end",
        observable_lag_months=2,
        units="Per cent",
        default_transform="year_ended_rate",
        category="wages",
    ),
    "GWPIQP": MacroSeriesSpec(
        series_id="GWPIQP",
        display_name="Wage Price Index quarterly growth",
        source_table="H4",
        native_frequency="quarterly",
        reference_date_rule="quarter_end",
        observable_lag_months=2,
        units="Per cent",
        default_transform="quarterly_rate",
        include_in_dec2025_core_pack=False,
        category="wages",
    ),
    "GLFSURSA": MacroSeriesSpec(
        series_id="GLFSURSA",
        display_name="Unemployment rate",
        source_table="H5",
        native_frequency="monthly",
        reference_date_rule="month_end",
        observable_lag_months=1,
        units="Per cent",
        default_transform="level_and_percentage_points_change",
        category="labour",
    ),
    "GLFSPRSA": MacroSeriesSpec(
        series_id="GLFSPRSA",
        display_name="Participation rate",
        source_table="H5",
        native_frequency="monthly",
        reference_date_rule="month_end",
        observable_lag_months=1,
        units="Per cent",
        default_transform="level_and_percentage_points_change",
        include_in_dec2025_core_pack=False,
        category="labour",
    ),
    "GLFSEPTPOP": MacroSeriesSpec(
        series_id="GLFSEPTPOP",
        display_name="Employment-to-population ratio",
        source_table="H5",
        native_frequency="monthly",
        reference_date_rule="month_end",
        observable_lag_months=1,
        units="Per cent",
        default_transform="level_and_percentage_points_change",
        include_in_dec2025_core_pack=False,
        category="labour",
    ),
    "GLFOSVT": MacroSeriesSpec(
        series_id="GLFOSVT",
        display_name="Job vacancies",
        source_table="H5",
        native_frequency="point_in_time_quarterly",
        reference_date_rule="survey_month_end",
        observable_lag_months=2,
        units="'000",
        default_transform="level",
        include_in_dec2025_core_pack=False,
        category="labour",
        note=(
            "ABS vacancies are point-in-time quarterly observations rather "
            "than a December quarter flow."
        ),
    ),
    "GLFOSVTLF": MacroSeriesSpec(
        series_id="GLFOSVTLF",
        display_name="Vacancies to labour force ratio",
        source_table="H5",
        native_frequency="point_in_time_quarterly",
        reference_date_rule="survey_month_end",
        observable_lag_months=2,
        units="Per cent",
        default_transform="level",
        include_in_dec2025_core_pack=False,
        category="labour",
    ),
    "GRCPAIAD": MacroSeriesSpec(
        series_id="GRCPAIAD",
        display_name="Commodity price index (A$)",
        source_table="I2",
        native_frequency="monthly",
        reference_date_rule="month_end",
        observable_lag_months=0,
        units="Index, 2024/25=100",
        default_transform="level_and_log_change",
        category="external",
    ),
}


CORE_COMPARISON_SERIES = [
    "GLFSURSA",
    "GCPIAGYP",
    "GWPIYP",
    "GGDPCVGDPY",
    "FIRMMCRT",
    "FCMYGBAG10",
    "FXRTWI",
    "GRCPAIAD",
]

CORE_SMALL_MULTIPLE_SERIES = [
    "GLFSURSA",
    "GCPIAGYP",
    "GCPIOCPMTMYP",
    "GWPIYP",
    "GGDPCVGDPY",
    "FIRMMCRT",
    "FCMYGBAG10",
    "FXRTWI",
]

LABOUR_TIGHTNESS_SERIES = [
    "GLFSURSA",
    "GLFSPRSA",
    "GLFSEPTPOP",
    "GLFOSVT",
]

TIGHTENING_EPISODE_SERIES = [
    "FIRMMCRT",
    "GLFSURSA",
    "GCPIAGYP",
    "GWPIYP",
]


def core_comparison_specs() -> list[MacroSeriesSpec]:
    """Return the display order for the common-endpoint comparison pack."""

    return [AUSTRALIA_MACRO_SPECS[series_id] for series_id in CORE_COMPARISON_SERIES]


def required_source_tables() -> list[str]:
    """Return the RBA source tables needed for the Week 2 Australia pack."""

    tables = {spec.source_table for spec in AUSTRALIA_MACRO_SPECS.values()}
    tables.add("F11HIST")
    return sorted(tables)
