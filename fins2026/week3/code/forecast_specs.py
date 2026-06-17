# ruff: noqa
"""Australia-first forecast series metadata."""

from __future__ import annotations

from fintools.apps import SeriesSpec

SAMPLE_PERIODS = {
    "Full": None,
    "20Y": 20,
    "10Y": 10,
    "5Y": 5,
}

AUSTRALIA_FORECAST_SPECS: dict[str, SeriesSpec] = {
    "Cash rate target": SeriesSpec(
        series_id="Cash rate target",
        label="Cash rate target",
        units="Percent",
        target="change",
        target_label="Monthly change in the cash rate target",
        target_units="Percentage points",
        role="policy",
        frequency="monthly",
        caveat=(
            "Forecasts are shown as month-to-month policy-rate changes mapped "
            "back into an implied level path."
        ),
    ),
    "10Y government bond yield": SeriesSpec(
        series_id="10Y government bond yield",
        label="10Y government bond yield",
        units="Percent",
        target="change",
        target_label="Monthly change in the 10Y government bond yield",
        target_units="Percentage points",
        role="rates",
        frequency="monthly",
        caveat=(
            "The bond-yield target is forecast as a monthly change and then "
            "rebuilt into an implied yield path."
        ),
    ),
    "Unemployment rate": SeriesSpec(
        series_id="Unemployment rate",
        label="Unemployment rate",
        units="Percent",
        target="change",
        target_label="Monthly change in the unemployment rate",
        target_units="Percentage points",
        role="labour",
        frequency="monthly",
    ),
    "Trade-weighted index": SeriesSpec(
        series_id="Trade-weighted index",
        label="Trade-weighted index",
        units="Index",
        target="log_change",
        target_label="Monthly log change in the trade-weighted index",
        target_units="Percent",
        role="external",
        frequency="monthly",
    ),
    "Commodity price index (A$)": SeriesSpec(
        series_id="Commodity price index (A$)",
        label="Commodity price index (A$)",
        units="Index",
        target="log_change",
        target_label="Monthly log change in the commodity price index (A$)",
        target_units="Percent",
        role="external",
        frequency="monthly",
    ),
    "Headline CPI inflation": SeriesSpec(
        series_id="Headline CPI inflation",
        label="Headline CPI inflation",
        units="Percent",
        target="level",
        target_label="Published year-ended headline CPI inflation",
        target_units="Percent",
        role="inflation",
        frequency="quarterly",
        caveat=(
            "Inflation is modeled on the published year-ended rate, not "
            "re-derived from the price level."
        ),
    ),
    "Wage Price Index growth": SeriesSpec(
        series_id="Wage Price Index growth",
        label="Wage Price Index growth",
        units="Percent",
        target="level",
        target_label="Published year-ended Wage Price Index growth",
        target_units="Percent",
        role="wages",
        frequency="quarterly",
    ),
    "Real GDP": SeriesSpec(
        series_id="Real GDP",
        label="Real GDP",
        units="$ million",
        target="annualized_growth",
        target_label="Annualized quarter-over-quarter real GDP growth",
        target_units="Percent",
        role="activity",
        frequency="quarterly",
        caveat=(
            "GDP is quarterly, lagged, and revised, so the output is "
            "latest-available analysis rather than a real-time vintage nowcast."
        ),
    ),
}

QUARTERLY_FORECAST_SERIES = [
    "Headline CPI inflation",
    "Wage Price Index growth",
    "Real GDP",
]

AUSTRALIA_CONTEXT_SERIES = [
    "Participation rate",
    "Employment-to-population ratio",
    "Trimmed mean inflation",
    "Vacancies to labour force ratio",
]

US_CONTEXT_LABELS = {
    "DGS10": "U.S. 10Y Treasury (%)",
    "DGS2": "U.S. 2Y Treasury (%)",
    "DTB3": "U.S. 3M Treasury bill (%)",
    "T10Y2Y": "U.S. 10Y-2Y spread (pp)",
    "VIXCLS": "U.S. VIX (%)",
    "UNRATE": "U.S. unemployment rate (%)",
    "INDPRO": "U.S. industrial production",
    "PAYEMS": "U.S. payroll employment (thousands)",
    "FEDFUNDS": "U.S. federal funds rate (%)",
    "SP500": "U.S. S&P 500 index level",
    "DGS10_CHANGE_BP": "U.S. 10Y Treasury monthly change (bp)",
    "FEDFUNDS_CHANGE_BP": "U.S. federal funds monthly change (bp)",
    "UNRATE_CHANGE_PP": "U.S. unemployment monthly change (pp)",
    "INDPRO_LOG_GROWTH_PCT": "U.S. industrial production monthly log growth (%)",
    "SP500_RETURN_PCT": "U.S. S&P 500 monthly return (%)",
}
