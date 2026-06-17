# ruff: noqa
"""Configuration constants for the Week 3 companion U.S. macro app."""

from __future__ import annotations

from fintools.apps import forecastable_specs, week2_gdp_specs, week2_market_specs

APP_TITLE = "U.S. Macro Stress and Forecast Monitor"
MARKET_SPECS = week2_market_specs()
GDP_SPECS = week2_gdp_specs()
ALL_SPECS = {**MARKET_SPECS, **GDP_SPECS}
FORECAST_SPECS = forecastable_specs(ALL_SPECS)
FRED_SERIES = tuple(ALL_SPECS)
SAMPLE_PERIODS = {
    "Full": None,
    "40Y": 40,
    "20Y": 20,
    "10Y": 10,
    "5Y": 5,
}
DEFAULT_SAMPLE_PERIOD = "20Y"
STRESS_CURRENT_WINDOW = 21
MODEL_LABELS = {
    "drift": "Drift",
    "ar1": "AR(1)",
    "naive": "Naive",
}
VIEW_OPTIONS = [
    "Overview",
    "Stress Score",
    "Yield Curve",
    "Forecasts",
    "Backtests",
    "GDP Outlook",
    "Data",
    "Methodology",
]
DATA_MODE_OPTIONS = ["Fixture", "Live FRED"]
FORECAST_LABELS = {spec.label: spec for spec in FORECAST_SPECS.values()}
DATA_LABELS = {
    "DGS10": "10-Year Treasury (%)",
    "DGS2": "2-Year Treasury (%)",
    "DTB3": "3-Month Treasury Bill (%)",
    "T10Y2Y": "10Y-2Y Treasury Spread (%)",
    "VIXCLS": "VIX (%)",
    "BAMLH0A0HYM2": "High-Yield OAS (%)",
    "GDPC1": "Real GDP (bn chained 2017 $)",
}
BACKTEST_LABELS = {
    "actual": "Actual target",
    "forecast": "Forecast target",
    "error": "Target error",
    "absolute_error": "Absolute target error",
    "actual_level": "Actual level",
    "forecast_level": "Forecast level",
    "level_error": "Level error",
    "absolute_level_error": "Absolute level error",
}
FORECAST_OUTPUT_LABELS = {
    "forecast": "Forecast",
    "lower": "Lower band",
    "upper": "Upper band",
}
