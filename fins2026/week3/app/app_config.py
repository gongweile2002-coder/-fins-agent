# ruff: noqa
"""Configuration constants for the Australia macro forecast app."""

from __future__ import annotations

from fins2026.week3.code import (
    AUSTRALIA_CONTEXT_SERIES,
    AUSTRALIA_FORECAST_SPECS,
    SAMPLE_PERIODS,
    US_CONTEXT_LABELS,
)
from fins2026.week3.code import (
    MODEL_LABELS as WEEK3_MODEL_LABELS,
)
from fins2026.week3.code import (
    ONE_STEP_ONLY_MODELS as WEEK3_ONE_STEP_ONLY_MODELS,
)
from fintools.apps import forecastable_specs

APP_TITLE = "Australia Macro Forecast Monitor"
ALL_SPECS = AUSTRALIA_FORECAST_SPECS
FORECAST_SPECS = forecastable_specs(ALL_SPECS)
FORECAST_LABELS = {spec.label: spec for spec in FORECAST_SPECS.values()}
SAMPLE_PERIOD_OPTIONS = SAMPLE_PERIODS
DEFAULT_SAMPLE_PERIOD = "20Y"
VIEW_OPTIONS = [
    "Overview",
    "Australia Snapshot",
    "Forecasts",
    "Model Comparison",
    "Backtests",
    "U.S. Context",
    "Data",
    "Methodology",
]
DATA_MODE_OPTIONS = ["Fixture", "Live"]
MONTHLY_FORECAST_SERIES = [
    label for label, spec in FORECAST_LABELS.items() if spec.frequency == "monthly"
]
QUARTERLY_FORECAST_SERIES = [
    label for label, spec in FORECAST_LABELS.items() if spec.frequency == "quarterly"
]
AUSTRALIA_STATE_OPTIONS = [
    *FORECAST_LABELS,
    *AUSTRALIA_CONTEXT_SERIES,
]
US_CONTEXT_OPTIONS = list(US_CONTEXT_LABELS)
US_CONTEXT_LABEL_MAP = US_CONTEXT_LABELS
DEFAULT_FORECAST_SERIES = "Cash rate target"
DEFAULT_MODEL = "drift"
DEFAULT_MONTHLY_HORIZON = 6
DEFAULT_QUARTERLY_HORIZON = 4
MODEL_LABELS = WEEK3_MODEL_LABELS
ONE_STEP_ONLY_MODELS = WEEK3_ONE_STEP_ONLY_MODELS
MODEL_SCOPE_NOTES = {
    "naive": "Repeats the latest target reading forward.",
    "drift": "Fits a simple recent trend in the transformed target.",
    "ar": "Chooses an autoregressive lag length by BIC.",
    "arma": "Chooses an ARMA order by BIC on the transformed target.",
    "armax": "Uses one-step ARMA + exogenous forecasts only in v1.",
    "enet": "Uses one-step lagged elastic-net regression only in v1.",
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
LEADERBOARD_LABELS = {
    "model_label": "Model",
    "status": "Status",
    "target_mae": "Target MAE",
    "target_rmse": "Target RMSE",
    "level_mae": "Level MAE",
    "level_rmse": "Level RMSE",
    "ranking_metric": "Ranking metric",
}
