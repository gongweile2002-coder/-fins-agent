# ruff: noqa
"""Week 3 Australia-first forecasting helpers."""

from __future__ import annotations

from .forecast_data import (
    AUSTRALIA_STAGE1_FIXTURE,
    build_forecast_input_bundle,
    build_us_monthly_panel,
    build_us_quarterly_panel,
    load_australia_stage1_fixture,
    load_australia_stage1_live,
    load_forecast_input_bundle,
    write_forecast_input_bundle,
)
from .forecast_pipeline import (
    MODEL_LABELS,
    ONE_STEP_ONLY_MODELS,
    benchmark_all_specs,
    build_exogenous_inputs,
    build_target_series,
    candidate_lags_for_spec,
    candidate_orders_for_spec,
    compute_spec_outputs,
    default_horizon_for_spec,
    load_saved_leaderboard,
    results_slug,
    write_benchmark_outputs,
)
from .forecast_specs import (
    AUSTRALIA_CONTEXT_SERIES,
    AUSTRALIA_FORECAST_SPECS,
    QUARTERLY_FORECAST_SERIES,
    SAMPLE_PERIODS,
    US_CONTEXT_LABELS,
)

__all__ = [
    "AUSTRALIA_CONTEXT_SERIES",
    "AUSTRALIA_FORECAST_SPECS",
    "AUSTRALIA_STAGE1_FIXTURE",
    "MODEL_LABELS",
    "ONE_STEP_ONLY_MODELS",
    "QUARTERLY_FORECAST_SERIES",
    "SAMPLE_PERIODS",
    "US_CONTEXT_LABELS",
    "benchmark_all_specs",
    "build_exogenous_inputs",
    "build_forecast_input_bundle",
    "build_target_series",
    "build_us_monthly_panel",
    "build_us_quarterly_panel",
    "candidate_lags_for_spec",
    "candidate_orders_for_spec",
    "compute_spec_outputs",
    "default_horizon_for_spec",
    "load_australia_stage1_fixture",
    "load_australia_stage1_live",
    "load_forecast_input_bundle",
    "load_saved_leaderboard",
    "results_slug",
    "write_benchmark_outputs",
    "write_forecast_input_bundle",
]
