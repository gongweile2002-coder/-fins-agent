# ruff: noqa
"""Cached data and model-loading helpers for the Australia macro forecast app."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fins2026.week3.app.app_config import ALL_SPECS, DEFAULT_SAMPLE_PERIOD, SAMPLE_PERIOD_OPTIONS
from fins2026.week3.code import (
    benchmark_all_specs,
    compute_spec_outputs,
    load_forecast_input_bundle,
    load_saved_leaderboard,
)


@st.cache_data(ttl=86400)
def _load_fixture_bundle() -> dict[str, pd.DataFrame]:
    return load_forecast_input_bundle(use_fixture=True, rebuild=False)


@st.cache_data(ttl=86400)
def _load_live_bundle() -> tuple[dict[str, pd.DataFrame], pd.Timestamp]:
    return load_forecast_input_bundle(use_fixture=False, rebuild=True), pd.Timestamp.now(tz="UTC")


def apply_sample_period(frame: pd.DataFrame, sample_period: str) -> pd.DataFrame:
    """Restrict a date-indexed dataframe to the selected analysis sample."""

    years = SAMPLE_PERIOD_OPTIONS[sample_period]
    if years is None or frame.empty:
        return frame.copy()
    cutoff = frame.index.max() - pd.DateOffset(years=years)
    return frame.loc[frame.index >= cutoff].copy()


def _sampled_bundle(bundle: dict[str, pd.DataFrame], sample_period: str) -> dict[str, pd.DataFrame]:
    return {
        key: apply_sample_period(frame, sample_period)
        for key, frame in bundle.items()
        if key != "australia_stage1"
    }


def load_week3_data(
    data_mode: str,
) -> tuple[dict[str, pd.DataFrame], str, str | None, pd.Timestamp | None]:
    """Load the requested data source, falling back to fixtures when needed."""

    if data_mode == "Fixture":
        return _load_fixture_bundle(), "Fixture", None, None
    try:
        bundle, loaded_at_utc = _load_live_bundle()
        return bundle, "Live", None, loaded_at_utc
    except Exception as exc:
        warning = (
            "Live Australia and U.S. refresh failed, so the app is showing the "
            f"frozen fixture bundle instead. Technical detail: {exc}"
        )
        return _load_fixture_bundle(), "Fixture", warning, None


def source_status_text(
    bundle: dict[str, pd.DataFrame],
    *,
    active_data_mode: str,
    loaded_at_utc: pd.Timestamp | None = None,
    warning: str | None = None,
) -> str:
    """Return app-facing source freshness text for the Australia/U.S. bundle."""

    au_monthly = bundle["australia_monthly"]
    au_quarterly = bundle["australia_quarterly"]
    us_monthly = bundle["us_monthly"]
    au_monthly_latest = "n/a" if au_monthly.empty else f"{au_monthly.index.max():%Y-%m-%d}"
    au_quarterly_latest = "n/a" if au_quarterly.empty else f"{au_quarterly.index.max():%Y-%m-%d}"
    us_latest = "n/a" if us_monthly.empty else f"{us_monthly.index.max():%Y-%m-%d}"
    if active_data_mode == "Live":
        loaded = "n/a" if loaded_at_utc is None else f"{loaded_at_utc:%Y-%m-%d %H:%M} UTC"
        return (
            f"Live RBA + FRED cache loaded at {loaded}; Australia monthly observable panel "
            f"through {au_monthly_latest}; Australia quarterly release surface through "
            f"{au_quarterly_latest}; U.S. month-end context through {us_latest}."
        )
    if warning:
        return (
            f"Fixture fallback loaded; Australia monthly observable panel through "
            f"{au_monthly_latest}; Australia quarterly release surface through "
            f"{au_quarterly_latest}; U.S. month-end context through {us_latest}."
        )
    return (
        f"Fixture snapshot through Australia monthly {au_monthly_latest}, "
        f"Australia quarterly {au_quarterly_latest}, and U.S. month-end context {us_latest}."
    )


@st.cache_data(ttl=86400)
def load_benchmark_leaderboard(data_mode: str, sample_period: str) -> pd.DataFrame:
    """Load a saved leaderboard when possible, or compute the sample-specific fallback."""

    if data_mode == "Fixture" and sample_period == DEFAULT_SAMPLE_PERIOD:
        saved = load_saved_leaderboard()
        if saved is not None:
            return saved
    bundle = _load_fixture_bundle() if data_mode == "Fixture" else _load_live_bundle()[0]
    sample = _sampled_bundle(bundle, sample_period)
    leaderboard, _ = benchmark_all_specs(
        sample["australia_monthly"],
        sample["australia_quarterly"],
        sample["us_monthly"],
    )
    return leaderboard


@st.cache_data(ttl=86400)
def load_model_outputs(
    data_mode: str,
    sample_period: str,
    spec_label: str,
    model: str,
    horizon: int,
):
    """Load one forecast/backtest pair for the selected sample and model."""

    bundle = _load_fixture_bundle() if data_mode == "Fixture" else _load_live_bundle()[0]
    sample = _sampled_bundle(bundle, sample_period)
    spec = ALL_SPECS[spec_label]
    return compute_spec_outputs(
        spec,
        sample["australia_monthly"],
        sample["australia_quarterly"],
        sample["us_monthly"],
        model=model,
        horizon=horizon,
    )
