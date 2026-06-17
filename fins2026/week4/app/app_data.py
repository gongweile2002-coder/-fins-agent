# ruff: noqa
"""Cached data loaders for the Week 4 portfolio app."""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from fins2026.week4.app.app_config import SAMPLE_PERIOD_OPTIONS
from fins2026.week4.code.stage4_app import (
    Stage4AppBundle,
    build_live_app_bundle,
    filter_feature_panel_for_sample_period,
    filter_price_panel_for_sample_period,
    load_fixture_app_bundle,
)


@st.cache_data(ttl=86400)
def _load_fixture_bundle() -> Stage4AppBundle:
    return load_fixture_app_bundle()


@st.cache_data(ttl=86400)
def _load_live_bundle() -> tuple[Stage4AppBundle, pd.Timestamp]:
    return build_live_app_bundle(), pd.Timestamp.now(tz="UTC")


def load_week4_app_bundle() -> tuple[Stage4AppBundle, str, str | None, pd.Timestamp | None]:
    """Load live Week 4 app data, falling back to the committed fixture."""

    if os.environ.get("WEEK4_APP_FORCE_FIXTURE") == "1":
        return _load_fixture_bundle(), "Fixture", None, None

    try:
        bundle, loaded_at = _load_live_bundle()
        return bundle, "Live", None, loaded_at
    except Exception as exc:
        warning = (
            "Live Yahoo or Kenneth French refresh failed, so the app is showing the "
            f"committed fallback snapshot instead. Technical detail: {exc}"
        )
        return _load_fixture_bundle(), "Fixture", warning, None


def apply_sample_period(bundle: Stage4AppBundle, sample_period: str) -> Stage4AppBundle:
    """Restrict the app bundle to the selected analysis window."""

    _ = SAMPLE_PERIOD_OPTIONS[sample_period]
    return Stage4AppBundle(
        price_panel=filter_price_panel_for_sample_period(bundle.price_panel, sample_period),
        feature_panel=filter_feature_panel_for_sample_period(bundle.feature_panel, sample_period),
        latest_observation_date=bundle.latest_observation_date,
    )


def source_status_text(
    bundle: Stage4AppBundle,
    *,
    active_source: str,
    loaded_at_utc: pd.Timestamp | None = None,
    warning: str | None = None,
) -> str:
    """Return client-facing source freshness text for the app."""

    latest_price_date = pd.to_datetime(bundle.price_panel["date"]).max()
    latest_feature_date = pd.to_datetime(bundle.feature_panel["date"]).max()
    if active_source == "Live":
        loaded_text = "n/a" if loaded_at_utc is None else f"{loaded_at_utc:%Y-%m-%d %H:%M} UTC"
        return (
            f"Live Yahoo prices and Kenneth French daily risk-free rates loaded at {loaded_text}; "
            f"price history through {latest_price_date:%Y-%m-%d}; return panel through "
            f"{latest_feature_date:%Y-%m-%d}."
        )
    if warning:
        return (
            f"Fallback snapshot through prices {latest_price_date:%Y-%m-%d} and return panel "
            f"{latest_feature_date:%Y-%m-%d}."
        )
    return (
        f"Fallback snapshot through prices {latest_price_date:%Y-%m-%d} and return panel "
        f"{latest_feature_date:%Y-%m-%d}."
    )
