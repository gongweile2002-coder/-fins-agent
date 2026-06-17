# ruff: noqa
"""Cached data loaders for the Week 5 client-facing crypto fund app."""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from fins2026.week5.code.stage4_app import (
    Stage4AppBundle,
    Stage4ScenarioBundle,
    build_app_scenario_bundle,
    build_design_config,
    build_live_app_bundle,
    build_published_config,
    load_fixture_app_bundle,
)


@st.cache_data(ttl=86400)
def _load_fixture_bundle() -> Stage4AppBundle:
    return load_fixture_app_bundle()


@st.cache_data(ttl=86400)
def _load_live_bundle() -> tuple[Stage4AppBundle, pd.Timestamp]:
    return build_live_app_bundle(), pd.Timestamp.now(tz="UTC")


def load_week5_app_bundle() -> tuple[Stage4AppBundle, str, str | None, pd.Timestamp | None]:
    """Load live Week 5 app data, falling back to the committed fixture."""

    if os.environ.get("WEEK5_APP_FORCE_FIXTURE") == "1":
        return _load_fixture_bundle(), "Fixture", None, None

    try:
        bundle, loaded_at = _load_live_bundle()
        return bundle, "Live", None, loaded_at
    except Exception:
        warning = (
            "Live market data are temporarily unavailable, so the app is showing the "
            "latest committed snapshot instead."
        )
        return _load_fixture_bundle(), "Fixture", warning, None


@st.cache_data(ttl=86400)
def build_published_scenario(feature_panel: pd.DataFrame) -> Stage4ScenarioBundle:
    """Build the official published Week 5 scenario from the feature panel."""

    return build_app_scenario_bundle(
        feature_panel,
        config=build_published_config(),
    )


@st.cache_data(ttl=86400)
def build_design_scenario(
    feature_panel: pd.DataFrame,
    *,
    estimation_frequency: str,
    initial_window: int,
    window_rule: str,
) -> Stage4ScenarioBundle:
    """Build one advanced-design Week 5 scenario from the feature panel."""

    config = build_design_config(
        estimation_frequency=estimation_frequency,
        initial_window=initial_window,
        window_rule=window_rule,
    )
    return build_app_scenario_bundle(
        feature_panel,
        config=config,
    )
