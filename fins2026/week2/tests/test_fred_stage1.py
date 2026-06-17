# ruff: noqa
"""Focused tests for the Week 2 FRED Stage 1 walkthrough helpers."""

from __future__ import annotations

import pandas as pd

from fins2026.week2.code.fred_stage1 import (
    FRED_SERIES,
    build_fixture_raw_fred_frame,
    build_fred_stage1_long_table,
    stage1_assertion_report,
)
from fins2026.week2.code.market_window import (
    WEEK2_FRED_LECTURE_END,
    WEEK2_FRED_LECTURE_START,
)


def test_fred_stage1_fixture_contract() -> None:
    raw = build_fixture_raw_fred_frame()
    stage1 = build_fred_stage1_long_table(raw)

    assert not stage1.empty
    assert set(stage1["series_id"].unique()) == set(FRED_SERIES)
    assert stage1_assertion_report(stage1) == {
        "typed_dates": True,
        "typed_values": True,
        "all_series_present": True,
        "ordered_dates": True,
        "unique_pairs": True,
    }
    assert pd.Timestamp(stage1["raw_date"].min()) >= WEEK2_FRED_LECTURE_START
    assert pd.Timestamp(stage1["raw_date"].max()) <= WEEK2_FRED_LECTURE_END


def test_fred_fixture_window_modes() -> None:
    raw = build_fixture_raw_fred_frame()
    frozen = build_fred_stage1_long_table(raw)
    rolling = build_fred_stage1_long_table(raw, rolling_window=True)

    assert pd.Timestamp(frozen["raw_date"].min()) >= WEEK2_FRED_LECTURE_START
    assert pd.Timestamp(frozen["raw_date"].max()) <= WEEK2_FRED_LECTURE_END
    assert pd.Timestamp(rolling["raw_date"].max()) > WEEK2_FRED_LECTURE_END
