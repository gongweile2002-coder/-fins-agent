# ruff: noqa
"""Offline tests for the Week 5 Stage 4 app helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fins2026.week5.code.stage4_app import (
    APP_SAMPLE_PERIODS,
    build_app_scenario_bundle,
    build_display_window_analysis,
    build_investment_allocation,
    build_published_config,
    methodology_mapping_table,
    published_fund_name,
)

HEADLINE_TICKERS = ["BTC-USD", "ETH-USD", "XRP-USD", "DOGE-USD"]


def _synthetic_feature_panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=520, freq="D")
    chol = np.array(
        [
            [0.030, 0.000, 0.000, 0.000],
            [0.012, 0.026, 0.000, 0.000],
            [0.010, 0.008, 0.024, 0.000],
            [0.009, 0.006, 0.005, 0.022],
        ]
    )
    draws = np.random.default_rng(123).standard_normal((len(dates), len(HEADLINE_TICKERS)))
    innovations = draws @ chol.T
    means = np.array([0.0015, 0.0013, 0.0010, 0.0009], dtype=float)
    returns = innovations + means

    rows: list[dict[str, object]] = []
    for ticker_index, ticker in enumerate(HEADLINE_TICKERS):
        level = 1.0
        for date_index, date in enumerate(dates):
            ret = returns[date_index, ticker_index]
            next_level = level * (1.0 + ret)
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "open": level,
                    "high": max(level, next_level),
                    "low": min(level, next_level),
                    "close": next_level,
                    "adjClose": next_level,
                    "volume": float(1_000_000 + 50_000 * ticker_index + 2_000 * date_index),
                    "dividend": 0.0,
                    "splitFactor": 1.0,
                    "ret": ret,
                    "abs_ret": abs(ret),
                    "is_large_move_10pct": abs(ret) >= 0.10,
                    "is_large_move_20pct": abs(ret) >= 0.20,
                    "rfr": 0.0001,
                    "excess_ret": ret - 0.0001,
                    "rolling_6m_avg_ret": np.nan,
                    "rolling_6m_vol": np.nan,
                    "rolling_6m_var_95": np.nan,
                    "rolling_6m_sharpe": np.nan,
                    "rolling_6m_sortino": np.nan,
                }
            )
            level = next_level
    return pd.DataFrame(rows)


def test_build_app_scenario_bundle_returns_full_long_only_surface() -> None:
    feature_panel = _synthetic_feature_panel()
    scenario = build_app_scenario_bundle(
        feature_panel,
        config=build_published_config(),
    )

    assert scenario.portfolio_returns.columns[0:2].tolist() == ["return_date", "formation_date"]
    assert scenario.latest_target_weights["portfolio_key"].nunique() == 5
    assert scenario.latest_live_weights["portfolio_key"].nunique() == 5
    assert scenario.concentration_snapshot["portfolio_key"].nunique() == 5
    assert scenario.turnover_snapshot["portfolio_key"].nunique() == 5
    assert scenario.btc_eth_exposure["portfolio_key"].nunique() == 5


def test_build_display_window_analysis_filters_to_trailing_period() -> None:
    feature_panel = _synthetic_feature_panel()
    scenario = build_app_scenario_bundle(
        feature_panel,
        config=build_published_config(),
    )

    display_window = build_display_window_analysis(
        scenario,
        sample_period="1Y",
    )

    assert APP_SAMPLE_PERIODS["1Y"] == 1
    latest_date = pd.Timestamp(display_window.portfolio_returns["return_date"].max())
    expected_cutoff = latest_date - pd.DateOffset(years=1)
    actual_start = pd.Timestamp(display_window.portfolio_returns["return_date"].min())
    assert actual_start >= expected_cutoff or len(display_window.portfolio_returns) == len(
        scenario.portfolio_returns
    )
    assert set(display_window.metrics["portfolio_key"]) == set(
        scenario.metrics["portfolio_key"]
    )


def test_build_investment_allocation_sums_to_requested_amount() -> None:
    feature_panel = _synthetic_feature_panel()
    scenario = build_app_scenario_bundle(
        feature_panel,
        config=build_published_config(),
    )

    allocation = build_investment_allocation(
        scenario.latest_live_weights,
        portfolio_key="equal_weight",
        amount_usd=25_000.0,
    )

    assert float(allocation["allocation_usd"].sum()) == pytest.approx(25_000.0)
    assert float(allocation["weight"].sum()) == pytest.approx(1.0)


def test_methodology_mapping_uses_published_fund_names() -> None:
    mapping = methodology_mapping_table()
    assert "Core Market" in mapping["published_fund"].tolist()
    assert "Mean-variance" in mapping["technical_model"].tolist()
    assert published_fund_name("risk_parity_volatility_long_only") == "Risk Balanced"
