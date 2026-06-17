# ruff: noqa
"""Tests for the Week 4 Stage 4 app helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fins2026.week4.app.app_insights import cumulative_growth_figure
from fins2026.week4.code.stage3_portfolios import Stage3Sample
from fins2026.week4.code.stage4_app import (
    APP_PORTFOLIO_LABELS,
    build_app_frontier,
    build_stage4_sample,
    compute_named_portfolio_returns,
    estimate_app_portfolio_weights,
    normalize_custom_weights,
    summarize_named_portfolio_metrics,
)


def _toy_feature_panel() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=140, freq="B")
    rows: list[dict[str, object]] = []
    tickers = ["AAPL", "MSFT", "NVDA"]
    base_paths = {
        "AAPL": 100.0 * np.cumprod(1.0 + 0.0006 + 0.004 * np.sin(np.arange(len(dates)) / 9)),
        "MSFT": 120.0 * np.cumprod(1.0 + 0.0005 + 0.003 * np.cos(np.arange(len(dates)) / 11)),
        "NVDA": 90.0 * np.cumprod(1.0 + 0.0009 + 0.006 * np.sin(np.arange(len(dates)) / 7)),
    }
    for ticker in tickers:
        prices = pd.Series(base_paths[ticker], index=dates)
        returns = prices.pct_change()
        for date, price, ret in zip(dates, prices, returns, strict=False):
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "adjClose": float(price),
                    "ret": float(ret) if pd.notna(ret) else np.nan,
                    "abs_ret": abs(float(ret)) if pd.notna(ret) else np.nan,
                    "rfr": 0.0001,
                    "excess_ret": float(ret - 0.0001) if pd.notna(ret) else np.nan,
                    "is_large_move_10pct": False,
                    "is_large_move_20pct": False,
                }
            )
    return pd.DataFrame(rows)


def _toy_sample() -> Stage3Sample:
    dates = pd.date_range("2020-01-01", periods=160, freq="B")
    returns = pd.DataFrame(
        {
            "AAPL": 0.0008 + 0.004 * np.sin(np.arange(len(dates)) / 12),
            "MSFT": 0.0006 + 0.003 * np.cos(np.arange(len(dates)) / 15),
            "NVDA": 0.0010 + 0.006 * np.sin(np.arange(len(dates)) / 9),
        },
        index=dates,
    )
    rfr = pd.Series(0.0001, index=dates)
    return Stage3Sample(
        provider="yahoo_app",
        display_name="Yahoo app test sample",
        returns_wide=returns,
        rfr=rfr,
    )


def test_normalize_custom_weights_zero_sum_falls_back_to_equal_weight() -> None:
    weights, warning = normalize_custom_weights(["AAPL", "MSFT"], {"AAPL": 0.0, "MSFT": 0.0})
    assert warning is not None
    assert np.allclose(weights["weight"].to_numpy(), [0.5, 0.5])


def test_build_stage4_sample_balances_selected_feature_panel() -> None:
    sample = build_stage4_sample(_toy_feature_panel(), selected_tickers=["AAPL", "MSFT"])
    assert sample.n_assets == 2
    assert sample.sample_days > 100
    assert sample.returns_wide.notna().all().all()


def test_estimate_app_portfolio_weights_long_only_are_nonnegative() -> None:
    sample = _toy_sample()
    custom_weights, _warning = normalize_custom_weights(
        sample.tickers,
        {"AAPL": 40.0, "MSFT": 30.0, "NVDA": 30.0},
    )
    weights, _methods = estimate_app_portfolio_weights(
        sample,
        custom_weights=custom_weights,
        constraint_mode="long_only",
    )
    matrix = weights.pivot(index="ticker", columns="portfolio", values="weight")
    for label in [
        APP_PORTFOLIO_LABELS["equal_weight"],
        APP_PORTFOLIO_LABELS["minimum_variance"],
        APP_PORTFOLIO_LABELS["mean_variance_tangency"],
    ]:
        assert np.isclose(float(matrix[label].sum()), 1.0)
        assert (matrix[label] >= -1e-10).all()


def test_estimate_app_portfolio_weights_unconstrained_supports_custom_portfolio() -> None:
    sample = _toy_sample()
    custom_weights, _warning = normalize_custom_weights(
        sample.tickers,
        {"AAPL": 10.0, "MSFT": 20.0, "NVDA": 70.0},
    )
    weights, _methods = estimate_app_portfolio_weights(
        sample,
        custom_weights=custom_weights,
        constraint_mode="unconstrained",
    )
    assert set(weights["portfolio"]) == {
        APP_PORTFOLIO_LABELS["custom"],
        APP_PORTFOLIO_LABELS["equal_weight"],
        APP_PORTFOLIO_LABELS["minimum_variance"],
        APP_PORTFOLIO_LABELS["mean_variance_tangency"],
    }


def test_stage4_portfolio_returns_and_metrics_include_all_portfolios() -> None:
    sample = _toy_sample()
    custom_weights, _warning = normalize_custom_weights(
        sample.tickers,
        {"AAPL": 50.0, "MSFT": 25.0, "NVDA": 25.0},
    )
    weights, _methods = estimate_app_portfolio_weights(
        sample,
        custom_weights=custom_weights,
        constraint_mode="long_only",
    )
    returns = compute_named_portfolio_returns(sample, weights)
    metrics = summarize_named_portfolio_metrics(returns)
    assert "rfr" in returns
    assert len(metrics) == 4
    assert APP_PORTFOLIO_LABELS["custom"] in metrics["portfolio"].tolist()


def test_build_app_frontier_returns_nonempty_grid() -> None:
    sample = _toy_sample()
    custom_weights, _warning = normalize_custom_weights(
        sample.tickers,
        {"AAPL": 40.0, "MSFT": 40.0, "NVDA": 20.0},
    )
    weights, _methods = estimate_app_portfolio_weights(
        sample,
        custom_weights=custom_weights,
        constraint_mode="long_only",
    )
    frontier = build_app_frontier(sample, weights, constraint_mode="long_only", n_points=30)
    assert not frontier.empty
    assert frontier["volatility_ann_pct"].notna().all()


def test_cumulative_growth_figure_uses_readable_log_ticks() -> None:
    sample = _toy_sample()
    custom_weights, _warning = normalize_custom_weights(
        sample.tickers,
        {"AAPL": 50.0, "MSFT": 25.0, "NVDA": 25.0},
    )
    weights, _methods = estimate_app_portfolio_weights(
        sample,
        custom_weights=custom_weights,
        constraint_mode="long_only",
    )
    returns = compute_named_portfolio_returns(sample, weights)
    fig = cumulative_growth_figure(returns, sample=sample)
    assert fig.layout.yaxis.type == "log"
    assert list(fig.layout.yaxis.tickvals)
    assert all(str(label).startswith("$") for label in fig.layout.yaxis.ticktext)
    assert fig.layout.yaxis.automargin
