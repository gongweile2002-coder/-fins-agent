# ruff: noqa
"""Offline tests for the Week 5 Stage 3 OOS figure pack."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fins2026.week5.code.stage3_oos_portfolios import (
    OOS_FIGURE_PORTFOLIO_COLUMN_ORDER,
    OPTIMIZED_PORTFOLIO_COLUMN_ORDER,
    SQRT_365,
    TOP_HOLDINGS_MODEL_ORDER,
    TRADING_DAYS_PER_YEAR,
    Stage3OOSConfig,
    Stage3OOSSample,
    build_balanced_stage3_sample,
    build_factsheet_btc_eth_exposure_snapshot,
    build_factsheet_concentration_snapshot,
    build_factsheet_current_drawdown_snapshot,
    build_factsheet_risk_contribution_snapshot,
    build_factsheet_trailing_return_snapshot,
    build_factsheet_trailing_risk_snapshot,
    build_factsheet_turnover_snapshot,
    build_latest_live_weight_snapshot,
    build_latest_target_weight_snapshot,
    build_oos_ex_post_frontier,
    build_oos_window_sample,
    build_top_target_weight_histories,
    compute_oos_portfolio_returns,
    generate_oos_weight_panels,
    summarize_oos_asset_statistics,
    summarize_oos_portfolio_metrics,
    wealth_index,
)
from fins2026.week5.code.stage3_portfolio_figures import (
    make_stage3_factsheet_figure_pack,
    make_stage3_figure_pack,
)

HEADLINE_TICKERS = ["BTC-USD", "ETH-USD", "XRP-USD", "DOGE-USD"]


def _synthetic_stage3_feature_panel() -> pd.DataFrame:
    """Create a deterministic daily crypto feature panel for Stage 3 tests."""

    dates = pd.date_range("2024-01-01", periods=480, freq="D")
    chol = np.array(
        [
            [0.030, 0.000, 0.000, 0.000],
            [0.012, 0.026, 0.000, 0.000],
            [0.010, 0.008, 0.024, 0.000],
            [0.009, 0.006, 0.005, 0.022],
        ]
    )
    draws = np.random.default_rng(42).standard_normal((len(dates), len(HEADLINE_TICKERS)))
    innovations = draws @ chol.T
    means = np.array([0.0017, 0.0013, 0.0011, 0.0009], dtype=float)
    returns = innovations + means
    rows: list[dict[str, object]] = []
    rfr = np.full(len(dates), 0.0001)
    for ticker_index, ticker in enumerate(HEADLINE_TICKERS):
        for date_index, date in enumerate(dates):
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "ret": returns[date_index, ticker_index],
                    "rfr": rfr[date_index],
                    "volume": float(
                        1_000_000
                        + 50_000 * (ticker_index + 1)
                        + 1_000 * date_index
                    ),
                }
            )
    return pd.DataFrame(rows)


def _build_oos_bundle() -> tuple[
    pd.DataFrame,
    Stage3OOSSample,
    pd.DataFrame,
    pd.DataFrame,
    Stage3OOSSample,
]:
    """Build one reusable synthetic Stage 3 OOS bundle."""

    feature_panel = _synthetic_stage3_feature_panel()
    sample = build_balanced_stage3_sample(
        feature_panel,
        provider="yahoo_crypto",
        display_name="Yahoo Finance Crypto",
    )
    config = Stage3OOSConfig(initial_window=120)
    _daily_weights, rebalance_audit, _solve_summary = generate_oos_weight_panels(
        sample,
        config=config,
    )
    portfolio_returns = compute_oos_portfolio_returns(
        sample,
        rebalance_audit,
        config=config,
    )
    oos_sample = build_oos_window_sample(sample, portfolio_returns)
    return feature_panel, sample, rebalance_audit, portfolio_returns, oos_sample


def test_oos_metric_helpers_use_aligned_rfr_and_365_day_annualization() -> None:
    _feature_panel, _sample, rebalance_audit, portfolio_returns, oos_sample = _build_oos_bundle()

    frontier = build_oos_ex_post_frontier(oos_sample, n_points=80)
    equal_weight_returns = portfolio_returns["equal_weight"].astype(float).reset_index(drop=True)
    aligned_rfr = oos_sample.rfr.astype(float).reset_index(drop=True)
    equal_weight_excess = equal_weight_returns - aligned_rfr
    expected_ann_return = float(equal_weight_returns.mean() * TRADING_DAYS_PER_YEAR)
    expected_ann_vol = float(equal_weight_returns.std(ddof=1) * SQRT_365)
    expected_sharpe = float(
        SQRT_365 * equal_weight_excess.mean() / equal_weight_excess.std(ddof=1)
    )
    expected_cumulative = float(wealth_index(equal_weight_returns).iloc[-1] - 1.0)

    metrics = summarize_oos_portfolio_metrics(oos_sample, portfolio_returns)
    assert set(metrics["portfolio_key"]) == set(OOS_FIGURE_PORTFOLIO_COLUMN_ORDER)
    equal_weight_row = metrics.loc[metrics["portfolio_key"] == "equal_weight"].iloc[0]
    assert equal_weight_row["annualized_return"] == pytest.approx(expected_ann_return)
    assert equal_weight_row["annualized_volatility"] == pytest.approx(expected_ann_vol)
    assert equal_weight_row["sharpe_ratio"] == pytest.approx(expected_sharpe)
    assert equal_weight_row["cumulative_return"] == pytest.approx(expected_cumulative)

    frontier_diffs = np.diff(frontier["volatility_ann"].to_numpy(dtype=float))
    assert (frontier_diffs >= -1e-10).all()

    histories = build_top_target_weight_histories(rebalance_audit, top_n=5)
    assert list(histories) == TOP_HOLDINGS_MODEL_ORDER
    for history in histories.values():
        assert history.columns[0] == "decision_date"
        assert len(history.columns) == 1 + min(5, len(HEADLINE_TICKERS))


def test_make_stage3_figure_pack_exports_expected_pngs(tmp_path: Path) -> None:
    _feature_panel, _sample, rebalance_audit, portfolio_returns, oos_sample = _build_oos_bundle()
    frontier = build_oos_ex_post_frontier(oos_sample, n_points=80)
    asset_summary = summarize_oos_asset_statistics(oos_sample)
    metrics = summarize_oos_portfolio_metrics(oos_sample, portfolio_returns)

    outputs = make_stage3_figure_pack(
        sample=oos_sample,
        rebalance_audit=rebalance_audit,
        portfolio_returns=portfolio_returns,
        frontier=frontier,
        metrics=metrics,
        asset_summary=asset_summary,
        output_dir=tmp_path,
    )

    assert set(outputs) == {
        "growth_of_one",
        "drawdowns",
        "scorecard",
        "top_holdings_over_time",
        "efficient_frontier",
    }
    for figure_paths in outputs.values():
        assert figure_paths["png"].exists()


def test_factsheet_snapshot_builders_return_consistent_latest_snapshots() -> None:
    _feature_panel, sample, rebalance_audit, portfolio_returns, oos_sample = _build_oos_bundle()

    latest_target = build_latest_target_weight_snapshot(rebalance_audit)
    latest_live = build_latest_live_weight_snapshot(sample, rebalance_audit)
    concentration = build_factsheet_concentration_snapshot(latest_live)
    risk_contributions = build_factsheet_risk_contribution_snapshot(
        sample,
        latest_target,
        rebalance_audit,
    )
    trailing_returns = build_factsheet_trailing_return_snapshot(portfolio_returns)
    current_drawdown = build_factsheet_current_drawdown_snapshot(portfolio_returns)
    trailing_risk = build_factsheet_trailing_risk_snapshot(oos_sample, portfolio_returns)
    turnover = build_factsheet_turnover_snapshot(latest_target)
    btc_eth = build_factsheet_btc_eth_exposure_snapshot(latest_live)

    assert latest_target["decision_date"].nunique() == 1
    assert latest_target["previous_decision_date"].nunique() == 1
    live_sums = latest_live.groupby("portfolio_key")["weight"].sum()
    assert live_sums.to_numpy() == pytest.approx(np.ones(len(live_sums)))
    assert set(concentration["portfolio_key"]) == set(OOS_FIGURE_PORTFOLIO_COLUMN_ORDER)
    assert (concentration["effective_n"] > 0.0).all()
    contribution_sums = risk_contributions.groupby("portfolio_key")["risk_contribution_pct"].sum()
    assert contribution_sums.to_numpy() == pytest.approx(np.full(len(contribution_sums), 100.0))
    assert set(trailing_returns["window_label"]) == {
        "30-day",
        "90-day",
        "180-day",
        "Since inception",
    }
    assert set(current_drawdown["portfolio_key"]) == set(OOS_FIGURE_PORTFOLIO_COLUMN_ORDER)
    assert set(trailing_risk["portfolio_key"]) == set(OOS_FIGURE_PORTFOLIO_COLUMN_ORDER)
    assert set(turnover["portfolio_key"]) == set(OOS_FIGURE_PORTFOLIO_COLUMN_ORDER)
    assert (turnover["turnover"] >= 0.0).all()
    assert set(btc_eth["portfolio_key"]) == set(OOS_FIGURE_PORTFOLIO_COLUMN_ORDER)
    exposure_sums = (
        btc_eth["btc_weight_pct"]
        + btc_eth["eth_weight_pct"]
        + btc_eth["other_weight_pct"]
    )
    assert exposure_sums.to_numpy() == pytest.approx(np.full(len(exposure_sums), 100.0))
    assert set(risk_contributions["portfolio_key"]) == set(OPTIMIZED_PORTFOLIO_COLUMN_ORDER)


def test_make_stage3_factsheet_figure_pack_exports_expected_pngs(tmp_path: Path) -> None:
    _feature_panel, sample, rebalance_audit, portfolio_returns, oos_sample = _build_oos_bundle()
    latest_target = build_latest_target_weight_snapshot(rebalance_audit)
    latest_live = build_latest_live_weight_snapshot(sample, rebalance_audit)
    concentration = build_factsheet_concentration_snapshot(latest_live)
    risk_contributions = build_factsheet_risk_contribution_snapshot(
        sample,
        latest_target,
        rebalance_audit,
    )
    trailing_returns = build_factsheet_trailing_return_snapshot(portfolio_returns)
    current_drawdown = build_factsheet_current_drawdown_snapshot(portfolio_returns)
    trailing_risk = build_factsheet_trailing_risk_snapshot(oos_sample, portfolio_returns)
    turnover = build_factsheet_turnover_snapshot(latest_target)
    btc_eth = build_factsheet_btc_eth_exposure_snapshot(latest_live)

    outputs = make_stage3_factsheet_figure_pack(
        sample=oos_sample,
        latest_target_weights=latest_target,
        latest_live_weights=latest_live,
        concentration_snapshot=concentration,
        risk_contributions=risk_contributions,
        trailing_returns=trailing_returns,
        current_drawdown=current_drawdown,
        trailing_risk=trailing_risk,
        turnover_snapshot=turnover,
        btc_eth_exposure=btc_eth,
        output_dir=tmp_path,
    )

    assert set(outputs) == {
        "latest_holdings_dumbbell",
        "live_holdings_snapshot",
        "concentration_scorecard",
        "risk_contributions",
        "trailing_returns",
        "current_drawdown",
        "trailing_risk",
        "turnover_and_changes",
        "btc_eth_exposure",
    }
    for figure_paths in outputs.values():
        assert figure_paths["png"].exists()
