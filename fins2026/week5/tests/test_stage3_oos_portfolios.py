# ruff: noqa
"""Offline tests for the Week 5 Stage 3 out-of-sample weight engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fins2026.week5.code.stage3_oos_portfolios import (
    Stage3OOSConfig,
    Stage3OOSSample,
    build_balanced_stage3_sample,
    build_prefix_moments,
    build_rebalance_schedule,
    compute_oos_portfolio_returns,
    compute_window_statistics,
    estimate_one_window_weights,
    generate_oos_weight_panels,
    historical_cvar,
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
                }
            )
    return pd.DataFrame(rows)


def test_build_balanced_stage3_sample_drops_missing_dates() -> None:
    feature_panel = _synthetic_stage3_feature_panel()
    missing_date = feature_panel["date"].iloc[25]
    feature_panel.loc[
        (feature_panel["ticker"] == "BTC-USD") & (feature_panel["date"] == missing_date),
        "ret",
    ] = np.nan

    sample = build_balanced_stage3_sample(
        feature_panel,
        provider="yahoo_crypto",
        display_name="Yahoo Finance Crypto",
    )

    assert sample.returns_wide.index.is_monotonic_increasing
    assert not sample.returns_wide.isna().any().any()
    assert not sample.rfr.isna().any()
    assert feature_panel["date"].nunique() - sample.sample_days == 1


def test_monthly_expanding_schedule_uses_calendar_month_end_and_next_day_hold() -> None:
    sample = build_balanced_stage3_sample(
        _synthetic_stage3_feature_panel(),
        provider="yahoo_crypto",
        display_name="Yahoo Finance Crypto",
    )

    schedule = build_rebalance_schedule(
        sample.dates,
        initial_window=120,
        estimation_frequency="monthly",
        window_rule="expanding",
    )

    assert schedule[0].window_observations >= 120
    assert schedule[0].decision_date == schedule[0].decision_date + pd.offsets.MonthEnd(0)
    assert schedule[0].effective_start_date == schedule[0].decision_date + pd.Timedelta(days=1)
    assert schedule[0].window_start_date == sample.start_date
    window_sizes = [window.window_observations for window in schedule]
    assert window_sizes == sorted(window_sizes)
    assert schedule[0].effective_end_date == schedule[1].decision_date


def test_rolling_schedule_keeps_a_constant_window_width() -> None:
    sample = build_balanced_stage3_sample(
        _synthetic_stage3_feature_panel(),
        provider="yahoo_crypto",
        display_name="Yahoo Finance Crypto",
    )

    schedule = build_rebalance_schedule(
        sample.dates,
        initial_window=90,
        estimation_frequency="weekly",
        window_rule="rolling",
    )

    assert {window.window_observations for window in schedule} == {90}


def test_daily_schedule_assigns_one_holding_day_per_decision() -> None:
    sample = build_balanced_stage3_sample(
        _synthetic_stage3_feature_panel(),
        provider="yahoo_crypto",
        display_name="Yahoo Finance Crypto",
    )
    config = Stage3OOSConfig(
        initial_window=120,
        estimation_frequency="daily",
        window_rule="expanding",
        models=("equal_weight",),
        constraint_modes=("long_only",),
    )

    daily_weights, rebalance_audit, solve_summary = generate_oos_weight_panels(
        sample,
        config=config,
    )

    holding_days = (
        daily_weights.loc[:, ["decision_date", "date"]]
        .drop_duplicates()
        .groupby("decision_date")["date"]
        .nunique()
    )
    assert (holding_days == 1).all()
    assert set(rebalance_audit["model"]) == {"equal_weight"}
    assert set(solve_summary["model"]) == {"equal_weight"}


def test_generate_oos_weight_panels_builds_long_only_weight_panels() -> None:
    sample = build_balanced_stage3_sample(
        _synthetic_stage3_feature_panel(),
        provider="yahoo_crypto",
        display_name="Yahoo Finance Crypto",
    )
    config = Stage3OOSConfig(initial_window=120)

    daily_weights, rebalance_audit, solve_summary = generate_oos_weight_panels(
        sample,
        config=config,
    )

    assert not daily_weights.empty
    assert not rebalance_audit.empty
    assert (daily_weights["date"] > daily_weights["decision_date"]).all()
    grouped_daily = (
        daily_weights.groupby(["date", "decision_date", "model", "constraint_mode"])[
            "weight"
        ].sum()
    )
    grouped_audit = (
        rebalance_audit.groupby(["decision_date", "model", "constraint_mode"])[
            "weight"
        ].sum()
    )
    assert grouped_daily.to_numpy() == pytest.approx(np.ones(len(grouped_daily)))
    assert grouped_audit.to_numpy() == pytest.approx(np.ones(len(grouped_audit)))
    assert (
        daily_weights.loc[daily_weights["constraint_mode"] == "long_only", "weight"] >= -1e-10
    ).all()
    assert set(daily_weights["constraint_mode"]) == {"long_only"}
    assert set(rebalance_audit["constraint_mode"]) == {"long_only"}
    assert set(solve_summary["constraint_mode"]) == {"long_only"}


def test_compute_oos_portfolio_returns_has_expected_columns_and_dates() -> None:
    sample = build_balanced_stage3_sample(
        _synthetic_stage3_feature_panel(),
        provider="yahoo_crypto",
        display_name="Yahoo Finance Crypto",
    )
    config = Stage3OOSConfig(
        initial_window=120,
        models=("equal_weight", "minimum_variance", "risk_parity_volatility"),
    )
    daily_weights, rebalance_audit, _solve_summary = generate_oos_weight_panels(
        sample,
        config=config,
    )
    portfolio_returns = compute_oos_portfolio_returns(
        sample,
        rebalance_audit,
        config=config,
    )

    assert portfolio_returns.columns.tolist() == [
        "return_date",
        "formation_date",
        "equal_weight",
        "minimum_variance_long_only",
        "risk_parity_volatility_long_only",
    ]
    assert portfolio_returns["return_date"].is_monotonic_increasing
    assert (portfolio_returns["return_date"] > portfolio_returns["formation_date"]).all()
    assert len(portfolio_returns) == daily_weights["date"].nunique()
    assert portfolio_returns["return_date"].tolist() == sorted(daily_weights["date"].unique())
    assert not portfolio_returns.drop(columns=["return_date", "formation_date"]).isna().any().any()


def test_mean_cvar_weights_produce_a_finite_training_window_ratio() -> None:
    sample = build_balanced_stage3_sample(
        _synthetic_stage3_feature_panel(),
        provider="yahoo_crypto",
        display_name="Yahoo Finance Crypto",
    )
    config = Stage3OOSConfig(
        initial_window=180,
        models=("mean_cvar_tangency",),
    )
    schedule = build_rebalance_schedule(
        sample.dates,
        initial_window=config.initial_window,
        estimation_frequency=config.estimation_frequency,
        window_rule=config.window_rule,
    )
    returns_array = sample.returns_wide.to_numpy(dtype=float)
    rfr_array = sample.rfr.to_numpy(dtype=float)
    prefix = build_prefix_moments(returns_array, rfr_array)
    stats = compute_window_statistics(
        prefix,
        returns_array,
        rfr_array,
        schedule[0],
        covariance_ridge=config.covariance_ridge,
    )

    weights, _solver, _status = estimate_one_window_weights(
        stats,
        model="mean_cvar_tangency",
        constraint_mode="long_only",
        config=config,
    )
    portfolio_excess = stats.excess_returns @ weights
    cvar = historical_cvar(-portfolio_excess, alpha=config.cvar_alpha)
    assert np.isfinite(portfolio_excess.mean())
    assert np.isfinite(cvar)
    assert cvar > 0.0
    assert np.isfinite(float(portfolio_excess.mean() / cvar))


def test_long_only_minimum_variance_reduces_training_window_variance() -> None:
    sample = build_balanced_stage3_sample(
        _synthetic_stage3_feature_panel(),
        provider="yahoo_crypto",
        display_name="Yahoo Finance Crypto",
    )
    config = Stage3OOSConfig(
        initial_window=180,
        models=("minimum_variance",),
        constraint_modes=("long_only",),
    )
    schedule = build_rebalance_schedule(
        sample.dates,
        initial_window=config.initial_window,
        estimation_frequency=config.estimation_frequency,
        window_rule=config.window_rule,
    )
    returns_array = sample.returns_wide.to_numpy(dtype=float)
    rfr_array = sample.rfr.to_numpy(dtype=float)
    prefix = build_prefix_moments(returns_array, rfr_array)
    stats = compute_window_statistics(
        prefix,
        returns_array,
        rfr_array,
        schedule[0],
        covariance_ridge=config.covariance_ridge,
    )

    weights, _solver, _status = estimate_one_window_weights(
        stats,
        model="minimum_variance",
        constraint_mode="long_only",
        config=config,
    )
    equal_weight = np.full(sample.n_assets, 1.0 / sample.n_assets)
    optimized_variance = float(weights @ stats.covariance @ weights)
    equal_weight_variance = float(equal_weight @ stats.covariance @ equal_weight)
    assert optimized_variance <= equal_weight_variance + 1e-12


def test_long_only_tangency_improves_on_equal_weight_sharpe() -> None:
    sample = build_balanced_stage3_sample(
        _synthetic_stage3_feature_panel(),
        provider="yahoo_crypto",
        display_name="Yahoo Finance Crypto",
    )
    config = Stage3OOSConfig(
        initial_window=180,
        models=("mean_variance_tangency",),
        constraint_modes=("long_only",),
    )
    schedule = build_rebalance_schedule(
        sample.dates,
        initial_window=config.initial_window,
        estimation_frequency=config.estimation_frequency,
        window_rule=config.window_rule,
    )
    returns_array = sample.returns_wide.to_numpy(dtype=float)
    rfr_array = sample.rfr.to_numpy(dtype=float)
    prefix = build_prefix_moments(returns_array, rfr_array)
    stats = compute_window_statistics(
        prefix,
        returns_array,
        rfr_array,
        schedule[0],
        covariance_ridge=config.covariance_ridge,
    )

    weights, _solver, _status = estimate_one_window_weights(
        stats,
        model="mean_variance_tangency",
        constraint_mode="long_only",
        config=config,
    )
    equal_weight = np.full(sample.n_assets, 1.0 / sample.n_assets)

    def sharpe_ratio(vector: np.ndarray) -> float:
        excess_mean = float(vector @ stats.mean_returns - stats.avg_daily_rfr)
        volatility = float(np.sqrt(vector @ stats.covariance @ vector))
        return excess_mean / volatility

    assert sharpe_ratio(weights) >= sharpe_ratio(equal_weight) - 1e-12


def test_risk_parity_weights_equalize_risk_contributions() -> None:
    sample = build_balanced_stage3_sample(
        _synthetic_stage3_feature_panel(),
        provider="yahoo_crypto",
        display_name="Yahoo Finance Crypto",
    )
    config = Stage3OOSConfig(
        initial_window=180,
        models=("risk_parity_volatility",),
        constraint_modes=("long_only",),
    )
    schedule = build_rebalance_schedule(
        sample.dates,
        initial_window=config.initial_window,
        estimation_frequency=config.estimation_frequency,
        window_rule=config.window_rule,
    )
    returns_array = sample.returns_wide.to_numpy(dtype=float)
    rfr_array = sample.rfr.to_numpy(dtype=float)
    prefix = build_prefix_moments(returns_array, rfr_array)
    stats = compute_window_statistics(
        prefix,
        returns_array,
        rfr_array,
        schedule[0],
        covariance_ridge=config.covariance_ridge,
    )

    weights, _solver, _status = estimate_one_window_weights(
        stats,
        model="risk_parity_volatility",
        constraint_mode="long_only",
        config=config,
    )
    portfolio_variance = float(weights @ stats.covariance @ weights)
    target_contribution = portfolio_variance / float(sample.n_assets)
    contributions = weights * (stats.covariance @ weights)
    assert np.max(np.abs(contributions - target_contribution)) <= 1e-6


def test_compute_oos_portfolio_returns_matches_exact_drift_recursion() -> None:
    return_dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    sample = Stage3OOSSample(
        provider="yahoo_crypto",
        display_name="Yahoo Finance Crypto",
        returns_wide=pd.DataFrame(
            {
                "AAA": [0.10, 0.00],
                "BBB": [0.00, 0.10],
            },
            index=return_dates,
        ),
        rfr=pd.Series([0.0, 0.0], index=return_dates),
    )
    rebalance_audit = pd.DataFrame(
        {
            "decision_date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
            "effective_start_date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02")],
            "effective_end_date": [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-03")],
            "window_start_date": [pd.Timestamp("2023-07-01"), pd.Timestamp("2023-07-01")],
            "window_end_date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
            "window_observations": [185, 185],
            "model": ["equal_weight", "equal_weight"],
            "constraint_mode": ["long_only", "long_only"],
            "ticker": ["AAA", "BBB"],
            "weight": [0.5, 0.5],
            "solver": ["direct", "direct"],
            "status": ["ok", "ok"],
            "elapsed_ms": [0.0, 0.0],
        }
    )
    config = Stage3OOSConfig(models=("equal_weight",), constraint_modes=("long_only",))

    portfolio_returns = compute_oos_portfolio_returns(
        sample,
        rebalance_audit,
        config=config,
    )

    assert portfolio_returns["return_date"].tolist() == list(return_dates)
    assert portfolio_returns["formation_date"].tolist() == [pd.Timestamp("2024-01-01")] * 2
    assert portfolio_returns["equal_weight"].to_numpy() == pytest.approx(
        np.array([0.05, 0.047619047619]),
    )
    assert portfolio_returns["equal_weight"].iloc[1] != pytest.approx(0.05)


def test_compute_oos_portfolio_returns_resets_to_new_weights_at_rebalance_boundary() -> None:
    return_dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    sample = Stage3OOSSample(
        provider="yahoo_crypto",
        display_name="Yahoo Finance Crypto",
        returns_wide=pd.DataFrame(
            {
                "AAA": [0.10, 0.00, 0.20],
                "BBB": [0.00, 0.20, 0.00],
            },
            index=return_dates,
        ),
        rfr=pd.Series([0.0, 0.0, 0.0], index=return_dates),
    )
    rebalance_audit = pd.DataFrame(
        {
            "decision_date": [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-03"),
                pd.Timestamp("2024-01-03"),
            ],
            "effective_start_date": [
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-04"),
                pd.Timestamp("2024-01-04"),
            ],
            "effective_end_date": [
                pd.Timestamp("2024-01-03"),
                pd.Timestamp("2024-01-03"),
                pd.Timestamp("2024-01-04"),
                pd.Timestamp("2024-01-04"),
            ],
            "window_start_date": [
                pd.Timestamp("2023-07-01"),
                pd.Timestamp("2023-07-01"),
                pd.Timestamp("2023-07-03"),
                pd.Timestamp("2023-07-03"),
            ],
            "window_end_date": [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-03"),
                pd.Timestamp("2024-01-03"),
            ],
            "window_observations": [185, 185, 185, 185],
            "model": [
                "minimum_variance",
                "minimum_variance",
                "minimum_variance",
                "minimum_variance",
            ],
            "constraint_mode": ["long_only", "long_only", "long_only", "long_only"],
            "ticker": ["AAA", "BBB", "AAA", "BBB"],
            "weight": [0.5, 0.5, 1.0, 0.0],
            "solver": ["SLSQP_jac"] * 4,
            "status": ["ok"] * 4,
            "elapsed_ms": [0.0] * 4,
        }
    )
    config = Stage3OOSConfig(
        models=("minimum_variance",),
        constraint_modes=("long_only",),
    )

    portfolio_returns = compute_oos_portfolio_returns(
        sample,
        rebalance_audit,
        config=config,
    )

    assert portfolio_returns["minimum_variance_long_only"].to_numpy() == pytest.approx(
        np.array([0.05, 0.095238095238, 0.20]),
    )
    assert portfolio_returns.loc[
        portfolio_returns["return_date"] == pd.Timestamp("2024-01-04"),
        "formation_date",
    ].iloc[0] == pd.Timestamp("2024-01-03")
