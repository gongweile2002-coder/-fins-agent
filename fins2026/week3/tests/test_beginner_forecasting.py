# ruff: noqa
"""Tests for the Week 3 beginner forecasting ladder."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from fins2026.week3.code.beginner_forecasting import (
    add_level_columns,
    adf_summary,
    build_beginner_macro_panel,
    build_us_beginner_macro_panel,
    equal_weight_ensemble_backtest,
    load_fixture_unemployment_rate,
    load_fixture_us_unemployment_rate,
    make_simulated_series_frame,
    metrics_table,
    split_dates,
)
from fins2026.week3.scripts.make_beginner_forecast_story_figures import (
    main as make_story_figures_main,
)
from fins2026.week3.scripts.make_beginner_forecasting_series import main as make_series_main
from fins2026.week3.scripts.make_us_beginner_forecast_story_figures import (
    main as make_us_story_figures_main,
)
from fins2026.week3.scripts.make_us_beginner_forecasting_series import main as make_us_series_main
from fins2026.week3.scripts.run_beginner_ar_forecast import main as run_ar_main
from fins2026.week3.scripts.run_beginner_arma_forecast import main as run_arma_main
from fins2026.week3.scripts.run_beginner_armax_forecast import main as run_armax_main
from fins2026.week3.scripts.run_beginner_arx_forecast import main as run_arx_main
from fins2026.week3.scripts.run_beginner_enet_forecast import main as run_enet_main
from fins2026.week3.scripts.run_beginner_ensemble_forecast import main as run_ensemble_main
from fins2026.week3.scripts.run_beginner_model_horse_race import main as run_horse_race_main
from fins2026.week3.scripts.run_beginner_naive_forecast import main as run_naive_main
from fins2026.week3.scripts.run_beginner_ols_forecast import main as run_ols_main
from fins2026.week3.scripts.run_beginner_unit_root_check import main as run_unit_root_main
from fins2026.week3.scripts.run_us_beginner_ar_forecast import main as run_us_ar_main
from fins2026.week3.scripts.run_us_beginner_arma_forecast import main as run_us_arma_main
from fins2026.week3.scripts.run_us_beginner_arx_forecast import main as run_us_arx_main
from fins2026.week3.scripts.run_us_beginner_model_horse_race import (
    main as run_us_horse_race_main,
)
from fins2026.week3.scripts.run_us_beginner_naive_forecast import main as run_us_naive_main
from fins2026.week3.scripts.run_us_beginner_unit_root_check import (
    main as run_us_unit_root_main,
)


def test_unemployment_series_is_monthly_and_long() -> None:
    series = load_fixture_unemployment_rate()

    assert series.index.is_monotonic_increasing
    assert series.index.min() == pd.Timestamp("2000-02-29")
    assert series.index.max() >= pd.Timestamp("2026-03-31")
    assert len(series) >= 300
    assert series.name == "unemployment_rate"


def test_us_unemployment_series_is_monthly_and_long() -> None:
    series = load_fixture_us_unemployment_rate()

    assert series.index.is_monotonic_increasing
    assert series.index.min() <= pd.Timestamp("1960-01-31")
    assert series.index.max() >= pd.Timestamp("2026-03-31")
    assert len(series) >= 700
    assert series.name == "unemployment_rate"


def test_default_week3_split_starts_out_of_sample_in_2020() -> None:
    change = load_fixture_unemployment_rate().diff().dropna()

    train_last_date, test_first_date = split_dates(change)

    assert train_last_date == pd.Timestamp("2019-12-31")
    assert test_first_date == pd.Timestamp("2020-01-31")


def test_simulated_series_bundle_is_deterministic() -> None:
    first = make_simulated_series_frame()
    second = make_simulated_series_frame()

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == ["date", "random_walk", "stationary_ar1", "stationary_arma11"]


def test_beginner_macro_panel_contains_real_and_transformed_series() -> None:
    panel = build_beginner_macro_panel()

    expected = {
        "unemployment_rate",
        "cash_rate_target",
        "commodity_price_index_aud",
        "trade_weighted_index",
        "unemployment_change_pp",
        "cash_rate_change_pp",
        "commodity_price_log_change_pct",
        "trade_weighted_index_log_change_pct",
        "participation_change_pp",
        "employment_to_population_change_pp",
        "vacancies_to_labour_force_change_pp",
        "headline_cpi_inflation",
        "trimmed_mean_inflation",
        "wage_price_index_growth",
        "us_fedfunds_change_pp",
        "us_unemployment_change_pp",
        "us_indpro_log_growth_pct",
        "us_vix_level",
        "us_yield_spread_pp",
    }
    assert expected.issubset(panel.columns)
    assert panel.index.is_monotonic_increasing
    assert panel["unemployment_rate"].notna().sum() >= 300


def test_us_beginner_macro_panel_contains_real_and_transformed_series() -> None:
    panel = build_us_beginner_macro_panel()

    expected = {
        "unemployment_rate",
        "fedfunds_rate",
        "treasury_10y_rate",
        "yield_spread_pp",
        "unemployment_change_pp",
        "fedfunds_change_pp",
        "treasury_10y_change_pp",
        "indpro_log_growth_pct",
        "vix_level",
        "sp500_return_pct",
    }
    assert expected.issubset(panel.columns)
    assert panel.index.is_monotonic_increasing
    assert panel["unemployment_rate"].notna().sum() >= 700


def test_adf_summary_distinguishes_random_walk_from_stationary_series() -> None:
    simulated = make_simulated_series_frame().set_index("date")

    random_walk = adf_summary(simulated["random_walk"], name="Random walk", version="Level")
    stationary = adf_summary(
        simulated["stationary_ar1"],
        name="Stationary AR(1)",
        version="Level",
    )

    assert random_walk["reject_unit_root_5pct"] is False
    assert stationary["reject_unit_root_5pct"] is True
    assert float(random_walk["p_value"]) > float(stationary["p_value"])


def test_add_level_columns_rebuilds_implied_forecast_path() -> None:
    dates = pd.date_range("2024-01-31", periods=4, freq="ME")
    level = pd.Series([6.0, 6.2, 6.1, 6.3], index=dates)
    backtest = pd.DataFrame(
        {
            "actual": [0.2, -0.1, 0.2],
            "forecast": [0.1, -0.2, 0.3],
            "error": [0.1, 0.1, -0.1],
            "absolute_error": [0.1, 0.1, 0.1],
        },
        index=dates[1:],
    )

    rebuilt = add_level_columns(backtest, level)

    assert rebuilt["previous_level"].tolist() == [6.0, 6.2, 6.1]
    assert rebuilt["actual_level"].tolist() == [6.2, 6.1, 6.3]
    assert rebuilt["forecast_level"].tolist() == [6.1, 6.0, 6.3999999999999995]


def test_metrics_table_sets_naive_oos_r2_to_zero() -> None:
    dates = pd.date_range("2024-01-31", periods=6, freq="ME")
    target = pd.Series([0.1, -0.2, 0.3, -0.1, 0.2, 0.0], index=dates)
    naive = pd.DataFrame(
        {
            "actual": target.iloc[-3:].to_list(),
            "forecast": [0.3, -0.1, 0.2],
            "error": [target.iloc[3] - 0.3, target.iloc[4] + 0.1, target.iloc[5] - 0.2],
            "absolute_error": [0.4, 0.3, 0.2],
        },
        index=dates[-3:],
    )
    better = pd.DataFrame(
        {
            "actual": target.iloc[-3:].to_list(),
            "forecast": [0.0, 0.1, 0.1],
            "error": [target.iloc[3] - 0.0, target.iloc[4] - 0.1, target.iloc[5] - 0.1],
            "absolute_error": [0.1, 0.1, 0.1],
        },
        index=dates[-3:],
    )

    metrics = metrics_table(
        {"Naive": naive, "Better": better},
        target_series=target,
        test_periods=3,
        align_common_dates=True,
    ).set_index("model")

    assert metrics.loc["Naive", "target_oos_r2_vs_naive"] == pytest.approx(0.0)
    assert metrics.loc["Better", "target_rmse"] < metrics.loc["Naive", "target_rmse"]


def test_equal_weight_ensemble_is_the_member_average() -> None:
    dates = pd.date_range("2024-01-31", periods=4, freq="ME")
    level = pd.Series([6.0, 6.1, 6.2, 6.4], index=dates)
    member_a = pd.DataFrame(
        {
            "actual": [0.1, 0.2, 0.0],
            "forecast": [0.0, 0.3, 0.1],
            "error": [0.1, -0.1, -0.1],
            "absolute_error": [0.1, 0.1, 0.1],
        },
        index=dates[1:],
    )
    member_b = pd.DataFrame(
        {
            "actual": [0.1, 0.2, 0.0],
            "forecast": [0.2, 0.1, -0.1],
            "error": [-0.1, 0.1, 0.1],
            "absolute_error": [0.1, 0.1, 0.1],
        },
        index=dates[1:],
    )

    ensemble = equal_weight_ensemble_backtest({"A": member_a, "B": member_b}, level_series=level)

    assert ensemble["forecast"].tolist() == pytest.approx([0.1, 0.2, 0.0])
    assert ensemble["forecast_level"].tolist() == pytest.approx([6.1, 6.3, 6.2])


def test_beginner_scripts_write_expected_outputs() -> None:
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        data_dir = root / "data"
        figures_dir = root / "figures"
        tables_dir = root / "tables"

        assert make_series_main(["--output-dir", str(data_dir)]) == 0
        us_data_dir = root / "us_data"
        us_figures_dir = root / "us_figures"
        us_tables_dir = root / "us_tables"
        assert make_us_series_main(["--output-dir", str(us_data_dir)]) == 0
        assert run_unit_root_main(
            [
                "--data-dir",
                str(data_dir),
                "--figures-dir",
                str(figures_dir),
                "--tables-dir",
                str(tables_dir),
            ]
        ) == 0
        assert run_naive_main(
            [
                "--data-dir",
                str(data_dir),
                "--figures-dir",
                str(figures_dir),
                "--tables-dir",
                str(tables_dir),
                "--test-periods",
                "6",
            ]
        ) == 0
        assert run_ar_main(
            [
                "--data-dir",
                str(data_dir),
                "--figures-dir",
                str(figures_dir),
                "--tables-dir",
                str(tables_dir),
                "--test-periods",
                "6",
            ]
        ) == 0
        assert run_arma_main(
            [
                "--data-dir",
                str(data_dir),
                "--figures-dir",
                str(figures_dir),
                "--tables-dir",
                str(tables_dir),
                "--test-periods",
                "6",
            ]
        ) == 0
        assert run_arx_main(
            [
                "--data-dir",
                str(data_dir),
                "--figures-dir",
                str(figures_dir),
                "--tables-dir",
                str(tables_dir),
                "--test-periods",
                "6",
            ]
        ) == 0
        assert run_armax_main(
            [
                "--data-dir",
                str(data_dir),
                "--figures-dir",
                str(figures_dir),
                "--tables-dir",
                str(tables_dir),
                "--test-periods",
                "6",
            ]
        ) == 0
        assert run_ols_main(
            [
                "--data-dir",
                str(data_dir),
                "--figures-dir",
                str(figures_dir),
                "--tables-dir",
                str(tables_dir),
                "--test-periods",
                "6",
            ]
        ) == 0
        assert run_enet_main(
            [
                "--data-dir",
                str(data_dir),
                "--figures-dir",
                str(figures_dir),
                "--tables-dir",
                str(tables_dir),
                "--test-periods",
                "6",
            ]
        ) == 0
        assert run_horse_race_main(
            [
                "--data-dir",
                str(data_dir),
                "--figures-dir",
                str(figures_dir),
                "--tables-dir",
                str(tables_dir),
                "--test-periods",
                "6",
            ]
        ) == 0
        assert run_ensemble_main(
            [
                "--data-dir",
                str(data_dir),
                "--figures-dir",
                str(figures_dir),
                "--tables-dir",
                str(tables_dir),
                "--test-periods",
                "6",
            ]
        ) == 0
        story_figures_dir = root / "story_figures"
        story_tables_dir = root / "story_tables"
        assert make_story_figures_main(
            [
                "--data-dir",
                str(data_dir),
                "--figures-dir",
                str(story_figures_dir),
                "--tables-dir",
                str(story_tables_dir),
                "--test-periods",
                "6",
            ]
        ) == 0
        assert run_us_unit_root_main(
            [
                "--data-dir",
                str(us_data_dir),
                "--figures-dir",
                str(us_figures_dir),
                "--tables-dir",
                str(us_tables_dir),
            ]
        ) == 0
        assert run_us_naive_main(
            [
                "--data-dir",
                str(us_data_dir),
                "--figures-dir",
                str(us_figures_dir),
                "--tables-dir",
                str(us_tables_dir),
                "--test-periods",
                "6",
            ]
        ) == 0
        assert run_us_ar_main(
            [
                "--data-dir",
                str(us_data_dir),
                "--figures-dir",
                str(us_figures_dir),
                "--tables-dir",
                str(us_tables_dir),
                "--test-periods",
                "6",
            ]
        ) == 0
        assert run_us_arma_main(
            [
                "--data-dir",
                str(us_data_dir),
                "--figures-dir",
                str(us_figures_dir),
                "--tables-dir",
                str(us_tables_dir),
                "--test-periods",
                "6",
            ]
        ) == 0
        assert run_us_arx_main(
            [
                "--data-dir",
                str(us_data_dir),
                "--figures-dir",
                str(us_figures_dir),
                "--tables-dir",
                str(us_tables_dir),
                "--test-periods",
                "6",
            ]
        ) == 0
        assert run_us_horse_race_main(
            [
                "--data-dir",
                str(us_data_dir),
                "--figures-dir",
                str(us_figures_dir),
                "--tables-dir",
                str(us_tables_dir),
                "--test-periods",
                "6",
            ]
        ) == 0
        us_story_figures_dir = root / "us_story_figures"
        us_story_tables_dir = root / "us_story_tables"
        assert make_us_story_figures_main(
            [
                "--data-dir",
                str(us_data_dir),
                "--figures-dir",
                str(us_story_figures_dir),
                "--tables-dir",
                str(us_story_tables_dir),
                "--test-periods",
                "6",
            ]
        ) == 0

        expected = [
            data_dir / "week3_beginner_unemployment_rate.csv",
            data_dir / "week3_beginner_macro_panel.csv",
            data_dir / "week3_beginner_simulated_series.csv",
            tables_dir / "week3_beginner_unit_root_summary.csv",
            tables_dir / "week3_beginner_naive_metrics.csv",
            tables_dir / "week3_beginner_ar_metrics.csv",
            tables_dir / "week3_beginner_arma_metrics.csv",
            tables_dir / "week3_beginner_arx_metrics.csv",
            tables_dir / "week3_beginner_armax_metrics.csv",
            tables_dir / "week3_beginner_ols_metrics.csv",
            tables_dir / "week3_beginner_enet_metrics.csv",
            tables_dir / "week3_beginner_ols_coefficients.csv",
            tables_dir / "week3_beginner_enet_coefficients.csv",
            tables_dir / "week3_beginner_enet_alpha_table.csv",
            tables_dir / "week3_beginner_horse_race_metrics.csv",
            tables_dir / "week3_beginner_horse_race_model_specs.csv",
            tables_dir / "week3_beginner_horse_race_backtest_wide.csv",
            tables_dir / "week3_beginner_arx_variant_race.csv",
            tables_dir / "week3_beginner_armax_variant_race.csv",
            tables_dir / "week3_beginner_ensemble_metrics.csv",
            tables_dir / "week3_beginner_selected_ensemble_backtest.csv",
            tables_dir / "week3_beginner_ensemble_one_step_forecasts.csv",
            figures_dir / "week3_beginner_unit_root_simulations.png",
            figures_dir / "week3_beginner_naive_backtest.png",
            figures_dir / "week3_beginner_ar_backtest.png",
            figures_dir / "week3_beginner_arma_backtest.png",
            figures_dir / "week3_beginner_arx_backtest.png",
            figures_dir / "week3_beginner_armax_backtest.png",
            figures_dir / "week3_beginner_ols_backtest.png",
            figures_dir / "week3_beginner_enet_backtest.png",
            figures_dir / "week3_beginner_horse_race.png",
            figures_dir / "week3_beginner_ensemble_backtest.png",
            story_tables_dir / "week3_beginner_story_scorecard_metrics.csv",
            story_tables_dir / "week3_beginner_story_comparison_metrics.csv",
            story_tables_dir / "week3_beginner_story_absolute_errors.csv",
            story_tables_dir / "week3_beginner_story_latest_forecasts.csv",
            story_tables_dir / "week3_beginner_story_backtest_wide.csv",
            story_figures_dir / "week3_beginner_story_level_change.png",
            story_figures_dir / "week3_beginner_story_scorecard.png",
            story_figures_dir / "week3_beginner_story_backtest.png",
            story_figures_dir / "week3_beginner_story_absolute_errors.png",
            story_figures_dir / "week3_beginner_story_latest_forecast.png",
            us_data_dir / "week3_beginner_unemployment_rate.csv",
            us_data_dir / "week3_beginner_macro_panel.csv",
            us_data_dir / "week3_beginner_simulated_series.csv",
            us_tables_dir / "week3_us_beginner_unit_root_summary.csv",
            us_tables_dir / "week3_us_beginner_naive_metrics.csv",
            us_tables_dir / "week3_us_beginner_ar_metrics.csv",
            us_tables_dir / "week3_us_beginner_arma_metrics.csv",
            us_tables_dir / "week3_us_beginner_arx_metrics.csv",
            us_tables_dir / "week3_us_beginner_horse_race_metrics.csv",
            us_tables_dir / "week3_us_beginner_horse_race_model_specs.csv",
            us_tables_dir / "week3_us_beginner_horse_race_backtest_wide.csv",
            us_tables_dir / "week3_us_beginner_arx_variant_race.csv",
            us_figures_dir / "week3_us_beginner_unit_root_simulations.png",
            us_figures_dir / "week3_us_beginner_naive_backtest.png",
            us_figures_dir / "week3_us_beginner_ar_backtest.png",
            us_figures_dir / "week3_us_beginner_arma_backtest.png",
            us_figures_dir / "week3_us_beginner_arx_backtest.png",
            us_figures_dir / "week3_us_beginner_horse_race.png",
            us_story_tables_dir / "week3_us_beginner_story_scorecard_metrics.csv",
            us_story_tables_dir / "week3_us_beginner_story_comparison_metrics.csv",
            us_story_tables_dir / "week3_us_beginner_story_absolute_errors.csv",
            us_story_tables_dir / "week3_us_beginner_story_latest_forecasts.csv",
            us_story_tables_dir / "week3_us_beginner_story_backtest_wide.csv",
            us_story_figures_dir / "week3_us_beginner_story_level_change.png",
            us_story_figures_dir / "week3_us_beginner_story_scorecard.png",
            us_story_figures_dir / "week3_us_beginner_story_backtest.png",
            us_story_figures_dir / "week3_us_beginner_story_absolute_errors.png",
            us_story_figures_dir / "week3_us_beginner_story_latest_forecast.png",
        ]
        for path in expected:
            assert path.exists(), path
            assert path.stat().st_size > 0
