# ruff: noqa
"""Run the Week 3 U.S. beginner ARX-forecast walkthrough."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repo root from this script location."""

    current = (start or Path(__file__)).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "tools" / "workflow_lib.py"
        ).is_file():
            return candidate
    raise RuntimeError("Could not find the fins-agent repo root.")


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fins2026.week3.code.beginner_forecasting import (  # noqa: E402
    DEFAULT_AR_LAGS,
    DEFAULT_US_ARX_COLUMNS,
    DEFAULT_US_ARX_PLUS_COLUMNS,
    DEFAULT_US_DATA_DIR,
    DEFAULT_US_FIGURES_DIR,
    DEFAULT_US_TABLES_DIR,
    add_holdout_arguments,
    ar_one_step_forecast,
    arx_one_step_forecast,
    build_backtest_comparison_figure,
    choose_best_ar_lag,
    contemporaneous_correlation_table,
    ensure_us_beginner_source_tables,
    lagged_exogenous_features,
    load_saved_beginner_macro_panel,
    metrics_table,
    naive_one_step_forecast,
    rolling_one_step_backtest,
    rolling_one_step_backtest_with_exog,
    save_figure_pair,
    save_table_csv,
    split_dates,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run the Week 3 U.S. beginner ARX-forecast walkthrough.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_US_DATA_DIR),
        help="Repo-relative or absolute source-data directory.",
    )
    parser.add_argument(
        "--figures-dir",
        default=str(DEFAULT_US_FIGURES_DIR),
        help="Repo-relative or absolute figures directory.",
    )
    parser.add_argument(
        "--tables-dir",
        default=str(DEFAULT_US_TABLES_DIR),
        help="Repo-relative or absolute tables directory.",
    )
    add_holdout_arguments(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Show what happens when outside U.S. information is added to the AR model."""

    args = parse_args(argv)

    ensure_us_beginner_source_tables(args.data_dir, repo_root=REPO_ROOT)

    panel = load_saved_beginner_macro_panel(args.data_dir, repo_root=REPO_ROOT)
    unemployment = panel["unemployment_rate"]
    unemployment_change = panel["unemployment_change_pp"].dropna()
    train_last_date, _ = split_dates(
        unemployment_change,
        train_end=args.train_end,
        test_periods=args.test_periods,
    )
    initial_train = unemployment_change.loc[:train_last_date]

    selected_lag, _ = choose_best_ar_lag(initial_train, DEFAULT_AR_LAGS)
    fedfunds_only = lagged_exogenous_features(panel, DEFAULT_US_ARX_COLUMNS)
    fedfunds_plus_spread = lagged_exogenous_features(panel, DEFAULT_US_ARX_PLUS_COLUMNS)

    naive_backtest = rolling_one_step_backtest(
        unemployment_change,
        unemployment,
        forecast_fn=naive_one_step_forecast,
        train_end=args.train_end,
        test_periods=args.test_periods,
    )
    ar_backtest = rolling_one_step_backtest(
        unemployment_change,
        unemployment,
        forecast_fn=lambda train: ar_one_step_forecast(train, lag=selected_lag),
        train_end=args.train_end,
        test_periods=args.test_periods,
    )
    arx_fedfunds_backtest = rolling_one_step_backtest_with_exog(
        unemployment_change,
        unemployment,
        fedfunds_only,
        forecast_fn=(
            lambda train, train_exog, future_exog: arx_one_step_forecast(
                train,
                train_exog,
                future_exog,
                lag=selected_lag,
            )
        ),
        train_end=args.train_end,
        test_periods=args.test_periods,
    )
    arx_fedfunds_spread_backtest = rolling_one_step_backtest_with_exog(
        unemployment_change,
        unemployment,
        fedfunds_plus_spread,
        forecast_fn=(
            lambda train, train_exog, future_exog: arx_one_step_forecast(
                train,
                train_exog,
                future_exog,
                lag=selected_lag,
            )
        ),
        train_end=args.train_end,
        test_periods=args.test_periods,
    )

    metrics = metrics_table(
        {
            "Naive": naive_backtest,
            f"AR({selected_lag})": ar_backtest,
            f"ARX({selected_lag}) fedfunds": arx_fedfunds_backtest,
            f"ARX({selected_lag}) fedfunds+spread": arx_fedfunds_spread_backtest,
        },
        target_series=unemployment_change,
        train_end=args.train_end,
        test_periods=args.test_periods,
        align_common_dates=True,
    )
    correlation_table = contemporaneous_correlation_table(
        panel,
        columns=[
            "unemployment_change_pp",
            "fedfunds_change_pp",
            "yield_spread_pp",
        ],
    )

    correlation_path = save_table_csv(
        correlation_table,
        args.tables_dir,
        "week3_us_beginner_arx_correlation_table",
        repo_root=REPO_ROOT,
    )
    metrics_path = save_table_csv(
        metrics.set_index("model"),
        args.tables_dir,
        "week3_us_beginner_arx_metrics",
        repo_root=REPO_ROOT,
    )
    arx_backtest_path = save_table_csv(
        arx_fedfunds_spread_backtest,
        args.tables_dir,
        "week3_us_beginner_selected_arx_backtest",
        repo_root=REPO_ROOT,
    )
    figure_paths = save_figure_pair(
        build_backtest_comparison_figure(
            {
                f"AR({selected_lag})": ar_backtest,
                f"ARX({selected_lag}) fedfunds": arx_fedfunds_backtest,
                f"ARX({selected_lag}) fedfunds+spread": arx_fedfunds_spread_backtest,
            },
            title="U.S. ARX backtest: unemployment changes with policy and curve signals",
        ),
        args.figures_dir,
        "week3_us_beginner_arx_backtest",
        repo_root=REPO_ROOT,
    )

    print("Week 3 U.S. beginner ARX model")
    print()
    print("Interpretation:")
    print("- ARX keeps the same autoregressive backbone and adds outside variables.")
    print("- Here we use last month's federal-funds change first, then add the yield spread.")
    print("- Model quality is evaluated on the target change, not on rebuilt unemployment levels.")
    print(f"- Fixed autoregressive backbone from the AR script: AR({selected_lag})")
    print()
    print("Simple contemporaneous correlations:")
    print(correlation_table.to_string())
    print()
    print(metrics.to_string(index=False))
    print()
    print(f"correlations: {correlation_path}")
    print(f"metrics: {metrics_path}")
    print(f"selected_arx_backtest: {arx_backtest_path}")
    for label, path in figure_paths.items():
        print(f"figure_{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
