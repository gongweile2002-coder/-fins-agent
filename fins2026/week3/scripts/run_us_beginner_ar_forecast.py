# ruff: noqa
"""Run the Week 3 U.S. beginner AR-forecast walkthrough."""

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
    DEFAULT_US_DATA_DIR,
    DEFAULT_US_FIGURES_DIR,
    DEFAULT_US_TABLES_DIR,
    add_holdout_arguments,
    ar_one_step_forecast,
    build_backtest_comparison_figure,
    build_selection_figure,
    choose_best_ar_lag,
    ensure_us_beginner_source_tables,
    load_saved_unemployment_rate,
    metrics_table,
    monthly_change,
    naive_one_step_forecast,
    rolling_one_step_backtest,
    save_figure_pair,
    save_table_csv,
    split_dates,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run the Week 3 U.S. beginner AR-forecast walkthrough.",
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
    """Show AR(1) and selected-AR forecasts for U.S. unemployment-rate changes."""

    args = parse_args(argv)

    ensure_us_beginner_source_tables(args.data_dir, repo_root=REPO_ROOT)

    unemployment = load_saved_unemployment_rate(args.data_dir, repo_root=REPO_ROOT)
    unemployment_change = monthly_change(unemployment)
    train_last_date, _ = split_dates(
        unemployment_change,
        train_end=args.train_end,
        test_periods=args.test_periods,
    )
    initial_train = unemployment_change.loc[:train_last_date]

    selected_lag, lag_table = choose_best_ar_lag(initial_train, DEFAULT_AR_LAGS)

    naive_backtest = rolling_one_step_backtest(
        unemployment_change,
        unemployment,
        forecast_fn=naive_one_step_forecast,
        train_end=args.train_end,
        test_periods=args.test_periods,
    )
    ar1_backtest = rolling_one_step_backtest(
        unemployment_change,
        unemployment,
        forecast_fn=lambda train: ar_one_step_forecast(train, lag=1),
        train_end=args.train_end,
        test_periods=args.test_periods,
    )
    selected_backtest = rolling_one_step_backtest(
        unemployment_change,
        unemployment,
        forecast_fn=lambda train: ar_one_step_forecast(train, lag=selected_lag),
        train_end=args.train_end,
        test_periods=args.test_periods,
    )

    metrics = metrics_table(
        {
            "Naive": naive_backtest,
            "AR(1)": ar1_backtest,
            f"AR({selected_lag})": selected_backtest,
        },
        target_series=unemployment_change,
        train_end=args.train_end,
        test_periods=args.test_periods,
        align_common_dates=True,
    )

    lag_table_path = save_table_csv(
        lag_table.set_index("lag"),
        args.tables_dir,
        "week3_us_beginner_ar_lag_table",
        repo_root=REPO_ROOT,
    )
    metrics_path = save_table_csv(
        metrics.set_index("model"),
        args.tables_dir,
        "week3_us_beginner_ar_metrics",
        repo_root=REPO_ROOT,
    )
    selected_backtest_path = save_table_csv(
        selected_backtest,
        args.tables_dir,
        "week3_us_beginner_selected_ar_backtest",
        repo_root=REPO_ROOT,
    )
    lag_figure_paths = save_figure_pair(
        build_selection_figure(
            lag_table,
            x_column="lag",
            title="U.S. AR lag ladder chosen by BIC",
        ),
        args.figures_dir,
        "week3_us_beginner_ar_lag_selection",
        repo_root=REPO_ROOT,
    )
    backtest_figure_paths = save_figure_pair(
        build_backtest_comparison_figure(
            {
                "Naive": naive_backtest,
                "AR(1)": ar1_backtest,
                f"AR({selected_lag})": selected_backtest,
            },
            title="U.S. AR backtest: unemployment-rate changes",
        ),
        args.figures_dir,
        "week3_us_beginner_ar_backtest",
        repo_root=REPO_ROOT,
    )

    print("Week 3 U.S. beginner autoregression")
    print()
    print("Interpretation:")
    print("- AR models let the recent pattern of U.S. unemployment changes matter explicitly.")
    print("- We keep the lag choice transparent by showing the small BIC ladder.")
    print("- The comparison table ranks target forecasts directly with target-space errors.")
    print(f"- Selected lag on the initial training sample: AR({selected_lag})")
    print()
    print(lag_table.to_string(index=False))
    print()
    print(metrics.to_string(index=False))
    print()
    print(f"lag_table: {lag_table_path}")
    print(f"metrics: {metrics_path}")
    print(f"selected_backtest: {selected_backtest_path}")
    for label, path in lag_figure_paths.items():
        print(f"lag_figure_{label}: {path}")
    for label, path in backtest_figure_paths.items():
        print(f"backtest_figure_{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
