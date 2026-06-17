# ruff: noqa
"""Run the Week 3 U.S. beginner ARMA-forecast walkthrough."""

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
    DEFAULT_ARMA_ORDERS,
    DEFAULT_US_DATA_DIR,
    DEFAULT_US_FIGURES_DIR,
    DEFAULT_US_TABLES_DIR,
    add_holdout_arguments,
    ar_one_step_forecast,
    arma_one_step_forecast,
    build_backtest_comparison_figure,
    build_selection_figure,
    choose_best_ar_lag,
    choose_best_arma_order,
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
        description="Run the Week 3 U.S. beginner ARMA-forecast walkthrough.",
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
    """Show the step from AR to ARMA on U.S. unemployment-rate changes."""

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

    selected_lag, _ = choose_best_ar_lag(initial_train, DEFAULT_AR_LAGS)
    selected_order, order_table = choose_best_arma_order(initial_train, DEFAULT_ARMA_ORDERS)

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
    arma_backtest = rolling_one_step_backtest(
        unemployment_change,
        unemployment,
        forecast_fn=lambda train: arma_one_step_forecast(train, order=selected_order),
        train_end=args.train_end,
        test_periods=args.test_periods,
    )
    metrics = metrics_table(
        {
            "Naive": naive_backtest,
            f"AR({selected_lag})": ar_backtest,
            f"ARMA{selected_order}": arma_backtest,
        },
        target_series=unemployment_change,
        train_end=args.train_end,
        test_periods=args.test_periods,
        align_common_dates=True,
    )

    order_table_path = save_table_csv(
        order_table.set_index("order"),
        args.tables_dir,
        "week3_us_beginner_arma_order_table",
        repo_root=REPO_ROOT,
    )
    metrics_path = save_table_csv(
        metrics.set_index("model"),
        args.tables_dir,
        "week3_us_beginner_arma_metrics",
        repo_root=REPO_ROOT,
    )
    arma_backtest_path = save_table_csv(
        arma_backtest,
        args.tables_dir,
        "week3_us_beginner_selected_arma_backtest",
        repo_root=REPO_ROOT,
    )
    order_figure_paths = save_figure_pair(
        build_selection_figure(
            order_table,
            x_column="order",
            title="U.S. ARMA order ladder chosen by BIC",
        ),
        args.figures_dir,
        "week3_us_beginner_arma_order_selection",
        repo_root=REPO_ROOT,
    )
    backtest_figure_paths = save_figure_pair(
        build_backtest_comparison_figure(
            {
                "Naive": naive_backtest,
                f"AR({selected_lag})": ar_backtest,
                f"ARMA{selected_order}": arma_backtest,
            },
            title="U.S. ARMA backtest: unemployment-rate changes",
        ),
        args.figures_dir,
        "week3_us_beginner_arma_backtest",
        repo_root=REPO_ROOT,
    )

    print("Week 3 U.S. beginner ARMA model")
    print()
    print("Interpretation:")
    print("- ARMA keeps the autoregressive lags and also models short-run shocks.")
    print("- We still keep the model search narrow and visible for students.")
    print("- The metric table stays in target space so the model race is apples-to-apples.")
    print(f"- Selected AR lag on the training sample: AR({selected_lag})")
    print(f"- Selected ARMA order on the training sample: {selected_order}")
    print()
    print(order_table.to_string(index=False))
    print()
    print(metrics.to_string(index=False))
    print()
    print(f"order_table: {order_table_path}")
    print(f"metrics: {metrics_path}")
    print(f"arma_backtest: {arma_backtest_path}")
    for label, path in order_figure_paths.items():
        print(f"order_figure_{label}: {path}")
    for label, path in backtest_figure_paths.items():
        print(f"backtest_figure_{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
