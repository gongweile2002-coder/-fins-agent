# ruff: noqa
"""Run the Week 3 beginner ARMAX-forecast walkthrough."""

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
    DEFAULT_ARMA_ORDERS,
    DEFAULT_ARX_COLUMNS,
    DEFAULT_ARX_PLUS_COLUMNS,
    DEFAULT_DATA_DIR,
    DEFAULT_FIGURES_DIR,
    DEFAULT_TABLES_DIR,
    add_holdout_arguments,
    arma_one_step_forecast,
    armax_one_step_forecast,
    build_backtest_comparison_figure,
    choose_best_arma_order,
    ensure_beginner_source_tables,
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
        description="Run the Week 3 beginner ARMAX-forecast walkthrough.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Repo-relative or absolute source-data directory.",
    )
    parser.add_argument(
        "--figures-dir",
        default=str(DEFAULT_FIGURES_DIR),
        help="Repo-relative or absolute figures directory.",
    )
    parser.add_argument(
        "--tables-dir",
        default=str(DEFAULT_TABLES_DIR),
        help="Repo-relative or absolute tables directory.",
    )
    add_holdout_arguments(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Show the step from ARMA to ARMAX on unemployment-rate changes."""

    args = parse_args(argv)

    # Step 1: make sure the lecture source tables exist.
    ensure_beginner_source_tables(args.data_dir, repo_root=REPO_ROOT)

    # Step 2: load the small real-data panel and keep the same target.
    panel = load_saved_beginner_macro_panel(args.data_dir, repo_root=REPO_ROOT)
    unemployment = panel["unemployment_rate"]
    unemployment_change = panel["unemployment_change_pp"].dropna()
    train_last_date, _ = split_dates(
        unemployment_change,
        train_end=args.train_end,
        test_periods=args.test_periods,
    )
    initial_train = unemployment_change.loc[:train_last_date]

    # Step 3: keep the ARMA order fixed, then add outside information.
    selected_order, order_table = choose_best_arma_order(initial_train, DEFAULT_ARMA_ORDERS)
    cash_only = lagged_exogenous_features(panel, DEFAULT_ARX_COLUMNS)
    cash_plus_commodity = lagged_exogenous_features(panel, DEFAULT_ARX_PLUS_COLUMNS)

    # Step 4: compare ARMA with ARMAX variants on the same evaluation window.
    naive_backtest = rolling_one_step_backtest(
        unemployment_change,
        unemployment,
        forecast_fn=naive_one_step_forecast,
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
    armax_cash_backtest = rolling_one_step_backtest_with_exog(
        unemployment_change,
        unemployment,
        cash_only,
        forecast_fn=(
            lambda train, train_exog, future_exog: armax_one_step_forecast(
                train,
                train_exog,
                future_exog,
                order=selected_order,
            )
        ),
        train_end=args.train_end,
        test_periods=args.test_periods,
    )
    armax_cash_commodity_backtest = rolling_one_step_backtest_with_exog(
        unemployment_change,
        unemployment,
        cash_plus_commodity,
        forecast_fn=(
            lambda train, train_exog, future_exog: armax_one_step_forecast(
                train,
                train_exog,
                future_exog,
                order=selected_order,
            )
        ),
        train_end=args.train_end,
        test_periods=args.test_periods,
    )

    metrics = metrics_table(
        {
            "Naive": naive_backtest,
            f"ARMA{selected_order}": arma_backtest,
            f"ARMAX{selected_order} cash": armax_cash_backtest,
            f"ARMAX{selected_order} cash+commodity": armax_cash_commodity_backtest,
        },
        target_series=unemployment_change,
        train_end=args.train_end,
        test_periods=args.test_periods,
        align_common_dates=True,
    )

    # Step 5: save the selected-order table, metrics, backtest, and figure.
    order_table_path = save_table_csv(
        order_table.set_index("order"),
        args.tables_dir,
        "week3_beginner_armax_order_table",
        repo_root=REPO_ROOT,
    )
    metrics_path = save_table_csv(
        metrics.set_index("model"),
        args.tables_dir,
        "week3_beginner_armax_metrics",
        repo_root=REPO_ROOT,
    )
    armax_backtest_path = save_table_csv(
        armax_cash_commodity_backtest,
        args.tables_dir,
        "week3_beginner_selected_armax_backtest",
        repo_root=REPO_ROOT,
    )
    figure_paths = save_figure_pair(
        build_backtest_comparison_figure(
            {
                f"ARMA{selected_order}": arma_backtest,
                f"ARMAX{selected_order} cash": armax_cash_backtest,
                f"ARMAX{selected_order} cash+commodity": armax_cash_commodity_backtest,
            },
            title="ARMAX backtest: ARMA plus outside information",
        ),
        args.figures_dir,
        "week3_beginner_armax_backtest",
        repo_root=REPO_ROOT,
    )

    print("Week 3 beginner ARMAX model")
    print()
    print("Interpretation:")
    print("- ARMAX keeps the ARMA structure and adds outside variables.")
    print("- Here we reuse the selected ARMA order and ask whether outside information helps.")
    print("- The metric table scores target forecasts directly so the comparison stays consistent.")
    print(f"- Selected ARMA order from the ARMA script: {selected_order}")
    print()
    print(order_table.to_string(index=False))
    print()
    print(metrics.to_string(index=False))
    print()
    print(f"order_table: {order_table_path}")
    print(f"metrics: {metrics_path}")
    print(f"selected_armax_backtest: {armax_backtest_path}")
    for label, path in figure_paths.items():
        print(f"figure_{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
