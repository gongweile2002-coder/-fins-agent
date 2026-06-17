# ruff: noqa
"""Run the Week 3 beginner OLS-forecast walkthrough."""

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
    DEFAULT_DATA_DIR,
    DEFAULT_FIGURES_DIR,
    DEFAULT_OLS_COLUMNS,
    DEFAULT_REGRESSION_TARGET_LAGS,
    DEFAULT_TABLES_DIR,
    add_holdout_arguments,
    build_backtest_comparison_figure,
    coefficient_table_from_fit,
    ensure_beginner_source_tables,
    fit_dynamic_ols,
    lagged_exogenous_features,
    load_saved_beginner_macro_panel,
    metrics_table,
    naive_one_step_forecast,
    ols_one_step_forecast,
    rolling_one_step_backtest,
    rolling_one_step_backtest_with_exog,
    save_figure_pair,
    save_table_csv,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run the Week 3 beginner OLS-forecast walkthrough.",
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
    """Show the step from ARX/ARMAX into a simple dynamic OLS regression."""

    args = parse_args(argv)

    # Step 1: make sure the lecture source tables exist.
    ensure_beginner_source_tables(args.data_dir, repo_root=REPO_ROOT)

    # Step 2: load the real-data panel and keep the same unemployment-change target.
    panel = load_saved_beginner_macro_panel(args.data_dir, repo_root=REPO_ROOT)
    unemployment = panel["unemployment_rate"]
    unemployment_change = panel["unemployment_change_pp"].dropna()

    # Step 3: build one lagged predictor set by hand.
    ols_exog = lagged_exogenous_features(panel, DEFAULT_OLS_COLUMNS)

    # Step 4: compare the naive benchmark with a dynamic OLS regression.
    naive_backtest = rolling_one_step_backtest(
        unemployment_change,
        unemployment,
        forecast_fn=naive_one_step_forecast,
        train_end=args.train_end,
        test_periods=args.test_periods,
    )
    ols_backtest = rolling_one_step_backtest_with_exog(
        unemployment_change,
        unemployment,
        ols_exog,
        forecast_fn=(
            lambda train, train_exog, future_exog: ols_one_step_forecast(
                train,
                train_exog,
                future_exog,
                target_lags=DEFAULT_REGRESSION_TARGET_LAGS,
            )
        ),
        train_end=args.train_end,
        test_periods=args.test_periods,
    )

    fit, design, response = fit_dynamic_ols(
        unemployment_change,
        ols_exog,
        target_lags=DEFAULT_REGRESSION_TARGET_LAGS,
    )
    coefficients = coefficient_table_from_fit(fit)
    metrics = metrics_table(
        {
            "Naive": naive_backtest,
            "OLS": ols_backtest,
        },
        target_series=unemployment_change,
        train_end=args.train_end,
        test_periods=args.test_periods,
        align_common_dates=True,
    )

    # Step 5: save the lecture artifacts.
    coefficients_path = save_table_csv(
        coefficients.set_index("term"),
        args.tables_dir,
        "week3_beginner_ols_coefficients",
        repo_root=REPO_ROOT,
    )
    metrics_path = save_table_csv(
        metrics.set_index("model"),
        args.tables_dir,
        "week3_beginner_ols_metrics",
        repo_root=REPO_ROOT,
    )
    backtest_path = save_table_csv(
        ols_backtest,
        args.tables_dir,
        "week3_beginner_selected_ols_backtest",
        repo_root=REPO_ROOT,
    )
    figure_paths = save_figure_pair(
        build_backtest_comparison_figure(
            {
                "OLS": ols_backtest,
            },
            title="OLS backtest: unemployment changes with hand-picked predictors",
        ),
        args.figures_dir,
        "week3_beginner_ols_backtest",
        repo_root=REPO_ROOT,
    )

    print("Week 3 beginner OLS model")
    print()
    print("Interpretation:")
    print("- OLS is the clean bridge from time-series models into regression-style forecasting.")
    print(
        "- We keep the target as monthly unemployment-rate changes and explain it with a "
        "few lagged outside variables."
    )
    print(
        "- The regression is judged on target errors, not on the rebuilt "
        "unemployment-rate level."
    )
    print(f"- Target lags: {DEFAULT_REGRESSION_TARGET_LAGS}")
    print("- Hand-picked lagged predictors:")
    for column in DEFAULT_OLS_COLUMNS:
        print(f"  - {column}")
    print()
    print(f"Effective regression sample: {len(response)} rows and {design.shape[1]} predictors")
    print()
    print("Coefficient table:")
    print(coefficients.to_string(index=False))
    print()
    print(metrics.to_string(index=False))
    print()
    print(f"coefficients: {coefficients_path}")
    print(f"metrics: {metrics_path}")
    print(f"selected_ols_backtest: {backtest_path}")
    for label, path in figure_paths.items():
        print(f"figure_{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
