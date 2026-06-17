# ruff: noqa
"""Run the Week 3 beginner elastic-net walkthrough."""

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
    DEFAULT_ENET_ALPHA_GRID,
    DEFAULT_ENET_COLUMNS,
    DEFAULT_ENET_L1_WT,
    DEFAULT_FIGURES_DIR,
    DEFAULT_OLS_COLUMNS,
    DEFAULT_REGRESSION_TARGET_LAGS,
    DEFAULT_TABLES_DIR,
    add_holdout_arguments,
    build_backtest_comparison_figure,
    coefficient_table_from_fit,
    enet_one_step_forecast,
    ensure_beginner_source_tables,
    fit_dynamic_enet,
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
        description="Run the Week 3 beginner elastic-net walkthrough.",
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
    """Show the step from a small OLS model to a broader penalized regression."""

    args = parse_args(argv)

    # Step 1: make sure the lecture source tables exist.
    ensure_beginner_source_tables(args.data_dir, repo_root=REPO_ROOT)

    # Step 2: keep the same target and build a larger feature bank.
    panel = load_saved_beginner_macro_panel(args.data_dir, repo_root=REPO_ROOT)
    unemployment = panel["unemployment_rate"]
    unemployment_change = panel["unemployment_change_pp"].dropna()
    ols_exog = lagged_exogenous_features(panel, DEFAULT_OLS_COLUMNS)
    enet_exog = lagged_exogenous_features(panel, DEFAULT_ENET_COLUMNS)

    # Step 3: compare naive, small OLS, and broader elastic net on the same evaluation window.
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
    enet_backtest = rolling_one_step_backtest_with_exog(
        unemployment_change,
        unemployment,
        enet_exog,
        forecast_fn=(
            lambda train, train_exog, future_exog: enet_one_step_forecast(
                train,
                train_exog,
                future_exog,
                target_lags=DEFAULT_REGRESSION_TARGET_LAGS,
                alpha_grid=DEFAULT_ENET_ALPHA_GRID,
                l1_wt=DEFAULT_ENET_L1_WT,
            )
        ),
        train_end=args.train_end,
        test_periods=args.test_periods,
    )

    fit, design, _response, _, _, alpha, alpha_table = fit_dynamic_enet(
        unemployment_change,
        enet_exog,
        target_lags=DEFAULT_REGRESSION_TARGET_LAGS,
        alpha_grid=DEFAULT_ENET_ALPHA_GRID,
        l1_wt=DEFAULT_ENET_L1_WT,
    )
    coefficients = coefficient_table_from_fit(fit)
    selected_coefficients = coefficients.loc[coefficients["selected"]].copy()
    metrics = metrics_table(
        {
            "Naive": naive_backtest,
            "OLS": ols_backtest,
            "ENet": enet_backtest,
        },
        target_series=unemployment_change,
        train_end=args.train_end,
        test_periods=args.test_periods,
        align_common_dates=True,
    )

    # Step 4: save the validation, coefficients, backtest, and figure outputs.
    alpha_path = save_table_csv(
        alpha_table.set_index("alpha"),
        args.tables_dir,
        "week3_beginner_enet_alpha_table",
        repo_root=REPO_ROOT,
    )
    coefficients_path = save_table_csv(
        coefficients.set_index("term"),
        args.tables_dir,
        "week3_beginner_enet_coefficients",
        repo_root=REPO_ROOT,
    )
    metrics_path = save_table_csv(
        metrics.set_index("model"),
        args.tables_dir,
        "week3_beginner_enet_metrics",
        repo_root=REPO_ROOT,
    )
    backtest_path = save_table_csv(
        enet_backtest,
        args.tables_dir,
        "week3_beginner_selected_enet_backtest",
        repo_root=REPO_ROOT,
    )
    figure_paths = save_figure_pair(
        build_backtest_comparison_figure(
            {
                "OLS": ols_backtest,
                "ENet": enet_backtest,
            },
            title="Elastic-net backtest: richer predictors with shrinkage",
        ),
        args.figures_dir,
        "week3_beginner_enet_backtest",
        repo_root=REPO_ROOT,
    )

    print("Week 3 beginner elastic net")
    print()
    print("Interpretation:")
    print("- Elastic net keeps the regression idea but allows a much larger predictor bank.")
    print("- The penalty shrinks weak predictors and helps when many inputs overlap.")
    print("- The scorecard stays in target space so shrinkage is judged on real forecast errors.")
    print(f"- Target lags: {DEFAULT_REGRESSION_TARGET_LAGS}")
    print(f"- Feature-bank size before shrinkage: {design.shape[1]} predictors")
    print(f"- Selected alpha from validation: {alpha}")
    print(f"- Elastic-net mixing weight L1_wt: {DEFAULT_ENET_L1_WT}")
    print()
    print("Validation table:")
    print(alpha_table.to_string(index=False))
    print()
    print("Selected coefficients:")
    if selected_coefficients.empty:
        print("No non-zero coefficients were selected.")
    else:
        print(selected_coefficients.to_string(index=False))
    print()
    print(metrics.to_string(index=False))
    print()
    print(f"alpha_table: {alpha_path}")
    print(f"coefficients: {coefficients_path}")
    print(f"metrics: {metrics_path}")
    print(f"selected_enet_backtest: {backtest_path}")
    for label, path in figure_paths.items():
        print(f"figure_{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
