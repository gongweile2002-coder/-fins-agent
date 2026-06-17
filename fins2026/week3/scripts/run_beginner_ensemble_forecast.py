# ruff: noqa
"""Run the Week 3 beginner ensemble walkthrough."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


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
    DEFAULT_TABLES_DIR,
    DEFAULT_VARIANT_VALIDATION_PERIODS,
    add_holdout_arguments,
    beginner_model_one_step_forecasts,
    build_beginner_horse_race_bundle,
    build_target_only_backtest_figure,
    ensure_beginner_source_tables,
    equal_weight_ensemble_backtest,
    load_saved_beginner_macro_panel,
    metrics_table,
    save_figure_pair,
    save_table_csv,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run the Week 3 beginner ensemble walkthrough.",
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
    parser.add_argument(
        "--variant-validation-periods",
        type=int,
        default=DEFAULT_VARIANT_VALIDATION_PERIODS,
        help=(
            "Months used to choose the ARX and ARMAX exogenous variants "
            "inside the training sample."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Compare the horse-race winner with an equal-weight ensemble."""

    args = parse_args(argv)

    ensure_beginner_source_tables(args.data_dir, repo_root=REPO_ROOT)
    panel = load_saved_beginner_macro_panel(args.data_dir, repo_root=REPO_ROOT)
    bundle = build_beginner_horse_race_bundle(
        panel,
        train_end=args.train_end,
        test_periods=args.test_periods,
        variant_validation_periods=args.variant_validation_periods,
    )

    common_backtests = bundle["common_backtests"]
    non_naive_members = {
        label: frame for label, frame in common_backtests.items() if label != "Naive"
    }
    ensemble_backtest = equal_weight_ensemble_backtest(
        non_naive_members,
        level_series=bundle["level_series"],
    )
    winner = bundle["winner_model"]
    comparison_backtests = {
        "Naive": common_backtests["Naive"],
        "Equal-weight ensemble": ensemble_backtest,
    }
    if winner != "Naive":
        comparison_backtests[winner] = common_backtests[winner]

    metrics = metrics_table(
        comparison_backtests,
        target_series=bundle["target_series"],
        train_end=args.train_end,
        test_periods=args.test_periods,
        naive_label="Naive",
        align_common_dates=True,
    ).sort_values(["target_rmse", "model"]).reset_index(drop=True)

    one_step_forecasts = beginner_model_one_step_forecasts(panel, bundle)
    ensemble_row = pd.DataFrame(
        {
            "forecast_date": [one_step_forecasts["forecast_date"].iloc[0]],
            "model": ["Equal-weight ensemble"],
            "target_forecast": [
                float(
                    one_step_forecasts.loc[
                        one_step_forecasts["model"] != "Naive",
                        "target_forecast",
                    ].mean()
                )
            ],
        }
    )
    latest_forecast_table = pd.concat([one_step_forecasts, ensemble_row], ignore_index=True)

    metrics_path = save_table_csv(
        metrics.set_index("model"),
        args.tables_dir,
        "week3_beginner_ensemble_metrics",
        repo_root=REPO_ROOT,
    )
    backtest_path = save_table_csv(
        ensemble_backtest,
        args.tables_dir,
        "week3_beginner_selected_ensemble_backtest",
        repo_root=REPO_ROOT,
    )
    latest_forecast_path = save_table_csv(
        latest_forecast_table.set_index("model"),
        args.tables_dir,
        "week3_beginner_ensemble_one_step_forecasts",
        repo_root=REPO_ROOT,
    )
    figure_paths = save_figure_pair(
        build_target_only_backtest_figure(
            comparison_backtests,
            title="Equal-weight ensemble versus the benchmark forecasts",
        ),
        args.figures_dir,
        "week3_beginner_ensemble_backtest",
        repo_root=REPO_ROOT,
    )

    print("Week 3 beginner forecast ensemble")
    print()
    print("Interpretation:")
    print("- The ensemble simply averages the successful non-naive target forecasts.")
    print("- We compare the ensemble with the naive benchmark and the horse-race winner.")
    print("- The scorecard stays in target space and uses the same evaluation dates.")
    print()
    print(f"Horse-race winner: {winner}")
    print()
    print(metrics.to_string(index=False))
    print()
    print(latest_forecast_table.to_string(index=False))
    print()
    print(f"metrics: {metrics_path}")
    print(f"selected_ensemble_backtest: {backtest_path}")
    print(f"one_step_forecasts: {latest_forecast_path}")
    for label, path in figure_paths.items():
        print(f"figure_{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
