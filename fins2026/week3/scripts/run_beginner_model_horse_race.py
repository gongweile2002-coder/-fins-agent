# ruff: noqa
"""Run the Week 3 beginner forecast horse race."""

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
    DEFAULT_TABLES_DIR,
    DEFAULT_VARIANT_VALIDATION_PERIODS,
    add_holdout_arguments,
    build_beginner_horse_race_bundle,
    build_target_metric_race_figure,
    build_wide_backtest_forecast_table,
    ensure_beginner_source_tables,
    load_saved_beginner_macro_panel,
    save_figure_pair,
    save_table_csv,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run the Week 3 beginner forecast horse race.",
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
    """Compare the beginner models on one target-space horse-race table."""

    args = parse_args(argv)

    ensure_beginner_source_tables(args.data_dir, repo_root=REPO_ROOT)
    panel = load_saved_beginner_macro_panel(args.data_dir, repo_root=REPO_ROOT)
    bundle = build_beginner_horse_race_bundle(
        panel,
        train_end=args.train_end,
        test_periods=args.test_periods,
        variant_validation_periods=args.variant_validation_periods,
    )

    metrics = bundle["metrics"]
    winner = bundle["winner_model"]
    wide_backtest = build_wide_backtest_forecast_table(bundle["common_backtests"])

    metrics_path = save_table_csv(
        metrics.set_index("model"),
        args.tables_dir,
        "week3_beginner_horse_race_metrics",
        repo_root=REPO_ROOT,
    )
    specs_path = save_table_csv(
        bundle["model_specs"].set_index("model"),
        args.tables_dir,
        "week3_beginner_horse_race_model_specs",
        repo_root=REPO_ROOT,
    )
    wide_backtest_path = save_table_csv(
        wide_backtest,
        args.tables_dir,
        "week3_beginner_horse_race_backtest_wide",
        repo_root=REPO_ROOT,
    )
    arx_variant_path = save_table_csv(
        bundle["arx_variant_table"].set_index("variant"),
        args.tables_dir,
        "week3_beginner_arx_variant_race",
        repo_root=REPO_ROOT,
    )
    armax_variant_path = save_table_csv(
        bundle["armax_variant_table"].set_index("variant"),
        args.tables_dir,
        "week3_beginner_armax_variant_race",
        repo_root=REPO_ROOT,
    )
    figure_paths = save_figure_pair(
        build_target_metric_race_figure(
            metrics,
            bundle["common_backtests"],
            metric_column="target_rmse",
            title="Beginner horse race: target RMSE across forecast families",
        ),
        args.figures_dir,
        "week3_beginner_horse_race",
        repo_root=REPO_ROOT,
    )

    best_row = metrics.sort_values(["target_rmse", "model"]).iloc[0]
    print("Week 3 beginner model horse race")
    print()
    print("Interpretation:")
    print("- Every model is judged on the target change, not on rebuilt unemployment levels.")
    print("- The winner rule is lowest target RMSE on the common one-step evaluation dates.")
    print(
        "- ARX and ARMAX first choose between the cash-only and cash-plus-commodity variants "
        f"using a {args.variant_validation_periods}-month internal validation window."
    )
    print()
    print(
        f"Winner: {winner} | target RMSE {best_row['target_rmse']:.4f} | "
        f"OOS R^2 vs naive {best_row['target_oos_r2_vs_naive']:.4f}"
    )
    print()
    print(metrics.to_string(index=False))
    print()
    print(bundle["model_specs"].to_string(index=False))
    print()
    print(f"metrics: {metrics_path}")
    print(f"model_specs: {specs_path}")
    print(f"wide_backtest: {wide_backtest_path}")
    print(f"arx_variant_race: {arx_variant_path}")
    print(f"armax_variant_race: {armax_variant_path}")
    for label, path in figure_paths.items():
        print(f"figure_{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
