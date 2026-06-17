# ruff: noqa
"""Run the Week 3 beginner naive-forecast walkthrough."""

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
    add_holdout_arguments,
    build_backtest_comparison_figure,
    ensure_beginner_source_tables,
    load_saved_unemployment_rate,
    metrics_table,
    monthly_change,
    naive_one_step_forecast,
    rolling_one_step_backtest,
    save_figure_pair,
    save_table_csv,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run the Week 3 beginner naive-forecast walkthrough.",
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
    """Show the naive one-step benchmark on unemployment-rate changes."""

    args = parse_args(argv)

    # Step 1: make sure the lecture source tables exist.
    ensure_beginner_source_tables(args.data_dir, repo_root=REPO_ROOT)

    # Step 2: load the real unemployment level series and transform it once.
    unemployment = load_saved_unemployment_rate(args.data_dir, repo_root=REPO_ROOT)
    unemployment_change = monthly_change(unemployment)

    # Step 3: run the one-step rolling benchmark on the transformed target.
    naive_backtest = rolling_one_step_backtest(
        unemployment_change,
        unemployment,
        forecast_fn=naive_one_step_forecast,
        train_end=args.train_end,
        test_periods=args.test_periods,
    )
    metrics = metrics_table(
        {"Naive": naive_backtest},
        target_series=unemployment_change,
        train_end=args.train_end,
        test_periods=args.test_periods,
    )

    # Step 4: save the backtest table, metrics table, and comparison figure.
    backtest_path = save_table_csv(
        naive_backtest,
        args.tables_dir,
        "week3_beginner_naive_backtest",
        repo_root=REPO_ROOT,
    )
    metrics_path = save_table_csv(
        metrics.set_index("model"),
        args.tables_dir,
        "week3_beginner_naive_metrics",
        repo_root=REPO_ROOT,
    )
    figure_paths = save_figure_pair(
        build_backtest_comparison_figure(
            {"Naive": naive_backtest},
            title="Naive one-step forecast: unemployment-rate changes",
        ),
        args.figures_dir,
        "week3_beginner_naive_backtest",
        repo_root=REPO_ROOT,
    )

    print("Week 3 beginner naive forecast")
    print()
    print("Interpretation:")
    print("- The naive model says next month's change will look like this month's change.")
    print("- It is simple, but it gives us a benchmark before we try richer models.")
    print("- The metric table evaluates the target directly, not the reconstructed level path.")
    print()
    print(metrics.to_string(index=False))
    print()
    print(f"backtest: {backtest_path}")
    print(f"metrics: {metrics_path}")
    for label, path in figure_paths.items():
        print(f"figure_{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
