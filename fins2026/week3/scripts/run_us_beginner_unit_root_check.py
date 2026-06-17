# ruff: noqa
"""Run the Week 3 U.S. beginner unit-root walkthrough."""

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
    DEFAULT_US_DATA_DIR,
    DEFAULT_US_FIGURES_DIR,
    DEFAULT_US_TABLES_DIR,
    adf_summary,
    build_rate_level_change_figure,
    build_simulated_unit_root_figure,
    ensure_us_beginner_source_tables,
    load_saved_simulated_series,
    load_saved_unemployment_rate,
    monthly_change,
    save_figure_pair,
    save_table_csv,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run the Week 3 U.S. beginner unit-root walkthrough.",
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Explain the first U.S. extension forecasting decision: levels or differences."""

    args = parse_args(argv)

    ensure_us_beginner_source_tables(args.data_dir, repo_root=REPO_ROOT)

    simulated = load_saved_simulated_series(args.data_dir, repo_root=REPO_ROOT)
    unemployment = load_saved_unemployment_rate(args.data_dir, repo_root=REPO_ROOT)
    unemployment_change = monthly_change(unemployment)

    rows = [
        adf_summary(simulated["random_walk"], name="Random walk", version="Level"),
        adf_summary(simulated["stationary_ar1"], name="Stationary AR(1)", version="Level"),
        adf_summary(
            simulated["stationary_arma11"],
            name="Stationary ARMA(1,1)",
            version="Level",
        ),
        adf_summary(unemployment, name="U.S. unemployment rate", version="Level"),
        adf_summary(
            unemployment_change,
            name="U.S. unemployment rate",
            version="Monthly change",
        ),
    ]
    summary = pd.DataFrame(rows)

    table_path = save_table_csv(
        summary,
        args.tables_dir,
        "week3_us_beginner_unit_root_summary",
        repo_root=REPO_ROOT,
    )
    simulation_paths = save_figure_pair(
        build_simulated_unit_root_figure(simulated),
        args.figures_dir,
        "week3_us_beginner_unit_root_simulations",
        repo_root=REPO_ROOT,
    )
    unemployment_paths = save_figure_pair(
        build_rate_level_change_figure(
            unemployment,
            unemployment_change,
            title_prefix="U.S. unemployment rate",
        ),
        args.figures_dir,
        "week3_us_beginner_unemployment_level_and_change",
        repo_root=REPO_ROOT,
    )

    print("Week 3 U.S. beginner unit-root check")
    print()
    print("Rule of thumb for this extension:")
    print("- If the level series looks non-stationary and the ADF p-value is high,")
    print("  forecast a transformed series instead of the level.")
    print("- For U.S. unemployment, we will forecast the monthly change in percentage points.")
    print()
    print(summary.to_string(index=False))
    print()
    print(f"table: {table_path}")
    for label, path in simulation_paths.items():
        print(f"simulation_{label}: {path}")
    for label, path in unemployment_paths.items():
        print(f"unemployment_{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
