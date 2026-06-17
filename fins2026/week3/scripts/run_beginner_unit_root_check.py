# ruff: noqa
"""Run the Week 3 beginner unit-root walkthrough."""

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
    adf_summary,
    build_simulated_unit_root_figure,
    build_unemployment_level_change_figure,
    ensure_beginner_source_tables,
    load_saved_simulated_series,
    load_saved_unemployment_rate,
    monthly_change,
    save_figure_pair,
    save_table_csv,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run the Week 3 beginner unit-root walkthrough.",
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Explain the first Week 3 forecasting decision: levels or differences."""

    args = parse_args(argv)

    # Step 1: make sure the lecture CSVs exist before loading them.
    ensure_beginner_source_tables(args.data_dir, repo_root=REPO_ROOT)

    # Step 2: load the deterministic simulations and the real unemployment series.
    simulated = load_saved_simulated_series(args.data_dir, repo_root=REPO_ROOT)
    unemployment = load_saved_unemployment_rate(args.data_dir, repo_root=REPO_ROOT)
    unemployment_change = monthly_change(unemployment)

    # Step 3: run one plain ADF table that students can read line by line.
    rows = [
        adf_summary(simulated["random_walk"], name="Random walk", version="Level"),
        adf_summary(simulated["stationary_ar1"], name="Stationary AR(1)", version="Level"),
        adf_summary(
            simulated["stationary_arma11"],
            name="Stationary ARMA(1,1)",
            version="Level",
        ),
        adf_summary(unemployment, name="Australia unemployment rate", version="Level"),
        adf_summary(
            unemployment_change,
            name="Australia unemployment rate",
            version="Monthly change",
        ),
    ]
    summary = pd.DataFrame(rows)

    # Step 4: save the summary table and two teaching figures.
    table_path = save_table_csv(
        summary,
        args.tables_dir,
        "week3_beginner_unit_root_summary",
        repo_root=REPO_ROOT,
    )
    simulation_paths = save_figure_pair(
        build_simulated_unit_root_figure(simulated),
        args.figures_dir,
        "week3_beginner_unit_root_simulations",
        repo_root=REPO_ROOT,
    )
    unemployment_paths = save_figure_pair(
        build_unemployment_level_change_figure(unemployment, unemployment_change),
        args.figures_dir,
        "week3_beginner_unemployment_level_and_change",
        repo_root=REPO_ROOT,
    )

    print("Week 3 beginner unit-root check")
    print()
    print("Rule of thumb for this lecture:")
    print("- If the level series looks non-stationary and the ADF p-value is high,")
    print("  forecast a transformed series instead of the level.")
    print("- For unemployment, we will forecast the monthly change in percentage points.")
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
