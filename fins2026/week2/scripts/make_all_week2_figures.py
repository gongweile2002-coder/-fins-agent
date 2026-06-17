# ruff: noqa
"""Run the complete Week 2 figure lab."""

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
SCRIPT_DIR = Path(__file__).resolve().parent
for path in [REPO_ROOT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from describe_week2_data import describe_week2_data  # noqa: E402
from make_australia_macro_figures import build_australia_macro_figures  # noqa: E402
from make_fred_market_figures import build_fred_market_figures  # noqa: E402
from make_ft_validation_figures import build_validation_gallery  # noqa: E402
from pull_australia_macro_data import (  # noqa: E402
    FIXTURE_LONG_TABLE as AUSTRALIA_FIXTURE_LONG_TABLE,
)
from pull_australia_macro_data import (  # noqa: E402
    build_output_bundle as build_australia_output_bundle,
)
from pull_australia_macro_data import (  # noqa: E402
    load_live_stage1_long_table,
    write_australia_macro_bundle,
)
from pull_fred_market_data import DEFAULT_OUTPUT as DEFAULT_FRED_OUTPUT  # noqa: E402
from pull_fred_market_data import load_fixture_data, write_fred_market_data  # noqa: E402

from fins2026.week2.code.australia_macro_panel import load_fixture_long_table  # noqa: E402
from fins2026.week2.code.beginner_plotting import (  # noqa: E402
    export_ft_time_series,
    export_simple_time_series,
    write_synthetic_prices_csv,
)


def run_week2_lab(
    *,
    skip_live: bool = False,
    use_fixture: bool = False,
    skip_australia: bool = False,
    rolling_fred_window: bool = False,
) -> dict[str, Path | None]:
    """Run the Week 2 style gallery and market-and-macro figure workflow."""

    print(describe_week2_data())
    print()

    beginner_data_path, beginner_frame = write_synthetic_prices_csv(repo_root=REPO_ROOT)
    beginner_simple_paths = export_simple_time_series(beginner_frame, repo_root=REPO_ROOT)
    beginner_ft_paths = export_ft_time_series(beginner_frame, repo_root=REPO_ROOT)

    validation_docx = build_validation_gallery()

    fred_data_path: Path | None = None
    if use_fixture or skip_live:
        fred_data_path = write_fred_market_data(
            load_fixture_data(rolling_window=rolling_fred_window),
            DEFAULT_FRED_OUTPUT,
        )
        print(f"FRED exercise data from fixture: {fred_data_path}")
    else:
        from pull_fred_market_data import download_fred_market_data

        fred_data_path = write_fred_market_data(
            download_fred_market_data(rolling_window=rolling_fred_window),
            DEFAULT_FRED_OUTPUT,
        )
        print(f"FRED exercise data from live download: {fred_data_path}")

    fred_paths = build_fred_market_figures(
        fred_data_path,
        use_fixture_if_missing=True,
        auto_suite=False,
    )
    australia_data_path: Path | None = None
    australia_docx: Path | None = None
    if not skip_australia:
        australia_stage1 = (
            load_fixture_long_table(REPO_ROOT / AUSTRALIA_FIXTURE_LONG_TABLE)
            if use_fixture or skip_live
            else load_live_stage1_long_table()
        )
        australia_paths = write_australia_macro_bundle(
            build_australia_output_bundle(australia_stage1)
        )
        australia_data_path = australia_paths["stage1_long_csv"]
        australia_docx = build_australia_macro_figures(
            australia_data_path,
            use_fixture_if_missing=True,
        )
    return {
        "beginner_data": beginner_data_path,
        "beginner_simple_png": beginner_simple_paths["png"],
        "beginner_ft_png": beginner_ft_paths["png"],
        "validation_docx": validation_docx,
        "fred_data": fred_data_path,
        "australia_data": australia_data_path,
        "australia_docx": australia_docx,
        **fred_paths,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Run all Week 2 figure examples.")
    parser.add_argument(
        "--skip-live",
        action="store_true",
        help="Do not make live network calls; use the frozen FRED fixture path.",
    )
    parser.add_argument(
        "--use-fixture",
        action="store_true",
        help="Use frozen FRED fixtures for the live-data exercise.",
    )
    parser.add_argument(
        "--skip-australia",
        action="store_true",
        help="Skip the Australia macro extension and run only the original Week 2 surfaces.",
    )
    parser.add_argument(
        "--rolling-fred-window",
        action="store_true",
        help=(
            "Use the latest rolling 10-year FRED window instead of the frozen "
            "2015-01-01 to 2025-12-31 lecture sample."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the complete Week 2 figure lab."""

    args = parse_args(argv)
    paths = run_week2_lab(
        skip_live=args.skip_live,
        use_fixture=args.use_fixture,
        skip_australia=args.skip_australia,
        rolling_fred_window=args.rolling_fred_window,
    )
    print()
    print("Week 2 outputs:")
    for label, path in paths.items():
        if path is not None:
            print(f"- {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
