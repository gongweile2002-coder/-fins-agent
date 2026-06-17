# ruff: noqa
"""Build the Week 3 U.S. forecast story figure pack."""

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
    DEFAULT_US_DATA_DIR,
    DEFAULT_VARIANT_VALIDATION_PERIODS,
    add_holdout_arguments,
)
from fins2026.week3.code.forecast_story_figures import (  # noqa: E402
    DEFAULT_US_STORY_FIGURES_DIR,
    DEFAULT_US_STORY_TABLES_DIR,
    build_and_save_us_story_pack,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Build the Week 3 U.S. forecast story figure pack.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_US_DATA_DIR),
        help="Repo-relative or absolute source-data directory.",
    )
    parser.add_argument(
        "--figures-dir",
        default=str(DEFAULT_US_STORY_FIGURES_DIR),
        help="Repo-relative or absolute figures directory.",
    )
    parser.add_argument(
        "--tables-dir",
        default=str(DEFAULT_US_STORY_TABLES_DIR),
        help="Repo-relative or absolute tables directory.",
    )
    add_holdout_arguments(parser)
    parser.add_argument(
        "--variant-validation-periods",
        type=int,
        default=DEFAULT_VARIANT_VALIDATION_PERIODS,
        help="Months used to choose the ARX variant inside the training sample.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build and save the U.S. narrative figures."""

    args = parse_args(argv)
    outputs = build_and_save_us_story_pack(
        data_dir=args.data_dir,
        figures_dir=args.figures_dir,
        tables_dir=args.tables_dir,
        train_end=args.train_end,
        test_periods=args.test_periods,
        variant_validation_periods=args.variant_validation_periods,
        repo_root=REPO_ROOT,
    )

    print("Week 3 U.S. forecast story figures")
    print()
    print("Interpretation:")
    print(
        "- The U.S. pack mirrors the Australia narrative: target choice, "
        "scorecard, out-of-sample fit, absolute errors, latest forecast."
    )
    print(
        "- The winner is always judged on the target change, not on the "
        "rebuilt unemployment level."
    )
    print()
    for label, path in sorted(outputs.items()):
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
