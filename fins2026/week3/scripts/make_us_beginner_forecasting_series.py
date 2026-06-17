# ruff: noqa
"""Create the Week 3 U.S. beginner forecasting source CSVs."""

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
    load_fixture_us_unemployment_rate,
    make_simulated_series_frame,
    resolve_repo_path,
    sample_label,
    write_us_beginner_source_tables,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Create the Week 3 U.S. beginner forecasting source CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_US_DATA_DIR),
        help="Repo-relative or absolute output directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Write the real and simulated CSVs used by the U.S. extension ladder."""

    args = parse_args(argv)
    output_dir = resolve_repo_path(args.output_dir, REPO_ROOT)
    written = write_us_beginner_source_tables(output_dir, repo_root=REPO_ROOT)

    unemployment = load_fixture_us_unemployment_rate()
    print(f"real series sample: {sample_label(unemployment)}")

    simulated = make_simulated_series_frame().set_index("date")
    print(f"simulated sample: {sample_label(simulated['random_walk'])}")

    for label, path in written.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
