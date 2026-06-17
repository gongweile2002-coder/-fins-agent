# ruff: noqa
"""Create the Week 2 beginner synthetic price CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_OUTPUT = Path("fins2026/week2/results/data/week2_beginner_synthetic_prices.csv")


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

from fins2026.week2.code.beginner_plotting import (  # noqa: E402
    beginner_sample_label,
    write_synthetic_prices_csv,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Create the Week 2 beginner synthetic price CSV.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=(
            "Repo-relative or absolute CSV path. Defaults to "
            "fins2026/week2/results/data/week2_beginner_synthetic_prices.csv."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Write the synthetic CSV and print a compact schema summary."""

    args = parse_args(argv)
    path, frame = write_synthetic_prices_csv(args.output, repo_root=REPO_ROOT)
    print(f"csv: {path}")
    print(f"rows: {len(frame)}")
    print(f"columns: {', '.join(frame.columns)}")
    print(f"sample: {beginner_sample_label(frame)}")
    print(frame.head().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
