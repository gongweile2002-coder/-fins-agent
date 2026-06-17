# ruff: noqa
"""Build the plain Matplotlib Week 2 beginner time-series chart."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)

DEFAULT_DATA = Path("fins2026/week2/results/data/week2_beginner_synthetic_prices.csv")
DEFAULT_OUTPUT = Path("fins2026/week2/results/figures/beginner_plotting")


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
    export_simple_time_series,
    load_synthetic_prices,
    resolve_repo_path,
    write_synthetic_prices_csv,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Build the plain Matplotlib Week 2 beginner time-series chart.",
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
        help="Repo-relative or absolute synthetic CSV path.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Repo-relative or absolute output figure directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Load the synthetic CSV and export the beginner line chart."""

    args = parse_args(argv)
    data_path = resolve_repo_path(args.data, REPO_ROOT)
    if not data_path.exists():
        print(f"synthetic csv missing, generating: {data_path}")
        write_synthetic_prices_csv(data_path, repo_root=REPO_ROOT)

    frame = load_synthetic_prices(data_path, repo_root=REPO_ROOT)
    paths = export_simple_time_series(frame, args.output, repo_root=REPO_ROOT)
    print(f"sample: {beginner_sample_label(frame)}")
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
