# ruff: noqa
"""Download the Week 4 daily risk-free rate from Kenneth French."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import REPO_ROOT

from fins2026.week4.code.risk_free_rate_french import (
    DEFAULT_FRENCH_RFR_OUTPUT_PATH,
    FRENCH_RFR_URL,
    extract_first_csv_text,
    fetch_french_rfr_zip,
    parse_french_daily_rfr,
    write_french_rfr_parquet,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Download Kenneth French daily factors, keep only RF, and write one final "
            "Parquet file."
        ),
    )
    parser.add_argument(
        "--url",
        default=FRENCH_RFR_URL,
        help="Direct download URL for the Kenneth French daily factors zip file.",
    )
    parser.add_argument(
        "--output-path",
        default=str(DEFAULT_FRENCH_RFR_OUTPUT_PATH),
        help="Repo-relative or absolute Parquet output path.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="Download timeout in seconds.",
    )
    return parser.parse_args(argv)


def resolve_path(path_text: str) -> Path:
    """Resolve repo-relative paths while still allowing absolute paths."""

    path = Path(path_text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    """Download and save the final daily risk-free series."""

    args = parse_args(argv)
    output_path = resolve_path(args.output_path)

    zip_bytes = fetch_french_rfr_zip(args.url, timeout_seconds=args.timeout_seconds)
    csv_text = extract_first_csv_text(zip_bytes)
    frame = parse_french_daily_rfr(csv_text)
    saved_path = write_french_rfr_parquet(frame, output_path)

    print("Week 4 daily risk-free rate download")
    print()
    print("What this shows:")
    print("- not every external source is a JSON API; some are direct file downloads")
    print("- we can still treat that download as part of the same data factory floor")
    print("- Week 4 keeps only the final clean dataset, not intermediate raw files")
    print("- Kenneth French stores RF in percent, so we divide by 100 before saving `rfr`")
    print()
    print(f"GET {args.url}")
    print(f"Rows saved: {len(frame)}")
    print(f"Date range: {frame['date'].min().date()} to {frame['date'].max().date()}")
    print(f"Saved file: {saved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
