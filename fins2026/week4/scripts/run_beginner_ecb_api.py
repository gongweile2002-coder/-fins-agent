# ruff: noqa
"""Run the Week 4 free-API walkthrough with the ECB."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import REPO_ROOT

from fins2026.week4.code.api_intro_ecb import (
    DEFAULT_ECB_OUTPUT_DIR,
    build_ecb_url,
    fetch_ecb_csv,
    tidy_ecb_exchange_rates,
    write_ecb_outputs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run the Week 4 ECB API introduction and save raw plus tidy outputs.",
    )
    parser.add_argument(
        "--base-currency",
        default="USD",
        help="Base currency code. Default: USD.",
    )
    parser.add_argument(
        "--quote-currency",
        default="EUR",
        help="Quote currency code. Default: EUR.",
    )
    parser.add_argument(
        "--frequency",
        default="D",
        choices=("D", "M", "Q", "A"),
        help="ECB frequency code. Default: D for daily.",
    )
    parser.add_argument(
        "--start-date",
        default="2026-05-01",
        help="Start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        default="2026-05-15",
        help="End date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_ECB_OUTPUT_DIR),
        help="Repo-relative or absolute output directory.",
    )
    return parser.parse_args(argv)


def resolve_path(path_text: str) -> Path:
    """Resolve repo-relative paths while still allowing absolute paths."""

    path = Path(path_text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    """Fetch a small ECB sample and explain the key API ideas."""

    args = parse_args(argv)
    output_dir = resolve_path(args.output_dir)

    url = build_ecb_url(
        base_currency=args.base_currency,
        quote_currency=args.quote_currency,
        frequency=args.frequency,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    csv_text = fetch_ecb_csv(url)
    tidy = tidy_ecb_exchange_rates(csv_text)
    outputs = write_ecb_outputs(output_dir, csv_text, tidy)

    print("Week 4 API intro: ECB free API")
    print()
    print("What this shows:")
    print("- an API endpoint is just a web address that returns data")
    print("- a GET request asks the server for that data")
    print("- query parameters let us change the time window and output format")
    print()
    print(f"GET {url}")
    print(f"Rows fetched: {len(tidy)}")
    print(f"Date range: {tidy['date'].min().date()} to {tidy['date'].max().date()}")
    print("Saved files:")
    print(f"- raw CSV: {outputs['raw_csv']}")
    print(f"- tidy CSV: {outputs['tidy_csv']}")
    print(f"- tidy Parquet: {outputs['tidy_parquet']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
