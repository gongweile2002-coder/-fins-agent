# ruff: noqa
"""Run the full 50-ticker Tiingo panel pull and save the long parquet output."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import REPO_ROOT

from fins2026.week4.code.equity_api_tiingo import (
    DEFAULT_TIINGO_FAMOUS_50_FILE,
    DEFAULT_TIINGO_FULL_OUTPUT_DIR,
    TiingoPullConfig,
    load_tickers_from_file,
    pull_tiingo_panel,
    resolve_api_key,
    write_tiingo_outputs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Download the Week 4 50-ticker Tiingo panel and save it in long format.",
    )
    parser.add_argument(
        "--api-key",
        help="Optional Tiingo API key. Defaults to TIINGO_API_KEY.",
    )
    parser.add_argument(
        "--tickers-file",
        default=str(DEFAULT_TIINGO_FAMOUS_50_FILE),
        help="Repo-relative or absolute 50-ticker universe file.",
    )
    parser.add_argument(
        "--start-date",
        default="2000-01-01",
        help="First requested date for the long panel. Default: 2000-01-01.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_TIINGO_FULL_OUTPUT_DIR),
        help="Repo-relative or absolute output directory.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=1.0,
        help="Pause between ticker requests. Default is more conservative than the small pull.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=30,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--with-metadata",
        action="store_true",
        help=(
            "Also fetch per-ticker metadata. The default skips metadata to "
            "conserve free-tier quota."
        ),
    )
    return parser.parse_args(argv)


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    """Pull the full Tiingo panel used later in the week."""

    args = parse_args(argv)
    config = TiingoPullConfig(
        api_key=resolve_api_key(args.api_key),
        tickers=load_tickers_from_file(resolve_path(args.tickers_file)),
        start_date=args.start_date,
        output_dir=resolve_path(args.output_dir),
        pause_seconds=args.pause_seconds,
        timeout_seconds=args.timeout_seconds,
        skip_metadata=not args.with_metadata,
    )
    panel, metadata = pull_tiingo_panel(config)
    outputs = write_tiingo_outputs(
        panel=panel,
        metadata=metadata,
        output_dir=config.output_dir,
        requested_start_date=config.start_date,
    )

    print("Week 4 Tiingo 50-ticker pull")
    print()
    print("What this shows:")
    print("- the same API pattern scales from a tiny warm-up to a large panel pull")
    print("- the long-format Parquet file is the main Week 4 equity dataset")
    print("- the coverage summary lets us audit which tickers truly reach the requested start date")
    print()
    print(f"Tickers requested: {len(config.tickers)}")
    print(f"Rows fetched: {len(panel):,}")
    print(f"Date range: {panel['date'].min().date()} to {panel['date'].max().date()}")
    print("Saved files:")
    print(f"- long CSV: {outputs['panel_csv']}")
    print(f"- long Parquet: {outputs['panel_parquet']}")
    print(f"- metadata CSV: {outputs['metadata_csv']}")
    print(f"- coverage CSV: {outputs['coverage_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
