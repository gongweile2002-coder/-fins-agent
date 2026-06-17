# ruff: noqa
"""Run the first authenticated Tiingo pull on a tiny ticker set."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import REPO_ROOT

from fins2026.week4.code.equity_api_tiingo import (
    DEFAULT_TIINGO_SMALL_FILE,
    DEFAULT_TIINGO_SMALL_OUTPUT_DIR,
    TiingoPullConfig,
    load_tickers_from_file,
    pull_tiingo_panel,
    resolve_api_key,
    write_tiingo_outputs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run the Week 4 Tiingo warm-up pull on a very small ticker set.",
    )
    parser.add_argument(
        "--api-key",
        help="Optional Tiingo API key. Defaults to TIINGO_API_KEY.",
    )
    parser.add_argument(
        "--tickers-file",
        default=str(DEFAULT_TIINGO_SMALL_FILE),
        help="Repo-relative or absolute ticker file for the warm-up pull.",
    )
    parser.add_argument(
        "--start-date",
        default="2024-01-01",
        help="First date to request. Default keeps the warm-up pull small and quick.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_TIINGO_SMALL_OUTPUT_DIR),
        help="Repo-relative or absolute output directory.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.5,
        help="Pause between ticker requests.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=30,
        help="Per-request timeout in seconds.",
    )
    return parser.parse_args(argv)


def resolve_path(path_text: str) -> Path:
    """Resolve repo-relative paths while still allowing absolute paths."""

    path = Path(path_text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    """Pull a small Tiingo panel and save the long-format teaching outputs."""

    args = parse_args(argv)
    config = TiingoPullConfig(
        api_key=resolve_api_key(args.api_key),
        tickers=load_tickers_from_file(resolve_path(args.tickers_file)),
        start_date=args.start_date,
        output_dir=resolve_path(args.output_dir),
        pause_seconds=args.pause_seconds,
        timeout_seconds=args.timeout_seconds,
        skip_metadata=False,
    )
    panel, metadata = pull_tiingo_panel(config)
    outputs = write_tiingo_outputs(
        panel=panel,
        metadata=metadata,
        output_dir=config.output_dir,
        requested_start_date=config.start_date,
    )

    print("Week 4 Tiingo warm-up pull")
    print()
    print("What this shows:")
    print("- authenticated APIs need a token before the request will work")
    print("- the raw response is JSON, but we save the final result as a long panel")
    print("- the long panel is our canonical equity dataset for Week 4")
    print()
    print(f"Tickers: {', '.join(config.tickers)}")
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
