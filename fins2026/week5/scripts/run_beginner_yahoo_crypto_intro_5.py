# ruff: noqa
"""Run the Week 5 Yahoo Finance crypto warm-up pull on five headline coins."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from _bootstrap import REPO_ROOT

from fins2026.week5.code.crypto_api_yahoo import (
    DEFAULT_YAHOO_CRYPTO_INTRO_OUTPUT_DIR,
    DEFAULT_YAHOO_CRYPTO_SMALL_FILE,
    YahooCryptoPullConfig,
    load_yahoo_crypto_tickers_from_file,
    pull_yahoo_crypto_panel,
    write_yahoo_crypto_outputs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run the Week 5 Yahoo crypto warm-up pull on a five-coin universe.",
    )
    parser.add_argument(
        "--tickers-file",
        default=str(DEFAULT_YAHOO_CRYPTO_SMALL_FILE),
        help="Repo-relative or absolute ticker file for the warm-up crypto pull.",
    )
    parser.add_argument(
        "--start-date",
        default="2024-01-01",
        help="First date to request. Default keeps the warm-up pull small and quick.",
    )
    parser.add_argument(
        "--end-date",
        default=dt.date.today().isoformat(),
        help="Last date to request. Default: today.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_YAHOO_CRYPTO_INTRO_OUTPUT_DIR),
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
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=6,
        help="Maximum attempts per ticker request across Yahoo hosts.",
    )
    parser.add_argument(
        "--backoff-seconds",
        type=float,
        default=2.0,
        help="Base backoff in seconds for 429 and 5xx retries.",
    )
    return parser.parse_args(argv)


def resolve_path(path_text: str) -> Path:
    """Resolve repo-relative paths while still allowing absolute paths."""

    path = Path(path_text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    """Pull a small Yahoo crypto panel and save the long-format teaching outputs."""

    args = parse_args(argv)
    config = YahooCryptoPullConfig(
        tickers=load_yahoo_crypto_tickers_from_file(resolve_path(args.tickers_file)),
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=resolve_path(args.output_dir),
        pause_seconds=args.pause_seconds,
        timeout_seconds=args.timeout_seconds,
        max_attempts=args.max_attempts,
        backoff_seconds=args.backoff_seconds,
        resume=True,
    )
    panel, metadata, failures = pull_yahoo_crypto_panel(config)
    outputs = write_yahoo_crypto_outputs(
        panel=panel,
        metadata=metadata,
        failures=failures,
        output_dir=config.output_dir,
        requested_start_date=config.start_date,
    )

    print("Week 5 Yahoo Finance crypto warm-up pull")
    print()
    print("What this shows:")
    print("- Yahoo can be queried directly through a public chart endpoint")
    print("- no API key is needed for this Week 5 crypto branch")
    print("- the saved crypto panel is long format with one row per ticker-day")
    print("- daily crypto bars are UTC-aligned and quoted in USD")
    print()
    print(f"Tickers: {', '.join(config.tickers)}")
    print(f"Rows fetched: {len(panel):,}")
    print(f"Date range: {panel['date'].min()} to {panel['date'].max()}")
    print("Saved files:")
    print(f"- long CSV: {outputs['panel_csv']}")
    print(f"- long Parquet: {outputs['panel_parquet']}")
    print(f"- metadata CSV: {outputs['metadata_csv']}")
    print(f"- coverage CSV: {outputs['coverage_csv']}")
    if "failures_csv" in outputs:
        print(f"- failures CSV: {outputs['failures_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
