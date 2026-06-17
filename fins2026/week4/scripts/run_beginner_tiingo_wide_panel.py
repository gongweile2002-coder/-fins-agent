# ruff: noqa
"""Convert a saved Tiingo long panel into a wide close-price table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from _bootstrap import REPO_ROOT

from fins2026.week4.code.equity_api_tiingo import (
    DEFAULT_TIINGO_SMALL_OUTPUT_DIR,
    build_wide_price_table,
    write_tiingo_wide_outputs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Read a Tiingo long panel and export a wide close-price table.",
    )
    parser.add_argument(
        "--input-parquet",
        default=str(DEFAULT_TIINGO_SMALL_OUTPUT_DIR / "tiingo_eod_panel_long.parquet"),
        help="Repo-relative or absolute long-panel Parquet path.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_TIINGO_SMALL_OUTPUT_DIR),
        help="Repo-relative or absolute output directory for the wide table.",
    )
    parser.add_argument(
        "--value-column",
        default="close",
        help="Panel column to pivot into wide format. Default: close.",
    )
    return parser.parse_args(argv)


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    """Show how long panel data becomes the wide table many students expect."""

    args = parse_args(argv)
    input_parquet = resolve_path(args.input_parquet)
    output_dir = resolve_path(args.output_dir)

    if not input_parquet.exists():
        raise SystemExit(
            "Missing long panel. Run run_beginner_tiingo_small_panel.py "
            "first so there is data to pivot."
        )

    panel = pd.read_parquet(input_parquet)
    wide = build_wide_price_table(panel, value_column=args.value_column)
    outputs = write_tiingo_wide_outputs(
        wide,
        output_dir=output_dir,
        stem=f"tiingo_{args.value_column}_wide",
    )

    print("Week 4 Tiingo long-to-wide reshape")
    print()
    print("What this shows:")
    print("- long data is easiest for filtering, merging, and panel operations")
    print("- wide data is often easier for spreadsheets and correlation work")
    print(
        "- we derive wide format from the saved long panel instead of "
        "treating it as the raw source"
    )
    print()
    print(f"Input panel: {input_parquet}")
    print(f"Rows in wide table: {len(wide):,}")
    print(f"Columns in wide table: {len(wide.columns):,}")
    print("Saved files:")
    print(f"- wide CSV: {outputs['wide_csv']}")
    print(f"- wide Parquet: {outputs['wide_parquet']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
