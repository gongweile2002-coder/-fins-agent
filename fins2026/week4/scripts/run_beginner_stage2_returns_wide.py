# ruff: noqa
"""Build wide adjusted-price and return tables for Week 4 Stage 2."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import REPO_ROOT

from fins2026.week4.code.stage2_equity_returns import (
    build_adjusted_close_wide,
    compute_wide_returns,
    load_stage1_equity_panel,
    stage2_data_paths,
    stage2_output_dir,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Turn the Stage 1 long panel into a wide adjusted-price matrix and a "
            "wide daily-return matrix."
        ),
    )
    parser.add_argument(
        "--provider",
        default="yahoo",
        choices=("tiingo", "yahoo"),
        help="Stage 1 provider to use. Default: yahoo.",
    )
    parser.add_argument(
        "--input-path",
        help="Optional repo-relative or absolute Stage 1 long-panel Parquet path.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional repo-relative or absolute Stage 2 output directory.",
    )
    return parser.parse_args(argv)


def resolve_path(path_text: str | None) -> Path | None:
    """Resolve repo-relative paths while still allowing absolute paths."""

    if path_text is None:
        return None
    path = Path(path_text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    """Build the wide adjusted-price and return tables."""

    args = parse_args(argv)
    input_path = resolve_path(args.input_path)
    panel, spec = load_stage1_equity_panel(args.provider, panel_path=input_path)

    output_dir = resolve_path(args.output_dir) or stage2_output_dir(args.provider)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = stage2_data_paths(args.provider)
    adjclose_path = output_dir / output_paths["adjclose_wide"].name
    returns_path = output_dir / output_paths["returns_wide"].name

    wide_prices = build_adjusted_close_wide(panel, price_column=spec.adjusted_price_column)
    wide_returns = compute_wide_returns(wide_prices)
    wide_prices.to_parquet(adjclose_path, index=False)
    wide_returns.to_parquet(returns_path, index=False)

    print("Week 4 Stage 2: wide adjusted prices and returns")
    print()
    print("What this shows:")
    print("- adjusted prices are the right base for return construction")
    print("- wide data makes row-wise return calculations very transparent")
    print("- this wide table is a derived Stage 2 object, not the raw source of truth")
    print()
    print(f"Provider: {spec.display_name}")
    print(f"Input panel: {input_path or spec.default_input_path}")
    print(f"Wide price rows: {len(wide_prices):,}")
    print(f"Wide price columns: {len(wide_prices.columns):,}")
    print(f"Saved wide prices: {adjclose_path}")
    print(f"Saved wide returns: {returns_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
