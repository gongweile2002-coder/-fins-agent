# ruff: noqa
"""Build long daily returns for Week 5 Stage 2 and verify parity with wide returns."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import REPO_ROOT

from fins2026.week5.code.stage2_crypto_returns import (
    assert_return_parity,
    build_adjusted_close_wide,
    compute_long_returns,
    compute_wide_returns,
    load_stage1_crypto_panel,
    stage2_data_paths,
    stage2_output_dir,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Compute long daily returns with groupby and verify that they match "
            "the wide-table return calculation."
        ),
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
    """Build the long return panel and confirm wide-long parity."""

    args = parse_args(argv)
    input_path = resolve_path(args.input_path)
    panel, spec = load_stage1_crypto_panel(panel_path=input_path)

    output_dir = resolve_path(args.output_dir) or stage2_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = stage2_data_paths()
    returns_long_path = output_dir / output_paths["returns_long"].name

    wide_prices = build_adjusted_close_wide(panel, price_column=spec.adjusted_price_column)
    wide_returns = compute_wide_returns(wide_prices)
    long_returns = compute_long_returns(panel, price_column=spec.adjusted_price_column)
    max_abs_diff = assert_return_parity(long_returns, wide_returns)
    long_returns.to_parquet(returns_long_path, index=False)

    print("Week 5 Stage 2: long returns with groupby")
    print()
    print("What this shows:")
    print("- long data is the natural shape for panel operations")
    print("- groupby + pct_change gives the same returns as the wide calculation")
    print("- parity checks are useful even when crypto trades every day")
    print()
    print(f"Provider: {spec.display_name}")
    print(f"Input panel: {input_path or spec.default_input_path}")
    print(f"Rows in long return panel: {len(long_returns):,}")
    print(f"Saved long returns: {returns_long_path}")
    print(f"Maximum wide-vs-long absolute difference: {max_abs_diff:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
