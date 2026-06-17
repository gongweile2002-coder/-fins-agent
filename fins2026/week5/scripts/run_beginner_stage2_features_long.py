# ruff: noqa
"""Add rolling Stage 2 features and summary metrics to the Week 5 long return panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from _bootstrap import REPO_ROOT

from fins2026.week5.code.stage2_crypto_returns import (
    build_feature_long_panel,
    compute_long_returns,
    load_stage1_crypto_panel,
    stage2_data_paths,
    stage2_output_dir,
    stage2_table_dir,
    stage2_table_paths,
    summarize_stage2_metrics,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Merge the daily risk-free rate, add rolling six-month crypto return "
            "features, and save the Week 5 summary metrics."
        ),
    )
    parser.add_argument(
        "--returns-path",
        help="Optional repo-relative or absolute Stage 2 long-return Parquet path.",
    )
    parser.add_argument(
        "--input-path",
        help="Optional repo-relative or absolute Stage 1 long-panel Parquet path.",
    )
    parser.add_argument(
        "--rfr-path",
        help="Optional repo-relative or absolute French risk-free Parquet path.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional repo-relative or absolute Stage 2 output directory.",
    )
    parser.add_argument(
        "--table-dir",
        help="Optional repo-relative or absolute Stage 2 table directory.",
    )
    return parser.parse_args(argv)


def resolve_path(path_text: str | None) -> Path | None:
    """Resolve repo-relative paths while still allowing absolute paths."""

    if path_text is None:
        return None
    path = Path(path_text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    """Create the Stage 2 feature-rich long panel and summary metrics."""

    args = parse_args(argv)
    output_dir = resolve_path(args.output_dir) or stage2_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = resolve_path(args.table_dir) or stage2_table_dir()
    table_dir.mkdir(parents=True, exist_ok=True)

    output_paths = stage2_data_paths()
    table_paths = stage2_table_paths()
    default_returns_path = output_dir / output_paths["returns_long"].name
    returns_path = resolve_path(args.returns_path) or default_returns_path

    if returns_path.exists():
        long_returns = pd.read_parquet(returns_path)
        long_returns["date"] = pd.to_datetime(long_returns["date"])
        provider_name = "Yahoo Finance Crypto"
    else:
        input_path = resolve_path(args.input_path)
        panel, spec = load_stage1_crypto_panel(panel_path=input_path)
        long_returns = compute_long_returns(panel, price_column=spec.adjusted_price_column)
        provider_name = spec.display_name

    rfr_path = resolve_path(args.rfr_path)
    features_path = output_dir / output_paths["returns_features_long"].name
    summary_csv_path = table_dir / table_paths["summary_csv"].name
    summary_parquet_path = table_dir / table_paths["summary_parquet"].name

    featured = build_feature_long_panel(long_returns, rfr_path=rfr_path)
    summary = summarize_stage2_metrics(featured)
    featured.to_parquet(features_path, index=False)
    summary.to_csv(summary_csv_path, index=False)
    summary.to_parquet(summary_parquet_path, index=False)

    print("Week 5 Stage 2: long return features and summary metrics")
    print()
    print("What this shows:")
    print("- we merge the business-day risk-free rate onto the full 24/7 crypto timeline")
    print("- RF is forward-filled across weekends, holidays, and tail dates")
    print("- rolling features and full-sample Sharpe/Sortino metrics both come from excess returns")
    print()
    print(f"Provider: {provider_name}")
    print(f"Rows in feature panel: {len(featured):,}")
    print(f"Date range: {featured['date'].min().date()} to {featured['date'].max().date()}")
    print(f"Saved feature panel: {features_path}")
    print(f"Saved summary CSV: {summary_csv_path}")
    print(f"Saved summary Parquet: {summary_parquet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
