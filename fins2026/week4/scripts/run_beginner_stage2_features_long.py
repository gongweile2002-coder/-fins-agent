# ruff: noqa
"""Add rolling Stage 2 features to the Week 4 long return panel."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from _bootstrap import REPO_ROOT

from fins2026.week4.code.stage2_equity_returns import (
    build_feature_long_panel,
    compute_long_returns,
    load_stage1_equity_panel,
    resolve_stage2_provider,
    stage2_data_paths,
    stage2_output_dir,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Merge the daily risk-free rate and add rolling six-month return "
            "features to the long Stage 2 panel."
        ),
    )
    parser.add_argument(
        "--provider",
        default="yahoo",
        choices=("tiingo", "yahoo"),
        help="Stage 1 provider to use. Default: yahoo.",
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
    return parser.parse_args(argv)


def resolve_path(path_text: str | None) -> Path | None:
    """Resolve repo-relative paths while still allowing absolute paths."""

    if path_text is None:
        return None
    path = Path(path_text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    """Create the Stage 2 feature-rich long panel."""

    args = parse_args(argv)
    output_dir = resolve_path(args.output_dir) or stage2_output_dir(args.provider)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = stage2_data_paths(args.provider)
    default_returns_path = output_dir / output_paths["returns_long"].name
    returns_path = resolve_path(args.returns_path) or default_returns_path

    if returns_path.exists():
        long_returns = pd.read_parquet(returns_path)
        long_returns["date"] = pd.to_datetime(long_returns["date"])
        provider_name = resolve_stage2_provider(args.provider).display_name
    else:
        input_path = resolve_path(args.input_path)
        panel, spec = load_stage1_equity_panel(args.provider, panel_path=input_path)
        long_returns = compute_long_returns(panel, price_column=spec.adjusted_price_column)
        provider_name = spec.display_name

    rfr_path = resolve_path(args.rfr_path)
    features_path = output_dir / output_paths["returns_features_long"].name
    featured = build_feature_long_panel(long_returns, rfr_path=rfr_path)
    featured.to_parquet(features_path, index=False)

    first_feature_date = featured["date"].min()
    last_feature_date = featured["date"].max()
    print("Week 4 Stage 2: long return features")
    print()
    print("What this shows:")
    print("- we merge the daily risk-free rate before building excess-return features")
    print("- rolling six-month features are computed ticker by ticker in long format")
    print("- these columns feed the later portfolio-construction stage")
    print()
    print(f"Provider: {provider_name}")
    print(f"Rows in feature panel: {len(featured):,}")
    print(f"Date range: {first_feature_date.date()} to {last_feature_date.date()}")
    print(f"Saved feature panel: {features_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
