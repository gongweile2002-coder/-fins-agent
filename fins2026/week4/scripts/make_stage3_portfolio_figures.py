# ruff: noqa
"""Export Week 4 Stage 3 FT-style portfolio figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from _bootstrap import REPO_ROOT

from fins2026.week4.code.stage3_portfolio_figures import (
    make_stage3_figure_pack,
    stage3_figure_dir,
)
from fins2026.week4.code.stage3_portfolios import (
    PROVIDER_SPECS,
    build_balanced_stage3_sample,
    load_stage2_feature_panel,
    stage3_data_dir,
    stage3_output_paths,
    stage3_table_dir,
    summarize_asset_statistics,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Build the Week 4 Stage 3 FT-style figure pack from the saved "
            "portfolio outputs."
        ),
    )
    parser.add_argument(
        "--provider",
        default="yahoo",
        choices=tuple(sorted(PROVIDER_SPECS)),
        help="Stage 3 provider surface to use. Default: yahoo.",
    )
    parser.add_argument(
        "--input-path",
        help="Optional repo-relative or absolute Stage 2 feature-panel Parquet path.",
    )
    parser.add_argument(
        "--data-dir",
        help="Optional repo-relative or absolute Stage 3 data directory.",
    )
    parser.add_argument(
        "--table-dir",
        help="Optional repo-relative or absolute Stage 3 table directory.",
    )
    parser.add_argument(
        "--figure-dir",
        help="Optional repo-relative or absolute Stage 3 figure directory.",
    )
    return parser.parse_args(argv)


def resolve_path(path_text: str | None) -> Path | None:
    """Resolve repo-relative paths while still allowing absolute paths."""

    if path_text is None:
        return None
    path = Path(path_text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    """Build the canonical Week 4 Stage 3 figure pack."""

    args = parse_args(argv)
    input_path = resolve_path(args.input_path)
    data_dir = resolve_path(args.data_dir) or stage3_data_dir(args.provider)
    table_dir = resolve_path(args.table_dir) or stage3_table_dir(args.provider)
    figure_dir = resolve_path(args.figure_dir) or stage3_figure_dir(args.provider)
    figure_dir.mkdir(parents=True, exist_ok=True)

    feature_panel, spec = load_stage2_feature_panel(args.provider, panel_path=input_path)
    sample = build_balanced_stage3_sample(
        feature_panel,
        provider=spec.provider,
        display_name=spec.display_name,
    )
    output_paths = stage3_output_paths(args.provider)
    weights = pd.read_parquet(data_dir / output_paths["weights"].name)
    portfolio_returns = pd.read_parquet(data_dir / output_paths["returns"].name)
    frontier = pd.read_parquet(data_dir / output_paths["frontier"].name)
    metrics = pd.read_csv(table_dir / output_paths["metrics"].name)
    asset_summary = summarize_asset_statistics(sample)

    outputs = make_stage3_figure_pack(
        provider=args.provider,
        sample=sample,
        weights=weights,
        portfolio_returns=portfolio_returns,
        frontier=frontier,
        metrics=metrics,
        asset_summary=asset_summary,
        output_dir=figure_dir,
    )

    print("Week 4 Stage 3: FT-style portfolio figures")
    print()
    print("What this shows:")
    print("- the portfolio weight structure behind each in-sample solution")
    print("- growth of $1 and drawdowns from daily rebalanced portfolio returns")
    print("- a compact in-sample scorecard for return, risk, Sharpe ratio, and drawdown")
    print("- the efficient frontier and the tangency line")
    print()
    print(f"Provider: {spec.display_name}")
    print(f"Figure folder: {figure_dir}")
    print("Exported figures:")
    for key, paths in outputs.items():
        print(f"- {key}: {paths['png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
