# ruff: noqa
"""Build FT-style Stage 2 crypto figures for Week 5."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from _bootstrap import REPO_ROOT

from fins2026.week5.code.stage2_crypto_figures import make_stage2_figure_pack, stage2_figure_dir
from fins2026.week5.code.stage2_crypto_returns import (
    load_stage1_crypto_panel,
    stage2_data_paths,
    stage2_output_dir,
    stage2_table_dir,
    stage2_table_paths,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Create FT-style Stage 2 crypto figures, including price small multiples, "
            "growth of $1, Sharpe/Sortino rankings, distribution diagnostics, "
            "a return correlation matrix, and optional appendix figures."
        ),
    )
    parser.add_argument(
        "--panel-path",
        help="Optional repo-relative or absolute Stage 1 panel Parquet path.",
    )
    parser.add_argument(
        "--features-path",
        help="Optional repo-relative or absolute Stage 2 features Parquet path.",
    )
    parser.add_argument(
        "--summary-path",
        help="Optional repo-relative or absolute Stage 2 summary Parquet path.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional repo-relative or absolute figure output directory.",
    )
    parser.add_argument(
        "--include-appendix",
        action="store_true",
        help="Also export the optional appendix figure pack.",
    )
    return parser.parse_args(argv)


def resolve_path(path_text: str | None) -> Path | None:
    """Resolve repo-relative paths while still allowing absolute paths."""

    if path_text is None:
        return None
    path = Path(path_text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    """Export the Stage 2 FT-style crypto figure pack."""

    args = parse_args(argv)
    stage1_path = resolve_path(args.panel_path)
    stage1_panel, spec = load_stage1_crypto_panel(panel_path=stage1_path)

    default_data_dir = stage2_output_dir()
    default_features_path = default_data_dir / stage2_data_paths()["returns_features_long"].name
    default_summary_path = stage2_table_dir() / stage2_table_paths()["summary_parquet"].name
    features_path = resolve_path(args.features_path) or default_features_path
    summary_path = resolve_path(args.summary_path) or default_summary_path

    if not features_path.exists():
        raise SystemExit(
            f"Missing Stage 2 feature panel: {features_path}. "
            "Run run_beginner_stage2_features_long.py first."
        )

    feature_panel = pd.read_parquet(features_path)
    feature_panel["date"] = pd.to_datetime(feature_panel["date"])
    summary = pd.read_parquet(summary_path) if summary_path.exists() else None

    output_dir = resolve_path(args.output_dir) or stage2_figure_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = make_stage2_figure_pack(
        stage1_panel,
        feature_panel,
        output_dir=output_dir,
        summary=summary,
        include_appendix=args.include_appendix,
    )

    print("Week 5 Stage 2: FT-style crypto figures")
    print()
    print("What this shows:")
    print(
        "- crypto prices and cumulative returns have very different scales, "
        "so layout choices matter"
    )
    print("- Sharpe and Sortino rankings require the Week 5 excess-return merge step")
    print(
        "- histogram and tail-share views show how often crypto returns exceed "
        "what a normal distribution would imply"
    )
    print("- the correlation matrix shows which headline coins tend to move together")
    if args.include_appendix:
        print(
            "- the appendix adds drawdown, dispersion, rolling-risk, "
            "and liquidity diagnostics"
        )
    print()
    print(f"Provider: {spec.display_name}")
    print(f"Stage 1 panel: {stage1_path or spec.default_input_path}")
    print(f"Feature panel: {features_path}")
    print(f"Figure output directory: {output_dir}")
    print(f"Include appendix: {args.include_appendix}")
    print("Saved figures:")
    for figure_name, paths in outputs.items():
        print(f"- {figure_name}: {paths['png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
