# ruff: noqa
"""Build FT-style Stage 2 return-diagnostic figures for Week 4."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from _bootstrap import REPO_ROOT

from fins2026.week4.code.stage2_equity_returns import (
    resolve_stage2_provider,
    stage2_data_paths,
    stage2_output_dir,
)
from fins2026.week4.code.stage2_return_figures import make_stage2_figure_pack, stage2_figure_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Create FT-style Stage 2 return-check figures, including volatility "
            "ranking and top-versus-bottom cumulative-return paths."
        ),
    )
    parser.add_argument(
        "--provider",
        default="yahoo",
        choices=("tiingo", "yahoo"),
        help="Stage 2 provider to use. Default: yahoo.",
    )
    parser.add_argument(
        "--features-path",
        help="Optional repo-relative or absolute Stage 2 features Parquet path.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional repo-relative or absolute figure output directory.",
    )
    return parser.parse_args(argv)


def resolve_path(path_text: str | None) -> Path | None:
    """Resolve repo-relative paths while still allowing absolute paths."""

    if path_text is None:
        return None
    path = Path(path_text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    """Export the Stage 2 FT-style diagnostic figure pack."""

    args = parse_args(argv)
    default_data_dir = stage2_output_dir(args.provider)
    default_features_path = (
        default_data_dir
        / stage2_data_paths(args.provider)["returns_features_long"].name
    )
    features_path = resolve_path(args.features_path) or default_features_path
    if not features_path.exists():
        raise SystemExit(
            f"Missing Stage 2 feature panel: {features_path}. "
            "Run run_beginner_stage2_features_long.py first."
        )

    output_dir = resolve_path(args.output_dir) or stage2_figure_dir(args.provider)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_panel = pd.read_parquet(features_path)
    feature_panel["date"] = pd.to_datetime(feature_panel["date"])
    outputs = make_stage2_figure_pack(
        feature_panel,
        provider=args.provider,
        output_dir=output_dir,
    )
    provider_name = resolve_stage2_provider(args.provider).display_name

    print("Week 4 Stage 2: FT-style return checks")
    print()
    print("What this shows:")
    print("- return construction is only the first half of Stage 2")
    print("- we also need to inspect tails, volatility, and panel-wide stress episodes")
    print("- the top-versus-bottom volatility chart compounds returns on a log scale")
    print("- recession shading helps separate broad market stress from isolated ticker events")
    print()
    print(f"Provider: {provider_name}")
    print(f"Feature panel: {features_path}")
    print(f"Figure output directory: {output_dir}")
    print("Saved figures:")
    for figure_name, paths in outputs.items():
        print(f"- {figure_name}: {paths['png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
