# ruff: noqa
"""Build Week 4 Stage 3 in-sample portfolios and efficient-frontier outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import REPO_ROOT

from fins2026.week4.code.stage3_portfolios import (
    PROVIDER_SPECS,
    balanced_sample_summary,
    build_balanced_stage3_sample,
    build_efficient_frontier,
    compute_portfolio_returns,
    estimate_portfolio_weights,
    load_stage2_feature_panel,
    stage3_data_dir,
    stage3_output_paths,
    stage3_table_dir,
    summarize_portfolio_metrics,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Build the Week 4 Stage 3 in-sample equal-weight, minimum-variance, "
            "and mean-variance portfolios."
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
        "--output-dir",
        help="Optional repo-relative or absolute Stage 3 data output directory.",
    )
    parser.add_argument(
        "--table-dir",
        help="Optional repo-relative or absolute Stage 3 table output directory.",
    )
    return parser.parse_args(argv)


def resolve_path(path_text: str | None) -> Path | None:
    """Resolve repo-relative paths while still allowing absolute paths."""

    if path_text is None:
        return None
    path = Path(path_text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    """Build the canonical Week 4 Stage 3 portfolio outputs."""

    args = parse_args(argv)
    input_path = resolve_path(args.input_path)
    data_dir = resolve_path(args.output_dir) or stage3_data_dir(args.provider)
    table_dir = resolve_path(args.table_dir) or stage3_table_dir(args.provider)
    data_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    feature_panel, spec = load_stage2_feature_panel(args.provider, panel_path=input_path)
    sample = build_balanced_stage3_sample(
        feature_panel,
        provider=spec.provider,
        display_name=spec.display_name,
    )
    weights, solve_methods = estimate_portfolio_weights(sample)
    portfolio_returns = compute_portfolio_returns(sample, weights)
    frontier = build_efficient_frontier(sample, weights)
    metrics = summarize_portfolio_metrics(portfolio_returns)

    output_paths = stage3_output_paths(args.provider)
    weights_path = data_dir / output_paths["weights"].name
    returns_path = data_dir / output_paths["returns"].name
    frontier_path = data_dir / output_paths["frontier"].name
    metrics_path = table_dir / output_paths["metrics"].name

    weights.to_parquet(weights_path, index=False)
    portfolio_returns.to_parquet(returns_path, index=False)
    frontier.to_parquet(frontier_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    sample_info = balanced_sample_summary(sample)
    print("Week 4 Stage 3: in-sample mean-variance portfolios")
    print()
    print("What this shows:")
    print("- Equal-weight is the naive 1/N benchmark.")
    print("- Minimum variance chooses the fully invested lowest-variance risky mix.")
    print("- Mean-variance is the tangency portfolio with the highest in-sample Sharpe ratio.")
    print(
        "- This script is in-sample only. Later out-of-sample work would "
        "re-estimate using past data only."
    )
    print()
    print(f"Provider: {sample_info['provider']}")
    print(
        f"Balanced sample: {sample_info['start_date'].date()} "
        f"to {sample_info['end_date'].date()}"
    )
    print(f"Assets: {sample_info['n_assets']}")
    print(f"Trading days: {sample_info['sample_days']}")
    print(f"Average daily risk-free rate: {sample_info['mean_daily_rfr']:.6f}")
    print("Linear-system methods:")
    for portfolio, method in solve_methods.items():
        print(f"- {portfolio}: {method}")
    print(f"Saved weights: {weights_path}")
    print(f"Saved portfolio returns: {returns_path}")
    print(f"Saved efficient frontier: {frontier_path}")
    print(f"Saved portfolio metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
