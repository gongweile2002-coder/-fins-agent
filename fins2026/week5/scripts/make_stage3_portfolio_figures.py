# ruff: noqa
"""Export Week 5 Stage 3 FT-style out-of-sample portfolio figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from _bootstrap import REPO_ROOT

from fins2026.week5.code.stage3_oos_portfolios import (
    build_balanced_stage3_sample,
    build_factsheet_btc_eth_exposure_snapshot,
    build_factsheet_concentration_snapshot,
    build_factsheet_current_drawdown_snapshot,
    build_factsheet_risk_contribution_snapshot,
    build_factsheet_trailing_return_snapshot,
    build_factsheet_trailing_risk_snapshot,
    build_factsheet_turnover_snapshot,
    build_latest_live_weight_snapshot,
    build_latest_target_weight_snapshot,
    build_oos_ex_post_frontier,
    build_oos_window_sample,
    load_stage2_feature_panel,
    stage3_data_dir,
    stage3_output_paths,
    stage3_table_dir,
    summarize_oos_asset_statistics,
    summarize_oos_portfolio_metrics,
)
from fins2026.week5.code.stage3_portfolio_figures import (
    make_stage3_factsheet_figure_pack,
    make_stage3_figure_pack,
    stage3_figure_dir,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Build the Week 5 Stage 3 FT-style out-of-sample crypto figure pack "
            "from the saved portfolio outputs."
        ),
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
    """Build the canonical Week 5 Stage 3 figure pack."""

    args = parse_args(argv)
    input_path = resolve_path(args.input_path)
    data_dir = resolve_path(args.data_dir) or stage3_data_dir()
    table_dir = resolve_path(args.table_dir) or stage3_table_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = resolve_path(args.figure_dir) or stage3_figure_dir()
    figure_dir.mkdir(parents=True, exist_ok=True)

    feature_panel, spec = load_stage2_feature_panel(panel_path=input_path)
    sample = build_balanced_stage3_sample(
        feature_panel,
        provider=spec.provider,
        display_name=spec.display_name,
    )
    output_paths = stage3_output_paths()
    rebalance_audit = pd.read_parquet(data_dir / output_paths["rebalance_audit"].name)
    daily_returns = pd.read_parquet(data_dir / output_paths["daily_returns"].name)
    oos_sample = build_oos_window_sample(sample, daily_returns)
    asset_summary = summarize_oos_asset_statistics(oos_sample)
    frontier = build_oos_ex_post_frontier(oos_sample)
    latest_target_weights = build_latest_target_weight_snapshot(rebalance_audit)
    latest_live_weights = build_latest_live_weight_snapshot(sample, rebalance_audit)
    concentration_snapshot = build_factsheet_concentration_snapshot(latest_live_weights)
    risk_contributions = build_factsheet_risk_contribution_snapshot(
        sample,
        latest_target_weights,
        rebalance_audit,
    )
    trailing_returns = build_factsheet_trailing_return_snapshot(daily_returns)
    current_drawdown = build_factsheet_current_drawdown_snapshot(daily_returns)
    trailing_risk = build_factsheet_trailing_risk_snapshot(oos_sample, daily_returns)
    turnover_snapshot = build_factsheet_turnover_snapshot(latest_target_weights)
    btc_eth_exposure = build_factsheet_btc_eth_exposure_snapshot(latest_live_weights)
    print("Week 5 Stage 3: FT-style out-of-sample portfolio figures")
    print()
    print("What this shows:")
    print("- growth of $1 and drawdowns for the long-only core portfolios")
    print("- a six-metric out-of-sample scorecard with Sharpe and Sortino ratios")
    print("- the top five formation-date target holdings through time for each optimized model")
    print("- realized OOS portfolio points over an ex post asset frontier reference surface")
    print(
        "- app-style point-in-time factsheet views for holdings, concentration, "
        "risk, turnover, and BTC/ETH exposure"
    )
    print()
    print(f"Provider: {spec.display_name}")
    print(
        f"OOS sample: {oos_sample.start_date:%Y-%m-%d} "
        f"to {oos_sample.end_date:%Y-%m-%d}"
    )

    metrics = summarize_oos_portfolio_metrics(oos_sample, daily_returns)
    frontier_path = data_dir / output_paths["ex_post_frontier"].name
    latest_target_path = data_dir / output_paths["latest_target_weights"].name
    latest_live_path = data_dir / output_paths["latest_live_weights"].name
    risk_contribution_path = data_dir / output_paths["latest_risk_contributions"].name
    concentration_csv_path = table_dir / output_paths["concentration_snapshot_csv"].name
    concentration_parquet_path = table_dir / output_paths["concentration_snapshot_parquet"].name
    trailing_return_csv_path = table_dir / output_paths["trailing_return_snapshot_csv"].name
    trailing_return_parquet_path = table_dir / output_paths["trailing_return_snapshot_parquet"].name
    current_drawdown_csv_path = table_dir / output_paths["current_drawdown_snapshot_csv"].name
    current_drawdown_parquet_path = (
        table_dir / output_paths["current_drawdown_snapshot_parquet"].name
    )
    trailing_risk_csv_path = table_dir / output_paths["trailing_risk_snapshot_csv"].name
    trailing_risk_parquet_path = table_dir / output_paths["trailing_risk_snapshot_parquet"].name
    turnover_csv_path = table_dir / output_paths["turnover_snapshot_csv"].name
    turnover_parquet_path = table_dir / output_paths["turnover_snapshot_parquet"].name
    btc_eth_csv_path = table_dir / output_paths["btc_eth_exposure_csv"].name
    btc_eth_parquet_path = table_dir / output_paths["btc_eth_exposure_parquet"].name
    metrics_csv_path = table_dir / output_paths["portfolio_metrics_csv"].name
    metrics_parquet_path = table_dir / output_paths["portfolio_metrics_parquet"].name
    frontier.to_parquet(frontier_path, index=False)
    latest_target_weights.to_parquet(latest_target_path, index=False)
    latest_live_weights.to_parquet(latest_live_path, index=False)
    risk_contributions.to_parquet(risk_contribution_path, index=False)
    concentration_snapshot.to_csv(concentration_csv_path, index=False)
    concentration_snapshot.to_parquet(concentration_parquet_path, index=False)
    trailing_returns.to_csv(trailing_return_csv_path, index=False)
    trailing_returns.to_parquet(trailing_return_parquet_path, index=False)
    current_drawdown.to_csv(current_drawdown_csv_path, index=False)
    current_drawdown.to_parquet(current_drawdown_parquet_path, index=False)
    trailing_risk.to_csv(trailing_risk_csv_path, index=False)
    trailing_risk.to_parquet(trailing_risk_parquet_path, index=False)
    turnover_snapshot.to_csv(turnover_csv_path, index=False)
    turnover_snapshot.to_parquet(turnover_parquet_path, index=False)
    btc_eth_exposure.to_csv(btc_eth_csv_path, index=False)
    btc_eth_exposure.to_parquet(btc_eth_parquet_path, index=False)
    metrics.to_csv(metrics_csv_path, index=False)
    metrics.to_parquet(metrics_parquet_path, index=False)

    outputs = make_stage3_figure_pack(
        sample=oos_sample,
        rebalance_audit=rebalance_audit,
        portfolio_returns=daily_returns,
        frontier=frontier,
        metrics=metrics,
        asset_summary=asset_summary,
        output_dir=figure_dir,
    )
    factsheet_outputs = make_stage3_factsheet_figure_pack(
        sample=oos_sample,
        latest_target_weights=latest_target_weights,
        latest_live_weights=latest_live_weights,
        concentration_snapshot=concentration_snapshot,
        risk_contributions=risk_contributions,
        trailing_returns=trailing_returns,
        current_drawdown=current_drawdown,
        trailing_risk=trailing_risk,
        turnover_snapshot=turnover_snapshot,
        btc_eth_exposure=btc_eth_exposure,
        output_dir=figure_dir,
    )

    print(f"Figure folder: {figure_dir}")
    print(f"Saved ex post frontier: {frontier_path}")
    print(f"Saved latest target weights: {latest_target_path}")
    print(f"Saved latest live weights: {latest_live_path}")
    print(f"Saved latest risk contributions: {risk_contribution_path}")
    print(f"Saved metrics CSV: {metrics_csv_path}")
    print(f"Saved metrics Parquet: {metrics_parquet_path}")
    print(f"Saved concentration snapshot CSV: {concentration_csv_path}")
    print(f"Saved trailing return snapshot CSV: {trailing_return_csv_path}")
    print(f"Saved trailing risk snapshot CSV: {trailing_risk_csv_path}")
    print(f"Saved turnover snapshot CSV: {turnover_csv_path}")
    print(f"Saved BTC/ETH exposure CSV: {btc_eth_csv_path}")
    print("Exported research figures:")
    for key, paths in outputs.items():
        print(f"- {key}: {paths['png']}")
    print("Exported factsheet figures:")
    for key, paths in factsheet_outputs.items():
        print(f"- {key}: {paths['png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
