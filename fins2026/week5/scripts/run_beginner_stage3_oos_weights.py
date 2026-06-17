# ruff: noqa
"""Build Week 5 Stage 3 out-of-sample crypto portfolio weight and return panels."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import REPO_ROOT

from fins2026.week5.code.stage3_oos_portfolios import (
    DEFAULT_COVARIANCE_RIDGE,
    DEFAULT_CVAR_ALPHA,
    DEFAULT_ESTIMATION_FREQUENCY,
    DEFAULT_INITIAL_WINDOW,
    DEFAULT_MODELS,
    DEFAULT_SOLVER_MAX_ITER,
    DEFAULT_SOLVER_TOLERANCE,
    DEFAULT_WINDOW_RULE,
    ESTIMATION_FREQUENCIES,
    MODEL_LABELS,
    Stage3OOSConfig,
    balanced_sample_summary,
    build_balanced_stage3_sample,
    build_rebalance_schedule,
    compute_oos_portfolio_returns,
    generate_oos_weight_panels,
    load_stage2_feature_panel,
    stage3_data_dir,
    stage3_output_paths,
    stage3_table_dir,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Build the Week 5 Stage 3 out-of-sample crypto portfolio weight panels "
            "from the saved Stage 2 feature panel."
        ),
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
    parser.add_argument(
        "--initial-window",
        type=int,
        default=DEFAULT_INITIAL_WINDOW,
        help=(
            "Initial number of daily observations before the first rebalance. "
            f"Default: {DEFAULT_INITIAL_WINDOW}."
        ),
    )
    parser.add_argument(
        "--estimation-frequency",
        default=DEFAULT_ESTIMATION_FREQUENCY,
        choices=ESTIMATION_FREQUENCIES,
        help=f"Re-estimation frequency. Default: {DEFAULT_ESTIMATION_FREQUENCY}.",
    )
    parser.add_argument(
        "--window-rule",
        default=DEFAULT_WINDOW_RULE,
        choices=("expanding", "rolling"),
        help=(
            "Window update rule after the initial sample. "
            f"Default: {DEFAULT_WINDOW_RULE}."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        choices=DEFAULT_MODELS,
        help="Portfolio models to estimate. Default: all Stage 3 models.",
    )
    parser.add_argument(
        "--cvar-alpha",
        type=float,
        default=DEFAULT_CVAR_ALPHA,
        help=(
            "Historical CVaR confidence level for the mean-CVaR model. "
            f"Default: {DEFAULT_CVAR_ALPHA}."
        ),
    )
    parser.add_argument(
        "--covariance-ridge",
        type=float,
        default=DEFAULT_COVARIANCE_RIDGE,
        help=(
            "Optional diagonal ridge added to covariance matrices. "
            f"Default: {DEFAULT_COVARIANCE_RIDGE}."
        ),
    )
    parser.add_argument(
        "--solver-tolerance",
        type=float,
        default=DEFAULT_SOLVER_TOLERANCE,
        help=(
            "Numerical solver tolerance for SLSQP and HiGHS. "
            f"Default: {DEFAULT_SOLVER_TOLERANCE}."
        ),
    )
    parser.add_argument(
        "--solver-max-iter",
        type=int,
        default=DEFAULT_SOLVER_MAX_ITER,
        help=(
            "Maximum solver iterations for SLSQP and HiGHS. "
            f"Default: {DEFAULT_SOLVER_MAX_ITER}."
        ),
    )
    return parser.parse_args(argv)


def resolve_path(path_text: str | None) -> Path | None:
    """Resolve repo-relative paths while still allowing absolute paths."""

    if path_text is None:
        return None
    path = Path(path_text)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _format_model_list(models: tuple[str, ...]) -> str:
    return ", ".join(MODEL_LABELS[model] for model in models)


def main(argv: list[str] | None = None) -> int:
    """Build the canonical Week 5 Stage 3 weight and return outputs."""

    args = parse_args(argv)
    input_path = resolve_path(args.input_path)
    data_dir = resolve_path(args.output_dir) or stage3_data_dir()
    table_dir = resolve_path(args.table_dir) or stage3_table_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    feature_panel, spec = load_stage2_feature_panel(panel_path=input_path)
    sample = build_balanced_stage3_sample(
        feature_panel,
        provider=spec.provider,
        display_name=spec.display_name,
    )
    config = Stage3OOSConfig(
        initial_window=args.initial_window,
        estimation_frequency=args.estimation_frequency,
        window_rule=args.window_rule,
        models=tuple(args.models),
        cvar_alpha=args.cvar_alpha,
        covariance_ridge=args.covariance_ridge,
        solver_tolerance=args.solver_tolerance,
        solver_max_iter=args.solver_max_iter,
    )
    schedule = build_rebalance_schedule(
        sample.dates,
        initial_window=config.initial_window,
        estimation_frequency=config.estimation_frequency,
        window_rule=config.window_rule,
    )
    daily_weights, rebalance_audit, solve_summary = generate_oos_weight_panels(
        sample,
        config=config,
    )
    daily_returns = compute_oos_portfolio_returns(
        sample,
        rebalance_audit,
        config=config,
    )

    output_paths = stage3_output_paths()
    daily_path = data_dir / output_paths["daily_weights"].name
    returns_path = data_dir / output_paths["daily_returns"].name
    audit_path = data_dir / output_paths["rebalance_audit"].name
    summary_csv_path = table_dir / output_paths["solve_summary_csv"].name
    summary_parquet_path = table_dir / output_paths["solve_summary_parquet"].name

    daily_weights.to_parquet(daily_path, index=False)
    daily_returns.to_parquet(returns_path, index=False)
    rebalance_audit.to_parquet(audit_path, index=False)
    solve_summary.to_csv(summary_csv_path, index=False)
    solve_summary.to_parquet(summary_parquet_path, index=False)

    sample_info = balanced_sample_summary(sample, schedule=schedule)
    print("Week 5 Stage 3: out-of-sample crypto portfolio weights and returns")
    print()
    print("What this shows:")
    print("- every rebalance uses only data available through the decision date")
    print("- formation_date records when the weights became known")
    print("- return_date records the daily date the portfolio return was earned")
    print("- weights drift within each holding block until the next re-estimation date")
    print(
        "- monthly and weekly schedules use the 24/7 crypto calendar, not "
        "business-day month-end conventions"
    )
    print()
    print(f"Provider: {sample_info['provider']}")
    print(
        f"Balanced sample: {sample_info['start_date'].date()} "
        f"to {sample_info['end_date'].date()}"
    )
    print(f"Assets: {sample_info['n_assets']}")
    print(f"Daily observations: {sample_info['sample_days']}")
    print(f"Initial window: {config.initial_window}")
    print(f"Estimation frequency: {config.estimation_frequency}")
    print(f"Window rule: {config.window_rule}")
    print("Constraint mode: long_only")
    print(f"Models: {_format_model_list(config.models)}")
    print(f"CVaR alpha: {config.cvar_alpha:.2f}")
    print(f"Rebalance dates: {sample_info['rebalance_count']}")
    print(f"Daily out-of-sample return rows: {len(daily_returns)}")
    print(
        f"First decision date: {sample_info['first_decision_date'].date()} | "
        f"Last decision date: {sample_info['last_decision_date'].date()}"
    )
    print("Average solve times (ms):")
    for row in solve_summary.itertuples(index=False):
        print(
            f"- {row.model} [{row.constraint_mode}]: "
            f"{row.mean_elapsed_ms:.2f} mean / {row.max_elapsed_ms:.2f} max"
        )
    print(f"Saved daily holding-date weights: {daily_path}")
    print(f"Saved daily out-of-sample returns: {returns_path}")
    print(f"Saved rebalance audit: {audit_path}")
    print(f"Saved solve summary CSV: {summary_csv_path}")
    print(f"Saved solve summary Parquet: {summary_parquet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
