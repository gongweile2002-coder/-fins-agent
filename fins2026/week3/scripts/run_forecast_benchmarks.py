# ruff: noqa
"""Run the Week 3 forecast benchmark suite."""

from __future__ import annotations

import argparse
from pathlib import Path

from fins2026.week3.code import (
    benchmark_all_specs,
    build_forecast_input_bundle,
    write_benchmark_outputs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Week 3 Australia-first forecast benchmarks.",
    )
    parser.add_argument(
        "--use-fixture",
        action="store_true",
        help="Use the committed Australia fixture and frozen U.S. validation datasets.",
    )
    parser.add_argument(
        "--output-dir",
        default="fins2026/week3/results/forecasts",
        help="Repo-relative or absolute output directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bundle = build_forecast_input_bundle(use_fixture=args.use_fixture)
    leaderboard, outputs = benchmark_all_specs(
        bundle["australia_monthly"],
        bundle["australia_quarterly"],
        bundle["us_monthly"],
    )
    written = write_benchmark_outputs(leaderboard, outputs, output_dir=args.output_dir)
    mode = "fixture" if args.use_fixture else "live"
    print(f"Ran Week 3 forecast benchmarks in {mode} mode.")
    print(f"- successful rows: {(leaderboard['status'] == 'ok').sum():,}")
    print(f"- failed rows: {(leaderboard['status'] == 'failed').sum():,}")
    for label, path in written.items():
        print(f"- {label}: {Path(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
