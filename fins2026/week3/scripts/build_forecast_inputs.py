# ruff: noqa
"""Build the Week 3 Australia/U.S. forecast input panels."""

from __future__ import annotations

import argparse
from pathlib import Path

from fins2026.week3.code import build_forecast_input_bundle, write_forecast_input_bundle


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Week 3 Australia/U.S. forecast input bundle.",
    )
    parser.add_argument(
        "--use-fixture",
        action="store_true",
        help="Use the committed Australia fixture and frozen U.S. validation datasets.",
    )
    parser.add_argument(
        "--output-dir",
        default="fins2026/week3/results/data",
        help="Repo-relative or absolute output directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bundle = build_forecast_input_bundle(use_fixture=args.use_fixture)
    written = write_forecast_input_bundle(bundle, output_dir=args.output_dir)
    mode = "fixture" if args.use_fixture else "live"
    print(f"Built Week 3 forecast inputs in {mode} mode.")
    for label, path in written.items():
        print(f"- {label}: {Path(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
