# ruff: noqa
"""Build the committed Week 5 app fallback fixture."""

from __future__ import annotations

import argparse

import pandas as pd

from fins2026.week5.code.stage3_oos_portfolios import load_stage2_feature_panel
from fins2026.week5.code.stage4_app import (
    APP_FEATURE_FIXTURE_PATH,
    build_live_app_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the committed Week 5 app fallback fixture from local Stage 2 "
            "outputs when available, or from live Yahoo/French pulls if needed."
        )
    )
    parser.add_argument(
        "--source",
        choices=["auto", "local", "live"],
        default="auto",
        help="Prefer local saved Stage 2 outputs, or force a live refresh.",
    )
    return parser.parse_args()


def _load_local_feature_panel():
    frame, _spec = load_stage2_feature_panel()
    return frame


def main() -> None:
    args = parse_args()

    if args.source == "local":
        feature_panel = _load_local_feature_panel()
        source_label = "local Stage 2 feature output"
    elif args.source == "live":
        bundle = build_live_app_bundle()
        feature_panel = bundle.feature_panel
        source_label = "live Yahoo/French pulls"
    else:
        try:
            feature_panel = _load_local_feature_panel()
            source_label = "local Stage 2 feature output"
        except Exception:
            bundle = build_live_app_bundle()
            feature_panel = bundle.feature_panel
            source_label = "live Yahoo/French pulls"

    APP_FEATURE_FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_panel.to_parquet(APP_FEATURE_FIXTURE_PATH, index=False)

    latest_date = pd.Timestamp(pd.to_datetime(feature_panel["date"]).max())
    latest_rfr = pd.Timestamp(feature_panel.loc[feature_panel["rfr"].notna(), "date"].max())
    print("Week 5 app fixture build")
    print(f"- source: {source_label}")
    print(f"- coins: {feature_panel['ticker'].nunique()}")
    print(f"- latest feature date: {latest_date:%Y-%m-%d}")
    print(f"- latest merged rfr date: {latest_rfr:%Y-%m-%d}")
    print(f"Saved feature fixture: {APP_FEATURE_FIXTURE_PATH}")


if __name__ == "__main__":
    main()
