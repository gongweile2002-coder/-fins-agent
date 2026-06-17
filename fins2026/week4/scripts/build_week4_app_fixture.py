# ruff: noqa
"""Build the committed Week 4 app fallback fixture."""

from __future__ import annotations

import argparse

import pandas as pd

from fins2026.week4.code.equity_api_yahoo import DEFAULT_YAHOO_FULL_OUTPUT_DIR
from fins2026.week4.code.risk_free_rate_french import DEFAULT_FRENCH_RFR_OUTPUT_PATH
from fins2026.week4.code.stage4_app import (
    APP_PRICE_FIXTURE_PATH,
    APP_RFR_FIXTURE_PATH,
    build_live_app_bundle,
    fetch_live_french_rfr,
    load_app_tickers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the committed Week 4 app fallback fixture from local Stage 1 outputs "
            "when available, or from live Yahoo/French pulls if needed."
        )
    )
    parser.add_argument(
        "--source",
        choices=["auto", "local", "live"],
        default="auto",
        help="Prefer local saved Stage 1 outputs, or force a live refresh.",
    )
    return parser.parse_args()


def _subset_local_yahoo_panel() -> pd.DataFrame:
    panel_path = DEFAULT_YAHOO_FULL_OUTPUT_DIR / "yahoo_chart_panel_long.parquet"
    if not panel_path.exists():
        raise SystemExit(f"Missing local Yahoo 50 panel: {panel_path}")
    tickers = set(load_app_tickers())
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    subset = panel.loc[panel["ticker"].isin(tickers)].copy()
    if set(subset["ticker"]) != tickers:
        missing = sorted(tickers.difference(set(subset["ticker"])))
        raise SystemExit(f"Local Yahoo 50 panel is missing app tickers: {', '.join(missing)}")
    return subset.sort_values(["ticker", "date"]).reset_index(drop=True)


def _load_local_rfr() -> pd.DataFrame:
    if not DEFAULT_FRENCH_RFR_OUTPUT_PATH.exists():
        raise SystemExit(f"Missing local French rfr parquet: {DEFAULT_FRENCH_RFR_OUTPUT_PATH}")
    frame = pd.read_parquet(DEFAULT_FRENCH_RFR_OUTPUT_PATH)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date").reset_index(drop=True)


def main() -> None:
    args = parse_args()

    if args.source == "local":
        price_panel = _subset_local_yahoo_panel()
        rfr_frame = _load_local_rfr()
        source_label = "local Stage 1 outputs"
    elif args.source == "live":
        bundle = build_live_app_bundle()
        price_panel = bundle.price_panel
        rfr_frame = fetch_live_french_rfr()
        source_label = "live Yahoo/French pulls"
    else:
        try:
            price_panel = _subset_local_yahoo_panel()
            rfr_frame = _load_local_rfr()
            source_label = "local Stage 1 outputs"
        except SystemExit:
            bundle = build_live_app_bundle()
            price_panel = bundle.price_panel
            rfr_frame = fetch_live_french_rfr()
            source_label = "live Yahoo/French pulls"

    APP_PRICE_FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    price_panel.to_parquet(APP_PRICE_FIXTURE_PATH, index=False)
    rfr_frame.to_parquet(APP_RFR_FIXTURE_PATH, index=False)

    latest_price_date = pd.to_datetime(price_panel["date"]).max()
    latest_rfr_date = pd.to_datetime(rfr_frame["date"]).max()
    print("Week 4 app fixture build")
    print(f"- source: {source_label}")
    print(f"- stocks: {price_panel['ticker'].nunique()}")
    print(f"- latest Yahoo date: {latest_price_date:%Y-%m-%d}")
    print(f"- latest French rfr date: {latest_rfr_date:%Y-%m-%d}")
    print(f"Saved price fixture: {APP_PRICE_FIXTURE_PATH}")
    print(f"Saved rfr fixture: {APP_RFR_FIXTURE_PATH}")


if __name__ == "__main__":
    main()
