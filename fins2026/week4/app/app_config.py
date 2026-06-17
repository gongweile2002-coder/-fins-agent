# ruff: noqa
"""Configuration constants for the Week 4 portfolio app."""

from __future__ import annotations

from fins2026.week4.code.stage4_app import (
    APP_CONSTRAINT_LABELS,
    APP_PORTFOLIO_KEYS,
    APP_PORTFOLIO_LABELS,
    APP_SAMPLE_PERIODS,
)

APP_TITLE = "U.S. Equity Portfolio Builder"
APP_SUBTITLE = (
    "Explore a compact 10-stock U.S. equity universe, compare standard in-sample "
    "portfolios, and test a custom allocation against the same opportunity set."
)
VIEW_OPTIONS = [
    "Overview",
    "Portfolio Builder",
    "Optimized Portfolios",
    "Historical Performance",
    "Efficient Frontier",
    "Data",
    "Methodology",
]
DEFAULT_VIEW = "Overview"
DEFAULT_SAMPLE_PERIOD = "10Y"
SAMPLE_PERIOD_OPTIONS = APP_SAMPLE_PERIODS
CONSTRAINT_OPTIONS = APP_CONSTRAINT_LABELS
DEFAULT_CONSTRAINT_MODE = "long_only"
DEFAULT_PORTFOLIO_KEY = "mean_variance_tangency"
PORTFOLIO_KEYS = APP_PORTFOLIO_KEYS
PORTFOLIO_LABELS = APP_PORTFOLIO_LABELS
PORTFOLIO_OPTIONS = {
    key: APP_PORTFOLIO_LABELS[key]
    for key in APP_PORTFOLIO_KEYS
}
PORTFOLIO_DISPLAY_ORDER = [APP_PORTFOLIO_LABELS[key] for key in APP_PORTFOLIO_KEYS]
PORTFOLIO_COLORS = {
    APP_PORTFOLIO_LABELS["custom"]: "#4F5D75",
    APP_PORTFOLIO_LABELS["equal_weight"]: "#7A746B",
    APP_PORTFOLIO_LABELS["minimum_variance"]: "#4E8B84",
    APP_PORTFOLIO_LABELS["mean_variance_tangency"]: "#8E3B46",
}
APP_TICKER_NAMES = {
    "AAPL": "Apple",
    "NVDA": "Nvidia",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "JPM": "JPMorgan Chase",
    "JNJ": "Johnson & Johnson",
    "WMT": "Walmart",
    "XOM": "Exxon Mobil",
    "KO": "Coca-Cola",
    "PG": "Procter & Gamble",
}
APP_TICKER_OPTIONS = list(APP_TICKER_NAMES)
APP_TICKER_LABELS = {
    ticker: f"{name} ({ticker})"
    for ticker, name in APP_TICKER_NAMES.items()
}
METHOD_NOTES = {
    "long_only": (
        "Long-only optimization constrains every optimized weight between 0% and 100% "
        "while keeping the portfolio fully invested."
    ),
    "unconstrained": (
        "Unconstrained optimization allows negative weights. Those allocations can create "
        "leveraged long-short portfolios, so they are useful for comparison but need "
        "more interpretation in practice."
    ),
}
