# ruff: noqa
"""Configuration constants for the Week 5 client-facing crypto fund app."""

from __future__ import annotations

from fins2026.week5.code.stage4_app import (
    APP_ADVANCED_ESTIMATION_FREQUENCIES,
    APP_INITIAL_WINDOW_OPTIONS,
    APP_PUBLISHED_ESTIMATION_FREQUENCY,
    APP_PUBLISHED_INITIAL_WINDOW,
    APP_PUBLISHED_PORTFOLIO_KEYS,
    APP_PUBLISHED_PORTFOLIO_LABELS,
    APP_PUBLISHED_WINDOW_RULE,
    APP_SAMPLE_PERIODS,
    APP_WINDOW_RULE_OPTIONS,
    methodology_mapping_table,
    published_fund_description,
    technical_fund_name,
)

APP_TITLE = "Digital Asset Fund Explorer"
APP_SUBTITLE = (
    "Compare a published lineup of systematic crypto funds, inspect current "
    "holdings, and review historical performance built only from information "
    "available at each rebalance date."
)
VIEW_OPTIONS = [
    "Overview",
    "Fund Details",
    "Invest",
    "Portfolio Design",
    "Data & Downloads",
    "Methodology",
]
DEFAULT_VIEW = "Overview"
DEFAULT_SAMPLE_PERIOD = "3Y"
DEFAULT_PORTFOLIO_KEY = "mean_variance_tangency_long_only"
DEFAULT_INVESTMENT_AMOUNT = 25_000
DEFAULT_DESIGN_FREQUENCY = APP_PUBLISHED_ESTIMATION_FREQUENCY
DEFAULT_DESIGN_INITIAL_WINDOW = APP_PUBLISHED_INITIAL_WINDOW
DEFAULT_DESIGN_WINDOW_RULE = APP_PUBLISHED_WINDOW_RULE

SAMPLE_PERIOD_OPTIONS = APP_SAMPLE_PERIODS
DESIGN_FREQUENCY_OPTIONS = APP_ADVANCED_ESTIMATION_FREQUENCIES
DESIGN_INITIAL_WINDOW_OPTIONS = list(APP_INITIAL_WINDOW_OPTIONS)
DESIGN_WINDOW_RULE_OPTIONS = APP_WINDOW_RULE_OPTIONS

PORTFOLIO_KEYS = APP_PUBLISHED_PORTFOLIO_KEYS
PORTFOLIO_OPTIONS = {
    key: APP_PUBLISHED_PORTFOLIO_LABELS[key]
    for key in APP_PUBLISHED_PORTFOLIO_KEYS
}
PORTFOLIO_DESCRIPTIONS = {
    key: published_fund_description(key)
    for key in APP_PUBLISHED_PORTFOLIO_KEYS
}
PORTFOLIO_METHOD_LABELS = {
    key: technical_fund_name(key)
    for key in APP_PUBLISHED_PORTFOLIO_KEYS
}
PORTFOLIO_COLORS = {
    "equal_weight": "#746e65",
    "minimum_variance_long_only": "#4b8a7f",
    "mean_variance_tangency_long_only": "#96506a",
    "mean_cvar_tangency_long_only": "#4f6f88",
    "risk_parity_volatility_long_only": "#8a6b45",
}
BTC_ETH_COLORS = {
    "BTC": "#b06b2f",
    "ETH": "#5a7d99",
    "Other": "#8d887f",
}
METHOD_MAPPING = methodology_mapping_table()
METHOD_NOTES = {
    "equal_weight": (
        "Core Market spreads capital evenly across the full 20-coin universe."
    ),
    "minimum_variance_long_only": (
        "Low Volatility uses the historical covariance matrix to search for the "
        "lowest-variance fully invested long-only mix."
    ),
    "mean_variance_tangency_long_only": (
        "Return Seeking maximizes expected excess return per unit of total "
        "volatility within a long-only fully invested portfolio."
    ),
    "mean_cvar_tangency_long_only": (
        "Downside Aware maximizes expected excess return relative to historical "
        "tail-loss risk rather than total volatility."
    ),
    "risk_parity_volatility_long_only": (
        "Risk Balanced aims to spread volatility contribution more evenly across "
        "the underlying holdings."
    ),
}
