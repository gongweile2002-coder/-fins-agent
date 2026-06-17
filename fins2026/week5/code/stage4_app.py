# ruff: noqa
"""Stage 4 helpers for the Week 5 client-facing crypto fund app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .crypto_api_yahoo import (
    DEFAULT_YAHOO_CRYPTO_PANEL_FILE,
    build_yahoo_session,
    load_yahoo_crypto_tickers_from_file,
    normalize_yahoo_crypto_chart_payload,
    request_yahoo_chart_json,
)
from .risk_free_rate_french import (
    extract_first_csv_text,
    fetch_french_rfr_zip,
    parse_french_daily_rfr,
)
from .stage2_crypto_returns import (
    _add_rolling_features_for_ticker,
    compute_long_returns,
)
from .stage3_oos_portfolios import (
    DEFAULT_INITIAL_WINDOW,
    DEFAULT_WINDOW_RULE,
    OOS_FIGURE_PORTFOLIO_COLUMN_ORDER,
    OOS_FIGURE_PORTFOLIO_LABELS,
    Stage3OOSConfig,
    Stage3OOSSample,
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
    compute_oos_portfolio_returns,
    generate_oos_weight_panels,
    summarize_oos_asset_statistics,
    summarize_oos_portfolio_metrics,
)

WEEK_ROOT = Path(__file__).resolve().parents[1]
APP_TICKER_FILE = DEFAULT_YAHOO_CRYPTO_PANEL_FILE
APP_FIXTURE_DIR = WEEK_ROOT / "app" / "fixtures"
APP_FEATURE_FIXTURE_PATH = APP_FIXTURE_DIR / "week5_app_features_long.parquet"
APP_PRICE_START_DATE = "2019-01-01"
APP_PUBLISHED_ESTIMATION_FREQUENCY = "monthly"
APP_PUBLISHED_INITIAL_WINDOW = DEFAULT_INITIAL_WINDOW
APP_PUBLISHED_WINDOW_RULE = DEFAULT_WINDOW_RULE
APP_SAMPLE_PERIODS = {
    "1Y": 1,
    "3Y": 3,
    "5Y": 5,
    "Since inception": None,
}
APP_ADVANCED_ESTIMATION_FREQUENCIES = {
    "monthly": "Monthly",
    "weekly": "Weekly",
}
APP_INITIAL_WINDOW_OPTIONS = (180, 365, 730)
APP_WINDOW_RULE_OPTIONS = {
    "expanding": "Expanding",
    "rolling": "Rolling",
}
APP_PUBLISHED_PORTFOLIO_KEYS = list(OOS_FIGURE_PORTFOLIO_COLUMN_ORDER)
APP_PUBLISHED_PORTFOLIO_LABELS = {
    "equal_weight": "Core Market",
    "minimum_variance_long_only": "Low Volatility",
    "mean_variance_tangency_long_only": "Return Seeking",
    "mean_cvar_tangency_long_only": "Downside Aware",
    "risk_parity_volatility_long_only": "Risk Balanced",
}
APP_PUBLISHED_PORTFOLIO_DESCRIPTIONS = {
    "equal_weight": (
        "A broad equal-weight fund that spreads capital evenly across the full "
        "crypto opportunity set."
    ),
    "minimum_variance_long_only": (
        "A lower-volatility fund that prioritizes steadier realised risk over "
        "headline upside."
    ),
    "mean_variance_tangency_long_only": (
        "A return-seeking fund that tilts toward higher expected excess return "
        "per unit of total volatility."
    ),
    "mean_cvar_tangency_long_only": (
        "A downside-aware fund that rewards expected return but penalizes the "
        "most severe historical loss days more heavily."
    ),
    "risk_parity_volatility_long_only": (
        "A balanced-risk fund that aims to spread volatility contribution more "
        "evenly across holdings."
    ),
}


@dataclass(frozen=True)
class Stage4AppBundle:
    """Runtime feature-panel bundle for the Week 5 client app."""

    feature_panel: pd.DataFrame
    latest_price_date: pd.Timestamp
    latest_rfr_date: pd.Timestamp

    @property
    def latest_feature_date(self) -> pd.Timestamp:
        return pd.Timestamp(pd.to_datetime(self.feature_panel["date"]).max())

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(sorted(self.feature_panel["ticker"].astype(str).unique().tolist()))


@dataclass(frozen=True)
class Stage4ScenarioBundle:
    """One cached Week 5 app scenario bundle."""

    config: Stage3OOSConfig
    sample: Stage3OOSSample
    daily_weights: pd.DataFrame
    rebalance_audit: pd.DataFrame
    solve_summary: pd.DataFrame
    portfolio_returns: pd.DataFrame
    oos_sample: Stage3OOSSample
    metrics: pd.DataFrame
    frontier: pd.DataFrame
    asset_summary: pd.DataFrame
    latest_target_weights: pd.DataFrame
    latest_live_weights: pd.DataFrame
    concentration_snapshot: pd.DataFrame
    risk_contributions: pd.DataFrame
    trailing_returns: pd.DataFrame
    current_drawdown: pd.DataFrame
    trailing_risk: pd.DataFrame
    turnover_snapshot: pd.DataFrame
    btc_eth_exposure: pd.DataFrame

    @property
    def latest_rebalance_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.latest_target_weights["decision_date"].iloc[0])

    @property
    def previous_rebalance_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.latest_target_weights["previous_decision_date"].iloc[0])

    @property
    def latest_return_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.portfolio_returns["return_date"].max())


@dataclass(frozen=True)
class Stage4DisplayWindow:
    """One display-window slice of a full app scenario."""

    sample_period: str
    portfolio_returns: pd.DataFrame
    oos_sample: Stage3OOSSample
    metrics: pd.DataFrame
    frontier: pd.DataFrame
    asset_summary: pd.DataFrame


def published_fund_name(portfolio_key: str) -> str:
    """Return the visible client-facing fund name for one portfolio key."""

    if portfolio_key not in APP_PUBLISHED_PORTFOLIO_LABELS:
        raise ValueError(f"Unknown Week 5 app portfolio key: {portfolio_key}.")
    return APP_PUBLISHED_PORTFOLIO_LABELS[portfolio_key]


def technical_fund_name(portfolio_key: str) -> str:
    """Return the technical model label for one published fund key."""

    if portfolio_key not in OOS_FIGURE_PORTFOLIO_LABELS:
        raise ValueError(f"Unknown Week 5 app portfolio key: {portfolio_key}.")
    return OOS_FIGURE_PORTFOLIO_LABELS[portfolio_key]


def published_fund_description(portfolio_key: str) -> str:
    """Return a one-sentence product description for one fund."""

    if portfolio_key not in APP_PUBLISHED_PORTFOLIO_DESCRIPTIONS:
        raise ValueError(f"Unknown Week 5 app portfolio key: {portfolio_key}.")
    return APP_PUBLISHED_PORTFOLIO_DESCRIPTIONS[portfolio_key]


def published_portfolio_label_map() -> dict[str, str]:
    """Return the visible fund-name map keyed by portfolio key."""

    return dict(APP_PUBLISHED_PORTFOLIO_LABELS)


def published_portfolio_display_order() -> list[str]:
    """Return the canonical app display order for the published fund shelf."""

    return list(APP_PUBLISHED_PORTFOLIO_KEYS)


def load_app_tickers(path: Path = APP_TICKER_FILE) -> tuple[str, ...]:
    """Load the committed 20-coin Week 5 app universe."""

    return load_yahoo_crypto_tickers_from_file(path)


def _merge_rfr_frame(frame: pd.DataFrame, rfr_frame: pd.DataFrame) -> pd.DataFrame:
    """Merge an in-memory daily risk-free frame into a 24/7 long crypto panel."""

    timeline = rfr_frame.copy()
    timeline["date"] = pd.to_datetime(timeline["date"])
    timeline["rfr"] = pd.to_numeric(timeline["rfr"], errors="coerce")
    timeline = timeline.sort_values("date").reset_index(drop=True)

    unique_dates = pd.DataFrame(
        {"date": pd.to_datetime(pd.Series(frame["date"]).dropna().unique())}
    )
    unique_dates = unique_dates.sort_values("date").reset_index(drop=True)
    unique_dates = unique_dates.merge(timeline, on="date", how="left")
    unique_dates["rfr"] = unique_dates["rfr"].ffill()

    merged = frame.merge(unique_dates, on="date", how="left")
    return merged.sort_values(["ticker", "date"]).reset_index(drop=True)


def build_feature_panel_from_frames(
    price_panel: pd.DataFrame,
    rfr_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Build the Stage 2-style Week 5 feature panel from in-memory inputs."""

    panel = price_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    long_returns = compute_long_returns(panel, price_column="adjClose")
    frame = long_returns.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    frame["ret"] = pd.to_numeric(frame["ret"], errors="coerce")
    frame["abs_ret"] = frame["ret"].abs()
    frame["is_large_move_10pct"] = frame["abs_ret"] >= 0.10
    frame["is_large_move_20pct"] = frame["abs_ret"] >= 0.20
    frame = _merge_rfr_frame(frame, rfr_frame)
    frame["excess_ret"] = frame["ret"] - frame["rfr"]
    groups = [
        _add_rolling_features_for_ticker(group)
        for _ticker, group in frame.groupby("ticker", sort=False)
    ]
    return pd.concat(groups, ignore_index=True)


def fetch_live_french_rfr() -> pd.DataFrame:
    """Fetch the Kenneth French daily risk-free series in memory."""

    zip_bytes = fetch_french_rfr_zip()
    csv_text = extract_first_csv_text(zip_bytes)
    return parse_french_daily_rfr(csv_text)


def fetch_live_yahoo_panel(
    tickers: tuple[str, ...],
    *,
    start_date: str = APP_PRICE_START_DATE,
    end_date: str | None = None,
    timeout_seconds: int = 30,
    max_attempts: int = 4,
    backoff_seconds: float = 1.0,
) -> pd.DataFrame:
    """Fetch the full Week 5 crypto app price panel directly into memory."""

    resolved_end_date = end_date or pd.Timestamp.today(tz="UTC").strftime("%Y-%m-%d")
    session = build_yahoo_session()
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        payload = request_yahoo_chart_json(
            session,
            ticker,
            start_date=start_date,
            end_date=resolved_end_date,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
        )
        frame, _metadata = normalize_yahoo_crypto_chart_payload(ticker, payload)
        frames.append(frame)
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )


def build_live_app_bundle() -> Stage4AppBundle:
    """Build the live Week 5 app bundle from Yahoo and Kenneth French."""

    tickers = load_app_tickers()
    price_panel = fetch_live_yahoo_panel(tickers)
    rfr_frame = fetch_live_french_rfr()
    feature_panel = build_feature_panel_from_frames(price_panel, rfr_frame)
    latest_price_date = pd.Timestamp(pd.to_datetime(price_panel["date"]).max())
    latest_rfr_date = pd.Timestamp(pd.to_datetime(rfr_frame["date"]).max())
    return Stage4AppBundle(
        feature_panel=feature_panel,
        latest_price_date=latest_price_date,
        latest_rfr_date=latest_rfr_date,
    )


def load_fixture_app_bundle() -> Stage4AppBundle:
    """Load the committed fallback Week 5 app bundle."""

    if not APP_FEATURE_FIXTURE_PATH.exists():
        raise SystemExit(
            "Missing Week 5 app fixture file. Run "
            "python fins2026/week5/scripts/build_week5_app_fixture.py first."
        )
    feature_panel = pd.read_parquet(APP_FEATURE_FIXTURE_PATH).copy()
    feature_panel["date"] = pd.to_datetime(feature_panel["date"])
    feature_panel["rfr"] = pd.to_numeric(feature_panel["rfr"], errors="coerce")
    latest_price_date = pd.Timestamp(pd.to_datetime(feature_panel["date"]).max())
    latest_rfr_date = pd.Timestamp(
        feature_panel.loc[feature_panel["rfr"].notna(), "date"].max()
    )
    return Stage4AppBundle(
        feature_panel=feature_panel,
        latest_price_date=latest_price_date,
        latest_rfr_date=latest_rfr_date,
    )


def build_published_config() -> Stage3OOSConfig:
    """Return the official published-fund Week 5 app configuration."""

    return Stage3OOSConfig(
        initial_window=APP_PUBLISHED_INITIAL_WINDOW,
        estimation_frequency=APP_PUBLISHED_ESTIMATION_FREQUENCY,
        window_rule=APP_PUBLISHED_WINDOW_RULE,
    )


def build_design_config(
    *,
    estimation_frequency: str,
    initial_window: int,
    window_rule: str,
) -> Stage3OOSConfig:
    """Return one validated advanced-design config for the app."""

    return Stage3OOSConfig(
        initial_window=initial_window,
        estimation_frequency=estimation_frequency,
        window_rule=window_rule,
    )


def build_app_scenario_bundle(
    feature_panel: pd.DataFrame,
    *,
    config: Stage3OOSConfig,
) -> Stage4ScenarioBundle:
    """Build the full Week 5 app scenario bundle from one feature panel."""

    sample = build_balanced_stage3_sample(
        feature_panel,
        provider="yahoo_crypto",
        display_name="Yahoo Finance Crypto",
    )
    daily_weights, rebalance_audit, solve_summary = generate_oos_weight_panels(
        sample,
        config=config,
    )
    portfolio_returns = compute_oos_portfolio_returns(
        sample,
        rebalance_audit,
        config=config,
    )
    oos_sample = build_oos_window_sample(sample, portfolio_returns)
    metrics = summarize_oos_portfolio_metrics(oos_sample, portfolio_returns)
    frontier = build_oos_ex_post_frontier(oos_sample, n_points=120)
    asset_summary = summarize_oos_asset_statistics(oos_sample)
    latest_target_weights = build_latest_target_weight_snapshot(rebalance_audit)
    latest_live_weights = build_latest_live_weight_snapshot(
        sample,
        rebalance_audit,
        config=config,
    )
    concentration_snapshot = build_factsheet_concentration_snapshot(latest_live_weights)
    risk_contributions = build_factsheet_risk_contribution_snapshot(
        sample,
        latest_target_weights,
        rebalance_audit,
    )
    trailing_returns = build_factsheet_trailing_return_snapshot(portfolio_returns)
    current_drawdown = build_factsheet_current_drawdown_snapshot(portfolio_returns)
    trailing_risk = build_factsheet_trailing_risk_snapshot(oos_sample, portfolio_returns)
    turnover_snapshot = build_factsheet_turnover_snapshot(latest_target_weights)
    btc_eth_exposure = build_factsheet_btc_eth_exposure_snapshot(latest_live_weights)
    return Stage4ScenarioBundle(
        config=config,
        sample=sample,
        daily_weights=daily_weights,
        rebalance_audit=rebalance_audit,
        solve_summary=solve_summary,
        portfolio_returns=portfolio_returns,
        oos_sample=oos_sample,
        metrics=metrics,
        frontier=frontier,
        asset_summary=asset_summary,
        latest_target_weights=latest_target_weights,
        latest_live_weights=latest_live_weights,
        concentration_snapshot=concentration_snapshot,
        risk_contributions=risk_contributions,
        trailing_returns=trailing_returns,
        current_drawdown=current_drawdown,
        trailing_risk=trailing_risk,
        turnover_snapshot=turnover_snapshot,
        btc_eth_exposure=btc_eth_exposure,
    )


def filter_portfolio_returns_for_sample_period(
    portfolio_returns: pd.DataFrame,
    sample_period: str,
) -> pd.DataFrame:
    """Filter a Stage 3 daily-return panel to one trailing display window."""

    years = APP_SAMPLE_PERIODS[sample_period]
    frame = portfolio_returns.copy()
    frame["return_date"] = pd.to_datetime(frame["return_date"])
    frame["formation_date"] = pd.to_datetime(frame["formation_date"])
    if years is None or frame.empty:
        return frame.sort_values("return_date").reset_index(drop=True)
    cutoff = frame["return_date"].max() - pd.DateOffset(years=years)
    return (
        frame.loc[frame["return_date"] >= cutoff]
        .sort_values("return_date")
        .reset_index(drop=True)
    )


def build_display_window_analysis(
    scenario: Stage4ScenarioBundle,
    *,
    sample_period: str,
) -> Stage4DisplayWindow:
    """Return one chart-ready display-window slice of a full app scenario."""

    portfolio_returns = filter_portfolio_returns_for_sample_period(
        scenario.portfolio_returns,
        sample_period,
    )
    if sample_period == "Since inception":
        return Stage4DisplayWindow(
            sample_period=sample_period,
            portfolio_returns=portfolio_returns,
            oos_sample=scenario.oos_sample,
            metrics=scenario.metrics,
            frontier=scenario.frontier,
            asset_summary=scenario.asset_summary,
        )
    oos_sample = build_oos_window_sample(scenario.sample, portfolio_returns)
    metrics = summarize_oos_portfolio_metrics(oos_sample, portfolio_returns)
    frontier = build_oos_ex_post_frontier(oos_sample, n_points=120)
    asset_summary = summarize_oos_asset_statistics(oos_sample)
    return Stage4DisplayWindow(
        sample_period=sample_period,
        portfolio_returns=portfolio_returns,
        oos_sample=oos_sample,
        metrics=metrics,
        frontier=frontier,
        asset_summary=asset_summary,
    )


def build_investment_allocation(
    latest_live_weights: pd.DataFrame,
    *,
    portfolio_key: str,
    amount_usd: float,
) -> pd.DataFrame:
    """Return an illustrative coin-allocation table for one chosen fund."""

    if amount_usd <= 0.0:
        raise ValueError("Investment amount must be strictly positive.")
    block = latest_live_weights.loc[
        latest_live_weights["portfolio_key"] == portfolio_key
    ].copy()
    if block.empty:
        raise ValueError(f"Missing latest live weights for {portfolio_key}.")
    block["weight_pct"] = block["weight"].astype(float) * 100.0
    block["allocation_usd"] = block["weight"].astype(float) * float(amount_usd)
    return (
        block.sort_values("weight", ascending=False)
        .reset_index(drop=True)
        .loc[
            :,
            [
                "return_date",
                "formation_date",
                "portfolio_key",
                "portfolio",
                "ticker",
                "weight",
                "weight_pct",
                "allocation_usd",
            ],
        ]
    )


def source_status_text(
    bundle: Stage4AppBundle,
    *,
    active_source: str,
    loaded_at_utc: pd.Timestamp | None = None,
    warning: str | None = None,
) -> str:
    """Return client-facing source and freshness text for the Week 5 app."""

    latest_feature_date = bundle.latest_feature_date
    if active_source == "Live":
        loaded_text = "n/a" if loaded_at_utc is None else f"{loaded_at_utc:%Y-%m-%d %H:%M} UTC"
        return (
            f"Live Yahoo crypto prices and Kenneth French daily cash rates loaded at "
            f"{loaded_text}; price history through {bundle.latest_price_date:%Y-%m-%d}; "
            f"portfolio analytics through {latest_feature_date:%Y-%m-%d}."
        )
    if warning:
        return (
            f"Fallback snapshot through prices {bundle.latest_price_date:%Y-%m-%d} and "
            f"portfolio analytics {latest_feature_date:%Y-%m-%d}."
        )
    return (
        f"Fallback snapshot through prices {bundle.latest_price_date:%Y-%m-%d} and "
        f"portfolio analytics {latest_feature_date:%Y-%m-%d}."
    )


def methodology_mapping_table() -> pd.DataFrame:
    """Return a compact mapping from published fund names to technical models."""

    rows = []
    for portfolio_key in APP_PUBLISHED_PORTFOLIO_KEYS:
        rows.append(
            {
                "published_fund": published_fund_name(portfolio_key),
                "technical_model": technical_fund_name(portfolio_key),
                "description": published_fund_description(portfolio_key),
            }
        )
    return pd.DataFrame(rows)


def latest_summary_cards(
    scenario: Stage4ScenarioBundle,
    *,
    portfolio_key: str,
) -> dict[str, object]:
    """Return current point-in-time values used in the app metric strip."""

    metrics_row = scenario.metrics.loc[scenario.metrics["portfolio_key"] == portfolio_key].iloc[0]
    trailing_row = scenario.trailing_risk.loc[
        scenario.trailing_risk["portfolio_key"] == portfolio_key
    ].iloc[0]
    drawdown_row = scenario.current_drawdown.loc[
        scenario.current_drawdown["portfolio_key"] == portfolio_key
    ].iloc[0]
    return {
        "portfolio": published_fund_name(portfolio_key),
        "cumulative_return_pct": float(metrics_row["cumulative_return_pct"]),
        "annualized_return_pct": float(metrics_row["annualized_return_pct"]),
        "annualized_volatility_pct": float(trailing_row["annualized_volatility_pct"]),
        "sharpe_ratio": float(trailing_row["sharpe_ratio"]),
        "sortino_ratio": float(trailing_row["sortino_ratio"]),
        "current_drawdown_pct": float(drawdown_row["current_drawdown_pct"]),
        "latest_rebalance_date": scenario.latest_rebalance_date,
        "latest_return_date": scenario.latest_return_date,
    }
