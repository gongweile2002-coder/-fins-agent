# ruff: noqa
"""Stage 4 helpers for the Week 4 client-facing portfolio app."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .equity_api_yahoo import (
    build_yahoo_session,
    load_yahoo_tickers_from_file,
    normalize_yahoo_chart_payload,
    request_yahoo_chart_json,
)
from .risk_free_rate_french import (
    extract_first_csv_text,
    fetch_french_rfr_zip,
    parse_french_daily_rfr,
)
from .stage2_equity_returns import (
    SQRT_252,
    TRADING_DAYS_PER_YEAR,
    _add_rolling_features_for_ticker,
    compute_long_returns,
)
from .stage3_portfolios import (
    PORTFOLIO_LABELS,
    Stage3Sample,
    build_balanced_stage3_sample,
    build_efficient_frontier,
    drawdown_series,
    equal_weight_vector,
    estimate_portfolio_weights,
    summarize_asset_statistics,
)

WEEK_ROOT = Path(__file__).resolve().parents[1]
APP_TICKER_FILE = WEEK_ROOT / "data" / "yahoo_app_10.txt"
APP_FIXTURE_DIR = WEEK_ROOT / "app" / "fixtures"
APP_PRICE_FIXTURE_PATH = APP_FIXTURE_DIR / "yahoo_app_10_long.parquet"
APP_RFR_FIXTURE_PATH = APP_FIXTURE_DIR / "french_daily_rfr.parquet"
APP_START_DATE = "2000-01-01"
APP_PORTFOLIO_KEYS = [
    "custom",
    "equal_weight",
    "minimum_variance",
    "mean_variance_tangency",
]
APP_PORTFOLIO_LABELS = {
    "custom": "Custom",
    "equal_weight": PORTFOLIO_LABELS["equal_weight"],
    "minimum_variance": PORTFOLIO_LABELS["minimum_variance"],
    "mean_variance_tangency": PORTFOLIO_LABELS["mean_variance_tangency"],
}
APP_CONSTRAINT_LABELS = {
    "long_only": "Long-only",
    "unconstrained": "Unconstrained",
}
APP_SAMPLE_PERIODS = {
    "5Y": 5,
    "10Y": 10,
    "15Y": 15,
    "20Y": 20,
    "Max": None,
}
DEFAULT_FRONTIER_POINTS = 120


@dataclass(frozen=True)
class Stage4AppBundle:
    """Runtime app data bundle for the 10-stock Week 4 app surface."""

    price_panel: pd.DataFrame
    feature_panel: pd.DataFrame
    latest_observation_date: pd.Timestamp


def load_app_tickers(path: Path = APP_TICKER_FILE) -> tuple[str, ...]:
    """Load the committed 10-stock app universe."""

    return load_yahoo_tickers_from_file(path)


def _merge_rfr_frame(frame: pd.DataFrame, rfr_frame: pd.DataFrame) -> pd.DataFrame:
    """Merge a supplied daily risk-free frame into a long ticker-date panel."""

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
    """Build the Stage 2-style feature panel from in-memory price and rfr frames."""

    price_panel = price_panel.copy()
    price_panel["date"] = pd.to_datetime(price_panel["date"])
    long_returns = compute_long_returns(price_panel, price_column="adjClose")
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
    """Fetch the daily French risk-free rate in memory."""

    zip_bytes = fetch_french_rfr_zip()
    csv_text = extract_first_csv_text(zip_bytes)
    return parse_french_daily_rfr(csv_text)


def fetch_live_yahoo_panel(
    tickers: tuple[str, ...],
    *,
    start_date: str = APP_START_DATE,
    end_date: str | None = None,
    timeout_seconds: int = 30,
    max_attempts: int = 4,
    backoff_seconds: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch the 10-stock Yahoo app panel directly into memory."""

    resolved_end_date = end_date or pd.Timestamp.today(tz="UTC").strftime("%Y-%m-%d")
    session = build_yahoo_session()
    frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, object]] = []
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
        frame, metadata = normalize_yahoo_chart_payload(ticker, payload)
        frames.append(frame)
        metadata_rows.append(metadata)
    panel = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    metadata = pd.DataFrame(metadata_rows).sort_values("ticker").reset_index(drop=True)
    return panel, metadata


def build_live_app_bundle() -> Stage4AppBundle:
    """Build the live 10-stock app bundle from Yahoo and Kenneth French."""

    tickers = load_app_tickers()
    price_panel, _metadata = fetch_live_yahoo_panel(tickers)
    rfr_frame = fetch_live_french_rfr()
    feature_panel = build_feature_panel_from_frames(price_panel, rfr_frame)
    latest = pd.to_datetime(price_panel["date"]).max()
    return Stage4AppBundle(
        price_panel=price_panel,
        feature_panel=feature_panel,
        latest_observation_date=pd.Timestamp(latest),
    )


def load_fixture_app_bundle() -> Stage4AppBundle:
    """Load the committed fallback app bundle."""

    if not APP_PRICE_FIXTURE_PATH.exists() or not APP_RFR_FIXTURE_PATH.exists():
        raise SystemExit(
            "Missing Week 4 app fixture files. Run "
            "python fins2026/week4/scripts/build_week4_app_fixture.py first."
        )

    price_panel = pd.read_parquet(APP_PRICE_FIXTURE_PATH).copy()
    price_panel["date"] = pd.to_datetime(price_panel["date"])
    rfr_frame = pd.read_parquet(APP_RFR_FIXTURE_PATH).copy()
    rfr_frame["date"] = pd.to_datetime(rfr_frame["date"])
    feature_panel = build_feature_panel_from_frames(price_panel, rfr_frame)
    latest = pd.to_datetime(price_panel["date"]).max()
    return Stage4AppBundle(
        price_panel=price_panel,
        feature_panel=feature_panel,
        latest_observation_date=pd.Timestamp(latest),
    )


def filter_feature_panel_for_sample_period(
    feature_panel: pd.DataFrame,
    sample_period: str,
) -> pd.DataFrame:
    """Restrict the feature panel to the selected trailing sample window."""

    years = APP_SAMPLE_PERIODS[sample_period]
    frame = feature_panel.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    if years is None or frame.empty:
        return frame
    cutoff = pd.to_datetime(frame["date"]).max() - pd.DateOffset(years=years)
    return frame.loc[pd.to_datetime(frame["date"]) >= cutoff].reset_index(drop=True)


def filter_price_panel_for_sample_period(
    price_panel: pd.DataFrame,
    sample_period: str,
) -> pd.DataFrame:
    """Restrict the long price panel to the selected sample window."""

    years = APP_SAMPLE_PERIODS[sample_period]
    frame = price_panel.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    if years is None or frame.empty:
        return frame
    cutoff = pd.to_datetime(frame["date"]).max() - pd.DateOffset(years=years)
    return frame.loc[pd.to_datetime(frame["date"]) >= cutoff].reset_index(drop=True)


def build_stage4_sample(
    feature_panel: pd.DataFrame,
    *,
    selected_tickers: list[str],
) -> Stage3Sample:
    """Build the balanced sample used by the Stage 4 portfolio app."""

    subset = feature_panel.loc[feature_panel["ticker"].isin(selected_tickers)].copy()
    if subset.empty:
        raise ValueError("No rows remain for the selected stock set.")
    return build_balanced_stage3_sample(
        subset,
        provider="yahoo_app",
        display_name="Yahoo Finance app universe",
    )


def normalize_custom_weights(
    selected_tickers: list[str],
    raw_weight_inputs: Mapping[str, float | int | str],
) -> tuple[pd.DataFrame, str | None]:
    """Normalize raw custom-weight inputs into a long weight table."""

    if not selected_tickers:
        raise ValueError("At least one stock must be selected.")

    values = pd.Series(
        {
            ticker: max(float(raw_weight_inputs.get(ticker, 0.0) or 0.0), 0.0)
            for ticker in selected_tickers
        },
        dtype=float,
    )
    warning: str | None = None
    if np.isclose(float(values.sum()), 0.0):
        values = pd.Series(
            equal_weight_vector(len(selected_tickers)),
            index=selected_tickers,
            dtype=float,
        )
        warning = (
            "The custom allocation summed to 0%, so the app is showing an equal-weight "
            "allocation across the selected stocks."
        )
    else:
        values = values / float(values.sum())

    frame = pd.DataFrame(
        {
            "ticker": selected_tickers,
            "portfolio": APP_PORTFOLIO_LABELS["custom"],
            "weight": values.to_numpy(dtype=float),
        }
    )
    return frame, warning


def _variance_objective(weights: np.ndarray, covariance: np.ndarray) -> float:
    return float(weights @ covariance @ weights)


def _negative_sharpe_objective(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    covariance: np.ndarray,
    risk_free_rate: float,
) -> float:
    excess_mean = float(weights @ mean_returns - risk_free_rate)
    volatility = float(np.sqrt(max(weights @ covariance @ weights, 0.0)))
    if np.isclose(volatility, 0.0):
        return 1e9
    return -(excess_mean / volatility)


def _slsqp_result_weights(
    result,
    *,
    n_assets: int,
    label: str,
) -> np.ndarray:
    if not result.success:
        raise ValueError(f"{label} optimization failed: {result.message}")
    weights = np.asarray(result.x, dtype=float)
    if weights.shape[0] != n_assets:
        raise ValueError(f"{label} optimization returned the wrong number of weights.")
    return weights / float(weights.sum())


def long_only_minimum_variance_weights(covariance: np.ndarray) -> np.ndarray:
    """Solve the long-only minimum-variance portfolio."""

    n_assets = covariance.shape[0]
    x0 = equal_weight_vector(n_assets)
    result = minimize(
        _variance_objective,
        x0=x0,
        args=(covariance,),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_assets,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"maxiter": 800, "ftol": 1e-12},
    )
    return _slsqp_result_weights(result, n_assets=n_assets, label="Long-only minimum variance")


def long_only_tangency_weights(
    mean_returns: np.ndarray,
    covariance: np.ndarray,
    risk_free_rate: float,
) -> np.ndarray:
    """Solve the long-only max-Sharpe portfolio."""

    n_assets = covariance.shape[0]
    x0 = equal_weight_vector(n_assets)
    result = minimize(
        _negative_sharpe_objective,
        x0=x0,
        args=(mean_returns, covariance, risk_free_rate),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_assets,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"maxiter": 800, "ftol": 1e-12},
    )
    return _slsqp_result_weights(result, n_assets=n_assets, label="Long-only mean-variance")


def estimate_long_only_portfolio_weights(
    sample: Stage3Sample,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Estimate the long-only equal, minimum-variance, and mean-variance portfolios."""

    returns = sample.returns_wide.to_numpy(dtype=float)
    mean_returns = returns.mean(axis=0)
    covariance = np.cov(returns, rowvar=False, ddof=1)
    avg_daily_rfr = float(sample.rfr.mean())

    eq_weights = equal_weight_vector(sample.n_assets)
    mv_weights = long_only_minimum_variance_weights(covariance)
    tan_weights = long_only_tangency_weights(mean_returns, covariance, avg_daily_rfr)

    weight_frame = pd.DataFrame(
        {
            "ticker": sample.tickers,
            APP_PORTFOLIO_LABELS["equal_weight"]: eq_weights,
            APP_PORTFOLIO_LABELS["minimum_variance"]: mv_weights,
            APP_PORTFOLIO_LABELS["mean_variance_tangency"]: tan_weights,
        }
    )
    long_weights = (
        weight_frame.melt(id_vars="ticker", var_name="portfolio", value_name="weight")
        .sort_values(["portfolio", "ticker"])
        .reset_index(drop=True)
    )
    methods = {
        APP_PORTFOLIO_LABELS["minimum_variance"]: "SLSQP",
        APP_PORTFOLIO_LABELS["mean_variance_tangency"]: "SLSQP",
    }
    return long_weights, methods


def estimate_app_portfolio_weights(
    sample: Stage3Sample,
    *,
    custom_weights: pd.DataFrame,
    constraint_mode: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Estimate the app portfolio set for the chosen constraint mode."""

    if constraint_mode == "unconstrained":
        optimized, methods = estimate_portfolio_weights(sample)
    elif constraint_mode == "long_only":
        optimized, methods = estimate_long_only_portfolio_weights(sample)
    else:
        raise ValueError(f"Unknown constraint mode: {constraint_mode}")

    combined = pd.concat([custom_weights, optimized], ignore_index=True)
    combined["portfolio"] = pd.Categorical(
        combined["portfolio"],
        categories=[APP_PORTFOLIO_LABELS[key] for key in APP_PORTFOLIO_KEYS],
        ordered=True,
    )
    combined = combined.sort_values(["portfolio", "ticker"]).reset_index(drop=True)
    return combined, methods


def compute_named_portfolio_returns(
    sample: Stage3Sample,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """Compute daily rebalanced returns for the named app portfolios."""

    matrix = weights.pivot(index="ticker", columns="portfolio", values="weight")
    matrix = matrix.reindex(index=sample.tickers)
    matrix = matrix.reindex(columns=[APP_PORTFOLIO_LABELS[key] for key in APP_PORTFOLIO_KEYS])
    returns_array = sample.returns_wide.to_numpy(dtype=float)
    portfolio_returns = returns_array @ matrix.to_numpy(dtype=float)
    output = pd.DataFrame(
        portfolio_returns,
        index=sample.returns_wide.index,
        columns=matrix.columns,
    )
    output = output.reset_index().rename(columns={"index": "date"})
    output["rfr"] = sample.rfr.to_numpy(dtype=float)
    return output


def summarize_named_portfolio_metrics(portfolio_returns: pd.DataFrame) -> pd.DataFrame:
    """Summarize annualized in-sample metrics for the app portfolio set."""

    rows: list[dict[str, float | str]] = []
    rfr = portfolio_returns["rfr"].astype(float)
    for label in [APP_PORTFOLIO_LABELS[key] for key in APP_PORTFOLIO_KEYS]:
        returns = portfolio_returns[label].astype(float)
        excess = returns - rfr
        annualized_return = float(returns.mean() * TRADING_DAYS_PER_YEAR)
        annualized_volatility = float(returns.std(ddof=1) * SQRT_252)
        sharpe_ratio = (
            float(SQRT_252 * excess.mean() / excess.std(ddof=1))
            if not np.isclose(float(excess.std(ddof=1)), 0.0)
            else np.nan
        )
        max_drawdown = float(drawdown_series(returns).min())
        rows.append(
            {
                "portfolio": label,
                "annualized_return": annualized_return,
                "annualized_return_pct": annualized_return * 100.0,
                "annualized_volatility": annualized_volatility,
                "annualized_volatility_pct": annualized_volatility * 100.0,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "max_drawdown_pct": max_drawdown * 100.0,
            }
        )
    return pd.DataFrame(rows)


def _long_only_frontier_target_weights(
    mean_returns: np.ndarray,
    covariance: np.ndarray,
    target_return: float,
    *,
    initial_guess: np.ndarray,
) -> np.ndarray | None:
    """Solve one long-only target-return frontier point."""

    n_assets = covariance.shape[0]
    result = minimize(
        _variance_objective,
        x0=initial_guess,
        args=(covariance,),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_assets,
        constraints=[
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {
                "type": "eq",
                "fun": lambda w, mu=mean_returns, target=target_return: float(w @ mu - target),
            },
        ],
        options={"maxiter": 800, "ftol": 1e-12},
    )
    if not result.success:
        return None
    weights = np.asarray(result.x, dtype=float)
    if np.isclose(float(weights.sum()), 0.0):
        return None
    return weights / float(weights.sum())


def build_long_only_frontier(
    sample: Stage3Sample,
    weights: pd.DataFrame,
    *,
    n_points: int = DEFAULT_FRONTIER_POINTS,
) -> pd.DataFrame:
    """Numerically build the long-only efficient frontier."""

    returns = sample.returns_wide.to_numpy(dtype=float)
    mean_returns = returns.mean(axis=0)
    covariance = np.cov(returns, rowvar=False, ddof=1)
    matrix = (
        weights.pivot(index="ticker", columns="portfolio", values="weight")
        .reindex(index=sample.tickers)
    )

    eq_weights = matrix[APP_PORTFOLIO_LABELS["equal_weight"]].to_numpy(dtype=float)
    mv_weights = matrix[APP_PORTFOLIO_LABELS["minimum_variance"]].to_numpy(dtype=float)
    tan_weights = matrix[APP_PORTFOLIO_LABELS["mean_variance_tangency"]].to_numpy(dtype=float)

    eq_return = float(mean_returns @ eq_weights)
    mv_return = float(mean_returns @ mv_weights)
    tan_return = float(mean_returns @ tan_weights)
    upper_anchor = max(float(mean_returns.max()), eq_return, tan_return)
    if np.isclose(upper_anchor, mv_return):
        upper_anchor = mv_return + 1e-4
    targets = np.linspace(mv_return, upper_anchor, n_points)

    rows: list[dict[str, float]] = []
    guess = mv_weights.copy()
    for target in targets:
        solved = _long_only_frontier_target_weights(
            mean_returns,
            covariance,
            target,
            initial_guess=guess,
        )
        if solved is None:
            continue
        guess = solved
        variance = float(solved @ covariance @ solved)
        daily_vol = float(np.sqrt(max(variance, 0.0)))
        rows.append(
            {
                "target_return_daily": target,
                "target_return_ann": target * TRADING_DAYS_PER_YEAR,
                "target_return_ann_pct": target * TRADING_DAYS_PER_YEAR * 100.0,
                "volatility_daily": daily_vol,
                "volatility_ann": daily_vol * SQRT_252,
                "volatility_ann_pct": daily_vol * SQRT_252 * 100.0,
            }
        )
    if not rows:
        raise ValueError("Long-only efficient frontier did not produce any successful points.")
    frontier = pd.DataFrame(rows).drop_duplicates(
        subset=["target_return_daily", "volatility_daily"]
    )
    return frontier.sort_values(["volatility_daily", "target_return_daily"]).reset_index(drop=True)


def build_app_frontier(
    sample: Stage3Sample,
    weights: pd.DataFrame,
    *,
    constraint_mode: str,
    n_points: int = DEFAULT_FRONTIER_POINTS,
) -> pd.DataFrame:
    """Build the efficient frontier for the selected constraint mode."""

    optimized_only = weights.loc[weights["portfolio"] != APP_PORTFOLIO_LABELS["custom"]].copy()
    if constraint_mode == "unconstrained":
        return build_efficient_frontier(sample, optimized_only, n_points=n_points)
    if constraint_mode == "long_only":
        return build_long_only_frontier(sample, weights, n_points=n_points)
    raise ValueError(f"Unknown constraint mode: {constraint_mode}")


def frontier_point_coordinates(
    metrics: pd.DataFrame,
    portfolio_label: str,
) -> tuple[float, float]:
    """Return annualized volatility and annualized return coordinates for one portfolio."""

    row = metrics.loc[metrics["portfolio"] == portfolio_label].iloc[0]
    return float(row["annualized_volatility_pct"]), float(row["annualized_return_pct"])


def mean_variance_sharpe(metrics: pd.DataFrame) -> float:
    """Return the mean-variance portfolio Sharpe ratio for app annotations."""

    row = metrics.loc[
        metrics["portfolio"] == APP_PORTFOLIO_LABELS["mean_variance_tangency"]
    ].iloc[0]
    return float(row["sharpe_ratio"])


def annualized_rfr_pct(sample: Stage3Sample) -> float:
    """Return the annualized risk-free rate for the selected sample window."""

    return float(sample.rfr.mean() * TRADING_DAYS_PER_YEAR * 100.0)


def asset_statistics_for_sample(sample: Stage3Sample) -> pd.DataFrame:
    """Return annualized asset statistics for the selected sample."""

    return summarize_asset_statistics(sample)
