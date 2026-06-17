# ruff: noqa
"""Stage 2 helpers for Week 4 return construction and feature engineering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .equity_api_tiingo import build_wide_price_table
from .risk_free_rate_french import DEFAULT_FRENCH_RFR_OUTPUT_PATH

WEEK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE2_OUTPUT_ROOT = WEEK_ROOT / "results" / "data" / "stage2"
ROLLING_WINDOW_DAYS = 126
TRADING_DAYS_PER_YEAR = 252
SQRT_252 = np.sqrt(TRADING_DAYS_PER_YEAR)
RETURN_PARITY_TOLERANCE = 1e-10


@dataclass(frozen=True)
class Stage2ProviderSpec:
    """One Stage 2 provider configuration."""

    provider: str
    display_name: str
    default_input_path: Path
    adjusted_price_column: str
    stage1_label: str


PROVIDER_SPECS: dict[str, Stage2ProviderSpec] = {
    "tiingo": Stage2ProviderSpec(
        provider="tiingo",
        display_name="Tiingo",
        default_input_path=(
            WEEK_ROOT / "results" / "data" / "tiingo_famous_50" / "tiingo_eod_panel_long.parquet"
        ),
        adjusted_price_column="adjClose",
        stage1_label="run_beginner_tiingo_famous_50.py",
    ),
    "yahoo": Stage2ProviderSpec(
        provider="yahoo",
        display_name="Yahoo Finance",
        default_input_path=(
            WEEK_ROOT / "results" / "data" / "yahoo_famous_50" / "yahoo_chart_panel_long.parquet"
        ),
        adjusted_price_column="adjClose",
        stage1_label="run_beginner_yahoo_famous_50.py",
    ),
}


def resolve_stage2_provider(provider: str) -> Stage2ProviderSpec:
    """Return the Stage 2 provider spec or raise a clear error."""

    key = provider.strip().lower()
    try:
        return PROVIDER_SPECS[key]
    except KeyError as exc:
        names = ", ".join(sorted(PROVIDER_SPECS))
        raise SystemExit(f"Unknown provider {provider!r}. Choose one of: {names}.") from exc


def stage2_output_dir(provider: str) -> Path:
    """Return the default Stage 2 output directory for a provider."""

    spec = resolve_stage2_provider(provider)
    return DEFAULT_STAGE2_OUTPUT_ROOT / spec.provider


def stage2_data_paths(provider: str) -> dict[str, Path]:
    """Return the canonical Stage 2 parquet paths for one provider."""

    output_dir = stage2_output_dir(provider)
    return {
        "adjclose_wide": output_dir / f"{provider}_adjclose_wide.parquet",
        "returns_wide": output_dir / f"{provider}_returns_wide.parquet",
        "returns_long": output_dir / f"{provider}_returns_long.parquet",
        "returns_features_long": output_dir / f"{provider}_returns_features_long.parquet",
    }


def load_stage1_equity_panel(
    provider: str,
    *,
    panel_path: Path | None = None,
) -> tuple[pd.DataFrame, Stage2ProviderSpec]:
    """Load the Stage 1 long panel used as input to Stage 2."""

    spec = resolve_stage2_provider(provider)
    source_path = panel_path or spec.default_input_path
    if not source_path.exists():
        raise SystemExit(
            f"Missing Stage 1 panel: {source_path}. Run {spec.stage1_label} first "
            f"or pass --input-path."
        )

    panel = pd.read_parquet(source_path)
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel[spec.adjusted_price_column] = pd.to_numeric(
        panel[spec.adjusted_price_column],
        errors="coerce",
    )
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    if panel[["ticker", "date"]].duplicated().any():
        raise ValueError("Stage 1 panel contains duplicate ticker-date keys.")
    return panel, spec


def build_adjusted_close_wide(panel: pd.DataFrame, *, price_column: str) -> pd.DataFrame:
    """Pivot the adjusted-price column into the canonical wide matrix."""

    wide = build_wide_price_table(
        panel[["ticker", "date", price_column]],
        value_column=price_column,
    )
    return wide


def compute_wide_returns(wide_prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily returns from a wide adjusted-price matrix."""

    frame = wide_prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    value_columns = [column for column in frame.columns if column != "date"]
    returns = frame.set_index("date")[value_columns].pct_change(fill_method=None)
    returns = returns.reset_index()
    return returns


def melt_wide_returns(wide_returns: pd.DataFrame) -> pd.DataFrame:
    """Convert wide returns back into long form for parity checks."""

    frame = wide_returns.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    melted = frame.melt(id_vars="date", var_name="ticker", value_name="ret_from_wide")
    return melted.sort_values(["ticker", "date"]).reset_index(drop=True)


def compute_long_returns(panel: pd.DataFrame, *, price_column: str) -> pd.DataFrame:
    """Compute simple daily returns from the long panel with groupby."""

    frame = panel.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    frame["ret"] = (
        frame.groupby("ticker", sort=False)[price_column].pct_change(fill_method=None)
    )
    return frame


def assert_return_parity(
    long_returns: pd.DataFrame,
    wide_returns: pd.DataFrame,
    *,
    tolerance: float = RETURN_PARITY_TOLERANCE,
) -> float:
    """Assert that long-groupby and wide-matrix returns agree on common keys."""

    left = long_returns[["ticker", "date", "ret"]].copy()
    left["date"] = pd.to_datetime(left["date"])
    right = melt_wide_returns(wide_returns)
    merged = left.merge(right, on=["ticker", "date"], how="outer", sort=False)

    mismatch_mask = merged["ret"].isna() ^ merged["ret_from_wide"].isna()
    if mismatch_mask.any():
        bad = merged.loc[mismatch_mask, ["ticker", "date", "ret", "ret_from_wide"]].head()
        raise ValueError(f"Return parity failed because of missing-pattern mismatch:\n{bad}")

    comparable = merged.dropna(subset=["ret", "ret_from_wide"]).copy()
    comparable["abs_diff"] = (comparable["ret"] - comparable["ret_from_wide"]).abs()
    max_abs_diff = float(comparable["abs_diff"].max()) if not comparable.empty else 0.0
    if max_abs_diff > tolerance:
        bad = comparable.nlargest(5, "abs_diff")[
            ["ticker", "date", "ret", "ret_from_wide", "abs_diff"]
        ]
        raise ValueError(
            f"Return parity failed with max abs diff {max_abs_diff:.3e}.\n{bad}"
        )
    return max_abs_diff


def load_daily_rfr(path: Path | None = None) -> pd.DataFrame:
    """Load the French daily risk-free series."""

    source_path = path or DEFAULT_FRENCH_RFR_OUTPUT_PATH
    if not source_path.exists():
        raise SystemExit(
            f"Missing daily risk-free file: {source_path}. "
            "Run run_beginner_french_rfr.py first."
        )
    frame = pd.read_parquet(source_path)
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["rfr"] = pd.to_numeric(frame["rfr"], errors="coerce")
    return frame.sort_values("date").reset_index(drop=True)


def build_rfr_timeline(
    dates: pd.Series,
    *,
    rfr_path: Path | None = None,
) -> pd.DataFrame:
    """Align the daily risk-free series to the requested dates and forward-fill the tail."""

    rfr = load_daily_rfr(rfr_path)
    timeline = pd.DataFrame({"date": pd.to_datetime(pd.Series(dates).dropna().unique())})
    timeline = timeline.sort_values("date").reset_index(drop=True)
    timeline = timeline.merge(rfr, on="date", how="left")
    timeline["rfr"] = timeline["rfr"].ffill()
    return timeline


def merge_daily_rfr(
    frame: pd.DataFrame,
    *,
    rfr_path: Path | None = None,
) -> pd.DataFrame:
    """Merge the daily risk-free series into a long ticker-date panel."""

    timeline = build_rfr_timeline(frame["date"], rfr_path=rfr_path)
    merged = frame.merge(timeline, on="date", how="left")
    return merged.sort_values(["ticker", "date"]).reset_index(drop=True)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return a ratio that keeps zero-denominator observations missing."""

    ratio = numerator / denominator.replace(0.0, np.nan)
    return ratio.replace([np.inf, -np.inf], np.nan)


def _add_rolling_features_for_ticker(frame: pd.DataFrame) -> pd.DataFrame:
    """Add trailing six-month features for one ticker."""

    group = frame.copy().sort_values("date").reset_index(drop=True)
    window = ROLLING_WINDOW_DAYS
    ret = group["ret"].astype(float)
    excess = group["excess_ret"].astype(float)

    group["rolling_6m_avg_ret"] = ret.rolling(window, min_periods=window).mean()
    group["rolling_6m_vol"] = ret.rolling(window, min_periods=window).std() * SQRT_252
    group["rolling_6m_var_95"] = ret.rolling(window, min_periods=window).quantile(0.05)

    rolling_mean_excess = excess.rolling(window, min_periods=window).mean()
    rolling_std_excess = excess.rolling(window, min_periods=window).std()
    group["rolling_6m_sharpe"] = SQRT_252 * _safe_ratio(
        rolling_mean_excess,
        rolling_std_excess,
    )

    downside = excess.clip(upper=0.0)
    downside_std = downside.rolling(window, min_periods=window).apply(
        lambda values: float(np.sqrt(np.mean(np.square(values)))),
        raw=True,
    )
    group["rolling_6m_sortino"] = SQRT_252 * _safe_ratio(
        rolling_mean_excess,
        downside_std,
    )
    return group


def build_feature_long_panel(
    long_returns: pd.DataFrame,
    *,
    rfr_path: Path | None = None,
) -> pd.DataFrame:
    """Add Stage 2 feature columns to the long return panel."""

    frame = long_returns.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    frame["ret"] = pd.to_numeric(frame["ret"], errors="coerce")
    frame["abs_ret"] = frame["ret"].abs()
    frame["is_large_move_10pct"] = frame["abs_ret"] >= 0.10
    frame["is_large_move_20pct"] = frame["abs_ret"] >= 0.20
    frame = merge_daily_rfr(frame, rfr_path=rfr_path)
    frame["excess_ret"] = frame["ret"] - frame["rfr"]
    groups = [
        _add_rolling_features_for_ticker(group)
        for _ticker, group in frame.groupby("ticker", sort=False)
    ]
    featured = pd.concat(groups, ignore_index=True)
    return featured


def summarize_full_sample_volatility(feature_panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize full-sample return, volatility, and Sharpe statistics."""

    frame = feature_panel.dropna(subset=["ret"]).copy()
    if "excess_ret" not in frame.columns:
        frame["excess_ret"] = frame["ret"]

    summary = (
        frame
        .groupby("ticker", as_index=False)
        .agg(
            mean_excess_return=("excess_ret", "mean"),
            std_excess_return=("excess_ret", "std"),
            ann_volatility=("ret", lambda values: pd.Series(values).std() * SQRT_252),
            mean_daily_return=("ret", "mean"),
            row_count=("ret", "size"),
        )
        .sort_values("ann_volatility", ascending=False)
        .reset_index(drop=True)
    )
    summary["ann_return"] = summary["mean_daily_return"] * TRADING_DAYS_PER_YEAR
    summary["ann_sharpe"] = SQRT_252 * _safe_ratio(
        summary["mean_excess_return"],
        summary["std_excess_return"],
    )
    summary["ann_volatility_pct"] = summary["ann_volatility"] * 100.0
    summary["ann_return_pct"] = summary["ann_return"] * 100.0
    summary["mean_daily_return_pct"] = summary["mean_daily_return"] * 100.0
    return summary


def select_top_bottom_volatility_tickers(
    volatility_summary: pd.DataFrame,
    *,
    n: int = 5,
) -> tuple[list[str], list[str]]:
    """Return bottom and top volatility tickers."""

    ordered = volatility_summary.dropna(subset=["ann_volatility"]).copy()
    if ordered.empty:
        raise ValueError("No volatility summary rows were available.")
    bottom = ordered.nsmallest(n, "ann_volatility")["ticker"].tolist()
    top = ordered.nlargest(n, "ann_volatility")["ticker"].tolist()
    return bottom, top
