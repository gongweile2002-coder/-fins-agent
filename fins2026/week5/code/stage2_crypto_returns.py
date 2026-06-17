# ruff: noqa
"""Stage 2 helpers for Week 5 crypto return construction and summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .risk_free_rate_french import DEFAULT_FRENCH_RFR_OUTPUT_PATH

WEEK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE1_PANEL_PATH = (
    WEEK_ROOT / "results" / "data" / "yahoo_crypto_20_since_2019" / "yahoo_chart_panel_long.parquet"
)
DEFAULT_STAGE2_OUTPUT_DIR = WEEK_ROOT / "results" / "data" / "stage2" / "yahoo_crypto"
DEFAULT_STAGE2_TABLE_DIR = WEEK_ROOT / "results" / "tables" / "stage2" / "yahoo_crypto"
ROLLING_WINDOW_DAYS = 180
TRADING_DAYS_PER_YEAR = 365
SQRT_365 = np.sqrt(TRADING_DAYS_PER_YEAR)
RETURN_PARITY_TOLERANCE = 1e-10
NORMAL_SHARE_ABS_Z_GT_2 = 0.0455
NORMAL_SHARE_ABS_Z_GT_3 = 0.0027


@dataclass(frozen=True)
class Stage2CryptoSpec:
    """One Stage 2 provider configuration for Week 5."""

    provider: str
    display_name: str
    default_input_path: Path
    adjusted_price_column: str
    stage1_label: str


CRYPTO_SPEC = Stage2CryptoSpec(
    provider="yahoo_crypto",
    display_name="Yahoo Finance Crypto",
    default_input_path=DEFAULT_STAGE1_PANEL_PATH,
    adjusted_price_column="adjClose",
    stage1_label="run_beginner_yahoo_crypto_20_since_2019.py",
)


def stage2_output_dir() -> Path:
    """Return the default Stage 2 output directory."""

    return DEFAULT_STAGE2_OUTPUT_DIR


def stage2_table_dir() -> Path:
    """Return the default Stage 2 table directory."""

    return DEFAULT_STAGE2_TABLE_DIR


def stage2_data_paths() -> dict[str, Path]:
    """Return the canonical Stage 2 parquet paths."""

    output_dir = stage2_output_dir()
    return {
        "adjclose_wide": output_dir / "yahoo_crypto_adjclose_wide.parquet",
        "returns_wide": output_dir / "yahoo_crypto_returns_wide.parquet",
        "returns_long": output_dir / "yahoo_crypto_returns_long.parquet",
        "returns_features_long": output_dir / "yahoo_crypto_returns_features_long.parquet",
    }


def stage2_table_paths() -> dict[str, Path]:
    """Return the canonical Stage 2 summary-table paths."""

    table_dir = stage2_table_dir()
    return {
        "summary_csv": table_dir / "yahoo_crypto_summary_metrics.csv",
        "summary_parquet": table_dir / "yahoo_crypto_summary_metrics.parquet",
    }


def load_stage1_crypto_panel(
    *,
    panel_path: Path | None = None,
) -> tuple[pd.DataFrame, Stage2CryptoSpec]:
    """Load the Stage 1 long panel used as input to Stage 2."""

    source_path = panel_path or CRYPTO_SPEC.default_input_path
    if not source_path.exists():
        raise SystemExit(
            f"Missing Stage 1 panel: {source_path}. Run {CRYPTO_SPEC.stage1_label} first "
            "or pass --input-path."
        )

    panel = pd.read_parquet(source_path).copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel[CRYPTO_SPEC.adjusted_price_column] = pd.to_numeric(
        panel[CRYPTO_SPEC.adjusted_price_column],
        errors="coerce",
    )
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    if panel[["ticker", "date"]].duplicated().any():
        raise ValueError("Stage 1 crypto panel contains duplicate ticker-date keys.")
    return panel, CRYPTO_SPEC


def build_adjusted_close_wide(panel: pd.DataFrame, *, price_column: str) -> pd.DataFrame:
    """Pivot the adjusted-price column into the canonical wide matrix."""

    wide = (
        panel.pivot(index="date", columns="ticker", values=price_column)
        .sort_index()
        .reset_index()
    )
    wide.columns.name = None
    return wide


def compute_wide_returns(wide_prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily returns from a wide adjusted-price matrix."""

    frame = wide_prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    value_columns = [column for column in frame.columns if column != "date"]
    returns = frame.set_index("date")[value_columns].pct_change(fill_method=None)
    return returns.reset_index()


def melt_wide_returns(wide_returns: pd.DataFrame) -> pd.DataFrame:
    """Convert wide returns back into long form for parity checks."""

    frame = wide_returns.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    melted = frame.melt(id_vars="date", var_name="ticker", value_name="ret_from_wide")
    return melted.sort_values(["ticker", "date"]).reset_index(drop=True)


def compute_long_returns(panel: pd.DataFrame, *, price_column: str) -> pd.DataFrame:
    """Compute simple daily returns from the long panel with groupby."""

    frame = panel.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    frame["ret"] = frame.groupby("ticker", sort=False)[price_column].pct_change(fill_method=None)
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
    frame = pd.read_parquet(source_path).copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["rfr"] = pd.to_numeric(frame["rfr"], errors="coerce")
    return frame.sort_values("date").reset_index(drop=True)


def build_rfr_timeline(
    dates: pd.Series,
    *,
    rfr_path: Path | None = None,
) -> pd.DataFrame:
    """Align business-day RF to the full crypto daily timeline with forward-fill."""

    rfr = load_daily_rfr(rfr_path)
    requested_dates = pd.to_datetime(pd.Series(dates).dropna().unique())
    timeline_dates = pd.Index(requested_dates).union(pd.Index(rfr["date"]))
    timeline = pd.DataFrame({"date": timeline_dates}).sort_values("date").reset_index(drop=True)
    timeline = timeline.merge(rfr, on="date", how="left")
    timeline["rfr"] = timeline["rfr"].ffill()
    timeline = timeline.loc[timeline["date"].isin(requested_dates)].reset_index(drop=True)
    if timeline["rfr"].isna().any():
        missing_start = timeline.loc[timeline["rfr"].isna(), "date"].min()
        raise ValueError(
            "Risk-free merge left leading missing values after forward-fill. "
            f"First missing date: {missing_start}."
        )
    return timeline


def merge_daily_rfr(
    frame: pd.DataFrame,
    *,
    rfr_path: Path | None = None,
) -> pd.DataFrame:
    """Merge the daily risk-free series into a long ticker-date crypto panel."""

    timeline = build_rfr_timeline(frame["date"], rfr_path=rfr_path)
    merged = frame.merge(timeline, on="date", how="left")
    return merged.sort_values(["ticker", "date"]).reset_index(drop=True)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Return a ratio that keeps zero-denominator observations missing."""

    ratio = numerator / denominator.replace(0.0, np.nan)
    return ratio.replace([np.inf, -np.inf], np.nan)


def _downside_deviation(values: pd.Series) -> float:
    """Return downside deviation for a series of excess returns."""

    clean = pd.Series(values).dropna().astype(float)
    if clean.empty:
        return np.nan
    downside = clean.clip(upper=0.0)
    return float(np.sqrt(np.mean(np.square(downside))))


def _add_rolling_features_for_ticker(frame: pd.DataFrame) -> pd.DataFrame:
    """Add trailing six-month features for one ticker."""

    group = frame.copy().sort_values("date").reset_index(drop=True)
    window = ROLLING_WINDOW_DAYS
    ret = group["ret"].astype(float)
    excess = group["excess_ret"].astype(float)

    group["rolling_6m_avg_ret"] = ret.rolling(window, min_periods=window).mean()
    group["rolling_6m_vol"] = ret.rolling(window, min_periods=window).std() * SQRT_365
    group["rolling_6m_var_95"] = ret.rolling(window, min_periods=window).quantile(0.05)

    rolling_mean_excess = excess.rolling(window, min_periods=window).mean()
    rolling_std_excess = excess.rolling(window, min_periods=window).std()
    group["rolling_6m_sharpe"] = SQRT_365 * _safe_ratio(
        rolling_mean_excess,
        rolling_std_excess,
    )

    downside_std = excess.rolling(window, min_periods=window).apply(
        lambda values: _downside_deviation(pd.Series(values)),
        raw=False,
    )
    group["rolling_6m_sortino"] = SQRT_365 * _safe_ratio(
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
    return pd.concat(groups, ignore_index=True)


def summarize_stage2_metrics(feature_panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize full-sample return, risk, and tail metrics by ticker."""

    frame = feature_panel.dropna(subset=["ret"]).copy()
    if frame.empty:
        raise ValueError("No non-missing returns were available for summary metrics.")

    if "excess_ret" not in frame.columns:
        frame["excess_ret"] = frame["ret"]

    rows: list[dict[str, float | int | str]] = []
    for ticker, group in frame.groupby("ticker", sort=True):
        ret = group["ret"].astype(float)
        excess = group["excess_ret"].astype(float)
        mean_daily_return = float(ret.mean())
        ann_return = mean_daily_return * TRADING_DAYS_PER_YEAR
        ann_volatility = float(ret.std()) * SQRT_365
        mean_excess_return = float(excess.mean())
        std_excess_return = float(excess.std())
        downside_dev = _downside_deviation(excess)
        zscore_denominator = float(ret.std())
        if zscore_denominator == 0.0 or np.isnan(zscore_denominator):
            tail_share_abs_z_gt_2 = np.nan
            tail_share_abs_z_gt_3 = np.nan
        else:
            zscores = (ret - mean_daily_return) / zscore_denominator
            tail_share_abs_z_gt_2 = float((zscores.abs() > 2.0).mean())
            tail_share_abs_z_gt_3 = float((zscores.abs() > 3.0).mean())

        rows.append(
            {
                "ticker": ticker,
                "row_count": len(group),
                "mean_daily_return": mean_daily_return,
                "ann_return": ann_return,
                "ann_volatility": ann_volatility,
                "full_sample_sharpe": (
                    float(SQRT_365 * mean_excess_return / std_excess_return)
                    if not np.isnan(std_excess_return) and std_excess_return != 0.0
                    else np.nan
                ),
                "full_sample_sortino": (
                    float(SQRT_365 * mean_excess_return / downside_dev)
                    if not np.isnan(downside_dev) and downside_dev != 0.0
                    else np.nan
                ),
                "skewness": float(ret.skew()),
                "excess_kurtosis": float(ret.kurt()),
                "max_abs_ret": float(ret.abs().max()),
                "tail_share_abs_z_gt_2": tail_share_abs_z_gt_2,
                "tail_share_abs_z_gt_3": tail_share_abs_z_gt_3,
            }
        )

    summary = (
        pd.DataFrame(rows)
        .sort_values("ann_volatility", ascending=False)
        .reset_index(drop=True)
    )
    summary["ann_return_pct"] = summary["ann_return"] * 100.0
    summary["ann_volatility_pct"] = summary["ann_volatility"] * 100.0
    summary["max_abs_ret_pct"] = summary["max_abs_ret"] * 100.0
    summary["tail_share_abs_z_gt_2_pct"] = summary["tail_share_abs_z_gt_2"] * 100.0
    summary["tail_share_abs_z_gt_3_pct"] = summary["tail_share_abs_z_gt_3"] * 100.0
    summary["normal_share_abs_z_gt_2_pct"] = NORMAL_SHARE_ABS_Z_GT_2 * 100.0
    summary["normal_share_abs_z_gt_3_pct"] = NORMAL_SHARE_ABS_Z_GT_3 * 100.0
    return summary
