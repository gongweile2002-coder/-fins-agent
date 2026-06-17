# ruff: noqa
"""Stage 3 helpers for Week 4 in-sample mean-variance portfolios."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fintools.portfolio_math import (
    equal_weight_vector,
    minimum_variance_weights,
    solve_markowitz_system,
    tangency_weights,
)

WEEK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE3_DATA_ROOT = WEEK_ROOT / "results" / "data" / "stage3"
DEFAULT_STAGE3_TABLE_ROOT = WEEK_ROOT / "results" / "tables" / "stage3"
TRADING_DAYS_PER_YEAR = 252
SQRT_252 = np.sqrt(TRADING_DAYS_PER_YEAR)

PORTFOLIO_COLUMN_ORDER = [
    "equal_weight",
    "minimum_variance",
    "mean_variance_tangency",
]
PORTFOLIO_LABELS = {
    "equal_weight": "Equal-weight",
    "minimum_variance": "Minimum variance",
    "mean_variance_tangency": "Mean-variance",
}


@dataclass(frozen=True)
class Stage3ProviderSpec:
    """One Stage 3 provider configuration."""

    provider: str
    display_name: str
    default_input_path: Path
    stage2_label: str


@dataclass(frozen=True)
class Stage3Sample:
    """Balanced in-sample return matrix plus aligned risk-free series."""

    provider: str
    display_name: str
    returns_wide: pd.DataFrame
    rfr: pd.Series

    @property
    def tickers(self) -> list[str]:
        return self.returns_wide.columns.tolist()

    @property
    def start_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.returns_wide.index.min())

    @property
    def end_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.returns_wide.index.max())

    @property
    def sample_days(self) -> int:
        return len(self.returns_wide)

    @property
    def n_assets(self) -> int:
        return int(self.returns_wide.shape[1])


PROVIDER_SPECS: dict[str, Stage3ProviderSpec] = {
    "yahoo": Stage3ProviderSpec(
        provider="yahoo",
        display_name="Yahoo Finance",
        default_input_path=(
            WEEK_ROOT
            / "results"
            / "data"
            / "stage2"
            / "yahoo"
            / "yahoo_returns_features_long.parquet"
        ),
        stage2_label="run_beginner_stage2_features_long.py --provider yahoo",
    ),
    "tiingo": Stage3ProviderSpec(
        provider="tiingo",
        display_name="Tiingo",
        default_input_path=(
            WEEK_ROOT
            / "results"
            / "data"
            / "stage2"
            / "tiingo"
            / "tiingo_returns_features_long.parquet"
        ),
        stage2_label="run_beginner_stage2_features_long.py --provider tiingo",
    ),
    "tiingo_small": Stage3ProviderSpec(
        provider="tiingo_small",
        display_name="Tiingo (small panel)",
        default_input_path=(
            WEEK_ROOT
            / "results"
            / "data"
            / "stage2"
            / "tiingo_small"
            / "tiingo_returns_features_long.parquet"
        ),
        stage2_label="run_beginner_stage2_features_long.py --provider tiingo --input-path ...",
    ),
}


def resolve_stage3_provider(provider: str) -> Stage3ProviderSpec:
    """Return the Stage 3 provider spec or raise a clear error."""

    key = provider.strip().lower()
    try:
        return PROVIDER_SPECS[key]
    except KeyError as exc:
        names = ", ".join(sorted(PROVIDER_SPECS))
        raise SystemExit(f"Unknown provider {provider!r}. Choose one of: {names}.") from exc


def stage3_data_dir(provider: str) -> Path:
    """Return the default Stage 3 data directory for one provider."""

    spec = resolve_stage3_provider(provider)
    return DEFAULT_STAGE3_DATA_ROOT / spec.provider


def stage3_table_dir(provider: str) -> Path:
    """Return the default Stage 3 table directory for one provider."""

    spec = resolve_stage3_provider(provider)
    return DEFAULT_STAGE3_TABLE_ROOT / spec.provider


def stage3_output_paths(provider: str) -> dict[str, Path]:
    """Return the canonical Stage 3 output paths for one provider."""

    data_dir = stage3_data_dir(provider)
    table_dir = stage3_table_dir(provider)
    return {
        "weights": data_dir / f"{provider}_portfolio_weights.parquet",
        "returns": data_dir / f"{provider}_portfolio_returns.parquet",
        "frontier": data_dir / f"{provider}_efficient_frontier.parquet",
        "metrics": table_dir / f"{provider}_portfolio_metrics.csv",
    }


def load_stage2_feature_panel(
    provider: str,
    *,
    panel_path: Path | None = None,
) -> tuple[pd.DataFrame, Stage3ProviderSpec]:
    """Load the Stage 2 feature-rich long panel used as Stage 3 input."""

    spec = resolve_stage3_provider(provider)
    source_path = panel_path or spec.default_input_path
    if not source_path.exists():
        raise SystemExit(
            f"Missing Stage 2 feature panel: {source_path}. "
            f"Run {spec.stage2_label} first or pass --input-path."
        )

    frame = pd.read_parquet(source_path)
    frame = frame.copy()
    required = {"ticker", "date", "ret", "rfr"}
    missing = required.difference(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Stage 2 feature panel is missing required columns: {missing_text}")

    frame["date"] = pd.to_datetime(frame["date"])
    frame["ret"] = pd.to_numeric(frame["ret"], errors="coerce")
    frame["rfr"] = pd.to_numeric(frame["rfr"], errors="coerce")
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    if frame[["ticker", "date"]].duplicated().any():
        raise ValueError("Stage 2 feature panel contains duplicate ticker-date keys.")
    return frame, spec


def build_balanced_stage3_sample(
    feature_panel: pd.DataFrame,
    *,
    provider: str,
    display_name: str,
) -> Stage3Sample:
    """Build the balanced in-sample return matrix and aligned risk-free series."""

    returns_wide = (
        feature_panel.pivot(index="date", columns="ticker", values="ret")
        .sort_index()
        .sort_index(axis=1)
    )
    rfr_timeline = (
        feature_panel.groupby("date", sort=True)["rfr"]
        .first()
        .sort_index()
    )
    valid_mask = (~returns_wide.isna().any(axis=1)) & rfr_timeline.notna()
    balanced_returns = returns_wide.loc[valid_mask].copy()
    balanced_rfr = rfr_timeline.loc[balanced_returns.index].copy()
    if balanced_returns.empty:
        raise ValueError("No balanced Stage 3 return sample remains after dropping missing dates.")
    return Stage3Sample(
        provider=provider,
        display_name=display_name,
        returns_wide=balanced_returns,
        rfr=balanced_rfr,
    )


def estimate_portfolio_weights(sample: Stage3Sample) -> tuple[pd.DataFrame, dict[str, str]]:
    """Estimate the three Stage 3 portfolio weight vectors."""

    returns = sample.returns_wide.to_numpy(dtype=float)
    mean_returns = returns.mean(axis=0)
    covariance = np.cov(returns, rowvar=False, ddof=1)
    avg_daily_rfr = float(sample.rfr.mean())

    eq_weights = equal_weight_vector(sample.n_assets)
    mv_weights, mv_method = minimum_variance_weights(covariance)
    tan_weights, tan_method = tangency_weights(mean_returns, covariance, avg_daily_rfr)

    weight_frame = pd.DataFrame(
        {
            "ticker": sample.tickers,
            PORTFOLIO_LABELS["equal_weight"]: eq_weights,
            PORTFOLIO_LABELS["minimum_variance"]: mv_weights,
            PORTFOLIO_LABELS["mean_variance_tangency"]: tan_weights,
        }
    )
    long_weights = (
        weight_frame.melt(id_vars="ticker", var_name="portfolio", value_name="weight")
        .sort_values(["portfolio", "ticker"])
        .reset_index(drop=True)
    )
    methods = {
        PORTFOLIO_LABELS["minimum_variance"]: mv_method,
        PORTFOLIO_LABELS["mean_variance_tangency"]: tan_method,
    }
    return long_weights, methods


def weights_to_matrix(weights: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Pivot the long weight table into a ticker-by-portfolio matrix."""

    matrix = weights.pivot(index="ticker", columns="portfolio", values="weight")
    matrix = matrix.reindex(index=tickers)
    matrix = matrix.reindex(columns=[PORTFOLIO_LABELS[key] for key in PORTFOLIO_COLUMN_ORDER])
    return matrix


def compute_portfolio_returns(
    sample: Stage3Sample,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """Compute daily rebalanced constant-weight returns for the three portfolios."""

    weight_matrix = weights_to_matrix(weights, sample.tickers)
    returns_array = sample.returns_wide.to_numpy(dtype=float)
    portfolio_returns = returns_array @ weight_matrix.to_numpy(dtype=float)
    output = pd.DataFrame(
        portfolio_returns,
        index=sample.returns_wide.index,
        columns=PORTFOLIO_COLUMN_ORDER,
    )
    output = output.reset_index().rename(columns={"index": "date"})
    output["rfr"] = sample.rfr.to_numpy(dtype=float)
    return output


def wealth_index(returns: pd.Series) -> pd.Series:
    """Convert simple returns into a growth-of-one-dollar wealth index."""

    series = returns.astype(float).fillna(0.0)
    return (1.0 + series).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Convert simple returns into a drawdown series."""

    wealth = wealth_index(returns)
    return wealth / wealth.cummax() - 1.0


def summarize_asset_statistics(sample: Stage3Sample) -> pd.DataFrame:
    """Summarize annualized asset return, volatility, and Sharpe ratio."""

    daily_mean = sample.returns_wide.mean(axis=0)
    daily_vol = sample.returns_wide.std(axis=0, ddof=1)
    daily_excess_mean = sample.returns_wide.sub(sample.rfr, axis=0).mean(axis=0)
    sharpe = SQRT_252 * (daily_excess_mean / daily_vol.replace(0.0, np.nan))
    summary = pd.DataFrame(
        {
            "ticker": daily_mean.index,
            "annualized_return": daily_mean.to_numpy(dtype=float) * TRADING_DAYS_PER_YEAR,
            "annualized_volatility": daily_vol.to_numpy(dtype=float) * SQRT_252,
            "sharpe_ratio": sharpe.to_numpy(dtype=float),
        }
    )
    summary["annualized_return_pct"] = summary["annualized_return"] * 100.0
    summary["annualized_volatility_pct"] = summary["annualized_volatility"] * 100.0
    return summary.sort_values("ticker").reset_index(drop=True)


def summarize_portfolio_metrics(portfolio_returns: pd.DataFrame) -> pd.DataFrame:
    """Summarize annualized performance metrics for the three portfolios."""

    rows: list[dict[str, float | str]] = []
    rfr = portfolio_returns["rfr"].astype(float)
    for column in PORTFOLIO_COLUMN_ORDER:
        returns = portfolio_returns[column].astype(float)
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
                "portfolio": PORTFOLIO_LABELS[column],
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


def build_efficient_frontier(
    sample: Stage3Sample,
    weights: pd.DataFrame,
    *,
    n_points: int = 250,
) -> pd.DataFrame:
    """Build the annualized efficient frontier and capital allocation line inputs."""

    returns = sample.returns_wide.to_numpy(dtype=float)
    mean_returns = returns.mean(axis=0)
    covariance = np.cov(returns, rowvar=False, ddof=1)
    ones = np.ones(sample.n_assets, dtype=float)
    inv_ones, _ = solve_markowitz_system(covariance, ones)
    inv_mu, _ = solve_markowitz_system(covariance, mean_returns)

    a_value = float(ones @ inv_ones)
    b_value = float(ones @ inv_mu)
    c_value = float(mean_returns @ inv_mu)
    determinant = a_value * c_value - b_value * b_value
    if np.isclose(determinant, 0.0):
        raise ValueError("Efficient frontier determinant is zero.")

    weight_matrix = weights_to_matrix(weights, sample.tickers)
    eq_return = float(mean_returns @ weight_matrix[PORTFOLIO_LABELS["equal_weight"]].to_numpy())
    tan_return = float(
        mean_returns @ weight_matrix[PORTFOLIO_LABELS["mean_variance_tangency"]].to_numpy()
    )
    gmv_return = float(
        mean_returns @ weight_matrix[PORTFOLIO_LABELS["minimum_variance"]].to_numpy()
    )
    upper_anchor = max(float(mean_returns.max()), eq_return, tan_return)
    spread = max(abs(upper_anchor - gmv_return), 1e-4)
    target_grid = np.linspace(gmv_return, upper_anchor + 0.20 * spread, n_points)
    variance_grid = (a_value * target_grid**2 - 2.0 * b_value * target_grid + c_value) / determinant
    volatility_grid = np.sqrt(np.maximum(variance_grid, 0.0))

    frontier = pd.DataFrame(
        {
            "target_return_daily": target_grid,
            "target_return_ann": target_grid * TRADING_DAYS_PER_YEAR,
            "target_return_ann_pct": target_grid * TRADING_DAYS_PER_YEAR * 100.0,
            "volatility_daily": volatility_grid,
            "volatility_ann": volatility_grid * SQRT_252,
            "volatility_ann_pct": volatility_grid * SQRT_252 * 100.0,
        }
    )
    return frontier


def balanced_sample_summary(sample: Stage3Sample) -> dict[str, object]:
    """Return a compact summary used in script output and context notes."""

    return {
        "provider": sample.display_name,
        "n_assets": sample.n_assets,
        "sample_days": sample.sample_days,
        "start_date": sample.start_date,
        "end_date": sample.end_date,
        "mean_daily_rfr": float(sample.rfr.mean()),
    }
