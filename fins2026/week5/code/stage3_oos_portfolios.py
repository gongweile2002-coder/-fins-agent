# ruff: noqa
"""Stage 3 helpers for Week 5 out-of-sample crypto portfolio weights and returns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize

from fintools.portfolio_math import (
    add_diagonal_ridge,
    equal_weight_vector,
    minimum_variance_weights,
    solve_markowitz_system,
    tangency_weights,
)

WEEK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE3_DATA_ROOT = WEEK_ROOT / "results" / "data" / "stage3" / "yahoo_crypto"
DEFAULT_STAGE3_TABLE_ROOT = WEEK_ROOT / "results" / "tables" / "stage3" / "yahoo_crypto"
DEFAULT_STAGE2_FEATURE_PATH = (
    WEEK_ROOT
    / "results"
    / "data"
    / "stage2"
    / "yahoo_crypto"
    / "yahoo_crypto_returns_features_long.parquet"
)

DEFAULT_INITIAL_WINDOW = 365
DEFAULT_ESTIMATION_FREQUENCY = "monthly"
DEFAULT_WINDOW_RULE = "expanding"
DEFAULT_CONSTRAINT_MODES = ("long_only",)
DEFAULT_MODELS = (
    "equal_weight",
    "minimum_variance",
    "mean_variance_tangency",
    "mean_cvar_tangency",
    "risk_parity_volatility",
)
DEFAULT_CVAR_ALPHA = 0.95
DEFAULT_COVARIANCE_RIDGE = 1e-8
DEFAULT_SOLVER_TOLERANCE = 1e-7
DEFAULT_SOLVER_MAX_ITER = 500
TAU_FLOOR = 1e-8
OPTIMIZER_EPSILON = 1e-12

ESTIMATION_FREQUENCIES = ("daily", "weekly", "monthly")
WINDOW_RULES = ("expanding", "rolling")
CONSTRAINT_MODES = ("long_only",)
MODEL_ORDER = list(DEFAULT_MODELS)
MODEL_LABELS = {
    "equal_weight": "Equal-weight",
    "minimum_variance": "Minimum variance",
    "mean_variance_tangency": "Mean-variance",
    "mean_cvar_tangency": "Mean-CVaR",
    "risk_parity_volatility": "Risk parity",
}
TRADING_DAYS_PER_YEAR = 365
SQRT_365 = np.sqrt(TRADING_DAYS_PER_YEAR)
FACTSHEET_TRAILING_WINDOWS = (30, 90, 180)
FACTSHEET_RISK_WINDOW_DAYS = 180
OOS_FIGURE_PORTFOLIO_COLUMN_ORDER = [
    "equal_weight",
    "minimum_variance_long_only",
    "mean_variance_tangency_long_only",
    "mean_cvar_tangency_long_only",
    "risk_parity_volatility_long_only",
]
OOS_FIGURE_PORTFOLIO_LABELS = {
    "equal_weight": "Equal-weight",
    "minimum_variance_long_only": "Minimum variance",
    "mean_variance_tangency_long_only": "Mean-variance",
    "mean_cvar_tangency_long_only": "Mean-CVaR",
    "risk_parity_volatility_long_only": "Risk parity",
}
OPTIMIZED_PORTFOLIO_COLUMN_ORDER = [
    column
    for column in OOS_FIGURE_PORTFOLIO_COLUMN_ORDER
    if column != "equal_weight"
]
TOP_HOLDINGS_MODEL_ORDER = [
    "minimum_variance",
    "mean_variance_tangency",
    "mean_cvar_tangency",
    "risk_parity_volatility",
]
PORTFOLIO_KEY_TO_MODEL = {
    "equal_weight": "equal_weight",
    "minimum_variance_long_only": "minimum_variance",
    "mean_variance_tangency_long_only": "mean_variance_tangency",
    "mean_cvar_tangency_long_only": "mean_cvar_tangency",
    "risk_parity_volatility_long_only": "risk_parity_volatility",
}
PORTFOLIO_MODEL_TO_KEY = {value: key for key, value in PORTFOLIO_KEY_TO_MODEL.items()}


@dataclass(frozen=True)
class Stage3CryptoSpec:
    """One Stage 3 provider configuration for Week 5."""

    provider: str
    display_name: str
    default_input_path: Path
    stage2_label: str


@dataclass(frozen=True)
class Stage3OOSSample:
    """Balanced daily crypto return matrix plus aligned risk-free series."""

    provider: str
    display_name: str
    returns_wide: pd.DataFrame
    rfr: pd.Series

    @property
    def tickers(self) -> list[str]:
        return self.returns_wide.columns.tolist()

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.returns_wide.index)

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


@dataclass(frozen=True)
class Stage3OOSConfig:
    """Configuration for the Week 5 out-of-sample weight engine."""

    initial_window: int = DEFAULT_INITIAL_WINDOW
    estimation_frequency: str = DEFAULT_ESTIMATION_FREQUENCY
    window_rule: str = DEFAULT_WINDOW_RULE
    constraint_modes: tuple[str, ...] = DEFAULT_CONSTRAINT_MODES
    models: tuple[str, ...] = DEFAULT_MODELS
    cvar_alpha: float = DEFAULT_CVAR_ALPHA
    covariance_ridge: float = DEFAULT_COVARIANCE_RIDGE
    solver_tolerance: float = DEFAULT_SOLVER_TOLERANCE
    solver_max_iter: int = DEFAULT_SOLVER_MAX_ITER

    def __post_init__(self) -> None:
        normalized_modes = tuple(dict.fromkeys(self.constraint_modes))
        normalized_models = tuple(dict.fromkeys(self.models))
        object.__setattr__(self, "constraint_modes", normalized_modes)
        object.__setattr__(self, "models", normalized_models)

        if self.initial_window < 2:
            raise ValueError("Initial window must be at least 2 daily observations.")
        if self.estimation_frequency not in ESTIMATION_FREQUENCIES:
            names = ", ".join(ESTIMATION_FREQUENCIES)
            raise ValueError(
                f"Unknown estimation frequency {self.estimation_frequency!r}. Use: {names}."
            )
        if self.window_rule not in WINDOW_RULES:
            names = ", ".join(WINDOW_RULES)
            raise ValueError(f"Unknown window rule {self.window_rule!r}. Use: {names}.")
        if not normalized_modes:
            raise ValueError("At least one constraint mode is required.")
        if not normalized_models:
            raise ValueError("At least one model is required.")
        bad_modes = sorted(set(normalized_modes).difference(CONSTRAINT_MODES))
        if bad_modes:
            raise ValueError(f"Unknown constraint modes: {', '.join(bad_modes)}.")
        bad_models = sorted(set(normalized_models).difference(MODEL_ORDER))
        if bad_models:
            raise ValueError(f"Unknown model names: {', '.join(bad_models)}.")
        if not 0.0 < self.cvar_alpha < 1.0:
            raise ValueError("CVaR alpha must lie strictly between 0 and 1.")
        if self.covariance_ridge < 0.0:
            raise ValueError("Covariance ridge must be non-negative.")
        if self.solver_tolerance <= 0.0:
            raise ValueError("Solver tolerance must be strictly positive.")
        if self.solver_max_iter < 1:
            raise ValueError("Solver max iter must be at least 1.")


@dataclass(frozen=True)
class RebalanceWindow:
    """One decision-date window plus the holding-period dates it controls."""

    decision_index: int
    decision_date: pd.Timestamp
    effective_start_index: int
    effective_start_date: pd.Timestamp
    effective_end_index: int
    effective_end_date: pd.Timestamp
    window_start_index: int
    window_start_date: pd.Timestamp
    window_end_index: int
    window_end_date: pd.Timestamp
    window_observations: int


@dataclass(frozen=True)
class PrefixMoments:
    """Prefix sums used to compute window moments quickly."""

    return_sum: np.ndarray
    return_cross_sum: np.ndarray
    rfr_sum: np.ndarray


@dataclass(frozen=True)
class WindowStatistics:
    """Training-window statistics for one rebalance decision."""

    mean_returns: np.ndarray
    mean_excess_returns: np.ndarray
    avg_daily_rfr: float
    covariance: np.ndarray
    excess_returns: np.ndarray


CRYPTO_STAGE3_SPEC = Stage3CryptoSpec(
    provider="yahoo_crypto",
    display_name="Yahoo Finance Crypto",
    default_input_path=DEFAULT_STAGE2_FEATURE_PATH,
    stage2_label="run_beginner_stage2_features_long.py",
)


class UnsupportedModelConstraintError(ValueError):
    """Raised when one model is incompatible with one constraint mode."""


def stage3_data_dir() -> Path:
    """Return the default Stage 3 data directory."""

    return DEFAULT_STAGE3_DATA_ROOT


def stage3_table_dir() -> Path:
    """Return the default Stage 3 table directory."""

    return DEFAULT_STAGE3_TABLE_ROOT


def stage3_output_paths() -> dict[str, Path]:
    """Return the canonical Week 5 Stage 3 output paths."""

    data_dir = stage3_data_dir()
    table_dir = stage3_table_dir()
    return {
        "daily_weights": data_dir / "yahoo_crypto_oos_weights_daily.parquet",
        "daily_returns": data_dir / "yahoo_crypto_oos_portfolio_returns_daily.parquet",
        "ex_post_frontier": data_dir / "yahoo_crypto_oos_ex_post_frontier.parquet",
        "latest_target_weights": data_dir / "yahoo_crypto_oos_latest_target_weights.parquet",
        "latest_live_weights": data_dir / "yahoo_crypto_oos_latest_live_weights.parquet",
        "latest_risk_contributions": data_dir
        / "yahoo_crypto_oos_latest_risk_contributions.parquet",
        "rebalance_audit": data_dir / "yahoo_crypto_oos_rebalance_audit.parquet",
        "btc_eth_exposure_csv": table_dir / "yahoo_crypto_oos_latest_btc_eth_exposure.csv",
        "btc_eth_exposure_parquet": table_dir
        / "yahoo_crypto_oos_latest_btc_eth_exposure.parquet",
        "concentration_snapshot_csv": table_dir
        / "yahoo_crypto_oos_latest_concentration_metrics.csv",
        "concentration_snapshot_parquet": table_dir
        / "yahoo_crypto_oos_latest_concentration_metrics.parquet",
        "current_drawdown_snapshot_csv": table_dir
        / "yahoo_crypto_oos_current_drawdown_snapshot.csv",
        "current_drawdown_snapshot_parquet": table_dir
        / "yahoo_crypto_oos_current_drawdown_snapshot.parquet",
        "portfolio_metrics_csv": table_dir / "yahoo_crypto_oos_portfolio_metrics.csv",
        "portfolio_metrics_parquet": table_dir / "yahoo_crypto_oos_portfolio_metrics.parquet",
        "trailing_return_snapshot_csv": table_dir
        / "yahoo_crypto_oos_trailing_return_snapshot.csv",
        "trailing_return_snapshot_parquet": table_dir
        / "yahoo_crypto_oos_trailing_return_snapshot.parquet",
        "trailing_risk_snapshot_csv": table_dir
        / "yahoo_crypto_oos_trailing_risk_snapshot.csv",
        "trailing_risk_snapshot_parquet": table_dir
        / "yahoo_crypto_oos_trailing_risk_snapshot.parquet",
        "turnover_snapshot_csv": table_dir / "yahoo_crypto_oos_latest_turnover_snapshot.csv",
        "turnover_snapshot_parquet": table_dir
        / "yahoo_crypto_oos_latest_turnover_snapshot.parquet",
        "solve_summary_csv": table_dir / "yahoo_crypto_oos_solve_summary.csv",
        "solve_summary_parquet": table_dir / "yahoo_crypto_oos_solve_summary.parquet",
    }


def oos_figure_portfolio_columns() -> list[str]:
    """Return the canonical long-only Stage 3 figure columns."""

    return list(OOS_FIGURE_PORTFOLIO_COLUMN_ORDER)


def optimized_portfolio_columns() -> list[str]:
    """Return the optimized long-only portfolio columns only."""

    return list(OPTIMIZED_PORTFOLIO_COLUMN_ORDER)


def portfolio_display_label(portfolio_key: str) -> str:
    """Return the display label for one canonical Stage 3 portfolio key."""

    if portfolio_key not in OOS_FIGURE_PORTFOLIO_LABELS:
        raise ValueError(f"Unknown Stage 3 portfolio key: {portfolio_key}.")
    return OOS_FIGURE_PORTFOLIO_LABELS[portfolio_key]


def portfolio_key_for_model(model: str) -> str:
    """Return the canonical long-only Stage 3 portfolio key for one model."""

    if model not in PORTFOLIO_MODEL_TO_KEY:
        raise ValueError(f"Unknown Stage 3 model name: {model}.")
    return PORTFOLIO_MODEL_TO_KEY[model]


def latest_rebalance_dates(rebalance_audit: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the latest and previous formation dates from the Stage 3 audit."""

    decision_dates = sorted(pd.to_datetime(rebalance_audit["decision_date"]).dropna().unique())
    if len(decision_dates) < 2:
        raise ValueError("Need at least two rebalance dates to compare latest holdings changes.")
    return pd.Timestamp(decision_dates[-1]), pd.Timestamp(decision_dates[-2])


def load_stage2_feature_panel(
    *,
    panel_path: Path | None = None,
) -> tuple[pd.DataFrame, Stage3CryptoSpec]:
    """Load the Week 5 Stage 2 feature panel used as Stage 3 input."""

    source_path = panel_path or CRYPTO_STAGE3_SPEC.default_input_path
    if not source_path.exists():
        raise SystemExit(
            f"Missing Stage 2 feature panel: {source_path}. Run {CRYPTO_STAGE3_SPEC.stage2_label} "
            "first or pass --input-path."
        )

    frame = pd.read_parquet(source_path).copy()
    required = {"ticker", "date", "ret", "rfr"}
    missing = required.difference(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Stage 2 feature panel is missing required columns: {missing_text}.")

    frame["date"] = pd.to_datetime(frame["date"])
    frame["ret"] = pd.to_numeric(frame["ret"], errors="coerce")
    frame["rfr"] = pd.to_numeric(frame["rfr"], errors="coerce")
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    if frame[["ticker", "date"]].duplicated().any():
        raise ValueError("Stage 2 feature panel contains duplicate ticker-date keys.")
    return frame, CRYPTO_STAGE3_SPEC


def build_balanced_stage3_sample(
    feature_panel: pd.DataFrame,
    *,
    provider: str,
    display_name: str,
) -> Stage3OOSSample:
    """Build the balanced daily crypto return matrix and aligned risk-free series."""

    returns_wide = (
        feature_panel.pivot(index="date", columns="ticker", values="ret")
        .sort_index()
        .sort_index(axis=1)
    )
    rfr_timeline = feature_panel.groupby("date", sort=True)["rfr"].first().sort_index()
    valid_mask = (~returns_wide.isna().any(axis=1)) & rfr_timeline.notna()
    balanced_returns = returns_wide.loc[valid_mask].copy()
    balanced_rfr = rfr_timeline.loc[balanced_returns.index].copy()
    if balanced_returns.empty:
        raise ValueError("No balanced Stage 3 sample remains after dropping missing dates.")
    return Stage3OOSSample(
        provider=provider,
        display_name=display_name,
        returns_wide=balanced_returns,
        rfr=balanced_rfr,
    )


def _decision_indices_from_frequency(
    dates: pd.DatetimeIndex,
    estimation_frequency: str,
) -> np.ndarray:
    """Return rebalance decision indices for the requested frequency."""

    if estimation_frequency == "daily":
        return np.arange(len(dates), dtype=int)

    frame = pd.DataFrame({"date": pd.to_datetime(dates), "row_index": np.arange(len(dates))})
    if estimation_frequency == "weekly":
        groups = frame["date"].dt.to_period("W-SUN")
    elif estimation_frequency == "monthly":
        groups = frame["date"].dt.to_period("M")
    else:
        raise ValueError(f"Unknown estimation frequency: {estimation_frequency}")
    return frame.groupby(groups, sort=True)["row_index"].max().to_numpy(dtype=int)


def build_rebalance_schedule(
    dates: pd.DatetimeIndex,
    *,
    initial_window: int,
    estimation_frequency: str,
    window_rule: str,
) -> list[RebalanceWindow]:
    """Build the full decision-date schedule on the crypto daily calendar."""

    if len(dates) <= initial_window:
        raise ValueError(
            "The balanced Stage 3 sample is too short for the requested initial window."
        )

    decision_indices = _decision_indices_from_frequency(dates, estimation_frequency)
    eligible_indices = decision_indices[decision_indices >= initial_window - 1]
    if eligible_indices.size == 0:
        raise ValueError("No rebalance dates remain after applying the initial-window rule.")

    windows: list[RebalanceWindow] = []
    for position, decision_index in enumerate(eligible_indices):
        if decision_index + 1 >= len(dates):
            continue
        next_decision_index = (
            int(eligible_indices[position + 1])
            if position + 1 < len(eligible_indices)
            else len(dates) - 1
        )
        if window_rule == "expanding":
            window_start_index = 0
        elif window_rule == "rolling":
            window_start_index = decision_index - initial_window + 1
        else:
            raise ValueError(f"Unknown window rule: {window_rule}")
        window_end_index = int(decision_index)
        window_observations = window_end_index - window_start_index + 1
        windows.append(
            RebalanceWindow(
                decision_index=int(decision_index),
                decision_date=pd.Timestamp(dates[int(decision_index)]),
                effective_start_index=int(decision_index) + 1,
                effective_start_date=pd.Timestamp(dates[int(decision_index) + 1]),
                effective_end_index=int(next_decision_index),
                effective_end_date=pd.Timestamp(dates[int(next_decision_index)]),
                window_start_index=int(window_start_index),
                window_start_date=pd.Timestamp(dates[int(window_start_index)]),
                window_end_index=window_end_index,
                window_end_date=pd.Timestamp(dates[window_end_index]),
                window_observations=int(window_observations),
            )
        )
    if not windows:
        raise ValueError("No usable holding periods remain after scheduling rebalances.")
    return windows


def build_prefix_moments(returns_array: np.ndarray, rfr_array: np.ndarray) -> PrefixMoments:
    """Build prefix sums for fast expanding and rolling moment calculations."""

    returns = np.asarray(returns_array, dtype=float)
    rfr = np.asarray(rfr_array, dtype=float)
    cross_products = np.einsum("ti,tj->tij", returns, returns)

    return_sum = np.vstack(
        [np.zeros((1, returns.shape[1]), dtype=float), returns.cumsum(axis=0)]
    )
    return_cross_sum = np.concatenate(
        [
            np.zeros((1, returns.shape[1], returns.shape[1]), dtype=float),
            cross_products.cumsum(axis=0),
        ],
        axis=0,
    )
    rfr_sum = np.concatenate([[0.0], rfr.cumsum()])
    return PrefixMoments(
        return_sum=return_sum,
        return_cross_sum=return_cross_sum,
        rfr_sum=rfr_sum,
    )


def compute_window_statistics(
    prefix: PrefixMoments,
    returns_array: np.ndarray,
    rfr_array: np.ndarray,
    window: RebalanceWindow,
    *,
    covariance_ridge: float,
) -> WindowStatistics:
    """Compute mean, covariance, and excess-return inputs for one window."""

    start = window.window_start_index
    stop = window.window_end_index + 1
    observations = window.window_observations

    sum_returns = prefix.return_sum[stop] - prefix.return_sum[start]
    mean_returns = sum_returns / float(observations)

    sum_rfr = prefix.rfr_sum[stop] - prefix.rfr_sum[start]
    avg_daily_rfr = float(sum_rfr / float(observations))

    if observations == 1:
        covariance = np.zeros((returns_array.shape[1], returns_array.shape[1]), dtype=float)
    else:
        sum_cross = prefix.return_cross_sum[stop] - prefix.return_cross_sum[start]
        centered = sum_cross - observations * np.outer(mean_returns, mean_returns)
        covariance = centered / float(observations - 1)
    covariance = add_diagonal_ridge(covariance, covariance_ridge)

    excess_returns = returns_array[start:stop] - rfr_array[start:stop, None]
    mean_excess_returns = mean_returns - avg_daily_rfr
    return WindowStatistics(
        mean_returns=mean_returns,
        mean_excess_returns=mean_excess_returns,
        avg_daily_rfr=avg_daily_rfr,
        covariance=covariance,
        excess_returns=excess_returns,
    )


def _normalize_weights(weights: np.ndarray, *, long_only: bool) -> np.ndarray:
    """Normalize a weight vector to a fully invested portfolio."""

    vector = np.asarray(weights, dtype=float).copy()
    if long_only:
        vector = np.clip(vector, 0.0, None)
    total = float(vector.sum())
    if np.isclose(total, 0.0):
        raise ValueError("Optimization returned a zero-sum weight vector.")
    return vector / total


def _project_to_simplex(weights: np.ndarray) -> np.ndarray:
    """Project one vector onto the unit simplex."""

    vector = np.asarray(weights, dtype=float)
    if vector.ndim != 1:
        raise ValueError("Simplex projection expects a one-dimensional vector.")
    if vector.size == 1:
        return np.array([1.0], dtype=float)
    sorted_desc = np.sort(vector)[::-1]
    cumulative = np.cumsum(sorted_desc) - 1.0
    indices = np.arange(1, vector.size + 1, dtype=float)
    rho_candidates = np.nonzero(sorted_desc - cumulative / indices > 0.0)[0]
    if rho_candidates.size == 0:
        return equal_weight_vector(vector.size)
    rho = int(rho_candidates[-1])
    theta = cumulative[rho] / float(rho + 1)
    projected = np.maximum(vector - theta, 0.0)
    return _normalize_weights(projected, long_only=True)


def _sum_to_one_constraint(n_assets: int) -> LinearConstraint:
    """Return the fully-invested equality constraint."""

    return LinearConstraint(np.ones((1, n_assets), dtype=float), lb=1.0, ub=1.0)


def _long_only_bounds(n_assets: int) -> Bounds:
    """Return long-only box bounds."""

    return Bounds(np.zeros(n_assets, dtype=float), np.ones(n_assets, dtype=float))


def _initial_long_only_weights(
    n_assets: int,
    *,
    initial_weights: np.ndarray | None = None,
    fallback_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return one feasible long-only starting point."""

    for candidate in (initial_weights, fallback_weights):
        if candidate is None:
            continue
        try:
            return _project_to_simplex(np.asarray(candidate, dtype=float))
        except ValueError:
            continue
    return equal_weight_vector(n_assets)


def _initial_fully_invested_weights(
    n_assets: int,
    *,
    initial_weights: np.ndarray | None = None,
    fallback_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return one feasible fully invested starting point."""

    for candidate in (initial_weights, fallback_weights):
        if candidate is None:
            continue
        try:
            return _normalize_weights(np.asarray(candidate, dtype=float), long_only=False)
        except ValueError:
            continue
    return equal_weight_vector(n_assets)


def _variance_objective(weights: np.ndarray, covariance: np.ndarray) -> float:
    return float(weights @ covariance @ weights)


def _variance_gradient(weights: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    return 2.0 * (covariance @ weights)


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


def _negative_sharpe_gradient(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    covariance: np.ndarray,
    risk_free_rate: float,
) -> np.ndarray:
    excess_mean_vector = np.asarray(mean_returns, dtype=float) - float(risk_free_rate)
    numerator = float(weights @ excess_mean_vector)
    variance = float(weights @ covariance @ weights)
    volatility = float(np.sqrt(max(variance, OPTIMIZER_EPSILON)))
    covariance_times_weights = covariance @ weights
    return -(
        excess_mean_vector / volatility
        - numerator * covariance_times_weights / float(volatility**3)
    )


def _risk_parity_objective(weights: np.ndarray, covariance: np.ndarray) -> float:
    variance = float(weights @ covariance @ weights)
    if np.isclose(variance, 0.0):
        return 1e9
    marginal = covariance @ weights
    contributions = weights * marginal
    target = variance / float(len(weights))
    return float(np.sum((contributions - target) ** 2))


def _risk_parity_residual(weights: np.ndarray, covariance: np.ndarray) -> float:
    variance = float(weights @ covariance @ weights)
    if variance <= 0.0:
        return np.inf
    contributions = weights * (covariance @ weights)
    target = variance / float(len(weights))
    return float(np.max(np.abs(contributions - target)))


def long_only_minimum_variance_weights(
    covariance: np.ndarray,
    *,
    initial_weights: np.ndarray | None = None,
    tolerance: float,
    max_iter: int,
) -> np.ndarray:
    """Solve the long-only minimum-variance portfolio with analytic gradients."""

    matrix = np.asarray(covariance, dtype=float)
    n_assets = matrix.shape[0]
    x0 = _initial_long_only_weights(n_assets, initial_weights=initial_weights)
    result = minimize(
        _variance_objective,
        x0=x0,
        args=(matrix,),
        method="SLSQP",
        jac=_variance_gradient,
        bounds=_long_only_bounds(n_assets),
        constraints=[_sum_to_one_constraint(n_assets)],
        options={"maxiter": max_iter, "ftol": tolerance},
    )
    if not result.success:
        raise ValueError(f"Long-only minimum-variance optimization failed: {result.message}")
    return _project_to_simplex(np.asarray(result.x, dtype=float))


def long_only_tangency_weights(
    mean_returns: np.ndarray,
    covariance: np.ndarray,
    risk_free_rate: float,
    *,
    initial_weights: np.ndarray | None = None,
    tolerance: float,
    max_iter: int,
) -> np.ndarray:
    """Solve the long-only tangency portfolio with analytic gradients."""

    n_assets = covariance.shape[0]
    fallback_weights: np.ndarray | None = None
    try:
        fallback_vector, _method = tangency_weights(mean_returns, covariance, risk_free_rate)
        fallback_weights = fallback_vector
    except ValueError:
        fallback_weights = None
    x0 = _initial_long_only_weights(
        n_assets,
        initial_weights=initial_weights,
        fallback_weights=fallback_weights,
    )
    result = minimize(
        _negative_sharpe_objective,
        x0=x0,
        args=(mean_returns, covariance, risk_free_rate),
        method="SLSQP",
        jac=_negative_sharpe_gradient,
        bounds=_long_only_bounds(n_assets),
        constraints=[_sum_to_one_constraint(n_assets)],
        options={"maxiter": max_iter, "ftol": tolerance},
    )
    if not result.success:
        raise ValueError(f"Long-only mean-variance tangency optimization failed: {result.message}")
    return _project_to_simplex(np.asarray(result.x, dtype=float))


def risk_parity_weights(
    covariance: np.ndarray,
    *,
    initial_weights: np.ndarray | None = None,
    tolerance: float,
    max_iter: int,
) -> np.ndarray:
    """Solve the long-only volatility risk-parity portfolio via Newton steps."""

    matrix = np.asarray(covariance, dtype=float)
    n_assets = matrix.shape[0]
    budgets = equal_weight_vector(n_assets)
    inverse_vol = 1.0 / np.sqrt(np.clip(np.diag(matrix), OPTIMIZER_EPSILON, None))
    fallback_weights = inverse_vol / float(inverse_vol.sum())
    x = _initial_long_only_weights(
        n_assets,
        initial_weights=initial_weights,
        fallback_weights=fallback_weights,
    )
    x = np.clip(x, 1e-10, None)

    def objective(candidate: np.ndarray) -> float:
        return float(0.5 * candidate @ matrix @ candidate - budgets @ np.log(candidate))

    current_objective = objective(x)
    for _ in range(max_iter):
        gradient = matrix @ x - budgets / x
        hessian = matrix + np.diag(budgets / np.square(x))
        direction, _method = solve_markowitz_system(hessian, gradient)
        step = 1.0
        accepted = False
        directional_slope = float(gradient @ direction)
        while step >= 1e-8:
            candidate = x - step * direction
            if np.all(candidate > 0.0):
                candidate_objective = objective(candidate)
                if candidate_objective <= current_objective - 1e-4 * step * directional_slope:
                    x = candidate
                    current_objective = candidate_objective
                    accepted = True
                    break
            step *= 0.5
        if not accepted:
            break
        weights = _normalize_weights(x, long_only=True)
        if _risk_parity_residual(weights, matrix) <= tolerance:
            return weights
    return _normalize_weights(x, long_only=True)


def historical_cvar(losses: np.ndarray, *, alpha: float) -> float:
    """Return historical CVaR for one vector of scenario losses."""

    clean = np.asarray(losses, dtype=float)
    if clean.ndim != 1 or clean.size == 0:
        raise ValueError("CVaR requires a non-empty one-dimensional loss vector.")
    tail_count = max(1, int(np.ceil((1.0 - alpha) * clean.size)))
    ordered = np.sort(clean)
    return float(ordered[-tail_count:].mean())


def _cvar_tail_indices(losses: np.ndarray, *, alpha: float) -> np.ndarray:
    """Return the tail indices used by the historical CVaR estimate."""

    clean = np.asarray(losses, dtype=float)
    tail_count = max(1, int(np.ceil((1.0 - alpha) * clean.size)))
    return np.argpartition(clean, -tail_count)[-tail_count:]


def _mean_cvar_objective_and_gradient(
    weights: np.ndarray,
    scenarios: np.ndarray,
    *,
    alpha: float,
) -> tuple[float, np.ndarray]:
    """Return the negative mean-CVaR ratio and one exact tail subgradient."""

    portfolio_excess = scenarios @ weights
    mean_excess = float(portfolio_excess.mean())
    losses = -portfolio_excess
    tail_indices = _cvar_tail_indices(losses, alpha=alpha)
    cvar = max(float(losses[tail_indices].mean()), TAU_FLOOR)
    gradient_mean = scenarios.mean(axis=0)
    gradient_cvar = -scenarios[tail_indices].mean(axis=0)
    objective = -(mean_excess / cvar)
    gradient = -(gradient_mean * cvar - mean_excess * gradient_cvar) / float(cvar**2)
    return objective, gradient


def _fast_mean_cvar_tangency_weights(
    excess_returns: np.ndarray,
    *,
    alpha: float,
    long_only: bool,
    tolerance: float,
    max_iter: int,
    initial_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Solve the mean-CVaR ratio with SLSQP and one exact tail subgradient."""

    scenarios = np.asarray(excess_returns, dtype=float)
    n_assets = scenarios.shape[1]
    fallback_weights, _method = tangency_weights(
        scenarios.mean(axis=0),
        np.cov(scenarios, rowvar=False, ddof=1),
        0.0,
    )
    if long_only:
        x0 = _initial_long_only_weights(
            n_assets,
            initial_weights=initial_weights,
            fallback_weights=fallback_weights,
        )
        bounds = _long_only_bounds(n_assets)
    else:
        x0 = _initial_fully_invested_weights(
            n_assets,
            initial_weights=initial_weights,
            fallback_weights=fallback_weights,
        )
        bounds = None

    result = minimize(
        lambda weights, scenario_matrix, alpha_value: _mean_cvar_objective_and_gradient(
            weights,
            scenario_matrix,
            alpha=alpha_value,
        )[0],
        x0=x0,
        args=(scenarios, alpha),
        method="SLSQP",
        jac=lambda weights, scenario_matrix, alpha_value: _mean_cvar_objective_and_gradient(
            weights,
            scenario_matrix,
            alpha=alpha_value,
        )[1],
        bounds=bounds,
        constraints=[_sum_to_one_constraint(n_assets)],
        options={"maxiter": max_iter, "ftol": tolerance},
    )
    if not result.success:
        raise ValueError(f"Fast mean-CVaR optimization failed: {result.message}")
    return _normalize_weights(np.asarray(result.x, dtype=float), long_only=long_only)


def _linear_program_mean_cvar_tangency_weights(
    excess_returns: np.ndarray,
    *,
    alpha: float,
    long_only: bool,
    tolerance: float,
) -> np.ndarray:
    """Solve the exact mean-CVaR ratio via one sparse linear program."""

    scenarios = np.asarray(excess_returns, dtype=float)
    n_scenarios, n_assets = scenarios.shape
    tail_scale = 1.0 / ((1.0 - alpha) * n_scenarios)
    mean_excess = scenarios.mean(axis=0)

    n_variables = n_assets + 1 + n_scenarios + 1
    y_slice = slice(0, n_assets)
    z_index = n_assets
    v_slice = slice(n_assets + 1, n_assets + 1 + n_scenarios)
    tau_index = n_variables - 1

    objective = np.zeros(n_variables, dtype=float)
    objective[y_slice] = -mean_excess

    equality_rows = sparse.lil_matrix((2, n_variables), dtype=float)
    equality_rows[0, y_slice] = 1.0
    equality_rows[0, tau_index] = -1.0
    equality_rows[1, z_index] = 1.0
    equality_rows[1, v_slice] = tail_scale
    equality_rhs = np.array([0.0, 1.0], dtype=float)

    inequality_rows = sparse.hstack(
        [
            sparse.csr_matrix(-scenarios),
            sparse.csr_matrix(-np.ones((n_scenarios, 1), dtype=float)),
            -sparse.identity(n_scenarios, format="csr", dtype=float),
            sparse.csr_matrix((n_scenarios, 1), dtype=float),
        ],
        format="csr",
    )
    inequality_rhs = np.zeros(n_scenarios, dtype=float)

    bounds: list[tuple[float | None, float | None]] = []
    for _ in range(n_assets):
        bounds.append((0.0, None) if long_only else (None, None))
    bounds.append((None, None))
    bounds.extend((0.0, None) for _ in range(n_scenarios))
    bounds.append((TAU_FLOOR, None))

    result = linprog(
        objective,
        A_ub=inequality_rows,
        b_ub=inequality_rhs,
        A_eq=equality_rows.tocsr(),
        b_eq=equality_rhs,
        bounds=bounds,
        method="highs",
        options={
            "dual_feasibility_tolerance": float(tolerance),
            "primal_feasibility_tolerance": float(tolerance),
        },
    )
    if not result.success:
        raise ValueError(f"Mean-CVaR optimization failed: {result.message}")

    solution = np.asarray(result.x, dtype=float)
    tau = float(solution[tau_index])
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("Mean-CVaR optimization returned a non-positive scaling variable.")
    weights = solution[y_slice] / tau
    return _normalize_weights(weights, long_only=long_only)


def mean_cvar_tangency_weights(
    excess_returns: np.ndarray,
    *,
    alpha: float,
    long_only: bool,
    tolerance: float,
    max_iter: int,
    initial_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Solve the tangency-style mean-CVaR portfolio quickly, with an exact fallback."""

    scenarios = np.asarray(excess_returns, dtype=float)
    if scenarios.ndim != 2 or scenarios.shape[0] < 2:
        raise ValueError(
            "Mean-CVaR needs a two-dimensional scenario matrix with at least two rows."
        )
    try:
        return _fast_mean_cvar_tangency_weights(
            scenarios,
            alpha=alpha,
            long_only=long_only,
            tolerance=tolerance,
            max_iter=max_iter,
            initial_weights=initial_weights,
        )
    except ValueError:
        return _linear_program_mean_cvar_tangency_weights(
            scenarios,
            alpha=alpha,
            long_only=long_only,
            tolerance=tolerance,
        )


def _supported_model_constraint_pairs(
    config: Stage3OOSConfig,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return supported model-mode combinations for the long-only Week 5 surface."""

    supported = [
        (model, constraint_mode)
        for model in config.models
        for constraint_mode in config.constraint_modes
    ]
    return supported, []


def _infer_stage3_config_from_audit(rebalance_audit: pd.DataFrame) -> Stage3OOSConfig:
    """Infer the model and constraint selections from one rebalance-audit panel."""

    model_order = {model: index for index, model in enumerate(MODEL_ORDER)}
    mode_order = {mode: index for index, mode in enumerate(CONSTRAINT_MODES)}
    models = tuple(
        sorted(
            rebalance_audit["model"].dropna().astype(str).unique().tolist(),
            key=lambda value: model_order.get(value, len(model_order)),
        )
    )
    constraint_modes = tuple(
        sorted(
            rebalance_audit["constraint_mode"].dropna().astype(str).unique().tolist(),
            key=lambda value: mode_order.get(value, len(mode_order)),
        )
    )
    return Stage3OOSConfig(models=models, constraint_modes=constraint_modes)


def _portfolio_return_column_name(model: str, constraint_mode: str) -> str | None:
    """Return the canonical wide-panel return column name for one model-mode pair."""

    if model == "equal_weight":
        return "equal_weight"
    return f"{model}_{constraint_mode}"


def _portfolio_return_specs(config: Stage3OOSConfig) -> list[tuple[str, str, str]]:
    """Return the deterministic wide-panel return columns for one Stage 3 config."""

    specs: list[tuple[str, str, str]] = []
    if "equal_weight" in config.models:
        equal_mode = (
            "long_only"
            if "long_only" in config.constraint_modes
            else config.constraint_modes[0]
        )
        specs.append(("equal_weight", equal_mode, "equal_weight"))

    for model in MODEL_ORDER:
        if model == "equal_weight" or model not in config.models:
            continue
        for constraint_mode in config.constraint_modes:
            if constraint_mode not in config.constraint_modes:
                continue
            column_name = _portfolio_return_column_name(model, constraint_mode)
            if column_name is None:
                continue
            specs.append((model, constraint_mode, column_name))
    return specs


def _drifted_block_portfolio_returns(
    start_weights: np.ndarray,
    asset_returns: np.ndarray,
    *,
    formation_date: pd.Timestamp,
    return_dates: pd.DatetimeIndex,
    portfolio_name: str,
) -> np.ndarray:
    """Apply one target-weight vector through one holding block with within-block drift."""

    weights = np.asarray(start_weights, dtype=float).copy()
    block_returns = np.empty(asset_returns.shape[0], dtype=float)
    for day_index, daily_asset_returns in enumerate(asset_returns):
        portfolio_return = float(weights @ daily_asset_returns)
        block_returns[day_index] = portfolio_return
        if day_index + 1 >= asset_returns.shape[0]:
            continue
        gross_portfolio_value = 1.0 + portfolio_return
        if np.isclose(gross_portfolio_value, 0.0):
            raise ValueError(
                "Cannot drift portfolio weights after a zero gross portfolio value for "
                f"{portfolio_name} formed on {formation_date.date()} and earned on "
                f"{pd.Timestamp(return_dates[day_index]).date()}."
            )
        weights = weights * (1.0 + daily_asset_returns) / gross_portfolio_value
    return block_returns


def _drift_to_block_end_weights(
    start_weights: np.ndarray,
    asset_returns: np.ndarray,
    *,
    formation_date: pd.Timestamp,
    return_dates: pd.DatetimeIndex,
    portfolio_name: str,
) -> np.ndarray:
    """Drift one target-weight vector through a holding block and return end weights."""

    weights = np.asarray(start_weights, dtype=float).copy()
    for day_index, daily_asset_returns in enumerate(asset_returns):
        portfolio_return = float(weights @ daily_asset_returns)
        gross_portfolio_value = 1.0 + portfolio_return
        if np.isclose(gross_portfolio_value, 0.0):
            raise ValueError(
                "Cannot drift portfolio weights after a zero gross portfolio value for "
                f"{portfolio_name} formed on {formation_date.date()} and earned on "
                f"{pd.Timestamp(return_dates[day_index]).date()}."
            )
        weights = weights * (1.0 + daily_asset_returns) / gross_portfolio_value
    return weights


def compute_oos_portfolio_returns(
    sample: Stage3OOSSample,
    rebalance_audit: pd.DataFrame,
    *,
    config: Stage3OOSConfig | None = None,
) -> pd.DataFrame:
    """Build the daily out-of-sample portfolio return panel from the Stage 3 audit."""

    audit = rebalance_audit.copy()
    required = {
        "decision_date",
        "effective_start_date",
        "effective_end_date",
        "model",
        "constraint_mode",
        "ticker",
        "weight",
    }
    missing = required.difference(audit.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Rebalance audit is missing required columns: {missing_text}.")
    if audit.empty:
        raise ValueError("Rebalance audit is empty. Cannot build out-of-sample portfolio returns.")

    for column in ("decision_date", "effective_start_date", "effective_end_date"):
        audit[column] = pd.to_datetime(audit[column])
    audit["weight"] = pd.to_numeric(audit["weight"], errors="coerce")
    if audit["weight"].isna().any():
        raise ValueError("Rebalance audit contains non-numeric weights.")

    run_config = config or _infer_stage3_config_from_audit(audit)
    return_specs = _portfolio_return_specs(run_config)
    if not return_specs:
        raise ValueError("The Stage 3 return panel has no supported portfolio specifications.")

    returns_wide = sample.returns_wide.copy()
    returns_wide.index = pd.to_datetime(returns_wide.index)
    ticker_order = sample.tickers
    frames: list[pd.DataFrame] = []

    decision_dates = (
        audit["decision_date"].drop_duplicates().sort_values().to_list()
    )
    for decision_date in decision_dates:
        block_slice = audit.loc[audit["decision_date"] == decision_date].copy()
        effective_start = pd.Timestamp(block_slice["effective_start_date"].iloc[0])
        effective_end = pd.Timestamp(block_slice["effective_end_date"].iloc[0])
        holding_returns = returns_wide.loc[effective_start:effective_end]
        if holding_returns.empty:
            continue

        block_frame = pd.DataFrame(
            {
                "return_date": pd.DatetimeIndex(holding_returns.index),
                "formation_date": np.repeat(
                    np.datetime64(pd.Timestamp(decision_date)),
                    len(holding_returns),
                ),
            }
        )
        asset_returns = holding_returns.to_numpy(dtype=float)

        for model, constraint_mode, column_name in return_specs:
            pair_weights = block_slice.loc[
                (block_slice["model"] == model)
                & (block_slice["constraint_mode"] == constraint_mode),
                ["ticker", "weight"],
            ]
            if pair_weights.empty:
                continue
            ordered_weights = (
                pair_weights.set_index("ticker")
                .reindex(ticker_order)["weight"]
            )
            if ordered_weights.isna().any():
                raise ValueError(
                    f"Missing ticker weights for {column_name} on decision date "
                    f"{pd.Timestamp(decision_date).date()}."
                )
            block_frame[column_name] = _drifted_block_portfolio_returns(
                ordered_weights.to_numpy(dtype=float),
                asset_returns,
                formation_date=pd.Timestamp(decision_date),
                return_dates=pd.DatetimeIndex(holding_returns.index),
                portfolio_name=column_name,
            )
        frames.append(block_frame)

    if not frames:
        raise ValueError(
            "No out-of-sample portfolio returns were produced from the rebalance audit."
        )

    daily_returns = pd.concat(frames, ignore_index=True)
    ordered_columns = [
        "return_date",
        "formation_date",
        *[
            column_name
            for _model, _constraint_mode, column_name in return_specs
            if column_name in daily_returns.columns
        ],
    ]
    return (
        daily_returns.loc[:, ordered_columns]
        .sort_values(["return_date", "formation_date"])
        .reset_index(drop=True)
    )


def build_oos_window_sample(
    sample: Stage3OOSSample,
    portfolio_returns: pd.DataFrame,
) -> Stage3OOSSample:
    """Subset the balanced Stage 3 sample to the realized OOS return window."""

    if "return_date" not in portfolio_returns.columns:
        raise ValueError("Portfolio returns must contain a return_date column.")
    oos_dates = pd.DatetimeIndex(
        pd.to_datetime(portfolio_returns["return_date"]).drop_duplicates().sort_values()
    )
    if oos_dates.empty:
        raise ValueError("Portfolio returns contain no out-of-sample dates.")
    returns_wide = sample.returns_wide.reindex(oos_dates)
    rfr = sample.rfr.reindex(oos_dates)
    if returns_wide.isna().any().any() or rfr.isna().any():
        raise ValueError("The OOS return window does not align cleanly with the balanced sample.")
    return Stage3OOSSample(
        provider=sample.provider,
        display_name=sample.display_name,
        returns_wide=returns_wide,
        rfr=rfr,
    )


def wealth_index(returns: pd.Series) -> pd.Series:
    """Convert simple daily returns into a growth-of-one-dollar wealth index."""

    series = returns.astype(float).fillna(0.0)
    return (1.0 + series).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Convert simple daily returns into a drawdown series."""

    wealth = wealth_index(returns)
    return wealth / wealth.cummax() - 1.0


def _require_oos_figure_portfolio_columns(
    portfolio_returns: pd.DataFrame,
) -> list[str]:
    """Return the canonical long-only figure columns or raise a clear error."""

    columns = oos_figure_portfolio_columns()
    missing = [column for column in columns if column not in portfolio_returns.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            "Portfolio returns are missing the canonical long-only Stage 3 columns: "
            f"{missing_text}."
        )
    return columns


def summarize_oos_portfolio_metrics(
    oos_sample: Stage3OOSSample,
    portfolio_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize cumulative, annualized, and downside-risk metrics for OOS portfolios."""

    columns = _require_oos_figure_portfolio_columns(portfolio_returns)
    rfr = oos_sample.rfr.astype(float).reset_index(drop=True)
    if len(rfr) != len(portfolio_returns):
        raise ValueError("OOS sample and portfolio returns must have the same daily row count.")

    rows: list[dict[str, float | str]] = []
    for column in columns:
        returns = portfolio_returns[column].astype(float).reset_index(drop=True)
        excess = returns - rfr
        downside = excess.clip(upper=0.0)
        downside_deviation = float(np.sqrt(np.square(downside).mean()))
        annualized_return = float(returns.mean() * TRADING_DAYS_PER_YEAR)
        annualized_volatility = float(returns.std(ddof=1) * SQRT_365)
        sharpe_ratio = (
            float(SQRT_365 * excess.mean() / excess.std(ddof=1))
            if not np.isclose(float(excess.std(ddof=1)), 0.0)
            else np.nan
        )
        sortino_ratio = (
            float(SQRT_365 * excess.mean() / downside_deviation)
            if not np.isclose(downside_deviation, 0.0)
            else np.nan
        )
        cumulative_return = float(wealth_index(returns).iloc[-1] - 1.0)
        max_drawdown = float(drawdown_series(returns).min())
        rows.append(
            {
                "portfolio": OOS_FIGURE_PORTFOLIO_LABELS[column],
                "portfolio_key": column,
                "cumulative_return": cumulative_return,
                "cumulative_return_pct": cumulative_return * 100.0,
                "annualized_return": annualized_return,
                "annualized_return_pct": annualized_return * 100.0,
                "annualized_volatility": annualized_volatility,
                "annualized_volatility_pct": annualized_volatility * 100.0,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "max_drawdown": max_drawdown,
                "max_drawdown_pct": max_drawdown * 100.0,
            }
        )
    order = {column: index for index, column in enumerate(columns)}
    summary = pd.DataFrame(rows)
    summary["portfolio_order"] = summary["portfolio_key"].map(order)
    return (
        summary.sort_values("portfolio_order")
        .drop(columns="portfolio_order")
        .reset_index(drop=True)
    )


def summarize_oos_asset_statistics(oos_sample: Stage3OOSSample) -> pd.DataFrame:
    """Summarize annualized return, volatility, and Sharpe ratio for OOS assets."""

    daily_mean = oos_sample.returns_wide.mean(axis=0)
    daily_vol = oos_sample.returns_wide.std(axis=0, ddof=1)
    daily_excess_mean = oos_sample.returns_wide.sub(oos_sample.rfr, axis=0).mean(axis=0)
    sharpe = SQRT_365 * (daily_excess_mean / daily_vol.replace(0.0, np.nan))
    summary = pd.DataFrame(
        {
            "ticker": daily_mean.index,
            "annualized_return": daily_mean.to_numpy(dtype=float) * TRADING_DAYS_PER_YEAR,
            "annualized_volatility": daily_vol.to_numpy(dtype=float) * SQRT_365,
            "sharpe_ratio": sharpe.to_numpy(dtype=float),
        }
    )
    summary["annualized_return_pct"] = summary["annualized_return"] * 100.0
    summary["annualized_volatility_pct"] = summary["annualized_volatility"] * 100.0
    return summary.sort_values("ticker").reset_index(drop=True)


def build_oos_ex_post_frontier(
    oos_sample: Stage3OOSSample,
    *,
    n_points: int = 250,
) -> pd.DataFrame:
    """Build the ex post static efficient frontier from the realized OOS asset window."""

    returns = oos_sample.returns_wide.to_numpy(dtype=float)
    mean_returns = returns.mean(axis=0)
    covariance = np.cov(returns, rowvar=False, ddof=1)
    ones = np.ones(oos_sample.n_assets, dtype=float)
    inv_ones, _ = solve_markowitz_system(covariance, ones)
    inv_mu, _ = solve_markowitz_system(covariance, mean_returns)

    a_value = float(ones @ inv_ones)
    b_value = float(ones @ inv_mu)
    c_value = float(mean_returns @ inv_mu)
    determinant = a_value * c_value - b_value * b_value
    if np.isclose(determinant, 0.0):
        raise ValueError("Ex post efficient frontier determinant is zero.")

    gmv_weights, _method = minimum_variance_weights(covariance)
    gmv_return = float(mean_returns @ gmv_weights)
    try:
        tangency, _method = tangency_weights(mean_returns, covariance, float(oos_sample.rfr.mean()))
        tangency_return = float(mean_returns @ tangency)
    except ValueError:
        tangency_return = float(mean_returns.max())
    upper_anchor = max(float(mean_returns.max()), tangency_return, gmv_return)
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
            "volatility_ann": volatility_grid * SQRT_365,
            "volatility_ann_pct": volatility_grid * SQRT_365 * 100.0,
        }
    )
    return frontier


def build_top_target_weight_histories(
    rebalance_audit: pd.DataFrame,
    *,
    top_n: int = 5,
) -> dict[str, pd.DataFrame]:
    """Return top-target-weight time series by model using formation-date weights."""

    required = {"decision_date", "model", "constraint_mode", "ticker", "weight"}
    missing = required.difference(rebalance_audit.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Rebalance audit is missing required columns: {missing_text}.")
    histories: dict[str, pd.DataFrame] = {}
    for model in TOP_HOLDINGS_MODEL_ORDER:
        block = rebalance_audit.loc[
            (rebalance_audit["model"] == model)
            & (rebalance_audit["constraint_mode"] == "long_only"),
            ["decision_date", "ticker", "weight"],
        ].copy()
        if block.empty:
            raise ValueError(f"Missing long-only formation-date weights for model {model}.")
        block["decision_date"] = pd.to_datetime(block["decision_date"])
        pivot = (
            block.pivot(index="decision_date", columns="ticker", values="weight")
            .sort_index()
            .sort_index(axis=1)
        )
        top_tickers = (
            pivot.mean(axis=0)
            .sort_values(ascending=False)
            .head(top_n)
            .index
            .tolist()
        )
        histories[model] = pivot.loc[:, top_tickers].reset_index()
    return histories


def build_latest_target_weight_snapshot(rebalance_audit: pd.DataFrame) -> pd.DataFrame:
    """Return latest vs previous target weights for every long-only portfolio and ticker."""

    required = {"decision_date", "model", "constraint_mode", "ticker", "weight"}
    missing = required.difference(rebalance_audit.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Rebalance audit is missing required columns: {missing_text}.")

    audit = rebalance_audit.copy()
    audit["decision_date"] = pd.to_datetime(audit["decision_date"])
    audit["weight"] = pd.to_numeric(audit["weight"], errors="coerce")
    latest_date, previous_date = latest_rebalance_dates(audit)

    frames: list[pd.DataFrame] = []
    models = [
        "equal_weight",
        "minimum_variance",
        "mean_variance_tangency",
        "mean_cvar_tangency",
        "risk_parity_volatility",
    ]
    for model in models:
        latest_block = audit.loc[
            (audit["decision_date"] == latest_date)
            & (audit["model"] == model)
            & (audit["constraint_mode"] == "long_only"),
            ["ticker", "weight"],
        ].rename(columns={"weight": "latest_weight"})
        previous_block = audit.loc[
            (audit["decision_date"] == previous_date)
            & (audit["model"] == model)
            & (audit["constraint_mode"] == "long_only"),
            ["ticker", "weight"],
        ].rename(columns={"weight": "previous_weight"})
        merged = latest_block.merge(previous_block, on="ticker", how="outer")
        if merged.empty:
            raise ValueError(f"Missing latest or previous target weights for {model}.")
        merged["latest_weight"] = merged["latest_weight"].fillna(0.0)
        merged["previous_weight"] = merged["previous_weight"].fillna(0.0)
        portfolio_key = portfolio_key_for_model(model)
        merged["decision_date"] = latest_date
        merged["previous_decision_date"] = previous_date
        merged["model"] = model
        merged["constraint_mode"] = "long_only"
        merged["portfolio_key"] = portfolio_key
        merged["portfolio"] = portfolio_display_label(portfolio_key)
        merged["weight_change"] = merged["latest_weight"] - merged["previous_weight"]
        frames.append(merged)

    order = {column: index for index, column in enumerate(OOS_FIGURE_PORTFOLIO_COLUMN_ORDER)}
    snapshot = pd.concat(frames, ignore_index=True)
    snapshot["portfolio_order"] = snapshot["portfolio_key"].map(order)
    return (
        snapshot.sort_values(
            ["portfolio_order", "latest_weight", "ticker"],
            ascending=[True, False, True],
        )
        .drop(columns="portfolio_order")
        .reset_index(drop=True)
    )


def build_latest_live_weight_snapshot(
    sample: Stage3OOSSample,
    rebalance_audit: pd.DataFrame,
    *,
    config: Stage3OOSConfig | None = None,
) -> pd.DataFrame:
    """Return end-of-day drifted portfolio weights on the latest earned return date."""

    audit = rebalance_audit.copy()
    required = {
        "decision_date",
        "effective_start_date",
        "effective_end_date",
        "model",
        "constraint_mode",
        "ticker",
        "weight",
    }
    missing = required.difference(audit.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Rebalance audit is missing required columns: {missing_text}.")

    for column in ("decision_date", "effective_start_date", "effective_end_date"):
        audit[column] = pd.to_datetime(audit[column])
    audit["weight"] = pd.to_numeric(audit["weight"], errors="coerce")
    if audit["weight"].isna().any():
        raise ValueError("Rebalance audit contains non-numeric weights.")

    run_config = config or _infer_stage3_config_from_audit(audit)
    return_specs = _portfolio_return_specs(run_config)
    latest_return_date = pd.Timestamp(sample.returns_wide.index.max())
    latest_blocks = audit.loc[audit["effective_end_date"] == latest_return_date].copy()
    if latest_blocks.empty:
        raise ValueError("Could not locate the latest holding block in the Stage 3 audit.")

    frames: list[pd.DataFrame] = []
    returns_wide = sample.returns_wide.copy()
    returns_wide.index = pd.to_datetime(returns_wide.index)
    ticker_order = sample.tickers
    for model, constraint_mode, portfolio_key in return_specs:
        block = latest_blocks.loc[
            (latest_blocks["model"] == model)
            & (latest_blocks["constraint_mode"] == constraint_mode)
        ]
        if block.empty:
            continue
        formation_date = pd.Timestamp(block["decision_date"].iloc[0])
        effective_start = pd.Timestamp(block["effective_start_date"].iloc[0])
        holding_returns = returns_wide.loc[effective_start:latest_return_date]
        if holding_returns.empty:
            raise ValueError(
                f"No holding returns remain for {portfolio_key} after {formation_date.date()}."
            )
        ordered_weights = (
            block.set_index("ticker").reindex(ticker_order)["weight"]
        )
        if ordered_weights.isna().any():
            raise ValueError(
                f"Missing ticker weights for {portfolio_key} on decision date "
                f"{formation_date.date()}."
            )
        live_weights = _drift_to_block_end_weights(
            ordered_weights.to_numpy(dtype=float),
            holding_returns.to_numpy(dtype=float),
            formation_date=formation_date,
            return_dates=pd.DatetimeIndex(holding_returns.index),
            portfolio_name=portfolio_key,
        )
        frame = pd.DataFrame(
            {
                "return_date": np.repeat(np.datetime64(latest_return_date), len(ticker_order)),
                "formation_date": np.repeat(np.datetime64(formation_date), len(ticker_order)),
                "model": np.repeat(model, len(ticker_order)),
                "constraint_mode": np.repeat(constraint_mode, len(ticker_order)),
                "portfolio_key": np.repeat(portfolio_key, len(ticker_order)),
                "portfolio": np.repeat(portfolio_display_label(portfolio_key), len(ticker_order)),
                "ticker": ticker_order,
                "weight": live_weights,
            }
        )
        frames.append(frame)

    if not frames:
        raise ValueError("No latest live Stage 3 weight snapshots were produced.")

    order = {column: index for index, column in enumerate(OOS_FIGURE_PORTFOLIO_COLUMN_ORDER)}
    live_snapshot = pd.concat(frames, ignore_index=True)
    live_snapshot["portfolio_order"] = live_snapshot["portfolio_key"].map(order)
    return (
        live_snapshot.sort_values(
            ["portfolio_order", "weight", "ticker"],
            ascending=[True, False, True],
        )
        .drop(columns="portfolio_order")
        .reset_index(drop=True)
    )


def build_factsheet_concentration_snapshot(
    latest_live_weights: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize concentration and diversification metrics from the latest live weights."""

    required = {"return_date", "portfolio_key", "portfolio", "ticker", "weight"}
    missing = required.difference(latest_live_weights.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Latest live weights are missing required columns: {missing_text}.")

    rows: list[dict[str, object]] = []
    for portfolio_key in OOS_FIGURE_PORTFOLIO_COLUMN_ORDER:
        block = latest_live_weights.loc[
            latest_live_weights["portfolio_key"] == portfolio_key
        ].copy()
        if block.empty:
            raise ValueError(f"Missing latest live weights for {portfolio_key}.")
        weights = np.sort(block["weight"].astype(float).to_numpy())[::-1]
        rows.append(
            {
                "return_date": pd.Timestamp(block["return_date"].iloc[0]),
                "portfolio_key": portfolio_key,
                "portfolio": portfolio_display_label(portfolio_key),
                "top_1_weight": float(weights[:1].sum()),
                "top_3_weight": float(weights[:3].sum()),
                "top_5_weight": float(weights[:5].sum()),
                "effective_n": float(1.0 / np.square(weights).sum()),
            }
        )
    summary = pd.DataFrame(rows)
    summary["top_1_weight_pct"] = summary["top_1_weight"] * 100.0
    summary["top_3_weight_pct"] = summary["top_3_weight"] * 100.0
    summary["top_5_weight_pct"] = summary["top_5_weight"] * 100.0
    return summary


def _window_covariance_from_audit(
    sample: Stage3OOSSample,
    audit_block: pd.DataFrame,
) -> np.ndarray:
    """Return the sample covariance implied by one audit block's training window."""

    start_date = pd.Timestamp(audit_block["window_start_date"].iloc[0])
    end_date = pd.Timestamp(audit_block["window_end_date"].iloc[0])
    window_returns = sample.returns_wide.loc[start_date:end_date]
    if len(window_returns) < 2:
        raise ValueError(
            "Need at least two observations to compute the latest training covariance."
        )
    return np.cov(window_returns.to_numpy(dtype=float), rowvar=False, ddof=1)


def build_factsheet_risk_contribution_snapshot(
    sample: Stage3OOSSample,
    latest_target_weights: pd.DataFrame,
    rebalance_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Return the latest target-weight risk-contribution table for optimized models."""

    latest_decision_date = pd.Timestamp(latest_target_weights["decision_date"].iloc[0])
    audit = rebalance_audit.copy()
    audit["decision_date"] = pd.to_datetime(audit["decision_date"])
    audit["window_start_date"] = pd.to_datetime(audit["window_start_date"])
    audit["window_end_date"] = pd.to_datetime(audit["window_end_date"])

    frames: list[pd.DataFrame] = []
    for portfolio_key in OPTIMIZED_PORTFOLIO_COLUMN_ORDER:
        model = PORTFOLIO_KEY_TO_MODEL[portfolio_key]
        weight_block = latest_target_weights.loc[
            latest_target_weights["portfolio_key"] == portfolio_key,
            ["ticker", "latest_weight", "portfolio"],
        ].copy()
        audit_block = audit.loc[
            (audit["decision_date"] == latest_decision_date)
            & (audit["model"] == model)
            & (audit["constraint_mode"] == "long_only"),
        ]
        if weight_block.empty or audit_block.empty:
            raise ValueError(f"Missing latest target block for {portfolio_key}.")
        covariance = _window_covariance_from_audit(sample, audit_block)
        ordered_weights = (
            weight_block.set_index("ticker")
            .reindex(sample.tickers)["latest_weight"]
            .to_numpy(dtype=float)
        )
        marginal = covariance @ ordered_weights
        portfolio_variance = float(ordered_weights @ marginal)
        if np.isclose(portfolio_variance, 0.0):
            raise ValueError(
                f"Latest training covariance implies zero variance for {portfolio_key}."
            )
        contribution = ordered_weights * marginal
        frame = pd.DataFrame(
            {
                "decision_date": np.repeat(
                    np.datetime64(latest_decision_date),
                    len(sample.tickers),
                ),
                "portfolio_key": np.repeat(portfolio_key, len(sample.tickers)),
                "portfolio": np.repeat(portfolio_display_label(portfolio_key), len(sample.tickers)),
                "model": np.repeat(model, len(sample.tickers)),
                "ticker": sample.tickers,
                "weight": ordered_weights,
                "risk_contribution": contribution,
                "risk_contribution_pct": contribution / portfolio_variance * 100.0,
            }
        )
        frames.append(frame)
    snapshot = pd.concat(frames, ignore_index=True)
    return (
        snapshot.sort_values(
            ["portfolio", "risk_contribution_pct"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )


def build_factsheet_trailing_return_snapshot(
    portfolio_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize trailing realized OOS returns over app-style factsheet windows."""

    columns = _require_oos_figure_portfolio_columns(portfolio_returns)
    latest_date = pd.Timestamp(pd.to_datetime(portfolio_returns["return_date"]).max())
    rows: list[dict[str, object]] = []
    for portfolio_key in columns:
        series = portfolio_returns[portfolio_key].astype(float).reset_index(drop=True)
        for horizon in FACTSHEET_TRAILING_WINDOWS:
            window = series.tail(horizon)
            cumulative = float((1.0 + window).prod() - 1.0)
            rows.append(
                {
                    "as_of_date": latest_date,
                    "portfolio_key": portfolio_key,
                    "portfolio": portfolio_display_label(portfolio_key),
                    "window_label": f"{horizon}-day",
                    "window_days": horizon,
                    "cumulative_return": cumulative,
                    "cumulative_return_pct": cumulative * 100.0,
                }
            )
        cumulative = float((1.0 + series).prod() - 1.0)
        rows.append(
            {
                "as_of_date": latest_date,
                "portfolio_key": portfolio_key,
                "portfolio": portfolio_display_label(portfolio_key),
                "window_label": "Since inception",
                "window_days": len(series),
                "cumulative_return": cumulative,
                "cumulative_return_pct": cumulative * 100.0,
            }
        )
    return pd.DataFrame(rows)


def build_factsheet_current_drawdown_snapshot(
    portfolio_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Return current drawdowns from the realized OOS portfolio paths."""

    columns = _require_oos_figure_portfolio_columns(portfolio_returns)
    latest_date = pd.Timestamp(pd.to_datetime(portfolio_returns["return_date"]).max())
    rows: list[dict[str, object]] = []
    for portfolio_key in columns:
        drawdown = drawdown_series(portfolio_returns[portfolio_key].astype(float))
        current_drawdown = float(drawdown.iloc[-1])
        rows.append(
            {
                "as_of_date": latest_date,
                "portfolio_key": portfolio_key,
                "portfolio": portfolio_display_label(portfolio_key),
                "current_drawdown": current_drawdown,
                "current_drawdown_pct": current_drawdown * 100.0,
            }
        )
    return pd.DataFrame(rows)


def build_factsheet_trailing_risk_snapshot(
    oos_sample: Stage3OOSSample,
    portfolio_returns: pd.DataFrame,
    *,
    window_days: int = FACTSHEET_RISK_WINDOW_DAYS,
) -> pd.DataFrame:
    """Return the latest trailing realized risk snapshot for all long-only portfolios."""

    columns = _require_oos_figure_portfolio_columns(portfolio_returns)
    latest_date = pd.Timestamp(pd.to_datetime(portfolio_returns["return_date"]).max())
    trailing_rfr = (
        oos_sample.rfr.astype(float)
        .reset_index(drop=True)
        .tail(window_days)
        .reset_index(drop=True)
    )
    rows: list[dict[str, object]] = []
    for portfolio_key in columns:
        returns = (
            portfolio_returns[portfolio_key].astype(float).reset_index(drop=True).tail(window_days).reset_index(drop=True)
        )
        excess = returns - trailing_rfr
        downside = excess.clip(upper=0.0)
        downside_deviation = float(np.sqrt(np.square(downside).mean()))
        daily_vol = float(returns.std(ddof=1))
        sharpe = (
            float(SQRT_365 * excess.mean() / excess.std(ddof=1))
            if not np.isclose(float(excess.std(ddof=1)), 0.0)
            else np.nan
        )
        sortino = (
            float(SQRT_365 * excess.mean() / downside_deviation)
            if not np.isclose(downside_deviation, 0.0)
            else np.nan
        )
        cvar_95 = float(historical_cvar(-returns.to_numpy(dtype=float), alpha=0.95))
        rows.append(
            {
                "as_of_date": latest_date,
                "window_days": window_days,
                "portfolio_key": portfolio_key,
                "portfolio": portfolio_display_label(portfolio_key),
                "annualized_volatility": daily_vol * SQRT_365,
                "annualized_volatility_pct": daily_vol * SQRT_365 * 100.0,
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "cvar_95_loss": cvar_95,
                "cvar_95_loss_pct": cvar_95 * 100.0,
            }
        )
    return pd.DataFrame(rows)


def build_factsheet_turnover_snapshot(
    latest_target_weights: pd.DataFrame,
) -> pd.DataFrame:
    """Return latest one-way turnover by fund from previous to current target weights."""

    required = {
        "portfolio_key",
        "portfolio",
        "decision_date",
        "previous_decision_date",
        "weight_change",
    }
    missing = required.difference(latest_target_weights.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Latest target weights are missing required columns: {missing_text}.")

    rows: list[dict[str, object]] = []
    for portfolio_key in OOS_FIGURE_PORTFOLIO_COLUMN_ORDER:
        block = latest_target_weights.loc[
            latest_target_weights["portfolio_key"] == portfolio_key
        ].copy()
        if block.empty:
            raise ValueError(f"Missing latest target weights for {portfolio_key}.")
        turnover = 0.5 * float(block["weight_change"].abs().sum())
        rows.append(
            {
                "decision_date": pd.Timestamp(block["decision_date"].iloc[0]),
                "previous_decision_date": pd.Timestamp(block["previous_decision_date"].iloc[0]),
                "portfolio_key": portfolio_key,
                "portfolio": portfolio_display_label(portfolio_key),
                "turnover": turnover,
                "turnover_pct": turnover * 100.0,
            }
        )
    return pd.DataFrame(rows)


def build_factsheet_btc_eth_exposure_snapshot(
    latest_live_weights: pd.DataFrame,
) -> pd.DataFrame:
    """Return current BTC, ETH, and other exposure from the latest live weights."""

    rows: list[dict[str, object]] = []
    latest_return_date = pd.Timestamp(latest_live_weights["return_date"].iloc[0])
    for portfolio_key in OOS_FIGURE_PORTFOLIO_COLUMN_ORDER:
        block = latest_live_weights.loc[
            latest_live_weights["portfolio_key"] == portfolio_key
        ].copy()
        btc_weight = float(block.loc[block["ticker"] == "BTC-USD", "weight"].sum())
        eth_weight = float(block.loc[block["ticker"] == "ETH-USD", "weight"].sum())
        btc_eth_weight = btc_weight + eth_weight
        rows.append(
            {
                "return_date": latest_return_date,
                "portfolio_key": portfolio_key,
                "portfolio": portfolio_display_label(portfolio_key),
                "btc_weight": btc_weight,
                "eth_weight": eth_weight,
                "btc_eth_weight": btc_eth_weight,
                "other_weight": float(1.0 - btc_eth_weight),
                "btc_weight_pct": btc_weight * 100.0,
                "eth_weight_pct": eth_weight * 100.0,
                "btc_eth_weight_pct": btc_eth_weight * 100.0,
                "other_weight_pct": float(1.0 - btc_eth_weight) * 100.0,
            }
        )
    return pd.DataFrame(rows)


def estimate_one_window_weights(
    statistics: WindowStatistics,
    *,
    model: str,
    constraint_mode: str,
    config: Stage3OOSConfig,
    previous_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, str, str]:
    """Estimate one model's weights for one rebalance window."""

    if constraint_mode != "long_only":
        raise UnsupportedModelConstraintError(
            "Week 5 Stage 3 now supports long-only estimation only."
        )
    if model == "equal_weight":
        return equal_weight_vector(statistics.covariance.shape[0]), "direct", "ok"

    if model == "minimum_variance":
        weights = long_only_minimum_variance_weights(
            statistics.covariance,
            initial_weights=previous_weights,
            tolerance=config.solver_tolerance,
            max_iter=config.solver_max_iter,
        )
        return weights, "SLSQP_jac", "ok"

    if model == "mean_variance_tangency":
        weights = long_only_tangency_weights(
            statistics.mean_returns,
            statistics.covariance,
            statistics.avg_daily_rfr,
            initial_weights=previous_weights,
            tolerance=config.solver_tolerance,
            max_iter=config.solver_max_iter,
        )
        return weights, "SLSQP_jac", "ok"

    if model == "mean_cvar_tangency":
        weights = mean_cvar_tangency_weights(
            statistics.excess_returns,
            alpha=config.cvar_alpha,
            long_only=True,
            tolerance=config.solver_tolerance,
            max_iter=config.solver_max_iter,
            initial_weights=previous_weights,
        )
        return weights, "SLSQP_subgradient_or_sparse_HiGHS", "ok"

    if model == "risk_parity_volatility":
        weights = risk_parity_weights(
            statistics.covariance,
            initial_weights=previous_weights,
            tolerance=config.solver_tolerance,
            max_iter=config.solver_max_iter,
        )
        return weights, "newton_risk_budgeting", "ok"

    raise ValueError(f"Unknown model name: {model}")


def _expand_holding_weight_rows(
    *,
    holding_dates: pd.DatetimeIndex,
    decision_date: pd.Timestamp,
    model: str,
    constraint_mode: str,
    tickers: list[str],
    weights: np.ndarray,
) -> pd.DataFrame:
    """Expand one rebalance vector into holding-date rows."""

    n_holding_dates = len(holding_dates)
    n_assets = len(tickers)
    return pd.DataFrame(
        {
            "date": np.repeat(holding_dates.to_numpy(), n_assets),
            "decision_date": np.repeat(
                np.datetime64(pd.Timestamp(decision_date)),
                n_holding_dates * n_assets,
            ),
            "model": np.repeat(model, n_holding_dates * n_assets),
            "constraint_mode": np.repeat(constraint_mode, n_holding_dates * n_assets),
            "ticker": np.tile(np.asarray(tickers, dtype=object), n_holding_dates),
            "weight": np.tile(np.asarray(weights, dtype=float), n_holding_dates),
        }
    )


def _rebalance_audit_rows(
    *,
    window: RebalanceWindow,
    model: str,
    constraint_mode: str,
    tickers: list[str],
    weights: np.ndarray,
    solver: str,
    status: str,
    elapsed_ms: float,
) -> pd.DataFrame:
    """Return one ticker-level rebalance audit frame."""

    n_assets = len(tickers)
    return pd.DataFrame(
        {
            "decision_date": np.repeat(np.datetime64(window.decision_date), n_assets),
            "effective_start_date": np.repeat(np.datetime64(window.effective_start_date), n_assets),
            "effective_end_date": np.repeat(np.datetime64(window.effective_end_date), n_assets),
            "window_start_date": np.repeat(np.datetime64(window.window_start_date), n_assets),
            "window_end_date": np.repeat(np.datetime64(window.window_end_date), n_assets),
            "window_observations": np.repeat(window.window_observations, n_assets),
            "model": np.repeat(model, n_assets),
            "constraint_mode": np.repeat(constraint_mode, n_assets),
            "ticker": tickers,
            "weight": np.asarray(weights, dtype=float),
            "solver": np.repeat(solver, n_assets),
            "status": np.repeat(status, n_assets),
            "elapsed_ms": np.repeat(float(elapsed_ms), n_assets),
        }
    )


def _build_solve_summary(
    solve_meta: pd.DataFrame,
    *,
    unsupported_pairs: list[tuple[str, str]],
    config: Stage3OOSConfig,
) -> pd.DataFrame:
    """Summarize solve counts and timings by model and constraint mode."""

    if solve_meta.empty:
        raise ValueError("Solve metadata is empty. No supported portfolio solves were produced.")

    summary = (
        solve_meta.groupby(["model", "constraint_mode"], as_index=False)
        .agg(
            rebalance_count=("decision_date", "nunique"),
            first_decision_date=("decision_date", "min"),
            last_decision_date=("decision_date", "max"),
            mean_window_observations=("window_observations", "mean"),
            min_window_observations=("window_observations", "min"),
            max_window_observations=("window_observations", "max"),
            mean_elapsed_ms=("elapsed_ms", "mean"),
            max_elapsed_ms=("elapsed_ms", "max"),
            solver=("solver", lambda values: ", ".join(sorted(set(values)))),
            status=("status", lambda values: ", ".join(sorted(set(values)))),
        )
        .reset_index(drop=True)
    )
    if unsupported_pairs:
        unsupported_frame = pd.DataFrame(
            {
                "model": [model for model, _mode in unsupported_pairs],
                "constraint_mode": [mode for _model, mode in unsupported_pairs],
                "rebalance_count": 0,
                "first_decision_date": pd.NaT,
                "last_decision_date": pd.NaT,
                "mean_window_observations": np.nan,
                "min_window_observations": np.nan,
                "max_window_observations": np.nan,
                "mean_elapsed_ms": np.nan,
                "max_elapsed_ms": np.nan,
                "solver": "n/a",
                "status": "unsupported",
            }
        )
        summary = pd.concat([summary, unsupported_frame], ignore_index=True)

    model_order = {model: index for index, model in enumerate(MODEL_ORDER)}
    mode_order = {mode: index for index, mode in enumerate(config.constraint_modes)}
    summary["model_order"] = summary["model"].map(model_order).fillna(len(model_order))
    summary["mode_order"] = summary["constraint_mode"].map(mode_order).fillna(len(mode_order))
    summary = (
        summary.sort_values(["model_order", "mode_order"])
        .drop(columns=["model_order", "mode_order"])
        .reset_index(drop=True)
    )
    return summary


def generate_oos_weight_panels(
    sample: Stage3OOSSample,
    *,
    config: Stage3OOSConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate the daily holding-date weights, rebalance audit, and solve summary."""

    run_config = config or Stage3OOSConfig()
    schedule = build_rebalance_schedule(
        sample.dates,
        initial_window=run_config.initial_window,
        estimation_frequency=run_config.estimation_frequency,
        window_rule=run_config.window_rule,
    )

    supported_pairs, unsupported_pairs = _supported_model_constraint_pairs(run_config)
    if not supported_pairs:
        raise ValueError(
            "The requested Stage 3 configuration contains no supported model combinations."
        )

    returns_array = sample.returns_wide.to_numpy(dtype=float)
    rfr_array = sample.rfr.to_numpy(dtype=float)
    prefix = build_prefix_moments(returns_array, rfr_array)

    holding_frames: list[pd.DataFrame] = []
    audit_frames: list[pd.DataFrame] = []
    solve_meta_records: list[dict[str, object]] = []
    previous_weights: dict[tuple[str, str], np.ndarray] = {}

    for window in schedule:
        statistics = compute_window_statistics(
            prefix,
            returns_array,
            rfr_array,
            window,
            covariance_ridge=run_config.covariance_ridge,
        )
        holding_dates = sample.dates[window.effective_start_index : window.effective_end_index + 1]

        for model, constraint_mode in supported_pairs:
            start_time = perf_counter()
            weights, solver, status = estimate_one_window_weights(
                statistics,
                model=model,
                constraint_mode=constraint_mode,
                config=run_config,
                previous_weights=previous_weights.get((model, constraint_mode)),
            )
            elapsed_ms = (perf_counter() - start_time) * 1000.0
            previous_weights[(model, constraint_mode)] = weights.copy()

            holding_frames.append(
                _expand_holding_weight_rows(
                    holding_dates=holding_dates,
                    decision_date=window.decision_date,
                    model=model,
                    constraint_mode=constraint_mode,
                    tickers=sample.tickers,
                    weights=weights,
                )
            )
            audit_frames.append(
                _rebalance_audit_rows(
                    window=window,
                    model=model,
                    constraint_mode=constraint_mode,
                    tickers=sample.tickers,
                    weights=weights,
                    solver=solver,
                    status=status,
                    elapsed_ms=elapsed_ms,
                )
            )
            solve_meta_records.append(
                {
                    "decision_date": window.decision_date,
                    "effective_start_date": window.effective_start_date,
                    "effective_end_date": window.effective_end_date,
                    "window_start_date": window.window_start_date,
                    "window_end_date": window.window_end_date,
                    "window_observations": window.window_observations,
                    "model": model,
                    "constraint_mode": constraint_mode,
                    "solver": solver,
                    "status": status,
                    "elapsed_ms": elapsed_ms,
                }
            )

    daily_weights = (
        pd.concat(holding_frames, ignore_index=True)
        .sort_values(["date", "model", "constraint_mode", "ticker"])
        .reset_index(drop=True)
    )
    rebalance_audit = (
        pd.concat(audit_frames, ignore_index=True)
        .sort_values(["decision_date", "model", "constraint_mode", "ticker"])
        .reset_index(drop=True)
    )
    solve_summary = _build_solve_summary(
        pd.DataFrame(solve_meta_records),
        unsupported_pairs=unsupported_pairs,
        config=run_config,
    )
    return daily_weights, rebalance_audit, solve_summary


def balanced_sample_summary(
    sample: Stage3OOSSample,
    *,
    schedule: list[RebalanceWindow] | None = None,
) -> dict[str, object]:
    """Return a compact Stage 3 sample summary for script output and docs."""

    summary: dict[str, object] = {
        "provider": sample.display_name,
        "n_assets": sample.n_assets,
        "sample_days": sample.sample_days,
        "start_date": sample.start_date,
        "end_date": sample.end_date,
        "mean_daily_rfr": float(sample.rfr.mean()),
    }
    if schedule:
        summary["first_decision_date"] = schedule[0].decision_date
        summary["last_decision_date"] = schedule[-1].decision_date
        summary["rebalance_count"] = len(schedule)
    return summary
