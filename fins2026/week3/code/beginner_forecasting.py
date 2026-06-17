# ruff: noqa
"""Beginner forecasting helpers for the Week 3 lecture ladder."""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
)
from statsmodels.tools.sm_exceptions import (
    ValueWarning as StatsmodelsValueWarning,
)
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from fins2026.week3.code.forecast_data import load_forecast_input_bundle

DEFAULT_REAL_SERIES_NAME = "Unemployment rate"
DEFAULT_SIMULATION_START = "2006-01-31"
DEFAULT_SIMULATION_PERIODS = 240
DEFAULT_RANDOM_SEED = 2026
DEFAULT_TRAIN_END = "2019-12-31"
DEFAULT_TEST_PERIODS = 24
DEFAULT_AR_LAGS = [1, 3, 6, 12]
DEFAULT_ARMA_ORDERS = [(1, 0), (2, 0), (1, 1), (2, 1)]
DEFAULT_ARX_COLUMNS = ["cash_rate_change_pp"]
DEFAULT_ARX_PLUS_COLUMNS = ["cash_rate_change_pp", "commodity_price_log_change_pct"]
DEFAULT_US_ARX_COLUMNS = ["fedfunds_change_pp"]
DEFAULT_US_ARX_PLUS_COLUMNS = ["fedfunds_change_pp", "yield_spread_pp"]
DEFAULT_REGRESSION_TARGET_LAGS = [1, 3, 6, 12]
DEFAULT_OLS_COLUMNS = [
    "cash_rate_change_pp",
    "commodity_price_log_change_pct",
    "trade_weighted_index_log_change_pct",
    "participation_change_pp",
    "vacancies_to_labour_force_change_pp",
]
DEFAULT_ENET_COLUMNS = [
    *DEFAULT_OLS_COLUMNS,
    "employment_to_population_change_pp",
    "headline_cpi_inflation",
    "trimmed_mean_inflation",
    "wage_price_index_growth",
    "us_fedfunds_change_pp",
    "us_unemployment_change_pp",
    "us_indpro_log_growth_pct",
    "us_vix_level",
    "us_yield_spread_pp",
]
DEFAULT_ENET_ALPHA_GRID = [0.001, 0.01, 0.1, 1.0]
DEFAULT_ENET_L1_WT = 0.5
DEFAULT_VARIANT_VALIDATION_PERIODS = 12
DEFAULT_DATA_DIR = Path("fins2026/week3/results/data/beginner_forecasting")
DEFAULT_FIGURES_DIR = Path("fins2026/week3/results/figures/beginner_forecasting")
DEFAULT_TABLES_DIR = Path("fins2026/week3/results/tables/beginner_forecasting")
DEFAULT_US_DATA_DIR = Path("fins2026/week3/results/data/us_beginner_forecasting")
DEFAULT_US_FIGURES_DIR = Path("fins2026/week3/results/figures/us_beginner_forecasting")
DEFAULT_US_TABLES_DIR = Path("fins2026/week3/results/tables/us_beginner_forecasting")


@contextmanager
def suppress_expected_statsmodels_warnings():
    """Hide routine statsmodels warnings from the student-facing lecture scripts."""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=StatsmodelsValueWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", message="No supported index is available.*")
        warnings.filterwarnings(
            "ignore",
            message=(
                "A date index has been provided, but it has no associated "
                "frequency information.*"
            ),
        )
        yield


def add_holdout_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the standard Week 3 evaluation-window arguments to a script parser."""

    parser.add_argument(
        "--train-end",
        default=DEFAULT_TRAIN_END,
        help=(
            "Final in-sample month-end. The first later observation begins the "
            "out-of-sample evaluation window."
        ),
    )
    parser.add_argument(
        "--test-periods",
        type=int,
        default=None,
        help=(
            "Optional override: use the final N monthly observations as the "
            "out-of-sample evaluation window instead of the fixed train-end split."
        ),
    )


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repo root from a file or directory inside the repo."""

    current = (start or Path(__file__)).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "tools" / "workflow_lib.py"
        ).is_file():
            return candidate
    raise RuntimeError("Could not find the fins-agent repo root.")


def resolve_repo_path(path: str | Path, repo_root: Path | None = None) -> Path:
    """Resolve a repo-relative or absolute path."""

    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return (repo_root or find_repo_root()).resolve() / resolved


def _clean_series(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if not isinstance(clean.index, pd.DatetimeIndex):
        clean.index = pd.to_datetime(clean.index)
    return clean.sort_index()


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.copy()
    if not isinstance(clean.index, pd.DatetimeIndex):
        clean.index = pd.to_datetime(clean.index)
    for column in clean.columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    return clean.sort_index()


def _backtest_start_offset(
    index: pd.Index,
    *,
    train_end: str | pd.Timestamp = DEFAULT_TRAIN_END,
    test_periods: int | None = None,
    min_train_periods: int = 12,
) -> int:
    """Return the first out-of-sample position for a date- or count-based split."""

    dates = pd.DatetimeIndex(pd.to_datetime(index)).sort_values()
    if test_periods is not None:
        if test_periods < 1:
            raise ValueError("test_periods must be at least 1")
        start = len(dates) - int(test_periods)
        if start < min_train_periods:
            raise ValueError("series is too short for the requested backtest window")
        return start

    cutoff = pd.Timestamp(train_end)
    positions = np.flatnonzero(dates > cutoff)
    if positions.size == 0:
        raise ValueError("no out-of-sample observations are available after train_end")
    start = int(positions[0])
    if start < min_train_periods:
        raise ValueError("series is too short before train_end for a stable backtest")
    return start


def split_dates(
    series: pd.Series,
    *,
    train_end: str | pd.Timestamp = DEFAULT_TRAIN_END,
    test_periods: int | None = None,
    min_train_periods: int = 12,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the last in-sample date and the first out-of-sample date."""

    clean = _clean_series(series)
    start = _backtest_start_offset(
        clean.index,
        train_end=train_end,
        test_periods=test_periods,
        min_train_periods=min_train_periods,
    )
    return pd.Timestamp(clean.index[start - 1]), pd.Timestamp(clean.index[start])


def sample_label(series: pd.Series) -> str:
    """Return a compact sample label for a time series."""

    clean = _clean_series(series)
    return f"{clean.index.min():%Y-%m-%d} to {clean.index.max():%Y-%m-%d}"


def load_fixture_unemployment_rate() -> pd.Series:
    """Load the Australia unemployment rate from the committed Week 3 fixture path."""

    bundle = load_forecast_input_bundle(use_fixture=True, rebuild=False)
    series = bundle["australia_monthly"][DEFAULT_REAL_SERIES_NAME]
    clean = _clean_series(series)
    clean.name = "unemployment_rate"
    return clean


def load_fixture_us_unemployment_rate() -> pd.Series:
    """Load the U.S. unemployment rate from the committed Week 3 fixture path."""

    bundle = load_forecast_input_bundle(use_fixture=True, rebuild=False)
    series = bundle["us_monthly"]["UNRATE"]
    clean = _clean_series(series)
    clean.name = "unemployment_rate"
    return clean


def build_beginner_macro_panel() -> pd.DataFrame:
    """Build the real-data lecture panel used for ARX, OLS, and elastic net."""

    bundle = load_forecast_input_bundle(use_fixture=True, rebuild=False)
    monthly = bundle["australia_monthly"].copy()
    us_monthly = bundle["us_monthly"].copy()
    panel = pd.DataFrame(index=monthly.index)
    panel["unemployment_rate"] = monthly["Unemployment rate"]
    panel["cash_rate_target"] = monthly["Cash rate target"]
    panel["commodity_price_index_aud"] = monthly["Commodity price index (A$)"]
    panel["trade_weighted_index"] = monthly["Trade-weighted index"]
    panel["participation_rate"] = monthly["Participation rate"]
    panel["employment_to_population_ratio"] = monthly["Employment-to-population ratio"]
    panel["vacancies_to_labour_force_ratio"] = monthly["Vacancies to labour force ratio"]
    panel["headline_cpi_inflation"] = monthly["Headline CPI inflation"]
    panel["trimmed_mean_inflation"] = monthly["Trimmed mean inflation"]
    panel["wage_price_index_growth"] = monthly["Wage Price Index growth"]
    panel["job_vacancies"] = monthly["Job vacancies"]
    panel["unemployment_change_pp"] = panel["unemployment_rate"].diff()
    panel["cash_rate_change_pp"] = panel["cash_rate_target"].diff()
    panel["commodity_price_log_change_pct"] = (
        np.log(panel["commodity_price_index_aud"] / panel["commodity_price_index_aud"].shift(1))
        * 100.0
    )
    panel["trade_weighted_index_log_change_pct"] = (
        np.log(panel["trade_weighted_index"] / panel["trade_weighted_index"].shift(1)) * 100.0
    )
    panel["participation_change_pp"] = panel["participation_rate"].diff()
    panel["employment_to_population_change_pp"] = (
        panel["employment_to_population_ratio"].diff()
    )
    panel["vacancies_to_labour_force_change_pp"] = (
        panel["vacancies_to_labour_force_ratio"].diff()
    )
    panel["job_vacancies_log_change_pct"] = (
        np.log(panel["job_vacancies"] / panel["job_vacancies"].shift(1)) * 100.0
    )
    panel["us_fedfunds_rate"] = us_monthly["FEDFUNDS"].reindex(panel.index)
    panel["us_fedfunds_change_pp"] = panel["us_fedfunds_rate"].diff()
    panel["us_unemployment_rate"] = us_monthly["UNRATE"].reindex(panel.index)
    panel["us_unemployment_change_pp"] = panel["us_unemployment_rate"].diff()
    panel["us_indpro_log_growth_pct"] = us_monthly["INDPRO_LOG_GROWTH_PCT"].reindex(panel.index)
    panel["us_vix_level"] = us_monthly["VIXCLS"].reindex(panel.index)
    panel["us_yield_spread_pp"] = us_monthly["T10Y2Y"].reindex(panel.index)
    panel.index.name = "date"
    return _clean_frame(panel)


def build_us_beginner_macro_panel() -> pd.DataFrame:
    """Build the U.S. lecture panel used for the Week 3 extension path."""

    bundle = load_forecast_input_bundle(use_fixture=True, rebuild=False)
    us_monthly = bundle["us_monthly"].copy()
    panel = pd.DataFrame(index=us_monthly.index)
    panel["unemployment_rate"] = us_monthly["UNRATE"]
    panel["fedfunds_rate"] = us_monthly["FEDFUNDS"]
    panel["treasury_10y_rate"] = us_monthly["DGS10"]
    panel["yield_spread_pp"] = us_monthly["T10Y2Y"]
    panel["vix_level"] = us_monthly["VIXCLS"]
    panel["industrial_production"] = us_monthly["INDPRO"]
    panel["sp500_level"] = us_monthly["SP500"]
    panel["unemployment_change_pp"] = panel["unemployment_rate"].diff()
    panel["fedfunds_change_pp"] = panel["fedfunds_rate"].diff()
    panel["treasury_10y_change_pp"] = panel["treasury_10y_rate"].diff()
    panel["indpro_log_growth_pct"] = us_monthly["INDPRO_LOG_GROWTH_PCT"]
    panel["sp500_return_pct"] = us_monthly["SP500_RETURN_PCT"]
    panel.index.name = "date"
    return _clean_frame(panel)


def monthly_change(series: pd.Series) -> pd.Series:
    """Return the one-period change of a monthly series."""

    return _clean_series(series).diff().dropna()


def make_simulated_series_frame(
    *,
    start: str = DEFAULT_SIMULATION_START,
    periods: int = DEFAULT_SIMULATION_PERIODS,
    seed: int = DEFAULT_RANDOM_SEED,
    burn_in: int = 60,
) -> pd.DataFrame:
    """Return deterministic simulated monthly series for the forecasting lecture."""

    if periods < 24:
        raise ValueError("periods must be at least 24")

    dates = pd.date_range(start=start, periods=periods, freq="ME")
    rng = np.random.default_rng(seed)

    random_walk_shocks = rng.normal(loc=0.0, scale=0.30, size=periods)
    random_walk = np.cumsum(random_walk_shocks)

    ar1_shocks = rng.normal(loc=0.0, scale=0.50, size=periods + burn_in)
    stationary_ar1 = np.zeros(periods + burn_in, dtype=float)
    for index in range(1, periods + burn_in):
        stationary_ar1[index] = 0.60 * stationary_ar1[index - 1] + ar1_shocks[index]

    arma_shocks = rng.normal(loc=0.0, scale=0.50, size=periods + burn_in)
    stationary_arma11 = np.zeros(periods + burn_in, dtype=float)
    for index in range(1, periods + burn_in):
        stationary_arma11[index] = (
            0.50 * stationary_arma11[index - 1]
            + arma_shocks[index]
            + 0.40 * arma_shocks[index - 1]
        )

    frame = pd.DataFrame(
        {
            "date": dates,
            "random_walk": random_walk,
            "stationary_ar1": stationary_ar1[burn_in:],
            "stationary_arma11": stationary_arma11[burn_in:],
        }
    )
    return frame


def write_beginner_source_tables(
    output_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    repo_root: Path | None = None,
) -> dict[str, Path]:
    """Write the canonical beginner lecture CSVs and return their paths."""

    output_root = resolve_repo_path(output_dir, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    unemployment = load_fixture_unemployment_rate().rename("unemployment_rate")
    unemployment_path = output_root / "week3_beginner_unemployment_rate.csv"
    unemployment.reset_index().rename(columns={"index": "date"}).to_csv(
        unemployment_path,
        index=False,
    )

    macro_panel = build_beginner_macro_panel()
    macro_panel_path = output_root / "week3_beginner_macro_panel.csv"
    macro_panel.reset_index().to_csv(macro_panel_path, index=False)

    simulated = make_simulated_series_frame()
    simulated_path = output_root / "week3_beginner_simulated_series.csv"
    simulated.to_csv(simulated_path, index=False)

    return {
        "unemployment_rate": unemployment_path,
        "macro_panel": macro_panel_path,
        "simulated_series": simulated_path,
    }


def write_us_beginner_source_tables(
    output_dir: str | Path = DEFAULT_US_DATA_DIR,
    *,
    repo_root: Path | None = None,
) -> dict[str, Path]:
    """Write the canonical U.S. beginner lecture CSVs and return their paths."""

    output_root = resolve_repo_path(output_dir, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)

    unemployment = load_fixture_us_unemployment_rate().rename("unemployment_rate")
    unemployment_path = output_root / "week3_beginner_unemployment_rate.csv"
    unemployment.reset_index().rename(columns={"index": "date"}).to_csv(
        unemployment_path,
        index=False,
    )

    macro_panel = build_us_beginner_macro_panel()
    macro_panel_path = output_root / "week3_beginner_macro_panel.csv"
    macro_panel.reset_index().to_csv(macro_panel_path, index=False)

    simulated = make_simulated_series_frame()
    simulated_path = output_root / "week3_beginner_simulated_series.csv"
    simulated.to_csv(simulated_path, index=False)

    return {
        "unemployment_rate": unemployment_path,
        "macro_panel": macro_panel_path,
        "simulated_series": simulated_path,
    }


def ensure_beginner_source_tables(
    output_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    repo_root: Path | None = None,
) -> dict[str, Path]:
    """Create the canonical beginner source tables if they are missing."""

    output_root = resolve_repo_path(output_dir, repo_root)
    unemployment_path = output_root / "week3_beginner_unemployment_rate.csv"
    macro_panel_path = output_root / "week3_beginner_macro_panel.csv"
    simulated_path = output_root / "week3_beginner_simulated_series.csv"
    if unemployment_path.exists() and macro_panel_path.exists() and simulated_path.exists():
        return {
            "unemployment_rate": unemployment_path,
            "macro_panel": macro_panel_path,
            "simulated_series": simulated_path,
        }
    return write_beginner_source_tables(output_root, repo_root=repo_root)


def ensure_us_beginner_source_tables(
    output_dir: str | Path = DEFAULT_US_DATA_DIR,
    *,
    repo_root: Path | None = None,
) -> dict[str, Path]:
    """Create the canonical U.S. beginner source tables if they are missing."""

    output_root = resolve_repo_path(output_dir, repo_root)
    unemployment_path = output_root / "week3_beginner_unemployment_rate.csv"
    macro_panel_path = output_root / "week3_beginner_macro_panel.csv"
    simulated_path = output_root / "week3_beginner_simulated_series.csv"
    if unemployment_path.exists() and macro_panel_path.exists() and simulated_path.exists():
        return {
            "unemployment_rate": unemployment_path,
            "macro_panel": macro_panel_path,
            "simulated_series": simulated_path,
        }
    return write_us_beginner_source_tables(output_root, repo_root=repo_root)


def load_saved_unemployment_rate(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    repo_root: Path | None = None,
) -> pd.Series:
    """Load the saved beginner unemployment-rate CSV."""

    path = resolve_repo_path(data_dir, repo_root) / "week3_beginner_unemployment_rate.csv"
    frame = pd.read_csv(path, parse_dates=["date"])
    series = frame.set_index("date")["unemployment_rate"]
    clean = _clean_series(series)
    clean.name = "unemployment_rate"
    return clean


def load_saved_simulated_series(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    repo_root: Path | None = None,
) -> pd.DataFrame:
    """Load the saved beginner simulated-series CSV."""

    path = resolve_repo_path(data_dir, repo_root) / "week3_beginner_simulated_series.csv"
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return frame.sort_index()


def load_saved_beginner_macro_panel(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    repo_root: Path | None = None,
) -> pd.DataFrame:
    """Load the saved real-data lecture panel."""

    path = resolve_repo_path(data_dir, repo_root) / "week3_beginner_macro_panel.csv"
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return _clean_frame(frame)


def adf_summary(series: pd.Series, *, name: str, version: str) -> dict[str, object]:
    """Return a plain-English ADF summary row."""

    clean = _clean_series(series)
    statistic, p_value, used_lag, observations, _, _ = adfuller(clean, autolag="AIC")
    reject = bool(p_value < 0.05)
    conclusion = (
        "Treat as stationary for this lecture."
        if reject
        else "Treat as non-stationary for this lecture."
    )
    return {
        "series": name,
        "version": version,
        "observations": int(observations),
        "used_lag": int(used_lag),
        "adf_statistic": float(statistic),
        "p_value": float(p_value),
        "reject_unit_root_5pct": reject,
        "teaching_conclusion": conclusion,
    }


def add_level_columns(
    backtest: pd.DataFrame,
    level_series: pd.Series,
) -> pd.DataFrame:
    """Attach previous, actual, and forecast level paths to a change backtest."""

    level = _clean_series(level_series)
    frame = backtest.copy()
    previous_level = level.shift(1).reindex(frame.index)
    actual_level = level.reindex(frame.index)
    frame["previous_level"] = previous_level
    frame["actual_level"] = actual_level
    frame["forecast_level"] = previous_level + frame["forecast"]
    frame["level_error"] = frame["actual_level"] - frame["forecast_level"]
    frame["absolute_level_error"] = frame["level_error"].abs()
    return frame


def rolling_one_step_backtest(
    target_series: pd.Series,
    level_series: pd.Series,
    *,
    forecast_fn: Callable[[pd.Series], float],
    train_end: str | pd.Timestamp = DEFAULT_TRAIN_END,
    test_periods: int | None = None,
) -> pd.DataFrame:
    """Run an expanding-window one-step backtest and rebuild implied levels."""

    target = _clean_series(target_series)
    level = _clean_series(level_series)
    start = _backtest_start_offset(target.index, train_end=train_end, test_periods=test_periods)

    rows: list[dict[str, float | pd.Timestamp]] = []
    for offset in range(start, len(target)):
        train = target.iloc[:offset]
        prediction = float(forecast_fn(train))
        actual = float(target.iloc[offset])
        actual_date = target.index[offset]
        rows.append(
            {
                "date": actual_date,
                "actual": actual,
                "forecast": prediction,
                "error": actual - prediction,
                "absolute_error": abs(actual - prediction),
            }
        )
    backtest = pd.DataFrame(rows).set_index("date")
    return add_level_columns(backtest, level)


def rolling_one_step_backtest_with_exog(
    target_series: pd.Series,
    level_series: pd.Series,
    exog_frame: pd.DataFrame,
    *,
    forecast_fn: Callable[[pd.Series, pd.DataFrame, pd.DataFrame], float],
    train_end: str | pd.Timestamp = DEFAULT_TRAIN_END,
    test_periods: int | None = None,
) -> pd.DataFrame:
    """Run an expanding-window one-step backtest with exogenous features."""

    target = _clean_series(target_series)
    level = _clean_series(level_series)
    exog = _clean_frame(exog_frame)
    merged = pd.concat([target.rename("__target__"), exog], axis=1, join="inner").dropna()

    aligned_target = merged.pop("__target__").astype(float)
    aligned_exog = merged.astype(float)
    start = _backtest_start_offset(
        aligned_target.index,
        train_end=train_end,
        test_periods=test_periods,
    )

    rows: list[dict[str, float | pd.Timestamp]] = []
    for offset in range(start, len(aligned_target)):
        train_target = aligned_target.iloc[:offset]
        train_exog = aligned_exog.iloc[:offset]
        future_exog = aligned_exog.iloc[offset : offset + 1]
        prediction = float(forecast_fn(train_target, train_exog, future_exog))
        actual = float(aligned_target.iloc[offset])
        actual_date = aligned_target.index[offset]
        rows.append(
            {
                "date": actual_date,
                "actual": actual,
                "forecast": prediction,
                "error": actual - prediction,
                "absolute_error": abs(actual - prediction),
            }
        )
    backtest = pd.DataFrame(rows).set_index("date")
    return add_level_columns(backtest, level)


def naive_one_step_forecast(train: pd.Series) -> float:
    """Return the one-step naive forecast."""

    clean = _clean_series(train)
    return float(clean.iloc[-1])


def ar_candidate_table(
    target_series: pd.Series,
    candidate_lags: list[int] | None = None,
) -> pd.DataFrame:
    """Return a BIC table for the AR lag ladder."""

    target = _clean_series(target_series)
    rows: list[dict[str, object]] = []
    for lag in candidate_lags or DEFAULT_AR_LAGS:
        if lag < 1 or len(target) <= lag + 4:
            rows.append(
                {
                    "lag": lag,
                    "aic": np.nan,
                    "bic": np.nan,
                    "status": "too_short",
                }
            )
            continue
        try:
            with suppress_expected_statsmodels_warnings():
                fit = AutoReg(target.to_numpy(dtype=float), lags=lag, old_names=False).fit()
        except Exception:
            rows.append({"lag": lag, "aic": np.nan, "bic": np.nan, "status": "fit_failed"})
            continue
        rows.append(
            {
                "lag": lag,
                "aic": float(fit.aic),
                "bic": float(fit.bic),
                "status": "ok",
            }
        )
    return pd.DataFrame(rows).sort_values("lag").reset_index(drop=True)


def choose_best_ar_lag(
    target_series: pd.Series,
    candidate_lags: list[int] | None = None,
) -> tuple[int, pd.DataFrame]:
    """Choose the best AR lag by BIC and return the full selection table."""

    table = ar_candidate_table(target_series, candidate_lags)
    valid = table.loc[table["status"] == "ok"].sort_values(["bic", "lag"])
    if valid.empty:
        raise ValueError("could not fit any AR candidate")
    return int(valid.iloc[0]["lag"]), table


def ar_one_step_forecast(train: pd.Series, *, lag: int) -> float:
    """Return a one-step forecast from a fixed-lag autoregression."""

    clean = _clean_series(train)
    with suppress_expected_statsmodels_warnings():
        fit = AutoReg(clean.to_numpy(dtype=float), lags=lag, old_names=False).fit()
        prediction = fit.predict(start=len(clean), end=len(clean))
    return float(np.asarray(prediction, dtype=float)[0])


def _arx_design(
    target_series: pd.Series,
    exog_frame: pd.DataFrame,
    *,
    lag: int,
) -> tuple[pd.DataFrame, pd.Series]:
    target = _clean_series(target_series)
    exog = _clean_frame(exog_frame)
    design = pd.DataFrame(index=target.index)
    for step in range(1, lag + 1):
        design[f"lag_{step}"] = target.shift(step)
    for column in exog.columns:
        design[column] = exog[column]
    merged = pd.concat([target.rename("__target__"), design], axis=1, join="inner").dropna()
    response = merged.pop("__target__").astype(float)
    return merged.astype(float), response


def arx_one_step_forecast(
    train_target: pd.Series,
    train_exog: pd.DataFrame,
    future_exog: pd.DataFrame,
    *,
    lag: int,
) -> float:
    """Return a one-step forecast from an ARX model fit by OLS."""

    design, response = _arx_design(train_target, train_exog, lag=lag)
    with suppress_expected_statsmodels_warnings():
        model = sm.OLS(response, sm.add_constant(design, has_constant="add")).fit()

    row = pd.DataFrame(index=future_exog.index)
    for step in range(1, lag + 1):
        row[f"lag_{step}"] = float(_clean_series(train_target).iloc[-step])
    for column in train_exog.columns:
        row[column] = float(future_exog.iloc[0][column])

    with suppress_expected_statsmodels_warnings():
        prediction = model.predict(sm.add_constant(row, has_constant="add"))
    return float(np.asarray(prediction, dtype=float)[0])


def arma_candidate_table(
    target_series: pd.Series,
    candidate_orders: list[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Return a BIC table for the ARMA order ladder."""

    target = _clean_series(target_series)
    rows: list[dict[str, object]] = []
    for order in candidate_orders or DEFAULT_ARMA_ORDERS:
        p_order, q_order = order
        if len(target) <= max(p_order, q_order) + 6:
            rows.append(
                {
                    "order": f"({p_order},{q_order})",
                    "p": p_order,
                    "q": q_order,
                    "aic": np.nan,
                    "bic": np.nan,
                    "status": "too_short",
                }
            )
            continue
        try:
            with suppress_expected_statsmodels_warnings():
                fit = ARIMA(
                    target,
                    order=(p_order, 0, q_order),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit()
        except Exception:
            rows.append(
                {
                    "order": f"({p_order},{q_order})",
                    "p": p_order,
                    "q": q_order,
                    "aic": np.nan,
                    "bic": np.nan,
                    "status": "fit_failed",
                }
            )
            continue
        rows.append(
            {
                "order": f"({p_order},{q_order})",
                "p": p_order,
                "q": q_order,
                "aic": float(fit.aic),
                "bic": float(fit.bic),
                "status": "ok",
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def choose_best_arma_order(
    target_series: pd.Series,
    candidate_orders: list[tuple[int, int]] | None = None,
) -> tuple[tuple[int, int], pd.DataFrame]:
    """Choose the best ARMA order by BIC and return the full selection table."""

    table = arma_candidate_table(target_series, candidate_orders)
    valid = table.loc[table["status"] == "ok"].sort_values(["bic", "p", "q"])
    if valid.empty:
        raise ValueError("could not fit any ARMA candidate")
    row = valid.iloc[0]
    return (int(row["p"]), int(row["q"])), table


def arma_one_step_forecast(train: pd.Series, *, order: tuple[int, int]) -> float:
    """Return a one-step forecast from a fixed ARMA(p, q) model."""

    clean = _clean_series(train)
    with suppress_expected_statsmodels_warnings():
        fit = ARIMA(
            clean,
            order=(order[0], 0, order[1]),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()
        forecast = fit.forecast(steps=1)
    return float(np.asarray(forecast, dtype=float)[0])


def armax_one_step_forecast(
    train_target: pd.Series,
    train_exog: pd.DataFrame,
    future_exog: pd.DataFrame,
    *,
    order: tuple[int, int],
) -> float:
    """Return a one-step forecast from an ARMAX model."""

    with suppress_expected_statsmodels_warnings():
        fit = ARIMA(
            _clean_series(train_target),
            exog=_clean_frame(train_exog),
            order=(order[0], 0, order[1]),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()
        prediction = fit.forecast(steps=1, exog=_clean_frame(future_exog))
    return float(np.asarray(prediction, dtype=float)[0])


def _dynamic_regression_design(
    target_series: pd.Series,
    exog_frame: pd.DataFrame,
    *,
    target_lags: list[int],
) -> tuple[pd.DataFrame, pd.Series]:
    """Build a lagged regression design with target lags plus outside variables."""

    target = _clean_series(target_series)
    exog = _clean_frame(exog_frame)
    design = pd.DataFrame(index=target.index)
    unique_lags = sorted(set(target_lags))
    for lag in unique_lags:
        design[f"target_lag_{lag}"] = target.shift(lag)
    for column in exog.columns:
        design[column] = exog[column]
    merged = pd.concat([target.rename("__target__"), design], axis=1, join="inner").dropna()
    response = merged.pop("__target__").astype(float)
    return merged.astype(float), response


def _dynamic_regression_future_row(
    train_target: pd.Series,
    future_exog: pd.DataFrame,
    *,
    target_lags: list[int],
) -> pd.DataFrame:
    """Build the one-row design matrix used for the next forecast date."""

    clean_target = _clean_series(train_target)
    clean_future_exog = _clean_frame(future_exog)
    row = pd.DataFrame(index=clean_future_exog.index)
    unique_lags = sorted(set(target_lags))
    for lag in unique_lags:
        row[f"target_lag_{lag}"] = float(clean_target.iloc[-lag])
    for column in clean_future_exog.columns:
        row[column] = float(clean_future_exog.iloc[0][column])
    ordered = [f"target_lag_{lag}" for lag in unique_lags] + list(clean_future_exog.columns)
    return row.reindex(columns=ordered).astype(float)


def fit_dynamic_ols(
    target_series: pd.Series,
    exog_frame: pd.DataFrame,
    *,
    target_lags: list[int] | None = None,
) -> tuple[object, pd.DataFrame, pd.Series]:
    """Fit a simple dynamic OLS model with lagged target terms."""

    design, response = _dynamic_regression_design(
        target_series,
        exog_frame,
        target_lags=target_lags or DEFAULT_REGRESSION_TARGET_LAGS,
    )
    with suppress_expected_statsmodels_warnings():
        fit = sm.OLS(response, sm.add_constant(design, has_constant="add")).fit()
    return fit, design, response


def ols_one_step_forecast(
    train_target: pd.Series,
    train_exog: pd.DataFrame,
    future_exog: pd.DataFrame,
    *,
    target_lags: list[int] | None = None,
) -> float:
    """Return a one-step forecast from a dynamic OLS model."""

    selected_lags = target_lags or DEFAULT_REGRESSION_TARGET_LAGS
    fit, _, _ = fit_dynamic_ols(train_target, train_exog, target_lags=selected_lags)
    row = _dynamic_regression_future_row(
        train_target,
        future_exog,
        target_lags=selected_lags,
    )
    prediction = fit.predict(sm.add_constant(row, has_constant="add"))
    return float(np.asarray(prediction, dtype=float)[0])


def _validation_window(length: int) -> int:
    return max(2, min(8, max(length // 5, 1)))


def _fit_enet_design(
    design: pd.DataFrame,
    response: pd.Series,
    *,
    alpha: float,
    l1_wt: float,
) -> tuple[object, pd.Series, pd.Series]:
    means = design.mean()
    stds = design.std(ddof=0).replace(0.0, 1.0)
    scaled = (design - means) / stds
    model = sm.OLS(response.astype(float), sm.add_constant(scaled, has_constant="add"))
    penalty = np.array([0.0, *([alpha] * scaled.shape[1])], dtype=float)
    with suppress_expected_statsmodels_warnings():
        fit = model.fit_regularized(alpha=penalty, L1_wt=l1_wt, refit=True)
    return fit, means, stds


def enet_alpha_table(
    target_series: pd.Series,
    exog_frame: pd.DataFrame,
    *,
    target_lags: list[int] | None = None,
    alpha_grid: list[float] | None = None,
    l1_wt: float = DEFAULT_ENET_L1_WT,
) -> pd.DataFrame:
    """Return the validation table used to choose the elastic-net penalty."""

    design, response = _dynamic_regression_design(
        target_series,
        exog_frame,
        target_lags=target_lags or DEFAULT_REGRESSION_TARGET_LAGS,
    )
    validation_size = _validation_window(len(design))
    if len(design) <= validation_size + 2:
        raise ValueError("not enough observations for elastic-net validation")

    train_design = design.iloc[:-validation_size]
    train_response = response.iloc[:-validation_size]
    valid_design = design.iloc[-validation_size:]
    valid_response = response.iloc[-validation_size:]

    rows: list[dict[str, float | str | int]] = []
    for alpha in alpha_grid or DEFAULT_ENET_ALPHA_GRID:
        try:
            fit, means, stds = _fit_enet_design(
                train_design,
                train_response,
                alpha=float(alpha),
                l1_wt=l1_wt,
            )
            scaled_valid = (valid_design - means) / stds
            preds = np.asarray(
                fit.predict(sm.add_constant(scaled_valid, has_constant="add")),
                dtype=float,
            )
            mae = float(np.mean(np.abs(valid_response.to_numpy(dtype=float) - preds)))
            params = pd.Series(fit.params, index=fit.model.exog_names, dtype=float)
            non_zero = int((params.drop(labels="const", errors="ignore").abs() > 1e-9).sum())
            rows.append(
                {
                    "alpha": float(alpha),
                    "validation_mae": mae,
                    "non_zero_terms": non_zero,
                    "status": "ok",
                }
            )
        except Exception:
            rows.append(
                {
                    "alpha": float(alpha),
                    "validation_mae": np.nan,
                    "non_zero_terms": np.nan,
                    "status": "fit_failed",
                }
            )
    return pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)


def choose_best_enet_alpha(
    target_series: pd.Series,
    exog_frame: pd.DataFrame,
    *,
    target_lags: list[int] | None = None,
    alpha_grid: list[float] | None = None,
    l1_wt: float = DEFAULT_ENET_L1_WT,
) -> tuple[float, pd.DataFrame]:
    """Choose the elastic-net penalty with the lowest validation MAE."""

    table = enet_alpha_table(
        target_series,
        exog_frame,
        target_lags=target_lags,
        alpha_grid=alpha_grid,
        l1_wt=l1_wt,
    )
    valid = table.loc[table["status"] == "ok"].sort_values(["validation_mae", "alpha"])
    if valid.empty:
        raise ValueError("could not fit any elastic-net candidate")
    return float(valid.iloc[0]["alpha"]), table


def fit_dynamic_enet(
    target_series: pd.Series,
    exog_frame: pd.DataFrame,
    *,
    target_lags: list[int] | None = None,
    alpha_grid: list[float] | None = None,
    l1_wt: float = DEFAULT_ENET_L1_WT,
) -> tuple[object, pd.DataFrame, pd.Series, pd.Series, pd.Series, float, pd.DataFrame]:
    """Fit the final elastic-net model after choosing the penalty by validation MAE."""

    selected_lags = target_lags or DEFAULT_REGRESSION_TARGET_LAGS
    alpha, alpha_table = choose_best_enet_alpha(
        target_series,
        exog_frame,
        target_lags=selected_lags,
        alpha_grid=alpha_grid,
        l1_wt=l1_wt,
    )
    design, response = _dynamic_regression_design(
        target_series,
        exog_frame,
        target_lags=selected_lags,
    )
    fit, means, stds = _fit_enet_design(
        design,
        response,
        alpha=alpha,
        l1_wt=l1_wt,
    )
    return fit, design, response, means, stds, alpha, alpha_table


def enet_one_step_forecast(
    train_target: pd.Series,
    train_exog: pd.DataFrame,
    future_exog: pd.DataFrame,
    *,
    target_lags: list[int] | None = None,
    alpha_grid: list[float] | None = None,
    l1_wt: float = DEFAULT_ENET_L1_WT,
) -> float:
    """Return a one-step forecast from the lecture elastic-net model."""

    selected_lags = target_lags or DEFAULT_REGRESSION_TARGET_LAGS
    fit, _, _, means, stds, _, _ = fit_dynamic_enet(
        train_target,
        train_exog,
        target_lags=selected_lags,
        alpha_grid=alpha_grid,
        l1_wt=l1_wt,
    )
    future_row = _dynamic_regression_future_row(
        train_target,
        future_exog,
        target_lags=selected_lags,
    )
    scaled_future = (future_row - means) / stds
    prediction = fit.predict(sm.add_constant(scaled_future, has_constant="add"))
    return float(np.asarray(prediction, dtype=float)[0])


def lagged_exogenous_features(
    panel: pd.DataFrame,
    columns: list[str],
    *,
    periods: int = 1,
) -> pd.DataFrame:
    """Return last-period exogenous features aligned for one-step forecasts."""

    exog = _clean_frame(panel[columns]).shift(periods)
    exog.columns = [f"{column}_lag{periods}" for column in columns]
    return exog


def next_one_step_exogenous_features(
    panel: pd.DataFrame,
    columns: list[str],
    *,
    periods: int = 1,
) -> pd.DataFrame:
    """Return the lagged exogenous row used for the next unseen month."""

    raw = _clean_frame(panel[columns]).dropna()
    if raw.empty:
        raise ValueError("panel has no usable exogenous observations")
    future = raw.iloc[[-1]].copy()
    future.index = pd.DatetimeIndex([raw.index[-1] + pd.offsets.MonthEnd(1)])
    future.columns = [f"{column}_lag{periods}" for column in columns]
    return future.astype(float)


def contemporaneous_correlation_table(
    frame: pd.DataFrame,
    *,
    columns: list[str],
) -> pd.DataFrame:
    """Return a compact correlation table for the lecture target and exogenous series."""

    subset = _clean_frame(frame[columns]).dropna()
    return subset.corr().round(3)


def coefficient_table_from_fit(
    fit: object,
    *,
    threshold: float = 1e-9,
    drop_const: bool = True,
) -> pd.DataFrame:
    """Return a compact coefficient table from a statsmodels regression fit."""

    params = pd.Series(fit.params, index=fit.model.exog_names, dtype=float)
    if drop_const:
        params = params.drop(labels="const", errors="ignore")
    frame = pd.DataFrame(
        {
            "term": params.index,
            "coefficient": params.to_numpy(dtype=float),
        }
    )
    frame["absolute_coefficient"] = frame["coefficient"].abs()
    frame["selected"] = frame["absolute_coefficient"] > threshold
    return frame.sort_values(
        ["selected", "absolute_coefficient", "term"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def align_backtests_to_common_dates(
    backtests: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
    """Align a set of backtests on the dates shared by every non-empty frame."""

    valid = {label: frame.copy() for label, frame in backtests.items() if not frame.empty}
    if not valid:
        return {}, pd.DatetimeIndex([])
    common_index: pd.DatetimeIndex | None = None
    for frame in valid.values():
        index = pd.DatetimeIndex(frame.index)
        common_index = index if common_index is None else common_index.intersection(index)
    if common_index is None:
        return {}, pd.DatetimeIndex([])
    aligned = {
        label: frame.loc[common_index].copy()
        for label, frame in valid.items()
    }
    return aligned, common_index


def target_mase_scale(
    target_series: pd.Series,
    *,
    train_end: str | pd.Timestamp = DEFAULT_TRAIN_END,
    test_periods: int | None = None,
) -> float:
    """Return the one-step naive scaling term from the pre-evaluation target sample."""

    target = _clean_series(target_series)
    try:
        start = _backtest_start_offset(target.index, train_end=train_end, test_periods=test_periods)
    except ValueError:
        return float("nan")
    train = target.iloc[:start]
    diffs = train.diff().abs().dropna()
    if diffs.empty:
        return float("nan")
    scale = float(diffs.mean())
    if scale <= 0:
        return float("nan")
    return scale


def _target_rmse(backtest: pd.DataFrame) -> float:
    errors = pd.to_numeric(backtest["error"], errors="coerce").dropna().astype(float)
    if errors.empty:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(errors))))


def _target_mae(backtest: pd.DataFrame) -> float:
    absolute_errors = (
        pd.to_numeric(backtest["absolute_error"], errors="coerce").dropna().astype(float)
    )
    if absolute_errors.empty:
        return float("nan")
    return float(absolute_errors.mean())


def metrics_row(
    backtest: pd.DataFrame,
    *,
    model_label: str,
    naive_backtest: pd.DataFrame,
    mase_scale: float,
) -> dict[str, float | str | int]:
    """Return target-only forecast metrics for one backtest."""

    common = pd.DatetimeIndex(backtest.index).intersection(pd.DatetimeIndex(naive_backtest.index))
    if common.empty:
        return {
            "model": model_label,
            "evaluation_rows": 0,
            "target_mae": float("nan"),
            "target_rmse": float("nan"),
            "target_mase": float("nan"),
            "target_oos_r2_vs_naive": float("nan"),
        }

    model_frame = backtest.loc[common]
    naive_frame = naive_backtest.loc[common]
    model_rmse = _target_rmse(model_frame)
    model_mae = _target_mae(model_frame)
    naive_errors = pd.to_numeric(naive_frame["error"], errors="coerce").dropna().astype(float)
    model_errors = pd.to_numeric(model_frame["error"], errors="coerce").dropna().astype(float)
    naive_mspe = float(
        np.mean(np.square(naive_errors))
    )
    model_mspe = float(
        np.mean(np.square(model_errors))
    )
    if not np.isfinite(mase_scale) or mase_scale <= 0:
        mase = float("nan")
    else:
        mase = model_mae / mase_scale
    if not np.isfinite(naive_mspe) or naive_mspe <= 0:
        oos_r2 = float("nan")
    else:
        oos_r2 = 1.0 - (model_mspe / naive_mspe)
    return {
        "model": model_label,
        "evaluation_rows": len(common),
        "target_mae": model_mae,
        "target_rmse": model_rmse,
        "target_mase": float(mase),
        "target_oos_r2_vs_naive": float(oos_r2),
    }


def metrics_table(
    backtests: dict[str, pd.DataFrame],
    *,
    target_series: pd.Series,
    train_end: str | pd.Timestamp = DEFAULT_TRAIN_END,
    test_periods: int | None = None,
    naive_label: str = "Naive",
    align_common_dates: bool = False,
) -> pd.DataFrame:
    """Return a target-only metrics table for a set of backtests."""

    if naive_label not in backtests:
        raise KeyError(f"missing naive benchmark label: {naive_label}")
    working = backtests
    if align_common_dates:
        working, _ = align_backtests_to_common_dates(backtests)
        if naive_label not in working:
            raise ValueError("naive benchmark dropped out during common-date alignment")
    naive_backtest = working[naive_label]
    mase_scale = target_mase_scale(
        target_series,
        train_end=train_end,
        test_periods=test_periods,
    )
    rows = [
        metrics_row(
            frame,
            model_label=label,
            naive_backtest=naive_backtest,
            mase_scale=mase_scale,
        )
        for label, frame in working.items()
    ]
    return pd.DataFrame(rows)


def choose_best_backtest_variant(
    variant_backtests: dict[str, pd.DataFrame],
) -> tuple[str, pd.DataFrame]:
    """Choose the best variant by target RMSE and return the comparison table."""

    rows: list[dict[str, float | str | int]] = []
    for label, frame in variant_backtests.items():
        rows.append(
            {
                "variant": label,
                "evaluation_rows": len(frame),
                "target_mae": _target_mae(frame),
                "target_rmse": _target_rmse(frame),
            }
        )
    table = pd.DataFrame(rows).sort_values(["target_rmse", "target_mae", "variant"]).reset_index(
        drop=True
    )
    if table.empty:
        raise ValueError("no variant backtests are available")
    return str(table.iloc[0]["variant"]), table


def build_wide_backtest_forecast_table(backtests: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return a wide target-space table of actuals and model forecasts."""

    aligned, common_index = align_backtests_to_common_dates(backtests)
    if common_index.empty:
        return pd.DataFrame()
    first = next(iter(aligned.values()))
    table = pd.DataFrame(index=common_index)
    table["actual"] = first["actual"].astype(float)
    for label, frame in aligned.items():
        table[f"{label} forecast"] = frame["forecast"].astype(float)
    return table


def build_beginner_horse_race_bundle(
    panel: pd.DataFrame,
    *,
    train_end: str | pd.Timestamp = DEFAULT_TRAIN_END,
    test_periods: int | None = None,
    variant_validation_periods: int = DEFAULT_VARIANT_VALIDATION_PERIODS,
) -> dict[str, object]:
    """Build the shared backtests and settings used by the horse-race scripts."""

    unemployment = _clean_series(panel["unemployment_rate"])
    unemployment_change = _clean_series(panel["unemployment_change_pp"])
    train_offset = _backtest_start_offset(
        unemployment_change.index,
        train_end=train_end,
        test_periods=test_periods,
    )
    initial_train = unemployment_change.iloc[:train_offset]
    initial_level = unemployment.reindex(initial_train.index)

    selected_lag, ar_lag_table = choose_best_ar_lag(initial_train, DEFAULT_AR_LAGS)
    selected_order, arma_order_table = choose_best_arma_order(initial_train, DEFAULT_ARMA_ORDERS)

    cash_only = lagged_exogenous_features(panel, DEFAULT_ARX_COLUMNS)
    cash_plus_commodity = lagged_exogenous_features(panel, DEFAULT_ARX_PLUS_COLUMNS)
    ols_exog = lagged_exogenous_features(panel, DEFAULT_OLS_COLUMNS)
    enet_exog = lagged_exogenous_features(panel, DEFAULT_ENET_COLUMNS)

    arx_validation_backtests = {
        "cash": rolling_one_step_backtest_with_exog(
            initial_train,
            initial_level,
            cash_only,
            forecast_fn=(
                lambda train, train_exog, future_exog: arx_one_step_forecast(
                    train,
                    train_exog,
                    future_exog,
                    lag=selected_lag,
                )
            ),
            test_periods=variant_validation_periods,
        ),
        "cash+commodity": rolling_one_step_backtest_with_exog(
            initial_train,
            initial_level,
            cash_plus_commodity,
            forecast_fn=(
                lambda train, train_exog, future_exog: arx_one_step_forecast(
                    train,
                    train_exog,
                    future_exog,
                    lag=selected_lag,
                )
            ),
            test_periods=variant_validation_periods,
        ),
    }
    selected_arx_variant, arx_variant_table = choose_best_backtest_variant(
        arx_validation_backtests
    )

    armax_validation_backtests = {
        "cash": rolling_one_step_backtest_with_exog(
            initial_train,
            initial_level,
            cash_only,
            forecast_fn=(
                lambda train, train_exog, future_exog: armax_one_step_forecast(
                    train,
                    train_exog,
                    future_exog,
                    order=selected_order,
                )
            ),
            test_periods=variant_validation_periods,
        ),
        "cash+commodity": rolling_one_step_backtest_with_exog(
            initial_train,
            initial_level,
            cash_plus_commodity,
            forecast_fn=(
                lambda train, train_exog, future_exog: armax_one_step_forecast(
                    train,
                    train_exog,
                    future_exog,
                    order=selected_order,
                )
            ),
            test_periods=variant_validation_periods,
        ),
    }
    selected_armax_variant, armax_variant_table = choose_best_backtest_variant(
        armax_validation_backtests
    )

    backtests = {
        "Naive": rolling_one_step_backtest(
            unemployment_change,
            unemployment,
            forecast_fn=naive_one_step_forecast,
            train_end=train_end,
            test_periods=test_periods,
        ),
        f"AR({selected_lag})": rolling_one_step_backtest(
            unemployment_change,
            unemployment,
            forecast_fn=lambda train: ar_one_step_forecast(train, lag=selected_lag),
            train_end=train_end,
            test_periods=test_periods,
        ),
        f"ARMA{selected_order}": rolling_one_step_backtest(
            unemployment_change,
            unemployment,
            forecast_fn=lambda train: arma_one_step_forecast(train, order=selected_order),
            train_end=train_end,
            test_periods=test_periods,
        ),
        f"ARX({selected_lag}) {selected_arx_variant}": rolling_one_step_backtest_with_exog(
            unemployment_change,
            unemployment,
            cash_only if selected_arx_variant == "cash" else cash_plus_commodity,
            forecast_fn=(
                lambda train, train_exog, future_exog: arx_one_step_forecast(
                    train,
                    train_exog,
                    future_exog,
                    lag=selected_lag,
                )
            ),
            train_end=train_end,
            test_periods=test_periods,
        ),
        f"ARMAX{selected_order} {selected_armax_variant}": rolling_one_step_backtest_with_exog(
            unemployment_change,
            unemployment,
            cash_only if selected_armax_variant == "cash" else cash_plus_commodity,
            forecast_fn=(
                lambda train, train_exog, future_exog: armax_one_step_forecast(
                    train,
                    train_exog,
                    future_exog,
                    order=selected_order,
                )
            ),
            train_end=train_end,
            test_periods=test_periods,
        ),
        "OLS": rolling_one_step_backtest_with_exog(
            unemployment_change,
            unemployment,
            ols_exog,
            forecast_fn=(
                lambda train, train_exog, future_exog: ols_one_step_forecast(
                    train,
                    train_exog,
                    future_exog,
                    target_lags=DEFAULT_REGRESSION_TARGET_LAGS,
                )
            ),
            train_end=train_end,
            test_periods=test_periods,
        ),
        "ENet": rolling_one_step_backtest_with_exog(
            unemployment_change,
            unemployment,
            enet_exog,
            forecast_fn=(
                lambda train, train_exog, future_exog: enet_one_step_forecast(
                    train,
                    train_exog,
                    future_exog,
                    target_lags=DEFAULT_REGRESSION_TARGET_LAGS,
                    alpha_grid=DEFAULT_ENET_ALPHA_GRID,
                    l1_wt=DEFAULT_ENET_L1_WT,
                )
            ),
            train_end=train_end,
            test_periods=test_periods,
        ),
    }

    common_backtests, common_index = align_backtests_to_common_dates(backtests)
    metrics = metrics_table(
        common_backtests,
        target_series=unemployment_change,
        train_end=train_end,
        test_periods=test_periods,
        naive_label="Naive",
        align_common_dates=False,
    ).sort_values(["target_rmse", "model"]).reset_index(drop=True)

    ols_fit, ols_design, _ = fit_dynamic_ols(
        initial_train,
        ols_exog,
        target_lags=DEFAULT_REGRESSION_TARGET_LAGS,
    )
    enet_fit, enet_design, _, _, _, enet_alpha, enet_alpha_table = fit_dynamic_enet(
        initial_train,
        enet_exog,
        target_lags=DEFAULT_REGRESSION_TARGET_LAGS,
        alpha_grid=DEFAULT_ENET_ALPHA_GRID,
        l1_wt=DEFAULT_ENET_L1_WT,
    )

    model_specs = pd.DataFrame(
        [
            {
                "model": "Naive",
                "family": "Naive",
                "selection": "Repeat last target change",
                "feature_count": 0,
            },
            {
                "model": f"AR({selected_lag})",
                "family": "AR",
                "selection": f"Lag {selected_lag} chosen by BIC",
                "feature_count": selected_lag,
            },
            {
                "model": f"ARMA{selected_order}",
                "family": "ARMA",
                "selection": f"Order {selected_order} chosen by BIC",
                "feature_count": sum(selected_order),
            },
            {
                "model": f"ARX({selected_lag}) {selected_arx_variant}",
                "family": "ARX",
                "selection": (
                    f"AR({selected_lag}) with {selected_arx_variant} chosen by "
                    f"target RMSE on a {variant_validation_periods}-month validation window"
                ),
                "feature_count": selected_lag
                + len(
                    DEFAULT_ARX_COLUMNS
                    if selected_arx_variant == "cash"
                    else DEFAULT_ARX_PLUS_COLUMNS
                ),
            },
            {
                "model": f"ARMAX{selected_order} {selected_armax_variant}",
                "family": "ARMAX",
                "selection": (
                    f"ARMA{selected_order} with {selected_armax_variant} chosen by "
                    f"target RMSE on a {variant_validation_periods}-month validation window"
                ),
                "feature_count": sum(selected_order)
                + len(
                    DEFAULT_ARX_COLUMNS
                    if selected_armax_variant == "cash"
                    else DEFAULT_ARX_PLUS_COLUMNS
                ),
            },
            {
                "model": "OLS",
                "family": "OLS",
                "selection": (
                    f"Target lags {DEFAULT_REGRESSION_TARGET_LAGS} plus hand-picked predictors"
                ),
                "feature_count": int(ols_design.shape[1]),
            },
            {
                "model": "ENet",
                "family": "ENet",
                "selection": (
                    f"Target lags {DEFAULT_REGRESSION_TARGET_LAGS}, alpha {enet_alpha}, "
                    f"L1_wt {DEFAULT_ENET_L1_WT}"
                ),
                "feature_count": int(enet_design.shape[1]),
            },
        ]
    )

    winner_row = metrics.sort_values(["target_rmse", "model"]).iloc[0]
    return {
        "panel": panel,
        "level_series": unemployment,
        "target_series": unemployment_change,
        "common_index": common_index,
        "backtests": backtests,
        "common_backtests": common_backtests,
        "metrics": metrics,
        "winner_model": str(winner_row["model"]),
        "model_specs": model_specs,
        "ar_lag_table": ar_lag_table,
        "arma_order_table": arma_order_table,
        "arx_variant_table": arx_variant_table,
        "armax_variant_table": armax_variant_table,
        "selected_lag": selected_lag,
        "selected_order": selected_order,
        "selected_arx_variant": selected_arx_variant,
        "selected_armax_variant": selected_armax_variant,
        "selected_enet_alpha": enet_alpha,
        "enet_alpha_table": enet_alpha_table,
        "ols_coefficients": coefficient_table_from_fit(ols_fit),
        "enet_coefficients": coefficient_table_from_fit(enet_fit),
        "ols_exog": ols_exog,
        "enet_exog": enet_exog,
        "cash_only_exog": cash_only,
        "cash_plus_commodity_exog": cash_plus_commodity,
    }


def build_us_beginner_horse_race_bundle(
    panel: pd.DataFrame,
    *,
    train_end: str | pd.Timestamp = DEFAULT_TRAIN_END,
    test_periods: int | None = None,
    variant_validation_periods: int = DEFAULT_VARIANT_VALIDATION_PERIODS,
) -> dict[str, object]:
    """Build the shared backtests and settings for the U.S. extension horse race."""

    unemployment = _clean_series(panel["unemployment_rate"])
    unemployment_change = _clean_series(panel["unemployment_change_pp"])
    train_offset = _backtest_start_offset(
        unemployment_change.index,
        train_end=train_end,
        test_periods=test_periods,
    )
    initial_train = unemployment_change.iloc[:train_offset]
    initial_level = unemployment.reindex(initial_train.index)

    selected_lag, ar_lag_table = choose_best_ar_lag(initial_train, DEFAULT_AR_LAGS)
    selected_order, arma_order_table = choose_best_arma_order(initial_train, DEFAULT_ARMA_ORDERS)

    fedfunds_only = lagged_exogenous_features(panel, DEFAULT_US_ARX_COLUMNS)
    fedfunds_plus_spread = lagged_exogenous_features(panel, DEFAULT_US_ARX_PLUS_COLUMNS)

    arx_validation_backtests = {
        "fedfunds": rolling_one_step_backtest_with_exog(
            initial_train,
            initial_level,
            fedfunds_only,
            forecast_fn=(
                lambda train, train_exog, future_exog: arx_one_step_forecast(
                    train,
                    train_exog,
                    future_exog,
                    lag=selected_lag,
                )
            ),
            test_periods=variant_validation_periods,
        ),
        "fedfunds+spread": rolling_one_step_backtest_with_exog(
            initial_train,
            initial_level,
            fedfunds_plus_spread,
            forecast_fn=(
                lambda train, train_exog, future_exog: arx_one_step_forecast(
                    train,
                    train_exog,
                    future_exog,
                    lag=selected_lag,
                )
            ),
            test_periods=variant_validation_periods,
        ),
    }
    selected_arx_variant, arx_variant_table = choose_best_backtest_variant(
        arx_validation_backtests
    )

    backtests = {
        "Naive": rolling_one_step_backtest(
            unemployment_change,
            unemployment,
            forecast_fn=naive_one_step_forecast,
            train_end=train_end,
            test_periods=test_periods,
        ),
        f"AR({selected_lag})": rolling_one_step_backtest(
            unemployment_change,
            unemployment,
            forecast_fn=lambda train: ar_one_step_forecast(train, lag=selected_lag),
            train_end=train_end,
            test_periods=test_periods,
        ),
        f"ARMA{selected_order}": rolling_one_step_backtest(
            unemployment_change,
            unemployment,
            forecast_fn=lambda train: arma_one_step_forecast(train, order=selected_order),
            train_end=train_end,
            test_periods=test_periods,
        ),
        f"ARX({selected_lag}) {selected_arx_variant}": rolling_one_step_backtest_with_exog(
            unemployment_change,
            unemployment,
            fedfunds_only
            if selected_arx_variant == "fedfunds"
            else fedfunds_plus_spread,
            forecast_fn=(
                lambda train, train_exog, future_exog: arx_one_step_forecast(
                    train,
                    train_exog,
                    future_exog,
                    lag=selected_lag,
                )
            ),
            train_end=train_end,
            test_periods=test_periods,
        ),
    }

    common_backtests, common_index = align_backtests_to_common_dates(backtests)
    metrics = metrics_table(
        common_backtests,
        target_series=unemployment_change,
        train_end=train_end,
        test_periods=test_periods,
        naive_label="Naive",
        align_common_dates=False,
    ).sort_values(["target_rmse", "model"]).reset_index(drop=True)

    model_specs = pd.DataFrame(
        [
            {
                "model": "Naive",
                "family": "Naive",
                "selection": "Repeat last target change",
                "feature_count": 0,
            },
            {
                "model": f"AR({selected_lag})",
                "family": "AR",
                "selection": f"Lag {selected_lag} chosen by BIC",
                "feature_count": selected_lag,
            },
            {
                "model": f"ARMA{selected_order}",
                "family": "ARMA",
                "selection": f"Order {selected_order} chosen by BIC",
                "feature_count": sum(selected_order),
            },
            {
                "model": f"ARX({selected_lag}) {selected_arx_variant}",
                "family": "ARX",
                "selection": (
                    f"AR({selected_lag}) with {selected_arx_variant} chosen by "
                    f"target RMSE on a {variant_validation_periods}-month validation window"
                ),
                "feature_count": selected_lag
                + len(
                    DEFAULT_US_ARX_COLUMNS
                    if selected_arx_variant == "fedfunds"
                    else DEFAULT_US_ARX_PLUS_COLUMNS
                ),
            },
        ]
    )

    winner_row = metrics.sort_values(["target_rmse", "model"]).iloc[0]
    return {
        "panel": panel,
        "level_series": unemployment,
        "target_series": unemployment_change,
        "common_index": common_index,
        "backtests": backtests,
        "common_backtests": common_backtests,
        "metrics": metrics,
        "winner_model": str(winner_row["model"]),
        "model_specs": model_specs,
        "ar_lag_table": ar_lag_table,
        "arma_order_table": arma_order_table,
        "arx_variant_table": arx_variant_table,
        "selected_lag": selected_lag,
        "selected_order": selected_order,
        "selected_arx_variant": selected_arx_variant,
        "fedfunds_only_exog": fedfunds_only,
        "fedfunds_spread_exog": fedfunds_plus_spread,
    }


def beginner_model_one_step_forecasts(
    panel: pd.DataFrame,
    race_bundle: dict[str, object],
) -> pd.DataFrame:
    """Return the next one-step target forecast for each horse-race model."""

    unemployment_change = _clean_series(panel["unemployment_change_pp"])
    selected_lag = int(race_bundle["selected_lag"])
    selected_order = tuple(race_bundle["selected_order"])
    selected_arx_variant = str(race_bundle["selected_arx_variant"])
    selected_armax_variant = str(race_bundle["selected_armax_variant"])

    cash_only_future = next_one_step_exogenous_features(panel, DEFAULT_ARX_COLUMNS)
    cash_plus_commodity_future = next_one_step_exogenous_features(panel, DEFAULT_ARX_PLUS_COLUMNS)
    ols_future = next_one_step_exogenous_features(panel, DEFAULT_OLS_COLUMNS)
    enet_future = next_one_step_exogenous_features(panel, DEFAULT_ENET_COLUMNS)

    cash_only_train_exog = _clean_frame(race_bundle["cash_only_exog"]).dropna()
    cash_only_train_target = unemployment_change.reindex(cash_only_train_exog.index).dropna()
    cash_only_train_exog = cash_only_train_exog.reindex(cash_only_train_target.index)

    cash_plus_train_exog = _clean_frame(race_bundle["cash_plus_commodity_exog"]).dropna()
    cash_plus_train_target = unemployment_change.reindex(cash_plus_train_exog.index).dropna()
    cash_plus_train_exog = cash_plus_train_exog.reindex(cash_plus_train_target.index)

    ols_train_exog = _clean_frame(race_bundle["ols_exog"]).dropna()
    ols_train_target = unemployment_change.reindex(ols_train_exog.index).dropna()
    ols_train_exog = ols_train_exog.reindex(ols_train_target.index)

    enet_train_exog = _clean_frame(race_bundle["enet_exog"]).dropna()
    enet_train_target = unemployment_change.reindex(enet_train_exog.index).dropna()
    enet_train_exog = enet_train_exog.reindex(enet_train_target.index)

    forecasts = {
        "Naive": naive_one_step_forecast(unemployment_change),
        f"AR({selected_lag})": ar_one_step_forecast(unemployment_change, lag=selected_lag),
        f"ARMA{selected_order}": arma_one_step_forecast(unemployment_change, order=selected_order),
        f"ARX({selected_lag}) {selected_arx_variant}": arx_one_step_forecast(
            cash_only_train_target if selected_arx_variant == "cash" else cash_plus_train_target,
            cash_only_train_exog if selected_arx_variant == "cash" else cash_plus_train_exog,
            cash_only_future if selected_arx_variant == "cash" else cash_plus_commodity_future,
            lag=selected_lag,
        ),
        f"ARMAX{selected_order} {selected_armax_variant}": armax_one_step_forecast(
            cash_only_train_target if selected_armax_variant == "cash" else cash_plus_train_target,
            cash_only_train_exog if selected_armax_variant == "cash" else cash_plus_train_exog,
            cash_only_future if selected_armax_variant == "cash" else cash_plus_commodity_future,
            order=selected_order,
        ),
        "OLS": ols_one_step_forecast(
            ols_train_target,
            ols_train_exog,
            ols_future,
            target_lags=DEFAULT_REGRESSION_TARGET_LAGS,
        ),
        "ENet": enet_one_step_forecast(
            enet_train_target,
            enet_train_exog,
            enet_future,
            target_lags=DEFAULT_REGRESSION_TARGET_LAGS,
            alpha_grid=DEFAULT_ENET_ALPHA_GRID,
            l1_wt=DEFAULT_ENET_L1_WT,
        ),
    }
    forecast_date = pd.Timestamp(unemployment_change.index[-1]) + pd.offsets.MonthEnd(1)
    return pd.DataFrame(
        {
            "forecast_date": forecast_date,
            "model": list(forecasts),
            "target_forecast": [float(value) for value in forecasts.values()],
        }
    )


def equal_weight_ensemble_backtest(
    member_backtests: dict[str, pd.DataFrame],
    *,
    level_series: pd.Series,
) -> pd.DataFrame:
    """Build an equal-weight target ensemble on the common backtest dates."""

    aligned, common_index = align_backtests_to_common_dates(member_backtests)
    if common_index.empty:
        return pd.DataFrame()
    first = next(iter(aligned.values()))
    target_forecasts = pd.concat(
        [frame["forecast"].rename(label) for label, frame in aligned.items()],
        axis=1,
    )
    forecast = target_forecasts.mean(axis=1)
    actual = first["actual"].astype(float)
    backtest = pd.DataFrame(
        {
            "actual": actual,
            "forecast": forecast.astype(float),
        },
        index=common_index,
    )
    backtest["error"] = backtest["actual"] - backtest["forecast"]
    backtest["absolute_error"] = backtest["error"].abs()
    return add_level_columns(backtest, level_series)


def save_table_csv(
    frame: pd.DataFrame,
    output_dir: str | Path,
    stem: str,
    *,
    repo_root: Path | None = None,
) -> Path:
    """Write a CSV table and return the path."""

    output_root = resolve_repo_path(output_dir, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"{stem}.csv"
    frame.to_csv(path, index=True)
    return path


def save_figure_pair(
    fig: plt.Figure,
    output_dir: str | Path,
    stem: str,
    *,
    repo_root: Path | None = None,
) -> dict[str, Path]:
    """Write PNG and PDF versions of a figure."""

    output_root = resolve_repo_path(output_dir, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": output_root / f"{stem}.png",
        "pdf": output_root / f"{stem}.pdf",
    }
    for path in paths.values():
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return paths


def _format_month_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def build_simulated_unit_root_figure(simulated: pd.DataFrame) -> plt.Figure:
    """Build a three-panel figure for the simulated unit-root examples."""

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.0), sharex=True)
    labels = {
        "random_walk": "Random walk: non-stationary level",
        "stationary_ar1": "Stationary AR(1): mean reversion",
        "stationary_arma11": "Stationary ARMA(1,1): short memory",
    }
    for axis, column in zip(axes, simulated.columns, strict=True):
        axis.plot(simulated.index, simulated[column], linewidth=1.8, color="#1F77B4")
        axis.set_title(labels[column], loc="left", fontweight="bold")
        axis.grid(True, axis="y", alpha=0.30)
        axis.grid(False, axis="x")
        _format_month_axis(axis)
    axes[-1].set_xlabel("Date")
    fig.suptitle("Week 3 beginner unit-root examples", x=0.05, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.98))
    return fig


def build_rate_level_change_figure(
    level: pd.Series,
    change: pd.Series,
    *,
    title_prefix: str,
) -> plt.Figure:
    """Build a two-panel figure for a rate level and its monthly change."""

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.5), sharex=True)
    axes[0].plot(level.index, level, linewidth=1.9, color="#1F77B4")
    axes[0].set_title(f"{title_prefix} level (%)", loc="left", fontweight="bold")
    axes[0].grid(True, axis="y", alpha=0.30)
    axes[0].grid(False, axis="x")

    axes[1].plot(change.index, change, linewidth=1.5, color="#C44E52")
    axes[1].axhline(0.0, color="#4A5568", linewidth=1.0, linestyle="--")
    axes[1].set_title(
        f"Monthly change in {title_prefix.lower()} (%)",
        loc="left",
        fontweight="bold",
    )
    axes[1].grid(True, axis="y", alpha=0.30)
    axes[1].grid(False, axis="x")
    axes[1].set_xlabel("Date")

    for axis in axes:
        _format_month_axis(axis)
    fig.tight_layout()
    return fig


def build_unemployment_level_change_figure(
    level: pd.Series,
    change: pd.Series,
) -> plt.Figure:
    """Build a two-panel figure for Australia unemployment levels and changes."""

    return build_rate_level_change_figure(
        level,
        change,
        title_prefix="Australia unemployment rate",
    )


def build_selection_figure(
    table: pd.DataFrame,
    *,
    x_column: str,
    title: str,
) -> plt.Figure:
    """Build a simple BIC comparison chart."""

    valid = table.loc[table["status"] == "ok"].copy()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    if valid.empty:
        ax.text(0.5, 0.5, "No valid models", ha="center", va="center")
        ax.set_axis_off()
        return fig

    labels = valid[x_column].astype(str).tolist()
    ax.bar(labels, valid["bic"], color="#1F77B4", alpha=0.85)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel("BIC")
    ax.grid(True, axis="y", alpha=0.30)
    ax.grid(False, axis="x")
    fig.tight_layout()
    return fig


def build_backtest_comparison_figure(
    backtests: dict[str, pd.DataFrame],
    *,
    title: str,
) -> plt.Figure:
    """Build a two-panel backtest comparison figure."""

    first = next(iter(backtests.values()))
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.8), sharex=True)

    axes[0].plot(
        first.index,
        first["actual_level"],
        color="#111111",
        linewidth=2.0,
        label="Actual unemployment rate",
    )
    for label, frame in backtests.items():
        axes[0].plot(
            frame.index,
            frame["forecast_level"],
            linewidth=1.6,
            label=f"{label} forecast",
        )
    axes[0].set_title(title, loc="left", fontweight="bold")
    axes[0].set_ylabel("Level (%)")
    axes[0].grid(True, axis="y", alpha=0.30)
    axes[0].grid(False, axis="x")
    axes[0].legend(loc="upper left", frameon=False)

    axes[1].plot(
        first.index,
        first["actual"],
        color="#111111",
        linewidth=2.0,
        label="Actual monthly change",
    )
    for label, frame in backtests.items():
        axes[1].plot(
            frame.index,
            frame["forecast"],
            linewidth=1.4,
            label=f"{label} forecast",
        )
    axes[1].axhline(0.0, color="#4A5568", linewidth=1.0, linestyle="--")
    axes[1].set_ylabel("Change (%)")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, axis="y", alpha=0.30)
    axes[1].grid(False, axis="x")

    for axis in axes:
        _format_month_axis(axis)
    fig.tight_layout()
    return fig


def build_target_only_backtest_figure(
    backtests: dict[str, pd.DataFrame],
    *,
    title: str,
) -> plt.Figure:
    """Build a one-panel target-space backtest figure."""

    first = next(iter(backtests.values()))
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(
        first.index,
        first["actual"],
        color="#111111",
        linewidth=2.0,
        label="Actual monthly change",
    )
    for label, frame in backtests.items():
        ax.plot(
            frame.index,
            frame["forecast"],
            linewidth=1.5,
            label=f"{label} forecast",
        )
    ax.axhline(0.0, color="#4A5568", linewidth=1.0, linestyle="--")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel("Change (%)")
    ax.set_xlabel("Date")
    ax.grid(True, axis="y", alpha=0.30)
    ax.grid(False, axis="x")
    ax.legend(loc="upper left", frameon=False)
    _format_month_axis(ax)
    fig.tight_layout()
    return fig


def build_target_metric_race_figure(
    metrics: pd.DataFrame,
    backtests: dict[str, pd.DataFrame],
    *,
    metric_column: str,
    title: str,
    top_n: int = 3,
) -> plt.Figure:
    """Build a metric bar chart plus target-space backtest lines for the top models."""

    ordered_metrics = metrics.sort_values([metric_column, "model"]).reset_index(drop=True)
    top_labels = ordered_metrics["model"].head(top_n).tolist()
    top_backtests = {label: backtests[label] for label in top_labels if label in backtests}

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=False)
    axes[0].bar(
        ordered_metrics["model"],
        ordered_metrics[metric_column],
        color="#1F77B4",
        alpha=0.85,
    )
    axes[0].set_title(title, loc="left", fontweight="bold")
    axes[0].set_ylabel(metric_column.replace("_", " ").upper())
    axes[0].grid(True, axis="y", alpha=0.30)
    axes[0].grid(False, axis="x")
    axes[0].tick_params(axis="x", rotation=35)

    first = next(iter(top_backtests.values()))
    axes[1].plot(
        first.index,
        first["actual"],
        color="#111111",
        linewidth=2.0,
        label="Actual monthly change",
    )
    for label, frame in top_backtests.items():
        axes[1].plot(
            frame.index,
            frame["forecast"],
            linewidth=1.5,
            label=f"{label} forecast",
        )
    axes[1].axhline(0.0, color="#4A5568", linewidth=1.0, linestyle="--")
    axes[1].set_ylabel("Change (%)")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, axis="y", alpha=0.30)
    axes[1].grid(False, axis="x")
    axes[1].legend(loc="upper left", frameon=False)
    _format_month_axis(axes[1])
    fig.tight_layout()
    return fig
