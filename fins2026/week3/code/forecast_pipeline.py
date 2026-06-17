# ruff: noqa
"""Week 3 Australia-first model inputs, benchmarks, and outputs."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from fins2026.week3.code.forecast_specs import AUSTRALIA_FORECAST_SPECS
from fins2026.week3.code.panel_support import log_change_percent
from fintools.apps import ForecastModel, forecast_series_spec, rolling_backtest_spec

MODEL_LABELS = {
    "naive": "Naive",
    "drift": "Drift",
    "ar": "AR",
    "arma": "ARMA",
    "armax": "ARMA + exog",
    "enet": "OLS + elastic net",
}
ONE_STEP_ONLY_MODELS = {"armax", "enet"}
DEFAULT_FORECASTS_DIR = Path("fins2026/week3/results/forecasts")
DEFAULT_LEADERBOARD_FIXTURE = Path("fins2026/week3/data/benchmark_leaderboard_fixture.csv")


def results_slug(text: str) -> str:
    """Convert a label into a stable lowercase file stem."""

    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "series"


def default_horizon_for_spec(spec) -> int:
    """Return the default forward horizon for a Week 3 series."""

    return 4 if spec.frequency == "quarterly" else 6


def candidate_lags_for_spec(spec) -> list[int]:
    """Return the lag grid used for AR and elastic-net models."""

    return [1, 2, 4] if spec.frequency == "quarterly" else [1, 3, 6, 12]


def candidate_orders_for_spec(spec) -> list[tuple[int, int]]:
    """Return the ARMA order grid used for Week 3 models."""

    if spec.frequency == "quarterly":
        return [(1, 0), (2, 0), (1, 1), (2, 1)]
    return [(1, 0), (2, 0), (1, 1), (2, 1), (2, 2)]


def _next_index(series: pd.Series) -> pd.DatetimeIndex:
    inferred = pd.infer_freq(series.index)
    if inferred:
        offset = pd.tseries.frequencies.to_offset(inferred)
        return pd.DatetimeIndex([series.index[-1] + offset])
    if len(series.index) >= 2:
        delta = series.index.to_series().diff().dropna().median()
        if pd.notna(delta) and delta > pd.Timedelta(0):
            return pd.DatetimeIndex([series.index[-1] + delta])
    return pd.DatetimeIndex([series.index[-1] + pd.offsets.MonthEnd(1)])


def build_target_series(
    spec,
    australia_monthly: pd.DataFrame,
    australia_quarterly: pd.DataFrame,
) -> pd.Series:
    """Return the raw series used as the model level for a Week 3 target."""

    frame = australia_quarterly if spec.frequency == "quarterly" else australia_monthly
    return frame[spec.series_id].dropna().astype(float)


def _monthly_australia_features(australia_monthly: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(index=australia_monthly.index)
    frame["AU_CASH_RATE_CHANGE_PP"] = australia_monthly["Cash rate target"].diff()
    frame["AU_10Y_CHANGE_PP"] = australia_monthly["10Y government bond yield"].diff()
    frame["AU_UNEMP_CHANGE_PP"] = australia_monthly["Unemployment rate"].diff()
    frame["AU_TWI_LOG_CHANGE_PCT"] = log_change_percent(australia_monthly["Trade-weighted index"])
    frame["AU_COMMODITY_LOG_CHANGE_PCT"] = log_change_percent(
        australia_monthly["Commodity price index (A$)"]
    )
    frame["AU_CPI_YOY_PCT"] = australia_monthly["Headline CPI inflation"]
    frame["AU_WPI_YOY_PCT"] = australia_monthly["Wage Price Index growth"]
    return frame


def _monthly_us_features(us_monthly: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(index=us_monthly.index)
    frame["US_FEDFUNDS_CHANGE_PP"] = us_monthly["FEDFUNDS"].diff()
    frame["US_10Y_CHANGE_PP"] = us_monthly["DGS10"].diff()
    frame["US_VIX_LEVEL"] = us_monthly["VIXCLS"]
    frame["US_UNEMP_CHANGE_PP"] = us_monthly["UNRATE"].diff()
    frame["US_INDPRO_LOG_GROWTH_PCT"] = us_monthly["INDPRO_LOG_GROWTH_PCT"]
    frame["US_SP500_RETURN_PCT"] = us_monthly["SP500_RETURN_PCT"]
    frame["US_YIELD_SPREAD_PP"] = us_monthly["T10Y2Y"]
    return frame


def _quarterly_australia_features(australia_monthly: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(index=australia_monthly.resample("QE").last().index)
    quarter_cash = australia_monthly["Cash rate target"].resample("QE").last()
    quarter_unemp = australia_monthly["Unemployment rate"].resample("QE").mean()
    quarter_twi = australia_monthly["Trade-weighted index"].resample("QE").last()
    quarter_commodity = australia_monthly["Commodity price index (A$)"].resample("QE").last()
    frame["AU_CASH_RATE_CHANGE_PP_Q"] = quarter_cash.diff()
    frame["AU_UNEMP_CHANGE_PP_Q"] = quarter_unemp.diff()
    frame["AU_TWI_LOG_CHANGE_PCT_Q"] = log_change_percent(quarter_twi)
    frame["AU_COMMODITY_LOG_CHANGE_PCT_Q"] = log_change_percent(quarter_commodity)
    return frame


def _quarterly_us_features(us_monthly: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(index=us_monthly.resample("QE").last().index)
    frame["US_FEDFUNDS_CHANGE_PP_Q"] = us_monthly["FEDFUNDS"].resample("QE").mean().diff()
    frame["US_INDPRO_LOG_GROWTH_PCT_Q"] = log_change_percent(
        us_monthly["INDPRO"].resample("QE").last()
    )
    sp500_quarter = us_monthly["SP500"].resample("QE").last()
    frame["US_SP500_RETURN_PCT_Q"] = sp500_quarter.pct_change() * 100.0
    frame["US_YIELD_SPREAD_PP_Q"] = us_monthly["T10Y2Y"].resample("QE").mean()
    frame["US_VIX_LEVEL_Q"] = us_monthly["VIXCLS"].resample("QE").mean()
    return frame


def _feature_columns_for_spec(spec) -> list[str]:
    if spec.series_id in {"Cash rate target", "10Y government bond yield"}:
        return [
            "AU_CPI_YOY_PCT",
            "AU_UNEMP_CHANGE_PP",
            "AU_COMMODITY_LOG_CHANGE_PCT",
            "AU_TWI_LOG_CHANGE_PCT",
            "US_FEDFUNDS_CHANGE_PP",
            "US_10Y_CHANGE_PP",
            "US_VIX_LEVEL",
        ]
    if spec.series_id == "Unemployment rate":
        return [
            "AU_CASH_RATE_CHANGE_PP",
            "AU_COMMODITY_LOG_CHANGE_PCT",
            "AU_TWI_LOG_CHANGE_PCT",
            "US_UNEMP_CHANGE_PP",
            "US_INDPRO_LOG_GROWTH_PCT",
            "US_SP500_RETURN_PCT",
        ]
    if spec.series_id == "Trade-weighted index":
        return [
            "AU_CASH_RATE_CHANGE_PP",
            "AU_COMMODITY_LOG_CHANGE_PCT",
            "US_10Y_CHANGE_PP",
            "US_VIX_LEVEL",
            "US_SP500_RETURN_PCT",
        ]
    if spec.series_id == "Commodity price index (A$)":
        return [
            "AU_CASH_RATE_CHANGE_PP",
            "AU_TWI_LOG_CHANGE_PCT",
            "US_10Y_CHANGE_PP",
            "US_VIX_LEVEL",
            "US_SP500_RETURN_PCT",
        ]
    if spec.series_id in {"Headline CPI inflation", "Wage Price Index growth"}:
        return [
            "AU_UNEMP_CHANGE_PP_Q",
            "AU_CASH_RATE_CHANGE_PP_Q",
            "AU_TWI_LOG_CHANGE_PCT_Q",
            "AU_COMMODITY_LOG_CHANGE_PCT_Q",
            "US_FEDFUNDS_CHANGE_PP_Q",
            "US_INDPRO_LOG_GROWTH_PCT_Q",
        ]
    return [
        "AU_CASH_RATE_CHANGE_PP_Q",
        "AU_UNEMP_CHANGE_PP_Q",
        "AU_TWI_LOG_CHANGE_PCT_Q",
        "AU_COMMODITY_LOG_CHANGE_PCT_Q",
        "US_INDPRO_LOG_GROWTH_PCT_Q",
        "US_SP500_RETURN_PCT_Q",
        "US_YIELD_SPREAD_PP_Q",
    ]


def build_exogenous_inputs(
    spec,
    australia_monthly: pd.DataFrame,
    australia_quarterly: pd.DataFrame,
    us_monthly: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Build the lag-safe exogenous design for one Week 3 target."""

    target = build_target_series(spec, australia_monthly, australia_quarterly)
    if spec.frequency == "quarterly":
        raw = _quarterly_australia_features(australia_monthly).join(
            _quarterly_us_features(us_monthly),
            how="outer",
        )
        aligned = raw.reindex(target.index, method="ffill")
    else:
        raw = _monthly_australia_features(australia_monthly).join(
            _monthly_us_features(us_monthly),
            how="outer",
        )
        aligned = raw.reindex(target.index)
    columns = _feature_columns_for_spec(spec)
    aligned = aligned.reindex(columns=columns)
    shifted = aligned.shift(1)
    future = aligned.dropna().iloc[[-1]].copy()
    future.index = _next_index(target)
    return target, shifted, future


def _min_train_for_spec(spec) -> int:
    return 20 if spec.frequency == "quarterly" else 36


def _backtest_step_for_spec(spec) -> int:
    return 2 if spec.frequency == "quarterly" else 6


def _ranking_metric(backtest: pd.DataFrame) -> float:
    if "absolute_level_error" in backtest and backtest["absolute_level_error"].notna().any():
        return float(backtest["absolute_level_error"].dropna().mean())
    return float(backtest["absolute_error"].dropna().mean())


def _rmse(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if clean.empty:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(clean))))


def compute_spec_outputs(
    spec,
    australia_monthly: pd.DataFrame,
    australia_quarterly: pd.DataFrame,
    us_monthly: pd.DataFrame,
    *,
    model: ForecastModel,
    horizon: int | None = None,
) -> tuple[object, pd.DataFrame]:
    """Compute one forward forecast plus the comparable one-step backtest."""

    target, exog, future_exog = build_exogenous_inputs(
        spec,
        australia_monthly,
        australia_quarterly,
        us_monthly,
    )
    candidate_lags = candidate_lags_for_spec(spec)
    candidate_orders = candidate_orders_for_spec(spec)
    if model in ONE_STEP_ONLY_MODELS:
        forecast_horizon = 1
    else:
        forecast_horizon = horizon or default_horizon_for_spec(spec)
    forecast_result = forecast_series_spec(
        target,
        spec,
        model=model,
        horizon=forecast_horizon,
        exog=exog if model in ONE_STEP_ONLY_MODELS else None,
        future_exog=future_exog if model in ONE_STEP_ONLY_MODELS else None,
        candidate_lags=candidate_lags,
        candidate_orders=candidate_orders,
    )
    backtest = rolling_backtest_spec(
        target,
        spec,
        model=model,
        horizon=1,
        min_train=_min_train_for_spec(spec),
        step=_backtest_step_for_spec(spec),
        exog=exog if model in ONE_STEP_ONLY_MODELS else None,
        candidate_lags=candidate_lags,
        candidate_orders=candidate_orders,
    )
    return forecast_result, backtest


def benchmark_all_specs(
    australia_monthly: pd.DataFrame,
    australia_quarterly: pd.DataFrame,
    us_monthly: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, dict[str, object]]]]:
    """Benchmark every approved model for every Week 3 target."""

    leaderboard_rows: list[dict[str, object]] = []
    outputs: dict[str, dict[str, dict[str, object]]] = {}
    for spec in AUSTRALIA_FORECAST_SPECS.values():
        label_outputs: dict[str, dict[str, object]] = {}
        for model in MODEL_LABELS:
            try:
                forecast_result, backtest = compute_spec_outputs(
                    spec,
                    australia_monthly,
                    australia_quarterly,
                    us_monthly,
                    model=model,
                )
            except Exception as exc:
                leaderboard_rows.append(
                    {
                        "series": spec.label,
                        "frequency": spec.frequency,
                        "target": spec.target,
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "status": "failed",
                        "error": str(exc),
                        "target_mae": np.nan,
                        "target_rmse": np.nan,
                        "level_mae": np.nan,
                        "level_rmse": np.nan,
                        "ranking_metric": np.nan,
                    }
                )
                continue
            label_outputs[model] = {
                "forecast_result": forecast_result,
                "backtest": backtest,
            }
            has_level_backtest = (
                "absolute_level_error" in backtest
                and backtest["absolute_level_error"].notna().any()
            )
            leaderboard_rows.append(
                {
                    "series": spec.label,
                    "frequency": spec.frequency,
                    "target": spec.target,
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "status": "ok",
                    "error": "",
                    "target_mae": float(backtest["absolute_error"].dropna().mean()),
                    "target_rmse": _rmse(backtest["error"]),
                    "level_mae": (
                        float(backtest["absolute_level_error"].dropna().mean())
                        if has_level_backtest
                        else np.nan
                    ),
                    "level_rmse": _rmse(backtest["level_error"])
                    if "level_error" in backtest and backtest["level_error"].notna().any()
                    else np.nan,
                    "ranking_metric": _ranking_metric(backtest),
                }
            )
        outputs[spec.label] = label_outputs
    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        ["series", "ranking_metric", "model_label"],
        na_position="last",
    )
    return leaderboard.reset_index(drop=True), outputs


def _resolve(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return Path(__file__).resolve().parents[3] / resolved


def write_benchmark_outputs(
    leaderboard: pd.DataFrame,
    outputs: dict[str, dict[str, dict[str, object]]],
    *,
    output_dir: str | Path = DEFAULT_FORECASTS_DIR,
) -> dict[str, Path]:
    """Write benchmark leaderboards plus per-series forecast and backtest outputs."""

    root = _resolve(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    leaderboard_path = root / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    written["leaderboard"] = leaderboard_path
    for label, model_outputs in outputs.items():
        label_slug = results_slug(label)
        for model, bundle in model_outputs.items():
            forecast_result = bundle["forecast_result"]
            backtest = bundle["backtest"]
            forecast_path = root / f"{label_slug}__{model}__forecast.csv"
            display_path = root / f"{label_slug}__{model}__display_forecast.csv"
            backtest_path = root / f"{label_slug}__{model}__backtest.csv"
            forecast_result.target_forecast.reset_index().to_csv(forecast_path, index=False)
            forecast_result.display_forecast.reset_index().to_csv(display_path, index=False)
            backtest.reset_index().to_csv(backtest_path, index=False)
            written[f"{label_slug}_{model}_forecast"] = forecast_path
            written[f"{label_slug}_{model}_display"] = display_path
            written[f"{label_slug}_{model}_backtest"] = backtest_path
    return written


def load_saved_leaderboard(
    *,
    output_dir: str | Path = DEFAULT_FORECASTS_DIR,
) -> pd.DataFrame | None:
    """Load the saved leaderboard when it exists."""

    primary = _resolve(output_dir) / "leaderboard.csv"
    if primary.exists():
        return pd.read_csv(primary)
    fallback = _resolve(DEFAULT_LEADERBOARD_FIXTURE)
    if fallback.exists():
        return pd.read_csv(fallback)
    return None
