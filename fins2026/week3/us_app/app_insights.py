# ruff: noqa
"""Pure insight, metric, and figure helpers for the Week 3 companion U.S. app."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from fins2026.week3.us_app.app_config import (
    ALL_SPECS,
    FORECAST_LABELS,
    STRESS_CURRENT_WINDOW,
)
from fintools.apps import (
    MetricCard,
    add_nber_recession_vrects,
    apply_app_plotly_theme,
    forecast_series_spec,
    latest_delta,
    latest_percentile,
    rolling_backtest_spec,
    target_name,
)


def one_year_delta_label(series: pd.Series, *, periods: int) -> str | None:
    """Return a formatted one-year change label."""

    delta = latest_delta(series, periods=periods)
    if delta is None:
        return None
    return f"{delta:+,.2f}"


def _backtest_settings(spec_label: str, series: pd.Series) -> tuple[int, int]:
    """Choose a backtest window that matches the data frequency."""

    spec = FORECAST_LABELS[spec_label]
    if spec.role == "macro":
        return max(8, min(24, max(len(series) // 2, 1))), 1
    return max(60, min(250, max(len(series) // 2, 1))), 21


def _metric_periods(spec_label: str) -> int:
    spec = FORECAST_LABELS[spec_label]
    return 4 if spec.role == "macro" else 252


def sample_window_label(frame: pd.DataFrame) -> str:
    """Return a compact date span for the selected sample."""

    if frame.empty:
        return "n/a"
    return f"{frame.index.min():%Y-%m-%d} to {frame.index.max():%Y-%m-%d}"


def series_window_label(series: pd.Series) -> str:
    """Return a compact date span for a date-indexed series."""

    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if clean.empty:
        return "n/a"
    return f"{clean.index.min():%Y-%m-%d} to {clean.index.max():%Y-%m-%d}"


def compact_table_height(
    frame: pd.DataFrame,
    *,
    row_height: int = 35,
    header_height: int = 38,
    min_height: int = 118,
    max_height: int = 520,
) -> int:
    """Return a compact Streamlit dataframe height without blank grid rows."""

    if frame.empty:
        return min_height
    return min(max_height, max(min_height, header_height + row_height * len(frame)))


def latest_observation(
    frame: pd.DataFrame,
    column: str,
) -> tuple[float | None, pd.Timestamp | None]:
    """Return the latest available numeric value and date for a column."""

    if column not in frame:
        return None, None
    series = frame[column].dropna()
    if series.empty:
        return None, None
    return float(series.iloc[-1]), pd.Timestamp(series.index[-1])


def format_observation_date(date: pd.Timestamp | None) -> str:
    """Format a latest-observation date for app copy."""

    if date is None or pd.isna(date):
        return "n/a"
    return f"{pd.Timestamp(date):%Y-%m-%d}"


def display_units(units: str) -> str:
    """Return compact app-facing unit labels."""

    if units == "Billions of chained 2017 dollars":
        return "bn chained 2017 $"
    if units == "Percentage points":
        return "percentage points"
    return units


def format_percent(value: float | None, *, signed: bool = False) -> str:
    """Format a percentage value for app-facing text."""

    if value is None or pd.isna(value):
        return "n/a"
    sign = "+" if signed else ""
    return f"{value:{sign},.1f}%"


def format_percentage_point(value: float | None, *, signed: bool = True) -> str:
    """Format a percentage-point value for compact cards."""

    if value is None or pd.isna(value):
        return "n/a"
    sign = "+" if signed else ""
    return f"{value:{sign},.2f} p.p."


def target_short_label(spec_label: str) -> str:
    """Return a short target label that fits in compact app cards."""

    spec = FORECAST_LABELS[spec_label]
    if spec.target == "change":
        return "Change"
    if spec.target == "annualized_growth":
        return "Growth"
    if spec.target == "level":
        return "Level"
    return "Not forecast"


def top_state_metrics(
    frame: pd.DataFrame,
    *,
    active_data_mode: str,
    sample_period: str,
) -> list[MetricCard]:
    """Build compact top-level app state cards."""

    score = build_stress_score(frame)
    current_stress = current_stress_average(score)
    current_percentile = current_stress_percentile(score)
    stress_window = stress_window_label(score)
    spread, spread_date = latest_observation(frame, "T10Y2Y")
    vix, vix_date = latest_observation(frame, "VIXCLS")
    gdp, gdp_date = latest_observation(frame, "GDPC1")
    return [
        MetricCard(
            "Data source",
            active_data_mode,
            help="Fixture is a stable built-in sample; Live FRED pulls the latest available data.",
        ),
        MetricCard("Sample", sample_period, help=sample_window_label(frame)),
        MetricCard(
            f"Current stress vs {sample_period} (z)",
            format_stress_value(current_stress),
            help=(
                f"Latest 21-trading-day average, using {stress_window}. This "
                "z-score is recalculated from the selected sample's mean and "
                "standard deviation, so it changes when the sample period "
                "control changes. "
                f"Current standing: {format_percentile_label(current_percentile)}."
            ),
        ),
        MetricCard(
            "Current 10Y-2Y spread",
            f"{spread:,.2f}%" if spread is not None else "n/a",
            help=(
                "10-year Treasury yield minus 2-year Treasury yield; "
                f"latest observation as of {format_observation_date(spread_date)}."
            ),
        ),
        MetricCard(
            "Current VIX",
            f"{vix:,.1f}%" if vix is not None else "n/a",
            help=(
                "CBOE VIX, quoted as annualized expected volatility; "
                "higher readings indicate more expected volatility; "
                f"latest observation as of {format_observation_date(vix_date)}."
            ),
        ),
        MetricCard(
            "Current real GDP (bn 2017 $)",
            f"{gdp:,.0f}" if gdp is not None else "n/a",
            help=(
                "Quarterly real GDP in billions of chained 2017 dollars; "
                f"latest observation as of {format_observation_date(gdp_date)}."
            ),
        ),
    ]


def latest_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    """Build an app-facing latest-state table."""

    rows: list[dict[str, object]] = []
    for series_id, spec in ALL_SPECS.items():
        series = frame[series_id].dropna() if series_id in frame else pd.Series(dtype=float)
        if series.empty:
            continue
        rows.append(
            {
                "indicator": spec.label,
                "latest_date": series.index[-1],
                "latest_value": float(series.iloc[-1]),
                "units": display_units(spec.units),
                "forecast_treatment": (
                    "Not forecast" if not spec.allow_forecast else target_name(spec)
                ),
                "percentile": latest_percentile(series),
            }
        )
    return pd.DataFrame(rows)


def build_stress_score(frame: pd.DataFrame) -> pd.Series:
    """Construct a transparent stress-state composite from available indicators."""

    pieces: list[pd.Series] = []
    inputs = {"VIXCLS": 1.0, "BAMLH0A0HYM2": 1.0, "T10Y2Y": -1.0}
    for column, sign in inputs.items():
        series = frame[column].dropna() if column in frame else pd.Series(dtype=float)
        if len(series) < 10:
            continue
        std = float(series.std(ddof=1))
        if std <= 0 or pd.isna(std):
            continue
        pieces.append(sign * (series - float(series.mean())) / std)
    if not pieces:
        return pd.Series(dtype=float)
    return (
        pd.concat(pieces, axis=1, sort=False)
        .mean(axis=1)
        .dropna()
        .sort_index()
        .rename("stress_score")
    )


def format_stress_value(value: float | None) -> str:
    """Format a stress z-score for compact app cards."""

    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:,.2f}"


def current_stress_average(
    score: pd.Series,
    *,
    window: int = STRESS_CURRENT_WINDOW,
) -> float | None:
    """Return the latest rolling average stress score."""

    rolling = stress_rolling_average(score, window=window)
    if rolling.empty:
        return None
    return float(rolling.iloc[-1])


def stress_rolling_average(
    score: pd.Series,
    *,
    window: int = STRESS_CURRENT_WINDOW,
) -> pd.Series:
    """Return historical rolling averages used for current-stress standing."""

    clean = pd.to_numeric(score, errors="coerce").dropna().sort_index()
    if clean.empty:
        return pd.Series(dtype=float, name="stress_21d_average")
    min_periods = min(window, len(clean))
    return clean.rolling(window, min_periods=min_periods).mean().dropna().rename(
        "stress_21d_average"
    )


def current_stress_window(
    score: pd.Series,
    *,
    window: int = STRESS_CURRENT_WINDOW,
) -> pd.Series:
    """Return the observations used for the current stress average."""

    clean = pd.to_numeric(score, errors="coerce").dropna().sort_index()
    if clean.empty:
        return pd.Series(dtype=float)
    return clean.tail(min(window, len(clean)))


def stress_window_label(
    score: pd.Series,
    *,
    window: int = STRESS_CURRENT_WINDOW,
) -> str:
    """Return the date span behind the current stress average."""

    current_window = current_stress_window(score, window=window)
    if current_window.empty:
        return "n/a"
    start = pd.Timestamp(current_window.index[0])
    end = pd.Timestamp(current_window.index[-1])
    if start == end:
        return f"{end:%Y-%m-%d}"
    return f"{start:%Y-%m-%d} to {end:%Y-%m-%d}"


def stress_latest_date(score: pd.Series) -> pd.Timestamp | None:
    """Return the latest stress-score date."""

    current_window = current_stress_window(score)
    if current_window.empty:
        return None
    return pd.Timestamp(current_window.index[-1])


def prior_stress_average(
    score: pd.Series,
    *,
    window: int = STRESS_CURRENT_WINDOW,
) -> float | None:
    """Return the previous rolling average stress score."""

    clean = pd.to_numeric(score, errors="coerce").dropna().sort_index()
    if len(clean) <= window:
        return None
    prior = clean.iloc[-2 * window : -window]
    if prior.empty:
        return None
    return float(prior.mean())


def percentile_rank_for_value(score: pd.Series, value: float | None) -> float | None:
    """Return the share of observations less than or equal to a supplied value."""

    if value is None or pd.isna(value):
        return None
    clean = pd.to_numeric(score, errors="coerce").dropna()
    if clean.empty:
        return None
    return float((clean <= value).mean() * 100.0)


def current_stress_percentile(score: pd.Series) -> float | None:
    """Compare current stress with historical 21-trading-day averages."""

    current_stress = current_stress_average(score)
    return percentile_rank_for_value(stress_rolling_average(score), current_stress)


def stress_band_label(percentile: float | None) -> str:
    """Describe stress percentile in client-facing language."""

    if percentile is None or pd.isna(percentile):
        return "n/a"
    if percentile >= 80:
        return "Elevated"
    if percentile >= 60:
        return "Above typical"
    if percentile <= 20:
        return "Low"
    if percentile <= 40:
        return "Below typical"
    return "Typical"


def format_percentile_label(percentile: float | None) -> str:
    """Format a percentile with an interpretation."""

    if percentile is None or pd.isna(percentile):
        return "n/a"
    return f"{stress_band_label(percentile)} ({percentile:,.0f}%)"


def stress_percentile_help(percentile: float | None, sample_period: str) -> str:
    """Explain the stress percentile without statistical shorthand."""

    if percentile is None or pd.isna(percentile):
        return "Not enough data to compare current stress with the selected sample."
    return (
        "This is the percentile of the current 21-trading-day average stress "
        "score versus historical 21-trading-day averages in the selected "
        f"{sample_period} sample. It is higher than {percentile:,.0f}% of "
        "those historical 21-trading-day averages."
    )


def stress_band_legend() -> str:
    """Return the score-band thresholds used in app copy."""

    return (
        "Bands: Low <=20%, Below typical 21-40%, Typical 41-59%, "
        "Above typical 60-79%, Elevated >=80%."
    )


def stress_overview_text(score: pd.Series, sample_period: str) -> str:
    """Summarize the current stress score for the Overview tab."""

    current_stress = current_stress_average(score)
    current_percentile = current_stress_percentile(score)
    if current_stress is None or current_percentile is None:
        return "Not enough stress data are available for the selected sample."
    return (
        f"Current stress vs {sample_period}: {format_stress_value(current_stress)} "
        f"through {format_observation_date(stress_latest_date(score))}, using "
        f"observations from {stress_window_label(score)}. This falls in the "
        f"{stress_band_label(current_percentile)} band and is higher than "
        f"{current_percentile:,.0f}% of historical 21-trading-day average stress "
        f"readings in the selected {sample_period} sample. Changing the sample "
        "period changes the selected-sample mean and standard deviation used "
        "for z-scores; it is not a data-availability change. "
        f"{stress_band_legend()}"
    )


def stress_score_figure(score: pd.Series) -> go.Figure:
    """Plot the stress composite with app chart defaults."""

    score = pd.to_numeric(score, errors="coerce").dropna().sort_index()
    rolling_score = stress_rolling_average(score)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=score.index,
            y=score,
            name="Daily score",
            mode="lines",
            line={"color": "rgba(53, 92, 125, 0.36)", "width": 1.25},
            hovertemplate="%{x|%Y-%m-%d}<br>Score: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rolling_score.index,
            y=rolling_score,
            name="1M average",
            mode="lines",
            line={"color": "#A51C30", "width": 2.5},
            hovertemplate="%{x|%Y-%m-%d}<br>1M avg: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line={"color": "#6B6F76", "width": 1, "dash": "dot"})
    if not score.empty:
        add_nber_recession_vrects(fig, start=score.index.min(), end=score.index.max())
    apply_app_plotly_theme(fig, yaxis_title="Stress score (z)", height=470)
    return fig


def yield_curve_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return latest and one-year-ago Treasury curve points."""

    curve_columns = [("3M", 0.25, "DTB3"), ("2Y", 2.0, "DGS2"), ("10Y", 10.0, "DGS10")]
    latest_date = frame[["DTB3", "DGS2", "DGS10"]].dropna(how="all").index.max()
    earlier_date = latest_date - pd.DateOffset(years=1)
    rows: list[dict[str, object]] = []
    for label, maturity, column in curve_columns:
        series = frame[column].dropna()
        current_values = series.loc[:latest_date]
        earlier_values = series.loc[:earlier_date]
        current = current_values.iloc[-1] if not current_values.empty else np.nan
        earlier = earlier_values.iloc[-1] if not earlier_values.empty else np.nan
        rows.append(
            {
                "maturity": label,
                "years": maturity,
                "latest": current,
                "one_year_ago": earlier,
            }
        )
    return pd.DataFrame(rows)


def _curve_value(curve: pd.DataFrame, maturity: str, column: str) -> float | None:
    """Return a curve value for a maturity and column."""

    values = curve.loc[curve["maturity"] == maturity, column]
    if values.empty or pd.isna(values.iloc[0]):
        return None
    return float(values.iloc[0])


def curve_shape_label(
    ten_year_two_year: float | None,
    ten_year_three_month: float | None,
) -> str:
    """Classify the Treasury curve shape from common spread measures."""

    spreads = [value for value in [ten_year_two_year, ten_year_three_month] if value is not None]
    if not spreads:
        return "n/a"
    if any(value < -0.25 for value in spreads):
        return "Inverted"
    if any(abs(value) <= 0.25 for value in spreads):
        return "Flat"
    if all(value > 0.25 for value in spreads):
        return "Upward sloping"
    return "Mixed"


def curve_shift_label(spread_change: float | None) -> str:
    """Classify how a yield spread changed over the comparison period."""

    if spread_change is None or pd.isna(spread_change):
        return "n/a"
    if spread_change >= 0.25:
        return "Steepened"
    if spread_change <= -0.25:
        return "Flattened"
    return "Little changed"


def yield_curve_insights(curve: pd.DataFrame) -> dict[str, float | str | None]:
    """Compute app-facing yield-curve diagnostics."""

    latest_3m = _curve_value(curve, "3M", "latest")
    latest_2y = _curve_value(curve, "2Y", "latest")
    latest_10y = _curve_value(curve, "10Y", "latest")
    prior_3m = _curve_value(curve, "3M", "one_year_ago")
    prior_2y = _curve_value(curve, "2Y", "one_year_ago")
    prior_10y = _curve_value(curve, "10Y", "one_year_ago")

    spread_10y_2y = None if latest_10y is None or latest_2y is None else latest_10y - latest_2y
    spread_10y_3m = None if latest_10y is None or latest_3m is None else latest_10y - latest_3m
    prior_spread_10y_2y = (
        None if prior_10y is None or prior_2y is None else prior_10y - prior_2y
    )
    prior_spread_10y_3m = (
        None if prior_10y is None or prior_3m is None else prior_10y - prior_3m
    )
    spread_10y_2y_change = (
        None
        if spread_10y_2y is None or prior_spread_10y_2y is None
        else spread_10y_2y - prior_spread_10y_2y
    )
    spread_10y_3m_change = (
        None
        if spread_10y_3m is None or prior_spread_10y_3m is None
        else spread_10y_3m - prior_spread_10y_3m
    )
    change_3m = None if latest_3m is None or prior_3m is None else latest_3m - prior_3m
    change_10y = None if latest_10y is None or prior_10y is None else latest_10y - prior_10y

    return {
        "shape": curve_shape_label(spread_10y_2y, spread_10y_3m),
        "spread_10y_2y": spread_10y_2y,
        "spread_10y_3m": spread_10y_3m,
        "spread_10y_2y_change": spread_10y_2y_change,
        "spread_10y_3m_change": spread_10y_3m_change,
        "change_3m": change_3m,
        "change_10y": change_10y,
        "shift": curve_shift_label(spread_10y_3m_change),
    }


def yield_curve_interpretation(curve: pd.DataFrame) -> str:
    """Return a client-facing yield-curve interpretation."""

    insights = yield_curve_insights(curve)
    shape = str(insights["shape"])
    spread_10y_2y = insights["spread_10y_2y"]
    spread_10y_3m = insights["spread_10y_3m"]
    change_3m = insights["change_3m"]
    change_10y = insights["change_10y"]
    shift = str(insights["shift"]).lower()

    if shape == "Inverted":
        shape_read = (
            "The curve is inverted, so at least one short rate is above the "
            "10-year yield. That is a warning sign for tight near-term policy "
            "and weaker growth expectations."
        )
    elif shape == "Flat":
        shape_read = (
            "The curve is flat, so long rates are only slightly above short "
            "rates. That usually points to an uncertain transition between "
            "near-term policy pressure and longer-run growth expectations."
        )
    elif shape == "Upward sloping":
        shape_read = (
            "The curve is upward sloping, so long rates are above short rates. "
            "That is less recession-warning than an inverted curve and suggests "
            "investors require extra yield for longer maturity risk."
        )
    else:
        shape_read = (
            "The curve has a mixed shape, so the 2-year and 3-month comparisons "
            "send different signals."
        )

    return (
        f"{shape_read} The latest 10Y-2Y slope is "
        f"{format_percentage_point(spread_10y_2y)} and the latest 10Y-3M slope "
        f"is {format_percentage_point(spread_10y_3m)}. Over the past year, the "
        f"3-month yield moved {format_percentage_point(change_3m)} and the "
        f"10-year yield moved {format_percentage_point(change_10y)}, so the "
        f"10Y-3M curve has {shift}."
    )


def yield_curve_figure(curve: pd.DataFrame) -> go.Figure:
    """Plot a compact Treasury curve comparison."""

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve["years"],
            y=curve["latest"],
            name="Latest",
            mode="lines+markers",
            line={"color": "#355C7D", "width": 3},
            marker={"size": 8},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=curve["years"],
            y=curve["one_year_ago"],
            name="One year ago",
            mode="lines+markers",
            line={"color": "#A51C30", "width": 2, "dash": "dot"},
            marker={"size": 7},
        )
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=curve["years"],
        ticktext=curve["maturity"],
        showgrid=False,
        title="Maturity",
    )
    fig.update_layout(
        template="plotly_white",
        height=430,
        margin={"l": 32, "r": 28, "t": 34, "b": 32},
        legend={
            "orientation": "h",
            "y": 1.08,
            "x": 0,
            "xanchor": "left",
            "title": None,
        },
        hovermode="x unified",
        font={"color": "#262A33"},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_yaxes(
        title="Percent",
        showgrid=True,
        gridcolor="#E2E6EA",
        zerolinecolor="#E2E6EA",
    )
    return fig


def forecast_and_backtest(
    series: pd.Series,
    spec_label: str,
    *,
    model: str,
    horizon: int,
):
    """Run the app forecast and matching backtest."""

    spec = FORECAST_LABELS[spec_label]
    min_train, step = _backtest_settings(spec_label, series)
    forecast_result = forecast_series_spec(series, spec, model=model, horizon=horizon)
    backtest = rolling_backtest_spec(
        series,
        spec,
        model=model,
        horizon=1,
        min_train=min_train,
        step=step,
    )
    return forecast_result, backtest


def latest_annualized_quarterly_growth(series: pd.Series) -> float | None:
    """Return latest annualized quarter-over-quarter growth in percent."""

    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if len(clean) < 2:
        return None
    return float(((clean.iloc[-1] / clean.iloc[-2]) ** 4 - 1.0) * 100.0)


def latest_year_over_year_growth(series: pd.Series) -> float | None:
    """Return latest year-over-year growth in percent."""

    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if len(clean) < 5:
        return None
    return float((clean.iloc[-1] / clean.iloc[-5] - 1.0) * 100.0)


def gdp_growth_band(value: float | None) -> str:
    """Classify GDP growth into client-facing bands."""

    if value is None or pd.isna(value):
        return "n/a"
    if value < 0:
        return "Contraction"
    if value < 1:
        return "Slow growth"
    if value < 3:
        return "Moderate growth"
    return "Strong growth"


def forecast_level_change_percent(result, latest_level: float) -> float | None:
    """Return the percent level change implied by a display forecast."""

    forecast = result.display_forecast["forecast"].dropna()
    if forecast.empty or pd.isna(latest_level) or latest_level == 0:
        return None
    return float((forecast.iloc[-1] / latest_level - 1.0) * 100.0)


def gdp_outlook_interpretation(
    series: pd.Series,
    result,
    *,
    model_label: str,
    horizon: int,
) -> str:
    """Return a client-facing GDP interpretation."""

    latest_growth = latest_annualized_quarterly_growth(series)
    yoy_growth = latest_year_over_year_growth(series)
    latest_level = float(series.dropna().iloc[-1])
    forecast_change = forecast_level_change_percent(result, latest_level)
    band = gdp_growth_band(latest_growth)
    forecast_direction = "higher" if (forecast_change or 0.0) >= 0 else "lower"
    forecast_change_text = format_percent(
        abs(forecast_change) if forecast_change is not None else None
    )
    return (
        f"Latest annualized quarterly growth is {format_percent(latest_growth)}, "
        f"which is a {band.lower()} reading. Year-over-year real GDP growth is "
        f"{format_percent(yoy_growth)}. The selected {model_label} path implies "
        f"real GDP is {forecast_change_text} {forecast_direction} after "
        f"{horizon} quarters. GDP is quarterly, lagged, "
        "and revised, so this is latest-available analysis rather than a "
        "real-time vintage nowcast."
    )
