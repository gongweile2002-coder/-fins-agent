# ruff: noqa
"""Pure insight, metric, and figure helpers for the Australia macro forecast app."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from fins2026.week3.app.app_config import (
    ALL_SPECS,
    AUSTRALIA_STATE_OPTIONS,
    FORECAST_LABELS,
    US_CONTEXT_LABEL_MAP,
)
from fintools.apps import (
    MetricCard,
    add_nber_recession_vrects,
    apply_app_plotly_theme,
    target_name,
)

CONTEXT_UNITS = {
    "Participation rate": "Percent",
    "Employment-to-population ratio": "Percent",
    "Trimmed mean inflation": "Percent",
    "Vacancies to labour force ratio": "Percent",
}


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
    frame: pd.DataFrame, column: str
) -> tuple[float | None, pd.Timestamp | None]:
    """Return the latest numeric value and date for a dataframe column."""

    if column not in frame:
        return None, None
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        return None, None
    return float(series.iloc[-1]), pd.Timestamp(series.index[-1])


def format_observation_date(date: pd.Timestamp | None) -> str:
    """Format a latest-observation date for app copy."""

    if date is None or pd.isna(date):
        return "n/a"
    return f"{pd.Timestamp(date):%Y-%m-%d}"


def format_percent(value: float | None, *, signed: bool = False) -> str:
    """Format a percentage value for app-facing text."""

    if value is None or pd.isna(value):
        return "n/a"
    sign = "+" if signed else ""
    return f"{value:{sign},.2f}%"


def format_percentage_point(value: float | None, *, signed: bool = True) -> str:
    """Format a percentage-point value for compact cards."""

    if value is None or pd.isna(value):
        return "n/a"
    sign = "+" if signed else ""
    return f"{value:{sign},.2f} p.p."


def format_basis_points(value: float | None) -> str:
    """Format basis-point changes for compact cards."""

    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:+,.0f} bp"


def display_units(units: str) -> str:
    """Return compact app-facing unit labels."""

    if units == "$ million":
        return "$ million"
    if units == "Percentage points":
        return "percentage points"
    return units


def target_short_label(spec_label: str) -> str:
    """Return a short target label that fits in compact app cards."""

    spec = FORECAST_LABELS[spec_label]
    if spec.target == "change":
        return "Change"
    if spec.target == "log_change":
        return "Log change"
    if spec.target == "annualized_growth":
        return "Ann. growth"
    if spec.target == "level":
        return "Level"
    return "Not forecast"


def _state_metric(
    frame: pd.DataFrame,
    column: str,
    *,
    label: str,
    formatter,
    help_text: str,
) -> MetricCard:
    value, date = latest_observation(frame, column)
    return MetricCard(
        label,
        formatter(value),
        help=f"{help_text} Latest observation: {format_observation_date(date)}.",
    )


def top_state_metrics(
    australia_monthly: pd.DataFrame,
    australia_quarterly: pd.DataFrame,
    *,
    active_data_mode: str,
    sample_period: str,
) -> list[MetricCard]:
    """Build compact top-level app state cards."""

    return [
        MetricCard(
            "Data source",
            active_data_mode,
            help=(
                "Fixture uses frozen Australia and U.S. panels; "
                "Live rebuilds them from RBA and FRED."
            ),
        ),
        MetricCard("Sample", sample_period, help=sample_window_label(australia_monthly)),
        _state_metric(
            australia_monthly,
            "Cash rate target",
            label="Cash rate target",
            formatter=lambda value: format_percent(value),
            help_text="Australia cash rate target at the latest observable month-end.",
        ),
        _state_metric(
            australia_monthly,
            "Unemployment rate",
            label="Unemployment rate",
            formatter=lambda value: format_percent(value),
            help_text="Australia unemployment rate at the latest observable month-end.",
        ),
        _state_metric(
            australia_quarterly,
            "Headline CPI inflation",
            label="Headline CPI inflation",
            formatter=lambda value: format_percent(value),
            help_text=(
                "Published year-ended headline CPI inflation at the latest release month-end."
            ),
        ),
        _state_metric(
            australia_quarterly,
            "Real GDP",
            label="Real GDP",
            formatter=lambda value: f"{value:,.0f}" if value is not None else "n/a",
            help_text="Real GDP level at the latest quarterly release month-end.",
        ),
    ]


def latest_snapshot(
    australia_monthly: pd.DataFrame, australia_quarterly: pd.DataFrame
) -> pd.DataFrame:
    """Build an app-facing latest-state table for Australia series."""

    rows: list[dict[str, object]] = []
    extra = {label: label for label in AUSTRALIA_STATE_OPTIONS if label not in ALL_SPECS}
    for label, spec in ALL_SPECS.items():
        frame = australia_quarterly if spec.frequency == "quarterly" else australia_monthly
        value, date = latest_observation(frame, spec.series_id)
        if value is None:
            continue
        rows.append(
            {
                "indicator": spec.label,
                "latest_date": date,
                "latest_value": value,
                "units": display_units(spec.units),
                "forecast_treatment": target_name(spec),
            }
        )
    for label in extra:
        value, date = latest_observation(australia_monthly, label)
        if value is None:
            continue
        rows.append(
            {
                "indicator": label,
                "latest_date": date,
                "latest_value": value,
                "units": display_units(CONTEXT_UNITS.get(label, "")),
                "forecast_treatment": "Context only",
            }
        )
    return pd.DataFrame(rows)


def us_context_snapshot(us_monthly: pd.DataFrame) -> pd.DataFrame:
    """Build a latest-state table for key U.S. context series."""

    rows: list[dict[str, object]] = []
    for column, label in US_CONTEXT_LABEL_MAP.items():
        value, date = latest_observation(us_monthly, column)
        if value is None:
            continue
        rows.append(
            {
                "indicator": label,
                "latest_date": date,
                "latest_value": value,
            }
        )
    return pd.DataFrame(rows)


def best_models_summary(leaderboard: pd.DataFrame) -> pd.DataFrame:
    """Return the best successful benchmark row for each Australia target."""

    ok = leaderboard.loc[leaderboard["status"] == "ok"].copy()
    if ok.empty:
        return ok
    best = ok.sort_values(["series", "ranking_metric"]).groupby("series", as_index=False).first()
    return best[["series", "model_label", "ranking_metric", "target_mae", "level_mae"]]


def comparison_table(leaderboard: pd.DataFrame, series_label: str) -> pd.DataFrame:
    """Return the benchmark rows for one selected series."""

    rows = leaderboard.loc[leaderboard["series"] == series_label].copy()
    if rows.empty:
        return rows
    return rows.sort_values(["status", "ranking_metric", "model_label"])[
        [
            "model_label",
            "status",
            "target_mae",
            "target_rmse",
            "level_mae",
            "level_rmse",
            "ranking_metric",
        ]
    ]


def level_or_target_backtest_frame(backtest: pd.DataFrame) -> pd.DataFrame:
    """Return the plotted actual-versus-forecast series for the backtest figure."""

    has_level_path = {"actual_level", "forecast_level"}.issubset(backtest.columns)
    if has_level_path and backtest["forecast_level"].notna().any():
        return pd.DataFrame(
            {
                "actual": backtest["actual_level"],
                "forecast": backtest["forecast_level"],
            },
            index=backtest.index,
        )
    return pd.DataFrame(
        {"actual": backtest["actual"], "forecast": backtest["forecast"]},
        index=backtest.index,
    )


def line_figure(
    series: pd.Series,
    *,
    indicator_name: str,
    units: str,
    shade_recessions: bool = False,
) -> go.Figure:
    """Build a standard line figure for one app-facing series."""

    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=clean.index,
            y=clean,
            name=indicator_name,
            mode="lines",
            line={"color": "#A51C30", "width": 2.5},
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.2f}<extra></extra>",
        )
    )
    if shade_recessions and not clean.empty:
        add_nber_recession_vrects(fig, start=clean.index.min(), end=clean.index.max())
    apply_app_plotly_theme(
        fig,
        yaxis_title=units,
        height=430,
    )
    return fig


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


def forecast_summary_text(spec_label: str, result, *, model_label: str, horizon: int) -> str:
    """Return a client-facing forecast summary."""

    spec = FORECAST_LABELS[spec_label]
    latest_level = float(result.observed_level.dropna().iloc[-1])
    implied_change = forecast_level_change_percent(result, latest_level)
    direction = "higher" if (implied_change or 0.0) >= 0 else "lower"
    change_text = format_percent(abs(implied_change) if implied_change is not None else None)
    horizon_label = "quarter" if spec.frequency == "quarterly" else "month"
    return (
        f"The selected {model_label} path treats the target as {target_name(spec).lower()} "
        f"and implies {spec.label.lower()} is {change_text} {direction} after "
        f"{horizon} {horizon_label}{'' if horizon == 1 else 's'}. "
        "Forecast bands are approximate uncertainty guides rather than confidence guarantees."
    )


def ranking_metric_text(row: pd.Series) -> str:
    """Return a short explanation of the benchmark metric."""

    if pd.notna(row.get("level_mae")):
        return f"Level MAE {row['level_mae']:.3f}"
    return f"Target MAE {row['target_mae']:.3f}"


def australia_overview_text(
    australia_monthly: pd.DataFrame,
    australia_quarterly: pd.DataFrame,
    sample_period: str,
) -> str:
    """Return a short overview of the Australia macro forecast surface."""

    cash_rate, cash_date = latest_observation(australia_monthly, "Cash rate target")
    unemployment, unemp_date = latest_observation(australia_monthly, "Unemployment rate")
    cpi, cpi_date = latest_observation(australia_quarterly, "Headline CPI inflation")
    return (
        f"The selected {sample_period} Australia sample runs from "
        f"{sample_window_label(australia_monthly)}. Latest cash rate target: "
        f"{format_percent(cash_rate)} as of {format_observation_date(cash_date)}. "
        f"Latest unemployment rate: {format_percent(unemployment)} as of "
        f"{format_observation_date(unemp_date)}. Latest headline CPI inflation: "
        f"{format_percent(cpi)} as of {format_observation_date(cpi_date)}."
    )
