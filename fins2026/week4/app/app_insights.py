# ruff: noqa
"""Pure figures, metrics, and formatting helpers for the Week 4 app."""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go

from fins2026.week4.app.app_config import (
    CONSTRAINT_OPTIONS,
    PORTFOLIO_COLORS,
    PORTFOLIO_DISPLAY_ORDER,
)
from fins2026.week4.code.stage3_portfolios import Stage3Sample, drawdown_series
from fins2026.week4.code.stage4_app import (
    annualized_rfr_pct,
    frontier_point_coordinates,
    mean_variance_sharpe,
)
from fintools.apps import MetricCard, add_nber_recession_vrects, apply_app_plotly_theme


def compact_table_height(
    frame: pd.DataFrame,
    *,
    row_height: int = 35,
    header_height: int = 38,
    min_height: int = 118,
    max_height: int = 520,
) -> int:
    """Return a compact Streamlit dataframe height without blank rows."""

    if frame.empty:
        return min_height
    return min(max_height, max(min_height, header_height + row_height * len(frame)))


def sample_window_label(sample: Stage3Sample) -> str:
    """Return a compact date span for the balanced in-sample window."""

    return f"{sample.start_date:%Y-%m-%d} to {sample.end_date:%Y-%m-%d}"


def format_percent(value: float | None, *, signed: bool = False, decimals: int = 1) -> str:
    """Format a decimal value as a percentage string."""

    if value is None or pd.isna(value):
        return "n/a"
    sign = "+" if signed else ""
    return f"{value:{sign},.{decimals}f}%"


def format_ratio(value: float | None) -> str:
    """Format a ratio metric such as the Sharpe ratio."""

    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:,.2f}"


def _format_growth_tick(value: float) -> str:
    """Format one log-scale growth tick as a readable dollar label."""

    if value >= 10:
        return f"${value:,.0f}"
    if value >= 1:
        return f"${value:,.1f}".rstrip("0").rstrip(".")
    return f"${value:,.2f}".rstrip("0").rstrip(".")


def _growth_axis_ticks(values: pd.Series) -> tuple[list[float], list[str]]:
    """Return readable log-scale ticks for a growth-of-one-dollar chart."""

    positive = values.loc[values > 0].dropna().astype(float)
    if positive.empty:
        return [1.0], ["$1"]

    lower = float(positive.min()) / 1.05
    upper = float(positive.max()) * 1.05
    start_exp = math.floor(math.log10(lower)) - 1
    end_exp = math.ceil(math.log10(upper)) + 1

    tickvals: list[float] = []
    for exponent in range(start_exp, end_exp + 1):
        scale = 10.0**exponent
        for mantissa in (1.0, 2.0, 5.0):
            value = mantissa * scale
            if lower <= value <= upper:
                tickvals.append(value)

    tickvals = sorted(set(tickvals))
    if not tickvals:
        tickvals = [float(positive.min()), float(positive.max())]

    ticktext = [_format_growth_tick(value) for value in tickvals]
    return tickvals, ticktext


def top_portfolio_metrics(
    metrics: pd.DataFrame,
    *,
    active_portfolio_label: str,
    sample: Stage3Sample,
    constraint_mode: str,
) -> list[MetricCard]:
    """Build the top-line portfolio metric cards."""

    row = metrics.loc[metrics["portfolio"] == active_portfolio_label].iloc[0]
    return [
        MetricCard(
            "Selected portfolio",
            active_portfolio_label,
            help="Active portfolio for the main comparison charts and summary cards.",
        ),
        MetricCard(
            "Sample",
            sample_window_label(sample),
            help="Balanced in-sample return window for the currently selected stock subset.",
        ),
        MetricCard(
            "Constraint mode",
            CONSTRAINT_OPTIONS[constraint_mode],
            help="Optimization rule used for the minimum-variance and mean-variance portfolios.",
        ),
        MetricCard(
            "Annualized return",
            format_percent(float(row["annualized_return_pct"])),
            help="Mean daily portfolio return scaled to an annualized rate.",
        ),
        MetricCard(
            "Annualized volatility",
            format_percent(float(row["annualized_volatility_pct"])),
            help="Daily return volatility scaled by the square root of 252 trading days.",
        ),
        MetricCard(
            "Sharpe ratio",
            format_ratio(float(row["sharpe_ratio"])),
            help=(
                "Annualized excess-return Sharpe ratio against the daily "
                "Kenneth French risk-free rate."
            ),
        ),
    ]


def portfolio_weight_figure(
    weights: pd.DataFrame,
    *,
    portfolio_label: str,
) -> go.Figure:
    """Plot the actual portfolio weights for one selected portfolio."""

    frame = (
        weights.loc[weights["portfolio"] == portfolio_label, ["ticker", "weight"]]
        .copy()
        .sort_values("weight")
        .reset_index(drop=True)
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame["weight"] * 100.0,
            y=frame["ticker"],
            orientation="h",
            marker_color=PORTFOLIO_COLORS[portfolio_label],
            hovertemplate="%{y}<br>Weight: %{x:.2f}%<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_vline(x=0, line_color="#9AA3AD", line_dash="dot")
    fig.update_layout(title={"text": f"{portfolio_label} weights", "x": 0, "xanchor": "left"})
    apply_app_plotly_theme(
        fig,
        yaxis_title=None,
        height=420,
        range_selector=False,
        range_slider=False,
    )
    fig.update_xaxes(title="Weight (%)", showgrid=True)
    fig.update_yaxes(title=None, automargin=True)
    return fig


def cumulative_growth_figure(
    portfolio_returns: pd.DataFrame,
    *,
    sample: Stage3Sample,
) -> go.Figure:
    """Plot growth of one dollar for all portfolio variants."""

    fig = go.Figure()
    wealth_series: list[pd.Series] = []
    for label in PORTFOLIO_DISPLAY_ORDER:
        wealth = (1.0 + portfolio_returns[label].astype(float)).cumprod()
        wealth_series.append(wealth)
        fig.add_trace(
            go.Scatter(
                x=portfolio_returns["date"],
                y=wealth,
                mode="lines",
                name=label,
                line={"width": 2.2, "color": PORTFOLIO_COLORS[label]},
                hovertemplate="%{x|%Y-%m-%d}<br>Growth of $1: %{y:.2f}<extra></extra>",
            )
        )
    add_nber_recession_vrects(fig, start=sample.start_date, end=sample.end_date)
    fig.update_layout(title={"text": "Historical growth of $1", "x": 0, "xanchor": "left"})
    apply_app_plotly_theme(fig, yaxis_title="Growth of $1", height=520, range_slider=False)
    tickvals, ticktext = _growth_axis_ticks(pd.concat(wealth_series, ignore_index=True))
    fig.update_yaxes(
        type="log",
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        title="Growth of $1 (log scale)",
        automargin=True,
    )
    return fig


def drawdown_figure(
    portfolio_returns: pd.DataFrame,
    *,
    sample: Stage3Sample,
) -> go.Figure:
    """Plot drawdowns for all portfolio variants."""

    fig = go.Figure()
    for label in PORTFOLIO_DISPLAY_ORDER:
        drawdown = drawdown_series(portfolio_returns[label].astype(float)) * 100.0
        fig.add_trace(
            go.Scatter(
                x=portfolio_returns["date"],
                y=drawdown,
                mode="lines",
                name=label,
                line={"width": 1.7, "color": PORTFOLIO_COLORS[label]},
                hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.1f}%<extra></extra>",
            )
        )
    add_nber_recession_vrects(fig, start=sample.start_date, end=sample.end_date)
    fig.update_layout(title={"text": "Historical drawdowns", "x": 0, "xanchor": "left"})
    apply_app_plotly_theme(fig, yaxis_title="Drawdown (%)", height=520, range_slider=False)
    fig.update_yaxes(ticksuffix="%")
    return fig


def efficient_frontier_figure(
    frontier: pd.DataFrame,
    asset_stats: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    sample: Stage3Sample,
    active_portfolio_label: str,
) -> go.Figure:
    """Plot the app efficient frontier, asset cloud, CAL line, and portfolio points."""

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=asset_stats["annualized_volatility_pct"],
            y=asset_stats["annualized_return_pct"],
            mode="markers+text",
            text=asset_stats["ticker"],
            textposition="top center",
            marker={
                "size": 10,
                "color": "rgba(120,120,120,0.35)",
                "line": {"color": "rgba(120,120,120,0.65)", "width": 1},
            },
            name="Selected stocks",
            hovertemplate="%{text}<br>Ann. vol: %{x:.1f}%<br>Ann. return: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frontier["volatility_ann_pct"],
            y=frontier["target_return_ann_pct"],
            mode="lines",
            name="Efficient frontier",
            line={"color": "#2F455C", "width": 3},
            hovertemplate="Ann. vol: %{x:.1f}%<br>Ann. return: %{y:.1f}%<extra></extra>",
        )
    )

    rf_ann_pct = annualized_rfr_pct(sample)
    mean_var_vol, mean_var_ret = frontier_point_coordinates(metrics, "Mean-variance")
    fig.add_trace(
        go.Scatter(
            x=[0.0, mean_var_vol * 1.08],
            y=[rf_ann_pct, rf_ann_pct + (mean_var_ret - rf_ann_pct) * 1.08],
            mode="lines",
            name="Tangency line",
            line={"color": "#8E3B46", "dash": "dash", "width": 2.5},
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0.0],
            y=[rf_ann_pct],
            mode="markers+text",
            marker={"size": 16, "color": "#6C675D"},
            text=["Risk-free"],
            textposition="middle right",
            name="Risk-free",
            hovertemplate="Risk-free<br>Ann. return: %{y:.2f}%<extra></extra>",
        )
    )

    point_offsets = {
        "Custom": (18, -18),
        "Equal-weight": (24, 8),
        "Minimum variance": (18, 18),
        "Mean-variance": (16, -26),
    }
    for label in PORTFOLIO_DISPLAY_ORDER:
        vol, ret = frontier_point_coordinates(metrics, label)
        dx, dy = point_offsets[label]
        fig.add_trace(
            go.Scatter(
                x=[vol],
                y=[ret],
                mode="markers",
                marker={
                    "size": 17 if label == active_portfolio_label else 14,
                    "color": PORTFOLIO_COLORS[label],
                },
                name=label,
                hovertemplate=(
                    f"{label}<br>Ann. vol: %{{x:.1f}}%<br>Ann. return: %{{y:.1f}}%"
                    "<extra></extra>"
                ),
            )
        )
        fig.add_annotation(
            x=vol,
            y=ret,
            ax=dx,
            ay=dy,
            text=label,
            showarrow=True,
            arrowhead=0,
            arrowsize=1,
            arrowwidth=1.1,
            arrowcolor=PORTFOLIO_COLORS[label],
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=PORTFOLIO_COLORS[label],
            borderwidth=1,
            font={"size": 12},
        )

    sharpe_text = f"Mean-variance Sharpe: {mean_variance_sharpe(metrics):.2f}"
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        text=sharpe_text,
        showarrow=False,
        font={"size": 12},
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="#B7BEC6",
        borderwidth=1,
    )
    fig.update_layout(
        title={"text": "Efficient frontier", "x": 0, "xanchor": "left"},
        margin={"l": 32, "r": 26, "t": 58, "b": 40},
        hovermode="closest",
        height=560,
        legend={
            "orientation": "h",
            "y": 1.05,
            "x": 1,
            "xanchor": "right",
            "yanchor": "bottom",
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(title="Annualized volatility (%)", showgrid=True, gridcolor="#E2E6EA")
    fig.update_yaxes(title="Annualized return (%)", showgrid=True, gridcolor="#E2E6EA")
    return fig


def portfolio_metric_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return a compact app-facing metrics table."""

    table = metrics.copy()
    table["Annualized return"] = table["annualized_return_pct"].map(
        lambda value: format_percent(value)
    )
    table["Annualized volatility"] = table["annualized_volatility_pct"].map(
        lambda value: format_percent(value)
    )
    table["Sharpe ratio"] = table["sharpe_ratio"].map(format_ratio)
    table["Max drawdown"] = table["max_drawdown_pct"].map(lambda value: format_percent(value))
    return table[
        [
            "portfolio",
            "Annualized return",
            "Annualized volatility",
            "Sharpe ratio",
            "Max drawdown",
        ]
    ].rename(columns={"portfolio": "Portfolio"})
