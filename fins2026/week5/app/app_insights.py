# ruff: noqa
"""Pure figures, metrics, and formatting helpers for the Week 5 client app."""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go

from fins2026.week5.app.app_config import (
    PORTFOLIO_COLORS,
    PORTFOLIO_DESCRIPTIONS,
    PORTFOLIO_OPTIONS,
)
from fins2026.week5.code.stage4_app import (
    Stage4DisplayWindow,
    Stage4ScenarioBundle,
    latest_summary_cards,
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


def format_percent(value: float | None, *, signed: bool = False, decimals: int = 1) -> str:
    """Format one percentage value."""

    if value is None or pd.isna(value):
        return "n/a"
    sign = "+" if signed else ""
    return f"{value:{sign},.{decimals}f}%"


def format_ratio(value: float | None, *, decimals: int = 2) -> str:
    """Format one unitless ratio."""

    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:,.{decimals}f}"


def format_currency(value: float | None) -> str:
    """Format a USD amount in a readable style."""

    if value is None or pd.isna(value):
        return "n/a"
    amount = float(value)
    if abs(amount) >= 1_000_000_000:
        return f"${amount / 1_000_000_000:,.1f}bn"
    if abs(amount) >= 1_000_000:
        return f"${amount / 1_000_000:,.1f}m"
    if abs(amount) >= 1_000:
        return f"${amount:,.0f}"
    return f"${amount:,.2f}"


def _portfolio_name(portfolio_key: str) -> str:
    return PORTFOLIO_OPTIONS[portfolio_key]


def _portfolio_color(portfolio_key: str) -> str:
    return PORTFOLIO_COLORS[portfolio_key]


def _format_growth_tick(value: float) -> str:
    if value >= 10:
        return f"${value:,.0f}"
    if value >= 1:
        return f"${value:,.1f}".rstrip("0").rstrip(".")
    return f"${value:,.2f}".rstrip("0").rstrip(".")


def _growth_axis_ticks(values: pd.Series) -> tuple[list[float], list[str]]:
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
    return tickvals, [_format_growth_tick(value) for value in tickvals]


def selected_fund_metric_cards(
    scenario: Stage4ScenarioBundle,
    *,
    portfolio_key: str,
) -> list[MetricCard]:
    """Build the top metric cards for one selected published fund."""

    summary = latest_summary_cards(scenario, portfolio_key=portfolio_key)
    return [
        MetricCard(
            "Published fund",
            summary["portfolio"],
            help=PORTFOLIO_DESCRIPTIONS[portfolio_key],
        ),
        MetricCard(
            "Since-inception return",
            format_percent(summary["cumulative_return_pct"]),
            help="Cumulative performance over the full historical live-model track record.",
        ),
        MetricCard(
            "Trailing 180-day volatility",
            format_percent(summary["annualized_volatility_pct"]),
            help="Annualized realised volatility over the most recent 180 daily observations.",
        ),
        MetricCard(
            "Trailing Sharpe ratio",
            format_ratio(summary["sharpe_ratio"]),
            help="Trailing 180-day excess-return Sharpe ratio versus the daily cash-rate series.",
        ),
        MetricCard(
            "Current drawdown",
            format_percent(summary["current_drawdown_pct"]),
            help="Current distance from the prior realised peak in the live-model wealth path.",
        ),
        MetricCard(
            "Latest rebalance",
            f"{summary['latest_rebalance_date']:%Y-%m-%d}",
            help="Most recent date when the published target weights were refreshed.",
        ),
    ]


def design_metric_cards(
    scenario: Stage4ScenarioBundle,
    *,
    portfolio_key: str,
) -> list[MetricCard]:
    """Build top metric cards for the advanced design-comparison tab."""

    summary = latest_summary_cards(scenario, portfolio_key=portfolio_key)
    return [
        MetricCard(
            "Rebalance schedule",
            scenario.config.estimation_frequency.capitalize(),
            help="Frequency used to refresh the model portfolio in the design view.",
        ),
        MetricCard(
            "Training window",
            f"{scenario.config.initial_window} days",
            help="Minimum number of daily observations before the first live rebalance.",
        ),
        MetricCard(
            "Window rule",
            scenario.config.window_rule.capitalize(),
            help="Expanding uses all data to date; rolling keeps a fixed-length trailing window.",
        ),
        MetricCard(
            "Selected fund return",
            format_percent(summary["cumulative_return_pct"]),
            help="Since-inception realised return for the selected alternative design.",
        ),
        MetricCard(
            "Selected fund volatility",
            format_percent(summary["annualized_volatility_pct"]),
            help="Trailing 180-day annualized realised volatility for the selected design.",
        ),
        MetricCard(
            "Latest performance date",
            f"{summary['latest_return_date']:%Y-%m-%d}",
            help="Latest daily performance date available for the current scenario.",
        ),
    ]


def growth_figure(display_window: Stage4DisplayWindow) -> go.Figure:
    """Plot growth of one dollar for the published fund lineup."""

    frame = display_window.portfolio_returns.copy()
    fig = go.Figure()
    wealth_values: list[pd.Series] = []
    for portfolio_key in PORTFOLIO_OPTIONS:
        label = _portfolio_name(portfolio_key)
        wealth = (1.0 + frame[portfolio_key].astype(float)).cumprod()
        wealth_values.append(wealth)
        fig.add_trace(
            go.Scatter(
                x=frame["return_date"],
                y=wealth,
                mode="lines",
                name=label,
                line={"color": _portfolio_color(portfolio_key), "width": 2.2},
                hovertemplate="%{x|%Y-%m-%d}<br>Growth of $1: %{y:.2f}<extra></extra>",
            )
        )
    add_nber_recession_vrects(
        fig,
        start=frame["return_date"].min(),
        end=frame["return_date"].max(),
    )
    fig.update_layout(
        title={"text": "Historical growth of $1", "x": 0, "xanchor": "left"},
    )
    apply_app_plotly_theme(
        fig,
        yaxis_title="Growth of $1 (log scale)",
        height=520,
        range_slider=False,
    )
    tickvals, ticktext = _growth_axis_ticks(pd.concat(wealth_values, ignore_index=True))
    fig.update_yaxes(type="log", tickmode="array", tickvals=tickvals, ticktext=ticktext)
    return fig


def drawdown_figure(display_window: Stage4DisplayWindow) -> go.Figure:
    """Plot realised drawdowns for the published fund lineup."""

    frame = display_window.portfolio_returns.copy()
    fig = go.Figure()
    for portfolio_key in PORTFOLIO_OPTIONS:
        label = _portfolio_name(portfolio_key)
        wealth = (1.0 + frame[portfolio_key].astype(float)).cumprod()
        running_peak = wealth.cummax()
        drawdown = (wealth / running_peak - 1.0) * 100.0
        fig.add_trace(
            go.Scatter(
                x=frame["return_date"],
                y=drawdown,
                mode="lines",
                name=label,
                line={"color": _portfolio_color(portfolio_key), "width": 1.8},
                hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.1f}%<extra></extra>",
            )
        )
    add_nber_recession_vrects(
        fig,
        start=frame["return_date"].min(),
        end=frame["return_date"].max(),
    )
    fig.update_layout(
        title={"text": "Historical drawdowns", "x": 0, "xanchor": "left"},
    )
    apply_app_plotly_theme(
        fig,
        yaxis_title="Drawdown (%)",
        height=500,
        range_slider=False,
    )
    fig.update_yaxes(ticksuffix="%")
    return fig


def performance_comparison_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return a presentation-safe comparison table for the published fund shelf."""

    table = metrics.copy()
    table["Fund"] = table["portfolio_key"].map(_portfolio_name)
    table["Cumulative return"] = table["cumulative_return_pct"].map(format_percent)
    table["Annualized return"] = table["annualized_return_pct"].map(format_percent)
    table["Annualized volatility"] = table["annualized_volatility_pct"].map(format_percent)
    table["Sharpe ratio"] = table["sharpe_ratio"].map(format_ratio)
    table["Sortino ratio"] = table["sortino_ratio"].map(format_ratio)
    table["Max drawdown"] = table["max_drawdown_pct"].map(format_percent)
    return table[
        [
            "Fund",
            "Cumulative return",
            "Annualized return",
            "Annualized volatility",
            "Sharpe ratio",
            "Sortino ratio",
            "Max drawdown",
        ]
    ]


def holdings_snapshot_figure(
    weights: pd.DataFrame,
    *,
    portfolio_key: str,
    weight_column: str,
    title: str,
    top_n: int = 10,
) -> go.Figure:
    """Plot the current or target holdings snapshot for one selected fund."""

    block = weights.loc[weights["portfolio_key"] == portfolio_key].copy()
    block = block.sort_values(weight_column, ascending=False).head(top_n).iloc[::-1]
    values = block[weight_column].astype(float) * 100.0
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=values,
            y=block["ticker"].str.replace("-USD", "", regex=False),
            orientation="h",
            marker_color=_portfolio_color(portfolio_key),
            hovertemplate="%{y}<br>Weight: %{x:.1f}%<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(title={"text": title, "x": 0, "xanchor": "left"})
    apply_app_plotly_theme(
        fig,
        yaxis_title=None,
        height=420,
        range_selector=False,
        range_slider=False,
    )
    fig.update_xaxes(title="Weight (%)", ticksuffix="%")
    fig.update_yaxes(title=None, automargin=True)
    return fig


def concentration_cards(
    concentration_snapshot: pd.DataFrame,
    turnover_snapshot: pd.DataFrame,
    btc_eth_exposure: pd.DataFrame,
    *,
    portfolio_key: str,
) -> list[MetricCard]:
    """Return the point-in-time concentration cards for one selected fund."""

    concentration = concentration_snapshot.loc[
        concentration_snapshot["portfolio_key"] == portfolio_key
    ].iloc[0]
    turnover = turnover_snapshot.loc[
        turnover_snapshot["portfolio_key"] == portfolio_key
    ].iloc[0]
    exposure = btc_eth_exposure.loc[
        btc_eth_exposure["portfolio_key"] == portfolio_key
    ].iloc[0]
    return [
        MetricCard(
            "Top holding",
            format_percent(float(concentration["top_1_weight_pct"])),
            help="Current live share held in the largest single coin.",
        ),
        MetricCard(
            "Top five holdings",
            format_percent(float(concentration["top_5_weight_pct"])),
            help="Current live combined share held in the five largest positions.",
        ),
        MetricCard(
            "Effective holdings",
            format_ratio(float(concentration["effective_n"]), decimals=1),
            help="A diversification count based on 1 / sum of squared live weights.",
        ),
        MetricCard(
            "Latest turnover",
            format_percent(float(turnover["turnover_pct"])),
            help="One-way change in target weights between the latest two rebalances.",
        ),
        MetricCard(
            "BTC + ETH share",
            format_percent(float(exposure["btc_eth_weight_pct"])),
            help="Current live combined share invested in Bitcoin and Ether.",
        ),
    ]


def trailing_return_snapshot_figure(
    trailing_returns: pd.DataFrame,
    *,
    portfolio_key: str,
) -> go.Figure:
    """Plot trailing realised returns for one selected fund."""

    block = trailing_returns.loc[trailing_returns["portfolio_key"] == portfolio_key].copy()
    block["window_rank"] = block["window_label"].map(
        {"30-day": 0, "90-day": 1, "180-day": 2, "Since inception": 3}
    )
    block = block.sort_values("window_rank")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=block["window_label"],
            y=block["cumulative_return_pct"].astype(float),
            marker_color=_portfolio_color(portfolio_key),
            hovertemplate="%{x}<br>Return: %{y:.1f}%<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        title={"text": "Trailing return profile", "x": 0, "xanchor": "left"},
    )
    apply_app_plotly_theme(
        fig,
        yaxis_title="Cumulative return (%)",
        height=360,
        range_selector=False,
        range_slider=False,
    )
    fig.update_xaxes(title=None)
    fig.update_yaxes(ticksuffix="%")
    return fig


def trailing_risk_table(
    trailing_risk: pd.DataFrame,
    *,
    portfolio_key: str,
) -> pd.DataFrame:
    """Return one compact trailing-risk table for the selected fund."""

    row = trailing_risk.loc[trailing_risk["portfolio_key"] == portfolio_key].iloc[0]
    return pd.DataFrame(
        {
            "Measure": [
                "Annualized volatility",
                "Sharpe ratio",
                "Sortino ratio",
                "1-day CVaR 95% loss",
            ],
            "Value": [
                format_percent(float(row["annualized_volatility_pct"])),
                format_ratio(float(row["sharpe_ratio"])),
                format_ratio(float(row["sortino_ratio"])),
                format_percent(float(row["cvar_95_loss_pct"])),
            ],
        }
    )


def risk_contribution_figure(
    risk_contributions: pd.DataFrame,
    *,
    portfolio_key: str,
    top_n: int = 10,
) -> go.Figure:
    """Plot the latest risk contributors for one selected fund."""

    block = risk_contributions.loc[risk_contributions["portfolio_key"] == portfolio_key].copy()
    block = block.sort_values("risk_contribution_pct", ascending=False).head(top_n).iloc[::-1]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=block["risk_contribution_pct"].astype(float),
            y=block["ticker"].str.replace("-USD", "", regex=False),
            orientation="h",
            marker_color=_portfolio_color(portfolio_key),
            hovertemplate="%{y}<br>Variance contribution: %{x:.1f}%<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        title={"text": "Current risk contributors", "x": 0, "xanchor": "left"},
    )
    apply_app_plotly_theme(
        fig,
        yaxis_title=None,
        height=420,
        range_selector=False,
        range_slider=False,
    )
    fig.update_xaxes(title="Variance contribution (%)", ticksuffix="%")
    fig.update_yaxes(title=None, automargin=True)
    return fig


def turnover_change_figure(
    latest_target_weights: pd.DataFrame,
    *,
    portfolio_key: str,
    top_n: int = 8,
) -> go.Figure:
    """Plot the largest latest rebalance changes for one selected fund."""

    block = latest_target_weights.loc[
        latest_target_weights["portfolio_key"] == portfolio_key
    ].copy()
    block["abs_change"] = block["weight_change"].abs()
    block = block.sort_values("abs_change", ascending=False).head(top_n)
    block = block.sort_values("weight_change", ascending=True)
    colors = ["#96506a" if value < 0 else "#4b8a7f" for value in block["weight_change"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=block["weight_change"].astype(float) * 100.0,
            y=block["ticker"].str.replace("-USD", "", regex=False),
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>Weight change: %{x:.1f}%<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_vline(x=0, line_color="#9AA3AD", line_dash="dot")
    fig.update_layout(
        title={"text": "Latest holdings changes", "x": 0, "xanchor": "left"},
    )
    apply_app_plotly_theme(
        fig,
        yaxis_title=None,
        height=380,
        range_selector=False,
        range_slider=False,
    )
    fig.update_xaxes(title="Weight change (%)", ticksuffix="%")
    fig.update_yaxes(title=None, automargin=True)
    return fig


def allocation_figure(allocation: pd.DataFrame, *, portfolio_key: str) -> go.Figure:
    """Plot the illustrative current allocation for one selected fund."""

    frame = allocation.copy()
    frame = frame.sort_values("allocation_usd", ascending=False)
    if len(frame) > 10:
        top = frame.head(10).copy()
        other = pd.DataFrame(
            {
                "ticker": ["Other"],
                "allocation_usd": [frame.iloc[10:]["allocation_usd"].sum()],
                "weight_pct": [frame.iloc[10:]["weight_pct"].sum()],
            }
        )
        frame = pd.concat(
            [top.loc[:, ["ticker", "allocation_usd", "weight_pct"]], other],
            ignore_index=True,
        )
    else:
        frame = frame.loc[:, ["ticker", "allocation_usd", "weight_pct"]]
    frame = frame.iloc[::-1]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame["allocation_usd"].astype(float),
            y=frame["ticker"].str.replace("-USD", "", regex=False),
            orientation="h",
            marker_color=_portfolio_color(portfolio_key),
            hovertemplate=(
                "%{y}<br>Allocation: $%{x:,.0f}<br>Weight: %{customdata:.1f}%"
                "<extra></extra>"
            ),
            customdata=frame["weight_pct"].astype(float),
            showlegend=False,
        )
    )
    fig.update_layout(
        title={"text": "Illustrative current allocation", "x": 0, "xanchor": "left"},
    )
    apply_app_plotly_theme(
        fig,
        yaxis_title=None,
        height=460,
        range_selector=False,
        range_slider=False,
    )
    fig.update_xaxes(title="Estimated allocation (USD)")
    fig.update_yaxes(title=None, automargin=True)
    return fig


def allocation_table(allocation: pd.DataFrame) -> pd.DataFrame:
    """Return the investment-allocation table in app-facing format."""

    table = allocation.copy()
    table["Coin"] = table["ticker"].str.replace("-USD", "", regex=False)
    table["Live weight"] = table["weight_pct"].map(format_percent)
    table["Estimated allocation"] = table["allocation_usd"].map(format_currency)
    return table[["Coin", "Live weight", "Estimated allocation"]]


def live_weight_table(latest_live_weights: pd.DataFrame, *, portfolio_key: str) -> pd.DataFrame:
    """Return a compact live-weight table for one selected fund."""

    block = latest_live_weights.loc[
        latest_live_weights["portfolio_key"] == portfolio_key
    ].copy()
    block["Coin"] = block["ticker"].str.replace("-USD", "", regex=False)
    block["Live weight"] = block["weight"].astype(float).mul(100.0).map(format_percent)
    return block[["Coin", "Live weight"]]


def target_weight_table(
    latest_target_weights: pd.DataFrame,
    *,
    portfolio_key: str,
) -> pd.DataFrame:
    """Return a compact target-weight table for one selected fund."""

    block = latest_target_weights.loc[
        latest_target_weights["portfolio_key"] == portfolio_key
    ].copy()
    block["Coin"] = block["ticker"].str.replace("-USD", "", regex=False)
    block["Target weight"] = block["latest_weight"].astype(float).mul(100.0).map(format_percent)
    block["Previous target"] = block["previous_weight"].astype(float).mul(100.0).map(format_percent)
    block["Change"] = block["weight_change"].astype(float).mul(100.0).map(
        lambda value: format_percent(value, signed=True)
    )
    return block[["Coin", "Target weight", "Previous target", "Change"]]


def latest_btc_eth_exposure_table(btc_eth_exposure: pd.DataFrame) -> pd.DataFrame:
    """Return the published-fund BTC and ETH exposure snapshot table."""

    table = btc_eth_exposure.copy()
    table["Fund"] = table["portfolio_key"].map(_portfolio_name)
    table["Bitcoin"] = table["btc_weight_pct"].map(format_percent)
    table["Ether"] = table["eth_weight_pct"].map(format_percent)
    table["BTC + ETH"] = table["btc_eth_weight_pct"].map(format_percent)
    table["Other coins"] = table["other_weight_pct"].map(format_percent)
    return table[["Fund", "Bitcoin", "Ether", "BTC + ETH", "Other coins"]]


def concentration_table(concentration_snapshot: pd.DataFrame) -> pd.DataFrame:
    """Return the current concentration snapshot table across the fund lineup."""

    table = concentration_snapshot.copy()
    table["Fund"] = table["portfolio_key"].map(_portfolio_name)
    table["Top holding"] = table["top_1_weight_pct"].map(format_percent)
    table["Top 3 holdings"] = table["top_3_weight_pct"].map(format_percent)
    table["Top 5 holdings"] = table["top_5_weight_pct"].map(format_percent)
    table["Effective holdings"] = table["effective_n"].map(
        lambda value: format_ratio(float(value), decimals=1)
    )
    return table[
        ["Fund", "Top holding", "Top 3 holdings", "Top 5 holdings", "Effective holdings"]
    ]


def trailing_return_table(trailing_returns: pd.DataFrame) -> pd.DataFrame:
    """Return the trailing-return table across the published fund shelf."""

    table = trailing_returns.copy()
    table["Fund"] = table["portfolio_key"].map(_portfolio_name)
    table["Return"] = table["cumulative_return_pct"].map(format_percent)
    return table[["Fund", "window_label", "Return"]].rename(
        columns={"window_label": "Window"}
    )


def trailing_risk_comparison_table(trailing_risk: pd.DataFrame) -> pd.DataFrame:
    """Return the latest trailing-risk comparison table across the fund shelf."""

    table = trailing_risk.copy()
    table["Fund"] = table["portfolio_key"].map(_portfolio_name)
    table["Annualized volatility"] = table["annualized_volatility_pct"].map(format_percent)
    table["Sharpe ratio"] = table["sharpe_ratio"].map(format_ratio)
    table["Sortino ratio"] = table["sortino_ratio"].map(format_ratio)
    table["1-day CVaR 95% loss"] = table["cvar_95_loss_pct"].map(format_percent)
    return table[
        [
            "Fund",
            "Annualized volatility",
            "Sharpe ratio",
            "Sortino ratio",
            "1-day CVaR 95% loss",
        ]
    ]


def turnover_table(turnover_snapshot: pd.DataFrame) -> pd.DataFrame:
    """Return the latest turnover table across the fund shelf."""

    table = turnover_snapshot.copy()
    table["Fund"] = table["portfolio_key"].map(_portfolio_name)
    table["Latest turnover"] = table["turnover_pct"].map(format_percent)
    table["Latest rebalance"] = pd.to_datetime(table["decision_date"]).dt.strftime("%Y-%m-%d")
    table["Previous rebalance"] = pd.to_datetime(
        table["previous_decision_date"]
    ).dt.strftime("%Y-%m-%d")
    return table[["Fund", "Latest rebalance", "Previous rebalance", "Latest turnover"]]


def methodology_table(mapping: pd.DataFrame) -> pd.DataFrame:
    """Return the published-to-technical fund mapping table."""

    table = mapping.copy()
    return table.rename(
        columns={
            "published_fund": "Published fund",
            "technical_model": "Technical model",
            "description": "Investment approach",
        }
    )


def frontier_figure(
    display_window: Stage4DisplayWindow,
    *,
    active_portfolio_key: str,
) -> go.Figure:
    """Plot the ex post opportunity set and realised fund points."""

    frontier = display_window.frontier
    asset_stats = display_window.asset_summary
    metrics = display_window.metrics
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=asset_stats["annualized_volatility_pct"],
            y=asset_stats["annualized_return_pct"],
            mode="markers+text",
            text=asset_stats["ticker"].str.replace("-USD", "", regex=False),
            textposition="top center",
            marker={
                "size": 10,
                "color": "rgba(120,120,120,0.28)",
                "line": {"color": "rgba(120,120,120,0.60)", "width": 1},
            },
            name="Underlying coins",
            hovertemplate=(
                "%{text}<br>Annualized volatility: %{x:.1f}%"
                "<br>Annualized return: %{y:.1f}%<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frontier["volatility_ann_pct"],
            y=frontier["target_return_ann_pct"],
            mode="lines",
            name="Ex post opportunity set",
            line={"color": "#2F455C", "width": 3},
            hovertemplate=(
                "Annualized volatility: %{x:.1f}%<br>Annualized return: %{y:.1f}%"
                "<extra></extra>"
            ),
        )
    )
    for portfolio_key in PORTFOLIO_OPTIONS:
        row = metrics.loc[metrics["portfolio_key"] == portfolio_key].iloc[0]
        label = _portfolio_name(portfolio_key)
        fig.add_trace(
            go.Scatter(
                x=[float(row["annualized_volatility_pct"])],
                y=[float(row["annualized_return_pct"])],
                mode="markers",
                name=label,
                marker={
                    "size": 17 if portfolio_key == active_portfolio_key else 14,
                    "color": _portfolio_color(portfolio_key),
                },
                hovertemplate=(
                    f"{label}<br>Annualized volatility: %{{x:.1f}}%"
                    "<br>Annualized return: %{y:.1f}%<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title={"text": "Ex post opportunity set", "x": 0, "xanchor": "left"},
        hovermode="closest",
    )
    apply_app_plotly_theme(
        fig,
        yaxis_title="Annualized return (%)",
        height=560,
        range_slider=False,
        range_selector=False,
    )
    fig.update_xaxes(title="Annualized volatility (%)")
    fig.update_yaxes(ticksuffix="%")
    return fig
