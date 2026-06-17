# ruff: noqa
"""FT-style Stage 3 figures for Week 5 out-of-sample crypto portfolios."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from fintools.figures import (
    FigureContext,
    add_nber_recession_shading,
    export_word_figure,
    figure_style,
)
from fintools.figures.plots import _format_date_axis, _format_growth_dollars, _line_color
from fintools.portfolio_math import tangency_weights

from .stage3_oos_portfolios import (
    MODEL_LABELS,
    OOS_FIGURE_PORTFOLIO_COLUMN_ORDER,
    OOS_FIGURE_PORTFOLIO_LABELS,
    OPTIMIZED_PORTFOLIO_COLUMN_ORDER,
    SQRT_365,
    TRADING_DAYS_PER_YEAR,
    Stage3OOSSample,
    build_top_target_weight_histories,
    drawdown_series,
    wealth_index,
)

WEEK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE3_FIGURE_ROOT = WEEK_ROOT / "results" / "figures" / "stage3" / "long_only"
PORTFOLIO_COLORS = {
    "Equal-weight": "#746e65",
    "Minimum variance": "#4b8a7f",
    "Mean-variance": "#96506a",
    "Mean-CVaR": "#4f6f88",
    "Risk parity": "#8a6b45",
}
TOP_HOLDING_COLORS = ["#96506a", "#4b8a7f", "#4f6f88", "#8a6b45", "#746e65"]
ASSET_EXPOSURE_COLORS = {
    "Bitcoin (BTC)": "#b06b2f",
    "Ether (ETH)": "#5a7d99",
    "Other coins": "#8d887f",
}


def stage3_figure_dir() -> Path:
    """Return the default Week 5 Stage 3 figure directory."""

    return DEFAULT_STAGE3_FIGURE_ROOT


def provider_source_note() -> str:
    """Return the source note used in Week 5 Stage 3 figure captions."""

    return "Yahoo Finance crypto prices; Kenneth French Data Library daily RF."


def provider_label(sample: Stage3OOSSample) -> str:
    """Return the display label used in visible Stage 3 figure text."""

    return sample.display_name


def sample_label(sample: Stage3OOSSample) -> str:
    """Return a caption-ready sample label for the OOS Stage 3 window."""

    return f"{sample.start_date:%Y-%m-%d} to {sample.end_date:%Y-%m-%d}"


def export_stage3_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    context: FigureContext,
    *,
    spec: str = "full_width",
) -> dict[str, Path]:
    """Export one Word-ready Stage 3 figure and close it."""

    paths = export_word_figure(fig, output_dir, stem, context=context, spec=spec)
    plt.close(fig)
    return paths


def _grid(ax: plt.Axes) -> None:
    """Apply a light FT-style horizontal grid."""

    ax.grid(axis="y", color="#d6d1c6", linewidth=0.8)
    ax.grid(axis="x", visible=False)


def _display_ticker(ticker: str) -> str:
    """Return a compact display label for one Yahoo crypto ticker."""

    return ticker[:-4] if ticker.endswith("-USD") else ticker


def _portfolio_labels(portfolio_keys: list[str]) -> list[str]:
    """Return display labels in canonical key order."""

    return [OOS_FIGURE_PORTFOLIO_LABELS[key] for key in portfolio_keys]


def _format_percent_axis(ax: plt.Axes, *, decimals: int = 0) -> None:
    """Format one axis in percent units."""

    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _pos: f"{value:.{decimals}f}%")
    )


def _portfolio_labelled_frame(
    portfolio_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Rename the long-only OOS portfolio-return columns to display labels."""

    rename_map = {
        column: OOS_FIGURE_PORTFOLIO_LABELS[column]
        for column in OOS_FIGURE_PORTFOLIO_COLUMN_ORDER
    }
    frame = (
        portfolio_returns.loc[:, ["return_date", *OOS_FIGURE_PORTFOLIO_COLUMN_ORDER]]
        .rename(columns=rename_map | {"return_date": "date"})
        .copy()
    )
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def make_growth_figure(
    portfolio_returns: pd.DataFrame,
    *,
    sample: Stage3OOSSample,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot growth of one dollar for the long-only OOS core portfolios."""

    frame = _portfolio_labelled_frame(portfolio_returns)
    with figure_style("word_a4", style="ft"):
        fig, ax = plt.subplots(figsize=(7.2, 4.5), layout="none")
        labels = [
            OOS_FIGURE_PORTFOLIO_LABELS[column]
            for column in OOS_FIGURE_PORTFOLIO_COLUMN_ORDER
        ]
        for label in labels:
            ax.plot(
                frame["date"],
                wealth_index(frame[label]),
                label=label,
                color=PORTFOLIO_COLORS[label],
                linewidth=1.9,
                alpha=0.95,
                zorder=3,
            )
        add_nber_recession_shading(
            ax,
            data_start=frame["date"].min(),
            data_end=frame["date"].max(),
            style="ft",
        )
        _format_date_axis(ax, date_start=frame["date"].min(), date_end=frame["date"].max())
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(_format_growth_dollars))
        _grid(ax)
        ax.set_title("Growth of $1", loc="left")
        ax.set_xlabel("Date")
        ax.set_ylabel("Growth of $1")
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=3,
            frameon=False,
        )
        fig.subplots_adjust(bottom=0.25)
    context = FigureContext(
        title=f"{provider_label(sample)} out-of-sample growth of $1",
        note=(
            "Daily out-of-sample portfolio returns with within-block weight drift, "
            "plotted on a log scale. This figure uses the long-only core portfolios."
        ),
        source=provider_source_note(),
        sample=sample_label(sample),
        units="Growth of $1.",
    )
    return export_stage3_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage3_oos_growth_of_one",
        context,
    )


def make_drawdown_figure(
    portfolio_returns: pd.DataFrame,
    *,
    sample: Stage3OOSSample,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot OOS portfolio drawdowns for the long-only core portfolios."""

    frame = _portfolio_labelled_frame(portfolio_returns)
    with figure_style("word_a4", style="ft"):
        fig, ax = plt.subplots(figsize=(7.2, 4.5), layout="none")
        labels = [
            OOS_FIGURE_PORTFOLIO_LABELS[column]
            for column in OOS_FIGURE_PORTFOLIO_COLUMN_ORDER
        ]
        for label in labels:
            ax.plot(
                frame["date"],
                drawdown_series(frame[label]),
                label=label,
                color=PORTFOLIO_COLORS[label],
                linewidth=1.1,
                alpha=0.9,
                zorder=3,
            )
        add_nber_recession_shading(
            ax,
            data_start=frame["date"].min(),
            data_end=frame["date"].max(),
            style="ft",
        )
        _format_date_axis(ax, date_start=frame["date"].min(), date_end=frame["date"].max())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0%}"))
        _grid(ax)
        ax.set_title("Out-of-sample drawdowns", loc="left")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=3,
            frameon=False,
        )
        fig.subplots_adjust(bottom=0.25)
    context = FigureContext(
        title=f"{provider_label(sample)} out-of-sample drawdowns",
        note=(
            "Drawdowns computed from realized daily out-of-sample portfolio returns. "
            "This figure uses the long-only core portfolios."
        ),
        source=provider_source_note(),
        sample=sample_label(sample),
        units="Drawdown (%).",
    )
    return export_stage3_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage3_oos_drawdowns",
        context,
    )


def make_scorecard_figure(
    metrics: pd.DataFrame,
    *,
    sample: Stage3OOSSample,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot a six-panel OOS portfolio scorecard."""

    plot_order = [
        OOS_FIGURE_PORTFOLIO_LABELS[column]
        for column in OOS_FIGURE_PORTFOLIO_COLUMN_ORDER
    ]
    plot_frame = metrics.set_index("portfolio").loc[plot_order].reset_index()
    specs = [
        ("cumulative_return_pct", "Cumulative return", "%", True),
        ("annualized_return_pct", "Annualized return", "%", False),
        ("annualized_volatility_pct", "Annualized volatility", "%", True),
        ("sharpe_ratio", "Sharpe ratio", "", False),
        ("sortino_ratio", "Sortino ratio", "", True),
        ("max_drawdown_pct", "Max drawdown", "%", False),
    ]
    with figure_style("word_a4", style="ft"):
        fig, axes = plt.subplots(3, 2, figsize=(7.3, 8.2), layout="none")
        for ax, (column, title, suffix, show_labels) in zip(axes.ravel(), specs, strict=False):
            values = plot_frame[column].astype(float)
            colors = [PORTFOLIO_COLORS[label] for label in plot_frame["portfolio"]]
            ax.barh(plot_frame["portfolio"], values, color=colors, alpha=0.92)
            ax.axvline(0, color=_line_color("ft", "zero"), linewidth=0.9)
            _grid(ax)
            ax.set_title(title, loc="left", pad=10)
            if suffix == "%":
                ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0f}%"))
            if not show_labels:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", left=False, labelleft=False)
            for y_pos, value in enumerate(values):
                offset = 1.5 if suffix == "%" else 0.05
                text_x = value + offset if value >= 0 else value - offset
                ha = "left" if value >= 0 else "right"
                formatted = f"{value:.1f}{suffix}" if suffix else f"{value:.2f}"
                ax.text(text_x, y_pos, formatted, va="center", ha=ha, fontsize=8.2)
        fig.subplots_adjust(hspace=0.54, wspace=0.32, bottom=0.08)
    context = FigureContext(
        title=f"{provider_label(sample)} out-of-sample portfolio scorecard",
        note=(
            "Higher is better for cumulative return, annualized return, Sharpe ratio, "
            "and Sortino ratio. Lower is better for annualized volatility and "
            "drawdown depth. This figure uses the long-only core portfolios."
        ),
        source=provider_source_note(),
        sample=sample_label(sample),
        units=(
            "Cumulative return (%), annualized return (%), annualized volatility (%), "
            "Sharpe ratio, Sortino ratio, and max drawdown (%)."
        ),
    )
    return export_stage3_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage3_oos_scorecard",
        context,
        spec="portrait_full",
    )


def make_top_holdings_figure(
    rebalance_audit: pd.DataFrame,
    *,
    sample: Stage3OOSSample,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot the top-5 formation-date target weights through time for each optimized model."""

    histories = build_top_target_weight_histories(rebalance_audit, top_n=5)
    with figure_style("word_a4", style="ft"):
        fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.8), layout="none")
        for panel_index, model in enumerate(histories):
            ax = axes.ravel()[panel_index]
            frame = histories[model].copy()
            frame["decision_date"] = pd.to_datetime(frame["decision_date"])
            for line_index, ticker in enumerate(frame.columns[1:]):
                ax.plot(
                    frame["decision_date"],
                    frame[ticker].astype(float) * 100.0,
                    color=TOP_HOLDING_COLORS[line_index % len(TOP_HOLDING_COLORS)],
                    linewidth=1.7,
                    alpha=0.95,
                    label=_display_ticker(ticker),
                    zorder=3,
                )
            add_nber_recession_shading(
                ax,
                data_start=frame["decision_date"].min(),
                data_end=frame["decision_date"].max(),
                style="ft",
            )
            _format_date_axis(
                ax,
                date_start=frame["decision_date"].min(),
                date_end=frame["decision_date"].max(),
            )
            ax.tick_params(axis="x", labelsize=8.0, pad=1.5)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0f}%"))
            _grid(ax)
            row_index, col_index = divmod(panel_index, 2)
            ax.set_title(MODEL_LABELS[model], loc="left", pad=8)
            ax.set_xlabel("Formation date" if row_index == 1 else "")
            ax.set_ylabel("Target weight (%)" if col_index == 0 else "")
            if row_index == 0:
                ax.tick_params(axis="x", labelbottom=False)
            ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=7.2)
        fig.subplots_adjust(hspace=0.28, wspace=0.18, top=0.95, bottom=0.10)
    context = FigureContext(
        title=f"{provider_label(sample)} top target holdings over time",
        note=(
            "Each panel keeps the five tickers with the highest average formation-date target "
            "weight for that long-only model. The lines show target weights, "
            "not drifted daily weights."
        ),
        source=provider_source_note(),
        sample=sample_label(sample),
        units="Target weight (%).",
    )
    return export_stage3_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage3_oos_top_holdings_over_time",
        context,
    )


def _point_from_metrics(metrics: pd.DataFrame, label: str) -> tuple[float, float]:
    """Return the annualized volatility and return coordinates for one portfolio."""

    row = metrics.loc[metrics["portfolio"] == label].iloc[0]
    return float(row["annualized_volatility_pct"]), float(row["annualized_return_pct"])


def make_frontier_figure(
    frontier: pd.DataFrame,
    metrics: pd.DataFrame,
    asset_summary: pd.DataFrame,
    *,
    sample: Stage3OOSSample,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot the ex post OOS asset frontier and overlay realized portfolio points."""

    returns = sample.returns_wide.to_numpy(dtype=float)
    mean_returns = returns.mean(axis=0)
    covariance = np.cov(returns, rowvar=False, ddof=1)
    avg_daily_rfr = float(sample.rfr.mean())
    rf_ann_pct = avg_daily_rfr * TRADING_DAYS_PER_YEAR * 100.0
    try:
        tan_weights, _method = tangency_weights(mean_returns, covariance, avg_daily_rfr)
        tan_return_daily = float(mean_returns @ tan_weights)
        tan_sigma_daily = float(np.sqrt(tan_weights @ covariance @ tan_weights))
        tan_return_pct = tan_return_daily * TRADING_DAYS_PER_YEAR * 100.0
        tan_sigma_pct = tan_sigma_daily * SQRT_365 * 100.0
    except ValueError:
        sharpe_like = (
            frontier["target_return_ann_pct"] - rf_ann_pct
        ) / frontier["volatility_ann_pct"].replace(0.0, np.nan)
        tangency_row = frontier.loc[sharpe_like.idxmax()]
        tan_return_pct = float(tangency_row["target_return_ann_pct"])
        tan_sigma_pct = float(tangency_row["volatility_ann_pct"])
    x_max = max(
        float(frontier["volatility_ann_pct"].max()),
        float(asset_summary["annualized_volatility_pct"].max()),
        tan_sigma_pct,
    ) * 1.08
    cal_x = np.linspace(0.0, x_max, 240)
    cal_slope = (
        (tan_return_pct - rf_ann_pct) / tan_sigma_pct if not np.isclose(tan_sigma_pct, 0.0) else 0.0
    )
    cal_y = rf_ann_pct + cal_slope * cal_x

    with figure_style("word_a4", style="ft"):
        fig, ax = plt.subplots(figsize=(7.25, 4.9), layout="none")
        ax.scatter(
            asset_summary["annualized_volatility_pct"],
            asset_summary["annualized_return_pct"],
            s=24,
            color="#8d887f",
            alpha=0.42,
            label="Individual coins",
            zorder=2,
        )
        ax.plot(
            frontier["volatility_ann_pct"],
            frontier["target_return_ann_pct"],
            color="#24364f",
            linewidth=2.0,
            label="Ex post asset frontier",
            zorder=3,
        )
        ax.plot(
            cal_x,
            cal_y,
            color="#8e2f4d",
            linewidth=1.6,
            linestyle="--",
            label="Ex post tangency line",
            zorder=2,
        )
        points = [
            ("Risk-free", 0.0, rf_ann_pct, "#6f6a61", "o", (18, 8)),
            (
                "Equal-weight",
                *_point_from_metrics(metrics, "Equal-weight"),
                PORTFOLIO_COLORS["Equal-weight"],
                "s",
                (42, -14),
            ),
            (
                "Minimum variance",
                *_point_from_metrics(metrics, "Minimum variance"),
                PORTFOLIO_COLORS["Minimum variance"],
                "D",
                (0, 24),
            ),
            (
                "Mean-variance",
                *_point_from_metrics(metrics, "Mean-variance"),
                PORTFOLIO_COLORS["Mean-variance"],
                "^",
                (-20, 18),
            ),
            (
                "Mean-CVaR",
                *_point_from_metrics(metrics, "Mean-CVaR"),
                PORTFOLIO_COLORS["Mean-CVaR"],
                "P",
                (20, 14),
            ),
            (
                "Risk parity",
                *_point_from_metrics(metrics, "Risk parity"),
                PORTFOLIO_COLORS["Risk parity"],
                "X",
                (0, -28),
            ),
        ]
        for label, x_value, y_value, color, marker, offset in points:
            ax.scatter([x_value], [y_value], s=74, color=color, marker=marker, zorder=4)
            if offset[0] > 0:
                horizontal_alignment = "left"
            elif offset[0] < 0:
                horizontal_alignment = "right"
            else:
                horizontal_alignment = "center"
            if offset[1] > 0:
                vertical_alignment = "bottom"
            elif offset[1] < 0:
                vertical_alignment = "top"
            else:
                vertical_alignment = "center"
            ax.annotate(
                label,
                (x_value, y_value),
                textcoords="offset points",
                xytext=offset,
                ha=horizontal_alignment,
                va=vertical_alignment,
                fontsize=8.4,
                color=_line_color("ft", "text"),
                arrowprops={
                    "arrowstyle": "-",
                    "color": color,
                    "linewidth": 0.8,
                    "shrinkA": 3,
                    "shrinkB": 3,
                },
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "#faf8f3",
                    "edgecolor": "none",
                    "alpha": 0.94,
                },
            )
        _grid(ax)
        ax.set_title("Ex post efficient frontier", loc="left")
        ax.set_xlabel("Annualized volatility (%)")
        ax.set_ylabel("Annualized return (%)")
        ax.legend(loc="lower right", frameon=False)
    context = FigureContext(
        title=f"{provider_label(sample)} ex post efficient frontier",
        note=(
            "The frontier and dashed tangency line are built from realized out-of-sample asset "
            "moments over the same window. The portfolio points are the realized dynamic "
            "out-of-sample strategies, overlaid as a diagnostic comparison surface. "
            "This figure uses the long-only core portfolios."
        ),
        source=provider_source_note(),
        sample=sample_label(sample),
        units="Annualized return (%) and annualized volatility (%).",
    )
    return export_stage3_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage3_oos_efficient_frontier",
        context,
    )


def make_stage3_figure_pack(
    *,
    sample: Stage3OOSSample,
    rebalance_audit: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    frontier: pd.DataFrame,
    metrics: pd.DataFrame,
    asset_summary: pd.DataFrame,
    output_dir: Path,
) -> dict[str, dict[str, Path]]:
    """Export the full Week 5 Stage 3 OOS figure pack."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "growth_of_one": make_growth_figure(
            portfolio_returns,
            sample=sample,
            output_dir=output_dir,
        ),
        "drawdowns": make_drawdown_figure(
            portfolio_returns,
            sample=sample,
            output_dir=output_dir,
        ),
        "scorecard": make_scorecard_figure(
            metrics,
            sample=sample,
            output_dir=output_dir,
        ),
        "top_holdings_over_time": make_top_holdings_figure(
            rebalance_audit,
            sample=sample,
            output_dir=output_dir,
        ),
        "efficient_frontier": make_frontier_figure(
            frontier,
            metrics,
            asset_summary,
            sample=sample,
            output_dir=output_dir,
        ),
    }
    return outputs


def _portfolio_color(portfolio_key: str) -> str:
    """Return the stable display color for one long-only portfolio key."""

    return PORTFOLIO_COLORS[OOS_FIGURE_PORTFOLIO_LABELS[portfolio_key]]


def make_factsheet_latest_holdings_dumbbell(
    latest_target_weights: pd.DataFrame,
    *,
    sample: Stage3OOSSample,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot latest target weights for the top holdings of each optimized fund."""

    latest_date = pd.Timestamp(latest_target_weights["decision_date"].iloc[0])
    with figure_style("word_a4", style="ft"):
        fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.9), layout="none")
        for panel_index, portfolio_key in enumerate(OPTIMIZED_PORTFOLIO_COLUMN_ORDER):
            ax = axes.ravel()[panel_index]
            block = latest_target_weights.loc[
                latest_target_weights["portfolio_key"] == portfolio_key
            ].copy()
            block = block.sort_values("latest_weight", ascending=False).head(5).iloc[::-1]
            latest_values = block["latest_weight"].astype(float).to_numpy() * 100.0
            y_positions = np.arange(len(block))
            x_max = max(float(latest_values.max()), 1.0) * 1.14
            ax.barh(
                y_positions,
                latest_values,
                color=_portfolio_color(portfolio_key),
                alpha=0.92,
                zorder=3,
            )
            for y_pos, value in zip(y_positions, latest_values, strict=False):
                ax.text(value, y_pos, f" {value:.1f}%", va="center", ha="left", fontsize=7.2)
            ax.set_yticks(y_positions)
            ax.set_yticklabels([_display_ticker(ticker) for ticker in block["ticker"]])
            ax.tick_params(axis="y", labelsize=9.0)
            ax.set_xlim(0.0, x_max)
            _grid(ax)
            _format_percent_axis(ax, decimals=0)
            row_index, col_index = divmod(panel_index, 2)
            ax.set_title(OOS_FIGURE_PORTFOLIO_LABELS[portfolio_key], loc="left", pad=8)
            ax.set_xlabel("Target weight (%)" if row_index == 1 else "")
            ax.set_ylabel("Coin" if col_index == 0 else "")
            if row_index == 0:
                ax.tick_params(axis="x", labelbottom=False)
            if col_index == 1:
                ax.tick_params(axis="y", labelleft=True)
        fig.subplots_adjust(hspace=0.28, wspace=0.26, top=0.95, bottom=0.10)
    context = FigureContext(
        title=f"{provider_label(sample)} latest target weights",
        note=(
            f"Top five holdings by target weight for each optimized long-only fund at the "
            f"latest formation date {latest_date:%Y-%m-%d}."
        ),
        source=provider_source_note(),
        sample=sample_label(sample),
        units="Target weight (%).",
    )
    return export_stage3_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage3_factsheet_latest_holdings_dumbbell",
        context,
    )


def make_factsheet_live_holdings_snapshot(
    latest_live_weights: pd.DataFrame,
    *,
    sample: Stage3OOSSample,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot the current live top holdings for each optimized long-only fund."""

    latest_return_date = pd.Timestamp(latest_live_weights["return_date"].iloc[0])
    with figure_style("word_a4", style="ft"):
        fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.9), layout="none")
        for panel_index, portfolio_key in enumerate(OPTIMIZED_PORTFOLIO_COLUMN_ORDER):
            ax = axes.ravel()[panel_index]
            block = latest_live_weights.loc[
                latest_live_weights["portfolio_key"] == portfolio_key
            ].copy()
            block = block.sort_values("weight", ascending=False).head(5).iloc[::-1]
            values = block["weight"].astype(float).to_numpy() * 100.0
            ax.barh(
                np.arange(len(block)),
                values,
                color=_portfolio_color(portfolio_key),
                alpha=0.92,
                zorder=3,
            )
            ax.set_yticks(np.arange(len(block)))
            ax.set_yticklabels([_display_ticker(ticker) for ticker in block["ticker"]])
            _grid(ax)
            _format_percent_axis(ax, decimals=0)
            row_index, col_index = divmod(panel_index, 2)
            ax.set_title(OOS_FIGURE_PORTFOLIO_LABELS[portfolio_key], loc="left", pad=8)
            ax.set_xlabel("Live weight (%)" if row_index == 1 else "")
            ax.set_ylabel("Coin" if col_index == 0 else "")
            if row_index == 0:
                ax.tick_params(axis="x", labelbottom=False)
        fig.subplots_adjust(hspace=0.28, wspace=0.20, top=0.95, bottom=0.10)
    context = FigureContext(
        title=f"{provider_label(sample)} current live holdings snapshot",
        note=(
            f"Top five current end-of-day drifted holdings for each optimized long-only "
            f"fund, evaluated as of {latest_return_date:%Y-%m-%d}."
        ),
        source=provider_source_note(),
        sample=sample_label(sample),
        units="Live weight (%).",
    )
    return export_stage3_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage3_factsheet_live_holdings_snapshot",
        context,
    )


def make_factsheet_concentration_scorecard(
    concentration_snapshot: pd.DataFrame,
    *,
    sample: Stage3OOSSample,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot the current concentration scorecard across long-only funds."""

    latest_return_date = pd.Timestamp(concentration_snapshot["return_date"].iloc[0])
    plot_order = _portfolio_labels(list(OOS_FIGURE_PORTFOLIO_COLUMN_ORDER))
    frame = concentration_snapshot.set_index("portfolio").loc[plot_order].reset_index()
    specs = [
        ("top_1_weight_pct", "Top holding", "%"),
        ("top_3_weight_pct", "Top 3 holdings", "%"),
        ("top_5_weight_pct", "Top 5 holdings", "%"),
        ("effective_n", "Effective number of holdings", ""),
    ]
    with figure_style("word_a4", style="ft"):
        fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.9), layout="none")
        for panel_index, (ax, (column, title, suffix)) in enumerate(
            zip(axes.ravel(), specs, strict=False)
        ):
            values = frame[column].astype(float)
            ax.barh(
                frame["portfolio"],
                values,
                color=[PORTFOLIO_COLORS[label] for label in frame["portfolio"]],
                alpha=0.92,
                zorder=3,
            )
            _grid(ax)
            ax.set_title(title, loc="left", pad=8)
            if suffix == "%":
                _format_percent_axis(ax, decimals=0)
            ax.set_ylabel("")
            _row_index, col_index = divmod(panel_index, 2)
            if col_index == 1:
                ax.tick_params(axis="y", labelleft=False, left=False)
            for y_pos, value in enumerate(values):
                formatted = f"{value:.1f}{suffix}" if suffix else f"{value:.1f}"
                ax.text(value, y_pos, f" {formatted}", va="center", ha="left", fontsize=8.0)
        fig.subplots_adjust(hspace=0.34, wspace=0.28, top=0.95, bottom=0.10)
    context = FigureContext(
        title=f"{provider_label(sample)} concentration scorecard",
        note=(
            f"Current live concentration metrics computed from end-of-day drifted weights "
            f"as of {latest_return_date:%Y-%m-%d}. Effective holdings = 1 / sum(w^2)."
        ),
        source=provider_source_note(),
        sample=sample_label(sample),
        units="Weight share (%) and effective number of holdings.",
    )
    return export_stage3_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage3_factsheet_concentration_scorecard",
        context,
        spec="portrait_full",
    )


def make_factsheet_risk_contributions(
    risk_contributions: pd.DataFrame,
    *,
    sample: Stage3OOSSample,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot top variance contributors at the latest formation date."""

    latest_decision_date = pd.Timestamp(risk_contributions["decision_date"].iloc[0])
    with figure_style("word_a4", style="ft"):
        fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.9), layout="none")
        for panel_index, portfolio_key in enumerate(OPTIMIZED_PORTFOLIO_COLUMN_ORDER):
            ax = axes.ravel()[panel_index]
            block = risk_contributions.loc[
                risk_contributions["portfolio_key"] == portfolio_key
            ].copy()
            block = block.sort_values("risk_contribution_pct", ascending=False).head(5).iloc[::-1]
            values = block["risk_contribution_pct"].astype(float).to_numpy()
            ax.barh(
                np.arange(len(block)),
                values,
                color=_portfolio_color(portfolio_key),
                alpha=0.92,
                zorder=3,
            )
            ax.set_yticks(np.arange(len(block)))
            ax.set_yticklabels([_display_ticker(ticker) for ticker in block["ticker"]])
            _grid(ax)
            _format_percent_axis(ax, decimals=1)
            row_index, col_index = divmod(panel_index, 2)
            ax.set_title(OOS_FIGURE_PORTFOLIO_LABELS[portfolio_key], loc="left", pad=8)
            ax.set_xlabel("Variance contribution (%)" if row_index == 1 else "", fontsize=12)
            ax.set_ylabel("Coin" if col_index == 0 else "")
            if row_index == 0:
                ax.tick_params(axis="x", labelbottom=False)
        fig.subplots_adjust(hspace=0.28, wspace=0.20, top=0.95, bottom=0.14)
    context = FigureContext(
        title=f"{provider_label(sample)} latest risk-contribution snapshot",
        note=(
            f"Top five contributions to portfolio variance at the latest formation date "
            f"{latest_decision_date:%Y-%m-%d}, using the same training-window covariance "
            f"estimate as the portfolio construction step."
        ),
        source=provider_source_note(),
        sample=sample_label(sample),
        units="Contribution to portfolio variance (%).",
    )
    return export_stage3_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage3_factsheet_risk_contributions",
        context,
    )


def make_factsheet_trailing_returns(
    trailing_returns: pd.DataFrame,
    *,
    sample: Stage3OOSSample,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot trailing realized return snapshots across app-style horizons."""

    latest_date = pd.Timestamp(trailing_returns["as_of_date"].iloc[0])
    plot_order = _portfolio_labels(list(OOS_FIGURE_PORTFOLIO_COLUMN_ORDER))
    horizon_order = ["30-day", "90-day", "180-day", "Since inception"]
    with figure_style("word_a4", style="ft"):
        fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.9), layout="none")
        for panel_index, (ax, horizon_label) in enumerate(
            zip(axes.ravel(), horizon_order, strict=False)
        ):
            block = trailing_returns.loc[trailing_returns["window_label"] == horizon_label].copy()
            block = block.set_index("portfolio").loc[plot_order].reset_index()
            values = block["cumulative_return_pct"].astype(float).to_numpy()
            ax.barh(
                block["portfolio"],
                values,
                color=[PORTFOLIO_COLORS[label] for label in block["portfolio"]],
                alpha=0.92,
                zorder=3,
            )
            ax.axvline(0, color=_line_color("ft", "zero"), linewidth=0.9)
            _grid(ax)
            _format_percent_axis(ax, decimals=1)
            ax.set_title(horizon_label, loc="left", pad=8)
            ax.set_ylabel("")
            _row_index, col_index = divmod(panel_index, 2)
            if col_index == 1:
                ax.tick_params(axis="y", labelleft=False, left=False)
        fig.subplots_adjust(hspace=0.34, wspace=0.28, top=0.95, bottom=0.10)
    context = FigureContext(
        title=f"{provider_label(sample)} trailing return snapshot",
        note=(
            f"Realized out-of-sample cumulative returns across trailing 30-day, 90-day, "
            f"180-day, and since-inception windows, evaluated through {latest_date:%Y-%m-%d}."
        ),
        source=provider_source_note(),
        sample=sample_label(sample),
        units="Cumulative return (%).",
    )
    return export_stage3_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage3_factsheet_trailing_returns",
        context,
        spec="portrait_full",
    )


def make_factsheet_current_drawdown(
    current_drawdown: pd.DataFrame,
    *,
    sample: Stage3OOSSample,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot current drawdown from peak for each long-only fund."""

    latest_date = pd.Timestamp(current_drawdown["as_of_date"].iloc[0])
    frame = current_drawdown.sort_values("current_drawdown_pct", ascending=True).copy()
    values = frame["current_drawdown_pct"].astype(float).to_numpy()
    y_positions = np.arange(len(frame))
    with figure_style("word_a4", style="ft"):
        fig, ax = plt.subplots(figsize=(7.1, 4.2), layout="none")
        ax.hlines(
            y_positions,
            xmin=values,
            xmax=np.zeros(len(values)),
            color="#d6d1c6",
            linewidth=2.2,
            zorder=2,
        )
        ax.scatter(
            values,
            y_positions,
            s=58,
            color=[PORTFOLIO_COLORS[label] for label in frame["portfolio"]],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.axvline(0, color=_line_color("ft", "zero"), linewidth=0.9)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(frame["portfolio"])
        _grid(ax)
        _format_percent_axis(ax, decimals=1)
        ax.set_title("Current drawdown from peak", loc="left")
        ax.set_xlabel("Drawdown (%)")
        ax.set_ylabel("")
    context = FigureContext(
        title=f"{provider_label(sample)} current drawdown snapshot",
        note=(
            f"Current drawdown from prior peak in the realized OOS wealth path, "
            f"evaluated as of {latest_date:%Y-%m-%d}."
        ),
        source=provider_source_note(),
        sample=sample_label(sample),
        units="Drawdown (%).",
    )
    return export_stage3_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage3_factsheet_current_drawdown",
        context,
    )


def make_factsheet_trailing_risk(
    trailing_risk: pd.DataFrame,
    *,
    sample: Stage3OOSSample,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot the latest trailing realized risk snapshot."""

    latest_date = pd.Timestamp(trailing_risk["as_of_date"].iloc[0])
    plot_order = _portfolio_labels(list(OOS_FIGURE_PORTFOLIO_COLUMN_ORDER))
    frame = trailing_risk.set_index("portfolio").loc[plot_order].reset_index()
    specs = [
        ("annualized_volatility_pct", "180-day annualized volatility", "%"),
        ("sharpe_ratio", "180-day Sharpe ratio", ""),
        ("sortino_ratio", "180-day Sortino ratio", ""),
        ("cvar_95_loss_pct", "1-day CVaR 95% loss", "%"),
    ]
    with figure_style("word_a4", style="ft"):
        fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.9), layout="none")
        for panel_index, (ax, (column, title, suffix)) in enumerate(
            zip(axes.ravel(), specs, strict=False)
        ):
            values = frame[column].astype(float)
            ax.barh(
                frame["portfolio"],
                values,
                color=[PORTFOLIO_COLORS[label] for label in frame["portfolio"]],
                alpha=0.92,
                zorder=3,
            )
            if values.min() < 0 < values.max():
                ax.axvline(0, color=_line_color("ft", "zero"), linewidth=0.9)
            _grid(ax)
            ax.set_title(title, loc="left", pad=8)
            ax.set_ylabel("")
            if suffix == "%":
                _format_percent_axis(ax, decimals=1)
            _row_index, col_index = divmod(panel_index, 2)
            if col_index == 1:
                ax.tick_params(axis="y", labelleft=False, left=False)
        fig.subplots_adjust(hspace=0.34, wspace=0.28, top=0.95, bottom=0.10)
    context = FigureContext(
        title=f"{provider_label(sample)} trailing risk snapshot",
        note=(
            f"Latest trailing 180-day realized risk metrics, evaluated through "
            f"{latest_date:%Y-%m-%d}. Historical CVaR is reported as the average "
            f"loss on the worst 5% of daily return outcomes."
        ),
        source=provider_source_note(),
        sample=sample_label(sample),
        units="Percent for volatility and CVaR; unitless for Sharpe and Sortino.",
    )
    return export_stage3_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage3_factsheet_trailing_risk",
        context,
        spec="portrait_full",
    )


def make_factsheet_turnover_and_changes(
    turnover_snapshot: pd.DataFrame,
    latest_target_weights: pd.DataFrame,
    *,
    sample: Stage3OOSSample,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot latest turnover by fund and the largest target-weight changes."""

    latest_date = pd.Timestamp(turnover_snapshot["decision_date"].iloc[0])
    previous_date = pd.Timestamp(turnover_snapshot["previous_decision_date"].iloc[0])
    turnover_frame = turnover_snapshot.set_index("portfolio").loc[
        _portfolio_labels(list(OOS_FIGURE_PORTFOLIO_COLUMN_ORDER))
    ].reset_index()
    change_frame = latest_target_weights.loc[
        latest_target_weights["portfolio_key"].isin(OPTIMIZED_PORTFOLIO_COLUMN_ORDER)
    ].copy()
    change_frame["label"] = (
        change_frame["portfolio"]
        + " | "
        + change_frame["ticker"].map(_display_ticker)
    )
    largest_increases = change_frame.nlargest(5, "weight_change")
    largest_decreases = change_frame.nsmallest(5, "weight_change")
    change_frame = pd.concat([largest_decreases, largest_increases], ignore_index=True)
    change_frame = change_frame.sort_values("weight_change", ascending=True)
    with figure_style("word_a4", style="ft"):
        fig, axes = plt.subplots(1, 2, figsize=(7.25, 4.9), layout="none")
        axes[0].barh(
            turnover_frame["portfolio"],
            turnover_frame["turnover_pct"].astype(float),
            color=[PORTFOLIO_COLORS[label] for label in turnover_frame["portfolio"]],
            alpha=0.92,
            zorder=3,
        )
        _grid(axes[0])
        _format_percent_axis(axes[0], decimals=0)
        axes[0].set_title("Latest one-way turnover", loc="left")
        axes[0].set_xlabel("Turnover (%)")
        axes[0].set_ylabel("")

        change_values = change_frame["weight_change"].astype(float).to_numpy() * 100.0
        change_colors = [
            "#2f7f73" if value >= 0 else "#8e2f4d"
            for value in change_values
        ]
        axes[1].barh(
            change_frame["label"],
            change_values,
            color=change_colors,
            alpha=0.92,
            zorder=3,
        )
        axes[1].axvline(0, color=_line_color("ft", "zero"), linewidth=0.9)
        _grid(axes[1])
        _format_percent_axis(axes[1], decimals=0)
        axes[1].set_title("Largest target-weight changes", loc="left")
        axes[1].set_xlabel("Weight change (%)")
        axes[1].set_ylabel("")
        axes[1].yaxis.tick_right()
        axes[1].tick_params(axis="y", labelright=True, labelleft=False, pad=6)
        fig.subplots_adjust(wspace=0.42, bottom=0.12)
    context = FigureContext(
        title=f"{provider_label(sample)} latest turnover and holdings changes",
        note=(
            f"Latest one-way turnover compares target weights on {latest_date:%Y-%m-%d} "
            f"against {previous_date:%Y-%m-%d}. The right panel highlights the largest "
            f"positive and negative target-weight changes across the optimized funds."
        ),
        source=provider_source_note(),
        sample=sample_label(sample),
        units="Percent of portfolio weight.",
    )
    return export_stage3_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage3_factsheet_turnover_and_changes",
        context,
    )


def make_factsheet_btc_eth_exposure(
    btc_eth_exposure: pd.DataFrame,
    *,
    sample: Stage3OOSSample,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot current combined BTC and ETH exposure across the long-only funds."""

    latest_date = pd.Timestamp(btc_eth_exposure["return_date"].iloc[0])
    frame = (
        btc_eth_exposure.set_index("portfolio")
        .loc[_portfolio_labels(list(OOS_FIGURE_PORTFOLIO_COLUMN_ORDER))]
        .reset_index()
        .copy()
    )
    frame = frame.sort_values("btc_eth_weight_pct", ascending=True)
    values = frame["btc_eth_weight_pct"].astype(float).to_numpy()
    y_positions = np.arange(len(frame))
    with figure_style("word_a4", style="ft"):
        fig, ax = plt.subplots(figsize=(7.15, 4.3), layout="none")
        ax.hlines(
            y_positions,
            xmin=np.zeros(len(values)),
            xmax=values,
            color="#d6d1c6",
            linewidth=2.2,
            zorder=2,
        )
        ax.scatter(
            values,
            y_positions,
            s=62,
            color=ASSET_EXPOSURE_COLORS["Bitcoin (BTC)"],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        for y_pos, value in zip(y_positions, values, strict=False):
            ax.text(value, y_pos, f" {value:.1f}%", va="center", ha="left", fontsize=8.5)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(frame["portfolio"])
        _grid(ax)
        _format_percent_axis(ax, decimals=0)
        ax.set_title("Current BTC + ETH share", loc="left")
        ax.set_xlabel("Combined portfolio weight (%)")
        ax.set_ylabel("")
        fig.subplots_adjust(bottom=0.12)
    context = FigureContext(
        title=f"{provider_label(sample)} BTC and ETH exposure snapshot",
        note=(
            f"Current end-of-day drifted combined exposure to Bitcoin and Ether, "
            f"evaluated as of {latest_date:%Y-%m-%d}. The remaining share is invested in "
            f"other coins."
        ),
        source=provider_source_note(),
        sample=sample_label(sample),
        units="Portfolio weight (%).",
    )
    return export_stage3_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage3_factsheet_btc_eth_exposure",
        context,
    )


def make_stage3_factsheet_figure_pack(
    *,
    sample: Stage3OOSSample,
    latest_target_weights: pd.DataFrame,
    latest_live_weights: pd.DataFrame,
    concentration_snapshot: pd.DataFrame,
    risk_contributions: pd.DataFrame,
    trailing_returns: pd.DataFrame,
    current_drawdown: pd.DataFrame,
    trailing_risk: pd.DataFrame,
    turnover_snapshot: pd.DataFrame,
    btc_eth_exposure: pd.DataFrame,
    output_dir: Path,
) -> dict[str, dict[str, Path]]:
    """Export the full Week 5 Stage 3 factsheet-style long-only figure pack."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "latest_holdings_dumbbell": make_factsheet_latest_holdings_dumbbell(
            latest_target_weights,
            sample=sample,
            output_dir=output_dir,
        ),
        "live_holdings_snapshot": make_factsheet_live_holdings_snapshot(
            latest_live_weights,
            sample=sample,
            output_dir=output_dir,
        ),
        "concentration_scorecard": make_factsheet_concentration_scorecard(
            concentration_snapshot,
            sample=sample,
            output_dir=output_dir,
        ),
        "risk_contributions": make_factsheet_risk_contributions(
            risk_contributions,
            sample=sample,
            output_dir=output_dir,
        ),
        "trailing_returns": make_factsheet_trailing_returns(
            trailing_returns,
            sample=sample,
            output_dir=output_dir,
        ),
        "current_drawdown": make_factsheet_current_drawdown(
            current_drawdown,
            sample=sample,
            output_dir=output_dir,
        ),
        "trailing_risk": make_factsheet_trailing_risk(
            trailing_risk,
            sample=sample,
            output_dir=output_dir,
        ),
        "turnover_and_changes": make_factsheet_turnover_and_changes(
            turnover_snapshot,
            latest_target_weights,
            sample=sample,
            output_dir=output_dir,
        ),
        "btc_eth_exposure": make_factsheet_btc_eth_exposure(
            btc_eth_exposure,
            sample=sample,
            output_dir=output_dir,
        ),
    }
    return outputs
