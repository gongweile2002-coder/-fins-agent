# ruff: noqa
"""FT-style Stage 3 figures for Week 4 portfolio construction."""

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

from .stage3_portfolios import (
    PORTFOLIO_COLUMN_ORDER,
    PORTFOLIO_LABELS,
    Stage3Sample,
    drawdown_series,
    resolve_stage3_provider,
    wealth_index,
)

WEEK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE3_FIGURE_ROOT = WEEK_ROOT / "results" / "figures" / "stage3"
PORTFOLIO_COLORS = {
    "Equal-weight": "#6f6a61",
    "Minimum variance": "#2f7f73",
    "Mean-variance": "#8e2f4d",
}


def stage3_figure_dir(provider: str) -> Path:
    """Return the default Stage 3 figure directory for one provider."""

    spec = resolve_stage3_provider(provider)
    return DEFAULT_STAGE3_FIGURE_ROOT / spec.provider


def provider_source_note(provider: str) -> str:
    """Return the source note used in Stage 3 figure captions."""

    spec = resolve_stage3_provider(provider)
    if spec.provider.startswith("tiingo"):
        return "Tiingo adjusted prices; Kenneth French Data Library daily RF."
    return "Yahoo Finance adjusted prices; Kenneth French Data Library daily RF."


def provider_label(provider: str) -> str:
    """Return the display label used in visible Stage 3 figure text."""

    return resolve_stage3_provider(provider).display_name


def sample_label(sample: Stage3Sample) -> str:
    """Return a caption-ready sample label for the balanced Stage 3 window."""

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


def _portfolio_labelled_frame(portfolio_returns: pd.DataFrame) -> pd.DataFrame:
    """Rename portfolio-return columns to their display labels."""

    rename_map = {column: PORTFOLIO_LABELS[column] for column in PORTFOLIO_COLUMN_ORDER}
    frame = portfolio_returns.rename(columns=rename_map).copy()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def make_weights_figure(
    weights: pd.DataFrame,
    *,
    provider: str,
    sample: Stage3Sample,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot panel-by-panel signed portfolio weights with bars."""

    pivot = weights.pivot(index="ticker", columns="portfolio", values="weight").copy()
    portfolios = [PORTFOLIO_LABELS[key] for key in PORTFOLIO_COLUMN_ORDER]
    pivot = pivot.reindex(columns=portfolios)
    max_abs_weight_pct = float(pivot.abs().max().max() * 100.0)
    x_limit = max(4.0, np.ceil(max_abs_weight_pct / 5.0) * 5.0)

    with figure_style("word_a4", style="ft"):
        fig, axes = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=(7.4, 13.8),
            layout="none",
        )
        panel_specs = [
            (PORTFOLIO_LABELS["equal_weight"], "Equal-weight"),
            (PORTFOLIO_LABELS["minimum_variance"], "Minimum-variance"),
            (PORTFOLIO_LABELS["mean_variance_tangency"], "Mean-variance"),
        ]
        for ax, (portfolio_name, title) in zip(axes, panel_specs, strict=False):
            series = pivot[portfolio_name].copy()
            if portfolio_name == PORTFOLIO_LABELS["equal_weight"]:
                series = series.sort_index()
            else:
                series = series.sort_values(ascending=False)
            weight_pct = series * 100.0
            y_pos = np.arange(len(series))
            bar_color = PORTFOLIO_COLORS[portfolio_name]
            negative_color = "#c9c4ba"
            colors = [bar_color if value >= 0 else negative_color for value in weight_pct]
            ax.barh(
                y_pos,
                weight_pct.to_numpy(dtype=float),
                color=colors,
                edgecolor="none",
                height=0.72,
                alpha=0.95,
            )
            ax.axvline(0, color=_line_color("ft", "zero"), linewidth=0.9)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(series.index.tolist(), fontsize=4.4)
            ax.invert_yaxis()
            ax.set_xlim(-x_limit, x_limit)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0f}%"))
            _grid(ax)
            ax.set_title(title, loc="left", pad=8, fontsize=9.2)
            if portfolio_name == PORTFOLIO_LABELS["equal_weight"]:
                ax.text(
                    0.99,
                    0.02,
                    "All names at 2.0%",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=8.2,
                    color=_line_color("ft", "text"),
                )
        axes[-1].set_xlabel("Portfolio weight (%)")
        fig.subplots_adjust(hspace=0.20, top=0.97, bottom=0.05, left=0.18, right=0.97)
    context = FigureContext(
        title=f"{provider_label(provider)} portfolio weights",
        note=(
            "Each panel shows the full signed weight vector. The optimized portfolios "
            "allow short sales, so negative bars appear to the left of zero."
        ),
        source=provider_source_note(provider),
        sample=sample_label(sample),
        units="Weight (%).",
    )
    return export_stage3_figure(
        fig,
        output_dir,
        f"{provider}_stage3_portfolio_weights",
        context,
        spec="portrait_full",
    )


def make_growth_figure(
    portfolio_returns: pd.DataFrame,
    *,
    provider: str,
    sample: Stage3Sample,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot growth of one dollar for the three in-sample portfolios."""

    frame = _portfolio_labelled_frame(portfolio_returns)
    with figure_style("word_a4", style="ft"):
        fig, ax = plt.subplots(figsize=(7.2, 4.4), layout="none")
        for label in [PORTFOLIO_LABELS[key] for key in PORTFOLIO_COLUMN_ORDER]:
            wealth = wealth_index(frame[label])
            ax.plot(
                frame["date"],
                wealth,
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
            bbox_to_anchor=(0.5, -0.16),
            ncol=3,
            frameon=False,
        )
        fig.subplots_adjust(bottom=0.23)
    context = FigureContext(
        title=f"{provider_label(provider)} portfolio growth of $1",
        note="Daily rebalanced constant-weight portfolio returns, plotted on a log scale.",
        source=provider_source_note(provider),
        sample=sample_label(sample),
        units="Growth of $1.",
    )
    return export_stage3_figure(
        fig,
        output_dir,
        f"{provider}_stage3_growth_of_one",
        context,
    )


def make_drawdown_figure(
    portfolio_returns: pd.DataFrame,
    *,
    provider: str,
    sample: Stage3Sample,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot portfolio drawdowns over the in-sample window."""

    frame = _portfolio_labelled_frame(portfolio_returns)
    with figure_style("word_a4", style="ft"):
        fig, ax = plt.subplots(figsize=(7.2, 4.4), layout="none")
        for label in [PORTFOLIO_LABELS[key] for key in PORTFOLIO_COLUMN_ORDER]:
            drawdown = drawdown_series(frame[label])
            ax.plot(
                frame["date"],
                drawdown,
                label=label,
                color=PORTFOLIO_COLORS[label],
                linewidth=1.0,
                alpha=0.88,
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
        ax.set_title("Portfolio drawdowns", loc="left")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=3,
            frameon=False,
        )
        fig.subplots_adjust(bottom=0.23)
    context = FigureContext(
        title=f"{provider_label(provider)} portfolio drawdowns",
        note="Drawdowns computed from daily rebalanced constant-weight portfolio returns.",
        source=provider_source_note(provider),
        sample=sample_label(sample),
        units="Drawdown (%).",
    )
    return export_stage3_figure(
        fig,
        output_dir,
        f"{provider}_stage3_drawdowns",
        context,
    )


def make_scorecard_figure(
    metrics: pd.DataFrame,
    *,
    provider: str,
    sample: Stage3Sample,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot a four-panel in-sample portfolio scorecard."""

    plot_order = [PORTFOLIO_LABELS[key] for key in PORTFOLIO_COLUMN_ORDER]
    plot_frame = metrics.set_index("portfolio").loc[plot_order].reset_index()
    specs = [
        ("annualized_return_pct", "Annualized return", "%", True),
        ("annualized_volatility_pct", "Annualized volatility", "%", False),
        ("sharpe_ratio", "Sharpe ratio", "", True),
        ("max_drawdown_pct", "Max drawdown", "%", False),
    ]
    with figure_style("word_a4", style="ft"):
        fig, axes = plt.subplots(2, 2, figsize=(7.3, 6.2), layout="none")
        for ax, (column, title, suffix, show_labels) in zip(axes.ravel(), specs, strict=False):
            values = plot_frame[column].astype(float)
            colors = [PORTFOLIO_COLORS[label] for label in plot_frame["portfolio"]]
            ax.barh(plot_frame["portfolio"], values, color=colors, alpha=0.92)
            ax.axvline(0, color=_line_color("ft", "zero"), linewidth=0.9)
            _grid(ax)
            ax.set_title(title, loc="left", pad=12)
            if suffix == "%":
                ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0f}%"))
            if not show_labels:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", left=False, labelleft=False)
            for y_pos, value in enumerate(values):
                offset = 1.5 if suffix == "%" else 0.04
                text_x = value + offset if value >= 0 else value - offset
                ha = "left" if value >= 0 else "right"
                formatted = f"{value:.1f}{suffix}" if suffix else f"{value:.2f}"
                ax.text(text_x, y_pos, formatted, va="center", ha=ha, fontsize=8.4)
        fig.subplots_adjust(hspace=0.48, wspace=0.36, bottom=0.12)
    context = FigureContext(
        title=f"{provider_label(provider)} in-sample portfolio scorecard",
        note=(
            "Higher is better for annualized return and Sharpe ratio. Lower is better "
            "for annualized volatility and drawdown depth."
        ),
        source=provider_source_note(provider),
        sample=sample_label(sample),
        units=(
            "Annualized return (%), annualized volatility (%), Sharpe ratio, "
            "and max drawdown (%)."
        ),
    )
    return export_stage3_figure(
        fig,
        output_dir,
        f"{provider}_stage3_scorecard",
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
    portfolio_returns: pd.DataFrame,
    *,
    provider: str,
    sample: Stage3Sample,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot the efficient frontier and capital allocation line."""

    rf_ann_pct = float(portfolio_returns["rfr"].mean() * 252.0 * 100.0)
    tan_sigma, tan_return = _point_from_metrics(metrics, PORTFOLIO_LABELS["mean_variance_tangency"])
    eq_sigma, eq_return = _point_from_metrics(metrics, PORTFOLIO_LABELS["equal_weight"])
    mv_sigma, mv_return = _point_from_metrics(metrics, PORTFOLIO_LABELS["minimum_variance"])
    tan_sharpe = float(
        metrics.loc[
            metrics["portfolio"] == PORTFOLIO_LABELS["mean_variance_tangency"],
            "sharpe_ratio",
        ].iloc[0]
    )
    x_max = max(
        float(frontier["volatility_ann_pct"].max()),
        float(asset_summary["annualized_volatility_pct"].max()),
        tan_sigma,
    ) * 1.08
    cal_x = np.linspace(0.0, x_max, 240)
    cal_slope = (tan_return - rf_ann_pct) / tan_sigma if not np.isclose(tan_sigma, 0.0) else 0.0
    cal_y = rf_ann_pct + cal_slope * cal_x

    with figure_style("word_a4", style="ft"):
        fig, ax = plt.subplots(figsize=(7.2, 4.8), layout="none")
        ax.scatter(
            asset_summary["annualized_volatility_pct"],
            asset_summary["annualized_return_pct"],
            s=24,
            color="#8d887f",
            alpha=0.45,
            label="Individual stocks",
            zorder=2,
        )
        ax.plot(
            frontier["volatility_ann_pct"],
            frontier["target_return_ann_pct"],
            color="#24364f",
            linewidth=2.0,
            label="Efficient frontier",
            zorder=3,
        )
        ax.plot(
            cal_x,
            cal_y,
            color="#8e2f4d",
            linewidth=1.6,
            linestyle="--",
            label="Tangency line",
            zorder=2,
        )
        points = [
            ("Risk-free", 0.0, rf_ann_pct, "#6f6a61", "o", (18, 8)),
            ("Equal-weight", eq_sigma, eq_return, PORTFOLIO_COLORS["Equal-weight"], "s", (42, -14)),
            (
                "Minimum variance",
                mv_sigma,
                mv_return,
                PORTFOLIO_COLORS["Minimum variance"],
                "D",
                (28, 16),
            ),
            (
                "Mean-variance",
                tan_sigma,
                tan_return,
                PORTFOLIO_COLORS["Mean-variance"],
                "^",
                (-20, 18),
            ),
        ]
        for label, x_value, y_value, color, marker, offset in points:
            ax.scatter([x_value], [y_value], s=72, color=color, marker=marker, zorder=4)
            horizontal_alignment = "left" if offset[0] >= 0 else "right"
            ax.annotate(
                label,
                (x_value, y_value),
                textcoords="offset points",
                xytext=offset,
                ha=horizontal_alignment,
                va="center",
                fontsize=8.6,
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
        ax.text(
            0.03,
            0.97,
            f"Mean-variance Sharpe: {tan_sharpe:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.6,
            color=_line_color("ft", "text"),
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "#faf8f3",
                "edgecolor": "#d6d1c6",
                "alpha": 0.96,
            },
        )
        _grid(ax)
        ax.set_title("Efficient frontier", loc="left")
        ax.set_xlabel("Annualized volatility (%)")
        ax.set_ylabel("Annualized expected return (%)")
        ax.legend(loc="lower right", frameon=False)
    context = FigureContext(
        title=f"{provider_label(provider)} efficient frontier",
        note=(
            "The efficient frontier uses sample daily means and covariances from the full "
            "balanced in-sample window. The dashed line is the capital allocation line "
            "through the tangency portfolio."
        ),
        source=provider_source_note(provider),
        sample=sample_label(sample),
        units="Annualized expected return (%) and annualized volatility (%).",
    )
    return export_stage3_figure(
        fig,
        output_dir,
        f"{provider}_stage3_efficient_frontier",
        context,
    )


def make_stage3_figure_pack(
    *,
    provider: str,
    sample: Stage3Sample,
    weights: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    frontier: pd.DataFrame,
    metrics: pd.DataFrame,
    asset_summary: pd.DataFrame,
    output_dir: Path,
) -> dict[str, dict[str, Path]]:
    """Export the full Stage 3 figure pack for one provider."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "weights": make_weights_figure(
            weights,
            provider=provider,
            sample=sample,
            output_dir=output_dir,
        ),
        "growth_of_one": make_growth_figure(
            portfolio_returns,
            provider=provider,
            sample=sample,
            output_dir=output_dir,
        ),
        "drawdowns": make_drawdown_figure(
            portfolio_returns,
            provider=provider,
            sample=sample,
            output_dir=output_dir,
        ),
        "scorecard": make_scorecard_figure(
            metrics,
            provider=provider,
            sample=sample,
            output_dir=output_dir,
        ),
        "efficient_frontier": make_frontier_figure(
            frontier,
            metrics,
            asset_summary,
            portfolio_returns,
            provider=provider,
            sample=sample,
            output_dir=output_dir,
        ),
    }
    return outputs
