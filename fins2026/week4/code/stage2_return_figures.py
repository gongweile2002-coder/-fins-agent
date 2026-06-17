# ruff: noqa
"""FT-style Stage 2 figures for Week 4 return diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from fintools.figures import (
    FigureContext,
    add_nber_recession_shading,
    distribution_plot,
    diverging_bar_plot,
    export_word_figure,
    lollipop_plot,
    time_series_plot,
)
from fintools.figures.plots import _categorical_colors, _format_date_axis, _format_growth_dollars

from .stage2_equity_returns import (
    resolve_stage2_provider,
    select_top_bottom_volatility_tickers,
    summarize_full_sample_volatility,
)

WEEK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE2_FIGURE_ROOT = WEEK_ROOT / "results" / "figures" / "stage2"


def stage2_figure_dir(provider: str) -> Path:
    """Return the default Stage 2 figure directory for a provider."""

    return DEFAULT_STAGE2_FIGURE_ROOT / provider


def provider_source_note(provider: str) -> str:
    """Return the source note used in Stage 2 figure captions."""

    if provider == "tiingo":
        return "Tiingo end-of-day prices; Kenneth French Data Library daily RF."
    if provider == "yahoo":
        return "Yahoo Finance chart history; Kenneth French Data Library daily RF."
    return "Stage 2 Week 4 provider data."


def provider_label(provider: str) -> str:
    """Return the display label used in visible figure text."""

    return resolve_stage2_provider(provider).display_name


def sample_label(frame: pd.DataFrame) -> str:
    """Return a caption-ready sample label from the date column."""

    dates = pd.to_datetime(frame["date"]).dropna()
    return f"{dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}"


def export_stage2_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    context: FigureContext,
    *,
    spec: str = "full_width",
) -> dict[str, Path]:
    """Export one Word-ready Stage 2 figure and close it."""

    paths = export_word_figure(fig, output_dir, stem, context=context, spec=spec)
    plt.close(fig)
    return paths


def make_return_distribution_figure(
    feature_panel: pd.DataFrame,
    *,
    provider: str,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot the pooled daily return distribution."""

    plot_frame = feature_panel.dropna(subset=["ret"]).copy()
    plot_frame["daily_return_pct"] = plot_frame["ret"] * 100.0
    label = provider_label(provider)
    fig, ax = distribution_plot(
        plot_frame,
        "daily_return_pct",
        title=f"{label} daily return distribution",
        profile="word_a4",
        style="ft",
    )
    ax.set_xlabel("Daily return (%)")
    context = FigureContext(
        title=f"{label} daily return distribution",
        note="Pooled simple daily returns from adjusted prices across all Stage 2 tickers.",
        source=provider_source_note(provider),
        sample=sample_label(plot_frame),
        units="Daily return (%).",
    )
    return export_stage2_figure(
        fig,
        output_dir,
        f"{provider}_stage2_return_distribution",
        context,
    )


def make_extreme_moves_figure(
    feature_panel: pd.DataFrame,
    *,
    provider: str,
    output_dir: Path,
    limit: int = 15,
) -> dict[str, Path]:
    """Plot the largest absolute one-day returns."""

    plot_frame = feature_panel.dropna(subset=["ret"]).copy()
    plot_frame = plot_frame.assign(
        ret_pct=plot_frame["ret"] * 100.0,
        abs_ret=plot_frame["ret"].abs(),
    )
    plot_frame = plot_frame.nlargest(limit, "abs_ret").sort_values("ret_pct")
    plot_frame["ticker_date"] = (
        plot_frame["ticker"] + " | " + pd.to_datetime(plot_frame["date"]).dt.strftime("%Y-%m-%d")
    )
    label = provider_label(provider)
    fig, ax = diverging_bar_plot(
        plot_frame,
        category="ticker_date",
        value="ret_pct",
        title=f"{label} largest one-day moves",
        xlabel="Daily return (%)",
        ylabel="Ticker | date",
        profile="word_a4",
        style="ft",
    )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0f}%"))
    context = FigureContext(
        title=f"{label} largest one-day moves",
        note="Largest signed single-day adjusted-price returns, ranked by absolute size.",
        source=provider_source_note(provider),
        sample=sample_label(feature_panel),
        units="Daily return (%).",
    )
    return export_stage2_figure(
        fig,
        output_dir,
        f"{provider}_stage2_extreme_moves",
        context,
        spec="portrait_tall",
    )


def make_volatility_ranking_figure(
    feature_panel: pd.DataFrame,
    *,
    provider: str,
    output_dir: Path,
    top_n: int = 5,
) -> tuple[dict[str, Path], pd.DataFrame]:
    """Plot the full-sample annualized volatility ranking."""

    summary = summarize_full_sample_volatility(feature_panel)
    bottom, top = select_top_bottom_volatility_tickers(summary, n=top_n)
    highlight = bottom + top
    label = provider_label(provider)
    fig, ax = lollipop_plot(
        summary,
        category="ticker",
        value="ann_volatility_pct",
        title=f"{label} annualized volatility ranking",
        xlabel="Annualized volatility (%)",
        ylabel="Ticker",
        highlight=highlight,
        profile="word_a4",
        style="ft",
    )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0f}%"))
    context = FigureContext(
        title=f"{label} annualized volatility ranking",
        note=(
            "Full-sample annualized volatility from simple daily adjusted-price "
            "returns. Highlighted names define the top and bottom five groups "
            "used in the cumulative-return comparison."
        ),
        source=provider_source_note(provider),
        sample=sample_label(feature_panel),
        units="Annualized volatility (%).",
    )
    paths = export_stage2_figure(
        fig,
        output_dir,
        f"{provider}_stage2_volatility_ranking",
        context,
        spec="portrait_full",
    )
    return paths, summary


def _top_bottom_highlight(summary: pd.DataFrame, column: str, *, n: int = 5) -> list[str]:
    """Return the top and bottom names for one summary metric."""

    metric = summary.dropna(subset=[column]).copy()
    bottom = metric.nsmallest(n, column)["ticker"].tolist()
    top = metric.nlargest(n, column)["ticker"].tolist()
    return bottom + top


def make_annualized_return_ranking_figure(
    volatility_summary: pd.DataFrame,
    *,
    provider: str,
    output_dir: Path,
    sample: str,
    top_n: int = 5,
) -> dict[str, Path]:
    """Plot the full-sample annualized return ranking."""

    label = provider_label(provider)
    highlight = _top_bottom_highlight(volatility_summary, "ann_return_pct", n=top_n)
    fig, ax = lollipop_plot(
        volatility_summary,
        category="ticker",
        value="ann_return_pct",
        title=f"{label} annualized return ranking",
        xlabel="Annualized return (%)",
        ylabel="Ticker",
        highlight=highlight,
        profile="word_a4",
        style="ft",
    )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0f}%"))
    context = FigureContext(
        title=f"{label} annualized return ranking",
        note=(
            "Full-sample annualized simple returns from adjusted prices. "
            "Highlighted names are the top and bottom five by annualized return."
        ),
        source=provider_source_note(provider),
        sample=sample,
        units="Annualized return (%).",
    )
    return export_stage2_figure(
        fig,
        output_dir,
        f"{provider}_stage2_annualized_return_ranking",
        context,
        spec="portrait_full",
    )


def make_sharpe_ranking_figure(
    volatility_summary: pd.DataFrame,
    *,
    provider: str,
    output_dir: Path,
    sample: str,
    top_n: int = 5,
) -> dict[str, Path]:
    """Plot the full-sample Sharpe ratio ranking."""

    label = provider_label(provider)
    highlight = _top_bottom_highlight(volatility_summary, "ann_sharpe", n=top_n)
    fig, _ax = lollipop_plot(
        volatility_summary,
        category="ticker",
        value="ann_sharpe",
        title=f"{label} Sharpe ratio ranking",
        xlabel="Annualized Sharpe ratio",
        ylabel="Ticker",
        highlight=highlight,
        profile="word_a4",
        style="ft",
    )
    context = FigureContext(
        title=f"{label} Sharpe ratio ranking",
        note=(
            "Full-sample annualized Sharpe ratios from daily excess returns. "
            "Highlighted names are the top and bottom five by Sharpe ratio."
        ),
        source=provider_source_note(provider),
        sample=sample,
        units="Annualized Sharpe ratio.",
    )
    return export_stage2_figure(
        fig,
        output_dir,
        f"{provider}_stage2_sharpe_ranking",
        context,
        spec="portrait_full",
    )


def _wealth_paths_from_return_panel(
    feature_panel: pd.DataFrame,
    tickers: list[str],
) -> pd.DataFrame:
    """Build growth-of-one-dollar paths for a selected ticker group."""

    frame = feature_panel.loc[
        feature_panel["ticker"].isin(tickers),
        ["date", "ticker", "ret"],
    ].copy()
    wide = frame.pivot(index="date", columns="ticker", values="ret").sort_index()
    wealth = (1.0 + wide.fillna(0.0)).cumprod()
    wealth = wealth.loc[:, tickers]
    return wealth.reset_index()


def make_top_bottom_growth_figure(
    feature_panel: pd.DataFrame,
    volatility_summary: pd.DataFrame,
    *,
    provider: str,
    output_dir: Path,
    top_n: int = 5,
) -> dict[str, Path]:
    """Plot growth-of-one-dollar paths for the least and most volatile names."""

    bottom, top = select_top_bottom_volatility_tickers(volatility_summary, n=top_n)
    low_wealth = _wealth_paths_from_return_panel(feature_panel, bottom)
    high_wealth = _wealth_paths_from_return_panel(feature_panel, top)
    label = provider_label(provider)

    fig, axes = plt.subplots(1, 2, figsize=(9.7, 5.45), sharey=False)
    palette = _categorical_colors("ft", len(bottom) + len(top))
    panels = [
        (axes[0], low_wealth, bottom, "Bottom 5 by volatility", palette[: len(bottom)]),
        (axes[1], high_wealth, top, "Top 5 by volatility", palette[len(bottom) :]),
    ]
    for ax, wealth_frame, tickers, title, colors in panels:
        wealth_frame = wealth_frame.copy()
        wealth_frame["date"] = pd.to_datetime(wealth_frame["date"])
        for index, ticker in enumerate(tickers):
            ax.plot(
                wealth_frame["date"],
                wealth_frame[ticker],
                label=ticker,
                color=colors[index],
                linewidth=1.35,
                alpha=0.92,
                zorder=3,
            )
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(_format_growth_dollars))
        add_nber_recession_shading(
            ax,
            data_start=wealth_frame["date"].min(),
            data_end=wealth_frame["date"].max(),
            style="ft",
        )
        _format_date_axis(
            ax,
            date_start=wealth_frame["date"].min(),
            date_end=wealth_frame["date"].max(),
            max_ticks=6,
        )
        ax.grid(False, axis="x")
        ax.grid(True, axis="y", color="#E6E2DC", linewidth=0.7, alpha=0.8)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Date")
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncols=3,
            frameon=False,
        )
    axes[0].set_ylabel("Growth of $1")
    axes[1].set_ylabel("")
    fig.subplots_adjust(bottom=0.26, wspace=0.18)
    context = FigureContext(
        title=f"{label} growth of $1 for the least and most volatile stocks",
        note=(
            "Each panel compounds simple daily adjusted-price returns. The "
            "least-volatile names are shown separately from the most-volatile "
            "names, with panel-specific legends and distinct color sets so "
            "group membership is unambiguous."
        ),
        source=provider_source_note(provider),
        sample=sample_label(feature_panel),
        units="Growth of one dollar, log scale.",
    )
    return export_stage2_figure(
        fig,
        output_dir,
        f"{provider}_stage2_top_bottom_volatility_growth",
        context,
        spec="landscape_wide",
    )


def make_extreme_move_count_figure(
    feature_panel: pd.DataFrame,
    *,
    provider: str,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot the count of names with absolute daily returns above 10%."""

    counts = (
        feature_panel.dropna(subset=["ret"])
        .groupby("date", as_index=False)["is_large_move_10pct"]
        .sum()
        .rename(columns={"is_large_move_10pct": "extreme_move_count"})
    )
    legend_label = "Return > |10%|"
    counts = counts.rename(columns={"extreme_move_count": legend_label})
    label = provider_label(provider)
    fig, _ax = time_series_plot(
        counts,
        legend_label,
        date="date",
        title=f"{label} extreme-move count",
        ylabel="Stocks with |daily return| above 10%",
        profile="word_a4",
        style="ft",
    )
    context = FigureContext(
        title=f"{label} count of stocks with absolute daily returns above 10%",
        note=(
            "Large single-day moves should cluster in broad stress episodes "
            "more than in random isolated dates. This is a quick panel-wide "
            "data-quality check."
        ),
        source=provider_source_note(provider),
        sample=sample_label(feature_panel),
        units="Count of stocks.",
    )
    return export_stage2_figure(
        fig,
        output_dir,
        f"{provider}_stage2_extreme_move_count",
        context,
    )


def make_stage2_figure_pack(
    feature_panel: pd.DataFrame,
    *,
    provider: str,
    output_dir: Path,
) -> dict[str, dict[str, Path]]:
    """Export the full Stage 2 diagnostic figure pack."""

    outputs: dict[str, dict[str, Path]] = {}
    sample = sample_label(feature_panel)
    outputs["distribution"] = make_return_distribution_figure(
        feature_panel,
        provider=provider,
        output_dir=output_dir,
    )
    outputs["extreme_moves"] = make_extreme_moves_figure(
        feature_panel,
        provider=provider,
        output_dir=output_dir,
    )
    ranking_paths, volatility_summary = make_volatility_ranking_figure(
        feature_panel,
        provider=provider,
        output_dir=output_dir,
    )
    outputs["volatility_ranking"] = ranking_paths
    outputs["annualized_return_ranking"] = make_annualized_return_ranking_figure(
        volatility_summary,
        provider=provider,
        output_dir=output_dir,
        sample=sample,
    )
    outputs["sharpe_ranking"] = make_sharpe_ranking_figure(
        volatility_summary,
        provider=provider,
        output_dir=output_dir,
        sample=sample,
    )
    outputs["top_bottom_growth"] = make_top_bottom_growth_figure(
        feature_panel,
        volatility_summary,
        provider=provider,
        output_dir=output_dir,
    )
    outputs["extreme_move_count"] = make_extreme_move_count_figure(
        feature_panel,
        provider=provider,
        output_dir=output_dir,
    )
    return outputs
