# ruff: noqa
"""FT-style Stage 2 figures for Week 5 crypto diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib.ticker import FuncFormatter

from fintools.figures import (
    FigureContext,
    add_nber_recession_shading,
    correlation_heatmap,
    export_word_figure,
    figure_style,
    lollipop_plot,
    small_multiples,
)
from fintools.figures.plots import _categorical_colors, _format_date_axis, _format_growth_dollars

from .stage2_crypto_returns import (
    summarize_stage2_metrics,
)

WEEK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE2_FIGURE_ROOT = WEEK_ROOT / "results" / "figures" / "stage2" / "yahoo_crypto"
HEADLINE_FOUR = ["BTC-USD", "ETH-USD", "DOGE-USD", "ADA-USD"]
HEADLINE_FIVE = ["BTC-USD", "ETH-USD", "XRP-USD", "ADA-USD", "DOGE-USD"]
HEADLINE_VOLUME_FOUR = ["BTC-USD", "ETH-USD", "XRP-USD", "DOGE-USD"]
HEADLINE_ALTS = [ticker for ticker in HEADLINE_FIVE if ticker != "BTC-USD"]
ROLLING_BTC_CORR_DAYS = 90
TRAILING_DOLLAR_VOLUME_DAYS = 30


def stage2_figure_dir() -> Path:
    """Return the default Stage 2 crypto figure directory."""

    return DEFAULT_STAGE2_FIGURE_ROOT


def provider_source_note() -> str:
    """Return the source note used in Stage 2 figure captions."""

    return (
        "Yahoo Finance chart history for USD-quoted cryptocurrencies; Kenneth French "
        "Data Library daily RF forward-filled across weekends and market holidays."
    )


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


def _apply_ft_time_series_grid(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(False, axis="x")
    ax.grid(True, axis="y", color="#E6E2DC", linewidth=0.7, alpha=0.8)


def _apply_ft_bar_grid(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, axis="x", color="#E6E2DC", linewidth=0.7, alpha=0.8)
    ax.grid(False, axis="y")


def _ordered_ticker_frame(frame: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    selected = frame.loc[frame["ticker"].isin(tickers)].copy()
    selected["ticker"] = pd.Categorical(selected["ticker"], categories=tickers, ordered=True)
    return selected.sort_values(["ticker", "date"]).reset_index(drop=True)


def _display_ticker(ticker: str) -> str:
    """Return a compact display label for a Yahoo crypto ticker."""

    return ticker.removesuffix("-USD")


def _format_usd_billions(value: float, _pos: int) -> str:
    """Format dollar-volume ticks expressed in billions of USD."""

    absolute = abs(float(value))
    if absolute >= 1_000.0:
        scaled = value / 1_000.0
        if abs(scaled) >= 100.0:
            return f"${scaled:,.0f}tn"
        return f"${scaled:,.1f}tn"
    if absolute >= 100.0:
        return f"${value:,.0f}bn"
    if absolute >= 1.0:
        return f"${value:,.1f}bn"
    if absolute >= 0.001:
        return f"${value * 1_000.0:,.0f}m"
    return f"${value * 1_000_000.0:,.0f}k"


def _return_wide(feature_panel: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Return a wide return panel for a selected ticker list."""

    frame = _ordered_ticker_frame(feature_panel, tickers)
    wide = frame.pivot(index="date", columns="ticker", values="ret").sort_index()
    return wide.reindex(columns=tickers)


def _wealth_wide(return_wide: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide return matrix into wealth paths."""

    return (1.0 + return_wide.fillna(0.0)).cumprod()


def _drawdown_wide(wealth_wide: pd.DataFrame) -> pd.DataFrame:
    """Convert wealth paths into drawdown paths."""

    return wealth_wide.div(wealth_wide.cummax()).sub(1.0)


def _build_headline_drawdown_wide(feature_panel: pd.DataFrame) -> pd.DataFrame:
    """Return headline-coin drawdown paths."""

    wealth = _wealth_wide(_return_wide(feature_panel, HEADLINE_FIVE))
    drawdown = _drawdown_wide(wealth)
    return drawdown.rename(columns={ticker: _display_ticker(ticker) for ticker in drawdown.columns})


def _build_relative_to_btc_wide(feature_panel: pd.DataFrame) -> pd.DataFrame:
    """Return altcoin wealth paths relative to BTC."""

    wealth = _wealth_wide(_return_wide(feature_panel, HEADLINE_FIVE))
    btc = wealth["BTC-USD"].replace(0.0, np.nan)
    relative = wealth.loc[:, HEADLINE_ALTS].div(btc, axis=0)
    return relative.rename(columns={ticker: _display_ticker(ticker) for ticker in relative.columns})


def _build_rolling_corr_to_btc_wide(feature_panel: pd.DataFrame) -> pd.DataFrame:
    """Return 90-day rolling altcoin correlations with BTC."""

    wide = _return_wide(feature_panel, HEADLINE_FIVE)
    btc = wide["BTC-USD"]
    columns: dict[str, pd.Series] = {}
    for ticker in HEADLINE_ALTS:
        columns[f"{_display_ticker(ticker)} vs BTC"] = wide[ticker].rolling(
            ROLLING_BTC_CORR_DAYS,
            min_periods=ROLLING_BTC_CORR_DAYS,
        ).corr(btc)
    return pd.DataFrame(columns)


def _build_cross_sectional_dispersion_frame(feature_panel: pd.DataFrame) -> pd.DataFrame:
    """Return daily cross-sectional dispersion measures for the crypto panel."""

    frame = feature_panel.dropna(subset=["ret"]).copy()
    dispersion = (
        frame.groupby("date")
        .agg(
            cross_sectional_vol=(
                "ret",
                lambda values: float(pd.Series(values).astype(float).std()),
            ),
            p90=("ret", lambda values: float(pd.Series(values).astype(float).quantile(0.9))),
            p10=("ret", lambda values: float(pd.Series(values).astype(float).quantile(0.1))),
        )
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )
    dispersion["interdecile_spread"] = dispersion["p90"] - dispersion["p10"]
    dispersion["date"] = pd.to_datetime(dispersion["date"])
    return dispersion[["date", "cross_sectional_vol", "interdecile_spread"]]


def _build_max_drawdown_summary(feature_panel: pd.DataFrame) -> pd.DataFrame:
    """Return max drawdown summaries by ticker."""

    wide = _return_wide(feature_panel, sorted(feature_panel["ticker"].dropna().unique().tolist()))
    drawdown = _drawdown_wide(_wealth_wide(wide))
    summary = (
        drawdown.min()
        .rename("max_drawdown")
        .reset_index()
        .rename(columns={"index": "ticker"})
        .sort_values("max_drawdown", ascending=True)
        .reset_index(drop=True)
    )
    summary["max_drawdown_pct"] = summary["max_drawdown"] * 100.0
    return summary


def _build_trailing_dollar_volume_frame(stage1_panel: pd.DataFrame) -> pd.DataFrame:
    """Return trailing median dollar volume by ticker for USD pairs."""

    required = {"ticker", "date", "volume"}
    missing = required.difference(stage1_panel.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(
            "Stage 1 panel is missing columns required for dollar volume: "
            f"{missing_text}."
        )

    frame = stage1_panel.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    frame = frame.dropna(subset=["volume"]).sort_values(["ticker", "date"])

    rows: list[dict[str, float | int | str]] = []
    for ticker, group in frame.groupby("ticker", sort=True):
        trailing = group.tail(TRAILING_DOLLAR_VOLUME_DAYS)
        if trailing.empty:
            continue
        rows.append(
            {
                "ticker": ticker,
                "trailing_median_dollar_volume_usd": float(trailing["volume"].median()),
                "window_end_date": pd.Timestamp(trailing["date"].max()).normalize(),
                "window_observations": len(trailing),
            }
        )

    ranking = (
        pd.DataFrame(rows)
        .sort_values("trailing_median_dollar_volume_usd", ascending=False)
        .reset_index(drop=True)
    )
    ranking["trailing_median_dollar_volume_bn"] = (
        ranking["trailing_median_dollar_volume_usd"] / 1_000_000_000.0
    )
    return ranking


def _build_headline_dollar_volume_wide(stage1_panel: pd.DataFrame) -> pd.DataFrame:
    """Return a wide daily dollar-volume panel for four headline coins."""

    frame = _ordered_ticker_frame(stage1_panel, HEADLINE_VOLUME_FOUR)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    wide = frame.pivot(index="date", columns="ticker", values="volume").sort_index()
    wide = wide.reindex(columns=HEADLINE_VOLUME_FOUR)
    return wide.rename(columns={ticker: _display_ticker(ticker) for ticker in wide.columns})


def _build_daily_volume_share_wide(stage1_panel: pd.DataFrame) -> pd.DataFrame:
    """Return daily dollar-volume shares for the headline five plus other."""

    frame = stage1_panel.loc[:, ["ticker", "date", "volume"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    frame = frame.dropna(subset=["volume"])
    wide = frame.pivot(index="date", columns="ticker", values="volume").sort_index()
    total = wide.sum(axis=1).replace(0.0, np.nan)
    share = wide.div(total, axis=0)
    share = share.fillna(0.0)

    headline_share = share.reindex(columns=HEADLINE_FIVE, fill_value=0.0)
    result = pd.DataFrame(index=share.index)
    for ticker in HEADLINE_FIVE:
        result[_display_ticker(ticker)] = headline_share[ticker]
    result["Other"] = (1.0 - headline_share.sum(axis=1)).clip(lower=0.0)
    return result


def _top_bottom_highlight(summary: pd.DataFrame, column: str, *, n: int = 5) -> list[str]:
    """Return the top and bottom names for one summary metric."""

    metric = summary.dropna(subset=[column]).copy()
    bottom = metric.nsmallest(n, column)["ticker"].tolist()
    top = metric.nlargest(n, column)["ticker"].tolist()
    return bottom + top


def make_headline_price_small_multiples(
    stage1_panel: pd.DataFrame,
    *,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot BTC, ETH, DOGE, and ADA price paths in a 2x2 grid."""

    plot_frame = _ordered_ticker_frame(stage1_panel, HEADLINE_FOUR)
    with figure_style("word_a4", style="ft"), plt.rc_context(
        {"figure.constrained_layout.use": False}
    ):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(9.7, 6.0),
            sharex=True,
            layout=None,
            gridspec_kw={"hspace": 0.14, "wspace": 0.14},
        )
        palette = _categorical_colors("ft", len(HEADLINE_FOUR))
        axes_array = axes.flatten()
        for index, (ax, ticker, color) in enumerate(
            zip(axes_array, HEADLINE_FOUR, palette, strict=True)
        ):
            ticker_frame = plot_frame.loc[plot_frame["ticker"] == ticker].copy()
            ticker_frame["date"] = pd.to_datetime(ticker_frame["date"])
            ax.plot(
                ticker_frame["date"],
                ticker_frame["adjClose"],
                color=color,
                linewidth=1.5,
                alpha=0.92,
            )
            add_nber_recession_shading(
                ax,
                data_start=ticker_frame["date"].min(),
                data_end=ticker_frame["date"].max(),
                style="ft",
            )
            _apply_ft_time_series_grid(ax)
            _format_date_axis(
                ax,
                date_start=ticker_frame["date"].min(),
                date_end=ticker_frame["date"].max(),
                max_ticks=6,
            )
            ax.set_title(ticker, loc="left")
            row_index, col_index = divmod(index, 2)
            ax.set_xlabel("Date" if row_index == 1 else "")
            ax.set_ylabel("Price (USD)" if col_index == 0 else "")
            if row_index == 0:
                ax.tick_params(axis="x", labelbottom=False)
        fig.subplots_adjust(bottom=0.11, top=0.95)
    context = FigureContext(
        title="Headline crypto prices",
        note="BTC, ETH, DOGE, and ADA daily adjusted closing prices from Yahoo Finance.",
        source=provider_source_note(),
        sample=sample_label(plot_frame),
        units="Price in USD.",
    )
    return export_stage2_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage2_headline_prices",
        context,
        spec="landscape_wide",
    )


def make_growth_of_one_dollar_figure(
    feature_panel: pd.DataFrame,
    *,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot log-scale growth of one dollar for five headline coins."""

    frame = _ordered_ticker_frame(feature_panel, HEADLINE_FIVE)
    wide = (
        frame.pivot(index="date", columns="ticker", values="ret")
        .sort_index()
        .fillna(0.0)
    )
    wealth = (1.0 + wide).cumprod().reset_index()

    with figure_style("word_a4", style="ft"), plt.rc_context(
        {"figure.constrained_layout.use": False}
    ):
        fig, ax = plt.subplots(figsize=(9.7, 5.45), layout=None)
        palette = _categorical_colors("ft", len(HEADLINE_FIVE))
        wealth["date"] = pd.to_datetime(wealth["date"])
        for ticker, color in zip(HEADLINE_FIVE, palette, strict=True):
            ax.plot(
                wealth["date"],
                wealth[ticker],
                label=ticker,
                color=color,
                linewidth=1.35,
                alpha=0.92,
            )
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(_format_growth_dollars))
        add_nber_recession_shading(
            ax,
            data_start=wealth["date"].min(),
            data_end=wealth["date"].max(),
            style="ft",
        )
        _apply_ft_time_series_grid(ax)
        _format_date_axis(
            ax,
            date_start=wealth["date"].min(),
            date_end=wealth["date"].max(),
            max_ticks=6,
        )
        ax.set_title("Growth of $1 across five headline cryptocurrencies", loc="left")
        ax.set_xlabel("Date")
        ax.set_ylabel("Growth of $1")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncols=3, frameon=False)
        fig.subplots_adjust(bottom=0.18, top=0.95)
    context = FigureContext(
        title="Growth of $1 across five headline cryptocurrencies",
        note="Each line compounds simple daily returns from adjusted prices.",
        source=provider_source_note(),
        sample=sample_label(feature_panel),
        units="Growth of one dollar, log scale.",
    )
    return export_stage2_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage2_growth_of_one_dollar",
        context,
        spec="landscape_wide",
    )


def make_headline_drawdown_figure(
    feature_panel: pd.DataFrame,
    *,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot drawdowns for the five headline cryptocurrencies."""

    drawdown = _build_headline_drawdown_wide(feature_panel)
    dates = pd.to_datetime(drawdown.index)

    with figure_style("word_a4", style="ft"), plt.rc_context(
        {"figure.constrained_layout.use": False}
    ):
        fig, ax = plt.subplots(figsize=(9.7, 4.9), layout=None)
        palette = _categorical_colors("ft", len(drawdown.columns))
        for column, color in zip(drawdown.columns, palette, strict=True):
            ax.plot(
                dates,
                drawdown[column],
                label=column,
                color=color,
                linewidth=1.35,
                alpha=0.92,
            )
        add_nber_recession_shading(
            ax,
            data_start=dates.min(),
            data_end=dates.max(),
            style="ft",
        )
        _apply_ft_time_series_grid(ax)
        _format_date_axis(
            ax,
            date_start=dates.min(),
            date_end=dates.max(),
            max_ticks=6,
        )
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0%}"))
        ax.set_title("Headline crypto drawdowns", loc="left")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncols=3, frameon=False)
        fig.subplots_adjust(bottom=0.18, top=0.95)
    context = FigureContext(
        title="Headline crypto drawdowns",
        note=(
            "Drawdowns are computed from compounded simple daily returns. The figure shows how "
            "far each coin fell below its own prior peak."
        ),
        source=provider_source_note(),
        sample=sample_label(feature_panel),
        units="Drawdown (%).",
    )
    return export_stage2_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage2_headline_drawdowns",
        context,
        spec="landscape_wide",
    )


def make_relative_to_btc_figure(
    feature_panel: pd.DataFrame,
    *,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot altcoin wealth paths relative to BTC."""

    relative = _build_relative_to_btc_wide(feature_panel)
    dates = pd.to_datetime(relative.index)

    with figure_style("word_a4", style="ft"), plt.rc_context(
        {"figure.constrained_layout.use": False}
    ):
        fig, ax = plt.subplots(figsize=(9.7, 4.9), layout=None)
        palette = _categorical_colors("ft", len(relative.columns))
        for column, color in zip(relative.columns, palette, strict=True):
            ax.plot(
                dates,
                relative[column],
                label=column,
                color=color,
                linewidth=1.35,
                alpha=0.92,
            )
        ax.set_yscale("log")
        add_nber_recession_shading(
            ax,
            data_start=dates.min(),
            data_end=dates.max(),
            style="ft",
        )
        _apply_ft_time_series_grid(ax)
        _format_date_axis(
            ax,
            date_start=dates.min(),
            date_end=dates.max(),
            max_ticks=6,
        )
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _pos: f"{value:.1f}" if value < 10 else f"{value:.0f}")
        )
        ax.set_title("Altcoin wealth relative to BTC", loc="left")
        ax.set_xlabel("Date")
        ax.set_ylabel("Relative wealth")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncols=4, frameon=False)
        fig.subplots_adjust(bottom=0.18, top=0.95)
    context = FigureContext(
        title="Altcoin wealth relative to BTC",
        note=(
            "Each line compounds simple daily returns and then divides the result by BTC's "
            "compounded path. Values above 1 mean the altcoin outperformed BTC over the sample."
        ),
        source=provider_source_note(),
        sample=sample_label(feature_panel),
        units="Relative wealth ratio.",
    )
    return export_stage2_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage2_relative_to_btc",
        context,
        spec="landscape_wide",
    )


def make_rolling_corr_to_btc_figure(
    feature_panel: pd.DataFrame,
    *,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot 90-day rolling correlations with BTC for the headline altcoins."""

    rolling_corr = _build_rolling_corr_to_btc_wide(feature_panel)
    dates = pd.to_datetime(rolling_corr.index)

    with figure_style("word_a4", style="ft"), plt.rc_context(
        {"figure.constrained_layout.use": False}
    ):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(9.7, 6.0),
            sharex=True,
            sharey=True,
            layout=None,
            gridspec_kw={"hspace": 0.16, "wspace": 0.14},
        )
        palette = _categorical_colors("ft", len(rolling_corr.columns))
        axes_array = axes.flatten()
        for index, (ax, column, color) in enumerate(
            zip(axes_array, rolling_corr.columns, palette, strict=True)
        ):
            ax.plot(
                dates,
                rolling_corr[column],
                color=color,
                linewidth=1.4,
                alpha=0.92,
            )
            ax.axhline(0.0, color="#4D4D4D", linewidth=0.8, zorder=1)
            add_nber_recession_shading(
                ax,
                data_start=dates.min(),
                data_end=dates.max(),
                style="ft",
            )
            _apply_ft_time_series_grid(ax)
            _format_date_axis(
                ax,
                date_start=dates.min(),
                date_end=dates.max(),
                max_ticks=6,
            )
            ax.set_ylim(-1.0, 1.0)
            ax.set_title(column, loc="left")
            row_index, col_index = divmod(index, 2)
            ax.set_xlabel("Date" if row_index == 1 else "")
            ax.set_ylabel("Corr with BTC" if col_index == 0 else "")
            if row_index == 0:
                ax.tick_params(axis="x", labelbottom=False)
        fig.subplots_adjust(bottom=0.11, top=0.95)
    context = FigureContext(
        title="Rolling correlations with BTC",
        note=(
            "Each panel reports a 90-day rolling correlation between one headline altcoin and "
            "BTC, using simple daily returns."
        ),
        source=provider_source_note(),
        sample=sample_label(feature_panel),
        units="Correlation coefficient.",
    )
    return export_stage2_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage2_rolling_corr_to_btc",
        context,
        spec="landscape_wide",
    )


def make_extreme_moves_figure(
    feature_panel: pd.DataFrame,
    *,
    output_dir: Path,
    limit: int = 15,
) -> dict[str, Path]:
    """Plot the largest downside and upside one-day returns."""

    plot_frame = feature_panel.dropna(subset=["ret"]).copy()
    plot_frame["ret_pct"] = plot_frame["ret"] * 100.0
    plot_frame["ticker_date"] = (
        plot_frame["ticker"].map(_display_ticker)
        + " | "
        + pd.to_datetime(plot_frame["date"]).dt.strftime("%Y-%m-%d")
    )
    downside = plot_frame.nsmallest(limit, "ret_pct").copy()
    upside = plot_frame.nlargest(limit, "ret_pct").copy()
    downside_floor = min(-100.0, float(downside["ret_pct"].min()) * 1.06)
    upside_ceiling = float(upside["ret_pct"].max()) * 1.08

    with figure_style("word_a4", style="ft"), plt.rc_context(
        {"figure.constrained_layout.use": False}
    ):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(9.7, 6.0),
            sharey=False,
            layout=None,
            gridspec_kw={"wspace": 0.28},
        )
        panels = [
            (axes[0], downside, "Largest downside moves", "#E6847A", True),
            (axes[1], upside, "Largest upside moves", "#0F5499", False),
        ]
        for ax, frame, title, color, is_downside in panels:
            ordered = frame.sort_values("ret_pct", ascending=is_downside).reset_index(drop=True)
            positions = np.arange(len(ordered))
            values = ordered["ret_pct"].astype(float).to_numpy()
            ax.hlines(
                positions,
                xmin=np.minimum(0.0, values),
                xmax=np.maximum(0.0, values),
                color="#CEC6B9",
                linewidth=2.0,
                zorder=2,
            )
            ax.scatter(
                values,
                positions,
                color=color,
                s=46,
                zorder=3,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.axvline(0.0, color="#4D4D4D", linewidth=0.8, zorder=1)
            ax.set_yticks(positions)
            ax.set_yticklabels(ordered["ticker_date"])
            ax.invert_yaxis()
            _apply_ft_bar_grid(ax)
            ax.tick_params(axis="y", labelsize=8.5)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0f}%"))
            ax.set_title(title, loc="left")
            ax.set_xlabel("Daily return (%)")
            ax.set_ylabel("Coin | date" if ax is axes[0] else "")
        axes[0].set_xlim(downside_floor, 0.0)
        axes[1].set_xlim(0.0, upside_ceiling)
        axes[1].yaxis.tick_right()
        axes[1].yaxis.set_label_position("right")
        axes[1].tick_params(axis="y", labelleft=False, labelright=True)
        fig.subplots_adjust(left=0.23, right=0.78, bottom=0.10, top=0.95, wspace=0.38)
    context = FigureContext(
        title="Largest crypto upside and downside moves",
        note=(
            "Left panel shows the largest one-day losses and right panel shows the largest "
            "one-day gains from simple daily adjusted-price returns. The downside panel uses "
            "its own axis because crypto losses are mechanically bounded near -100%, while "
            "upside moves are not."
        ),
        source=provider_source_note(),
        sample=sample_label(feature_panel),
        units="Daily return (%).",
    )
    return export_stage2_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage2_extreme_moves",
        context,
        spec="landscape_wide",
    )


def make_return_correlation_matrix_figure(
    feature_panel: pd.DataFrame,
    *,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot an FT-style daily-return correlation matrix for headline coins."""

    plot_frame = _ordered_ticker_frame(feature_panel.dropna(subset=["ret"]), HEADLINE_FIVE)
    wide = plot_frame.pivot(index="date", columns="ticker", values="ret").sort_index()
    wide = wide.rename(columns={ticker: _display_ticker(ticker) for ticker in wide.columns})
    wide.columns.name = None
    fig, _ax = correlation_heatmap(
        wide,
        title="Headline crypto daily-return correlations",
        xlabel="Crypto pair",
        ylabel="Crypto pair",
        profile="word_a4",
        style="ft",
    )
    context = FigureContext(
        title="Headline crypto daily-return correlation matrix",
        note=(
            "Pairwise correlations of simple daily returns across BTC, ETH, XRP, ADA, and DOGE. "
            "Higher positive values indicate coins that tend to move together on the same day."
        ),
        source=provider_source_note(),
        sample=sample_label(plot_frame),
        units="Correlation coefficient.",
    )
    return export_stage2_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage2_correlation_matrix",
        context,
        spec="landscape_wide",
    )


def make_volatility_ranking_figure(
    summary: pd.DataFrame,
    *,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot the full-sample annualized volatility ranking."""

    ordered = summary.sort_values("ann_volatility", ascending=False).reset_index(drop=True)
    highlight = ordered.head(5)["ticker"].tolist() + ordered.tail(5)["ticker"].tolist()
    fig, ax = lollipop_plot(
        ordered,
        category="ticker",
        value="ann_volatility_pct",
        title="Annualized crypto volatility ranking",
        xlabel="Annualized volatility (%)",
        ylabel="Ticker",
        highlight=highlight,
        profile="word_a4",
        style="ft",
    )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0f}%"))
    context = FigureContext(
        title="Annualized crypto volatility ranking",
        note=(
            "Full-sample annualized volatility from simple daily crypto returns, "
            "scaled with 365 days."
        ),
        source=provider_source_note(),
        sample="",
        units="Annualized volatility (%).",
    )
    return export_stage2_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage2_volatility_ranking",
        context,
        spec="portrait_full",
    )


def make_annualized_return_ranking_figure(
    summary: pd.DataFrame,
    *,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot the full-sample annualized return ranking."""

    ordered = summary.sort_values("ann_return", ascending=False).reset_index(drop=True)
    highlight = ordered.head(5)["ticker"].tolist() + ordered.tail(5)["ticker"].tolist()
    fig, ax = lollipop_plot(
        ordered,
        category="ticker",
        value="ann_return_pct",
        title="Annualized crypto return ranking",
        xlabel="Annualized return (%)",
        ylabel="Ticker",
        highlight=highlight,
        profile="word_a4",
        style="ft",
    )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0f}%"))
    context = FigureContext(
        title="Annualized crypto return ranking",
        note="Full-sample annualized simple returns from adjusted prices, scaled with 365 days.",
        source=provider_source_note(),
        sample=None,
        units="Annualized return (%).",
    )
    return export_stage2_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage2_annualized_return_ranking",
        context,
        spec="portrait_full",
    )


def _metric_ranking_figure(
    summary: pd.DataFrame,
    *,
    metric: str,
    title: str,
    xlabel: str,
    output_dir: Path,
    stem: str,
    units: str,
    note: str,
) -> dict[str, Path]:
    """Export a ranked FT-style lollipop chart for one summary metric."""

    highlight = _top_bottom_highlight(summary, metric, n=5)
    fig, _ax = lollipop_plot(
        summary,
        category="ticker",
        value=metric,
        title=title,
        xlabel=xlabel,
        ylabel="Ticker",
        highlight=highlight,
        profile="word_a4",
        style="ft",
    )
    context = FigureContext(
        title=title,
        note=note,
        source=provider_source_note(),
        sample=None,
        units=units,
    )
    return export_stage2_figure(
        fig,
        output_dir,
        stem,
        context,
        spec="portrait_full",
    )


def make_sharpe_ranking_figure(summary: pd.DataFrame, *, output_dir: Path) -> dict[str, Path]:
    """Plot the full-sample Sharpe ratio ranking."""

    return _metric_ranking_figure(
        summary,
        metric="full_sample_sharpe",
        title="Full-sample Sharpe ratio ranking",
        xlabel="Annualized Sharpe ratio",
        output_dir=output_dir,
        stem="yahoo_crypto_stage2_sharpe_ratios",
        units="Annualized Sharpe ratio.",
        note=(
            "Full-sample annualized Sharpe ratios computed from daily excess returns after "
            "forward-filling business-day RF across crypto weekends and holidays."
        ),
    )


def make_sortino_ranking_figure(summary: pd.DataFrame, *, output_dir: Path) -> dict[str, Path]:
    """Plot the full-sample Sortino ratio ranking."""

    return _metric_ranking_figure(
        summary,
        metric="full_sample_sortino",
        title="Full-sample Sortino ratio ranking",
        xlabel="Annualized Sortino ratio",
        output_dir=output_dir,
        stem="yahoo_crypto_stage2_sortino_ratios",
        units="Annualized Sortino ratio.",
        note="Full-sample annualized Sortino ratios computed from daily excess returns.",
    )


def make_distribution_vs_normal_figure(
    feature_panel: pd.DataFrame,
    *,
    output_dir: Path,
) -> dict[str, Path]:
    """Compare daily return histograms to fitted normal curves for headline coins."""

    plot_frame = _ordered_ticker_frame(feature_panel.dropna(subset=["ret"]), HEADLINE_FOUR)
    with figure_style("word_a4", style="ft"), plt.rc_context(
        {"figure.constrained_layout.use": False}
    ):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(9.7, 6.0),
            sharex=False,
            sharey=False,
            layout=None,
            gridspec_kw={"hspace": 0.14, "wspace": 0.14},
        )
        palette = _categorical_colors("ft", len(HEADLINE_FOUR))
        axes_array = axes.flatten()
        for index, (ax, ticker, color) in enumerate(
            zip(axes_array, HEADLINE_FOUR, palette, strict=True)
        ):
            ticker_frame = plot_frame.loc[plot_frame["ticker"] == ticker]
            values = (ticker_frame["ret"].astype(float) * 100.0).dropna().to_numpy()
            ax.hist(values, bins=40, density=True, color=color, alpha=0.7, edgecolor="none")
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1))
            if np.isfinite(std) and std > 0.0:
                x_grid = np.linspace(values.min(), values.max(), 250)
                normal_density = (
                    np.exp(-0.5 * ((x_grid - mean) / std) ** 2)
                    / (std * np.sqrt(2.0 * np.pi))
                )
                ax.plot(x_grid, normal_density, color="#4D4D4D", linewidth=1.5, linestyle="--")
            _apply_ft_time_series_grid(ax)
            ax.set_title(ticker, loc="left")
            row_index, col_index = divmod(index, 2)
            ax.set_xlabel("Daily return (%)" if row_index == 1 else "")
            ax.set_ylabel("Density" if col_index == 0 else "")
            if row_index == 0:
                ax.tick_params(axis="x", labelbottom=False)
            stats_text = (
                f"Skew {ticker_frame['ret'].skew():.2f}\n"
                f"Ex. kurt {ticker_frame['ret'].kurt():.2f}"
            )
            ax.text(
                0.98,
                0.95,
                stats_text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#666666",
            )
        fig.subplots_adjust(bottom=0.11, top=0.95)
    context = FigureContext(
        title="Crypto return distributions versus fitted normals",
        note=(
            "Histogram densities use simple daily returns in percent. The dashed line is a normal "
            "curve fitted to each coin's sample mean and standard deviation."
        ),
        source=provider_source_note(),
        sample=sample_label(plot_frame),
        units="Daily return (%).",
    )
    return export_stage2_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage2_distribution_vs_normal",
        context,
        spec="landscape_wide",
    )


def make_tail_share_comparison_figure(
    summary: pd.DataFrame,
    *,
    output_dir: Path,
) -> dict[str, Path]:
    """Compare observed crypto tail shares with normal-distribution benchmarks."""

    plot_frame = summary.loc[summary["ticker"].isin(HEADLINE_FIVE)].copy()
    plot_frame["ticker"] = pd.Categorical(
        plot_frame["ticker"],
        categories=HEADLINE_FIVE,
        ordered=True,
    )
    plot_frame = plot_frame.sort_values("tail_share_abs_z_gt_3_pct", ascending=True).reset_index(
        drop=True
    )
    positions = np.arange(len(plot_frame))

    with figure_style("word_a4", style="ft"), plt.rc_context(
        {"figure.constrained_layout.use": False}
    ):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(9.7, 4.9),
            sharey=True,
            layout=None,
            gridspec_kw={"wspace": 0.08},
        )
        panels = [
            (
                axes[0],
                "tail_share_abs_z_gt_2_pct",
                "normal_share_abs_z_gt_2_pct",
                "Days with |z| > 2",
            ),
            (
                axes[1],
                "tail_share_abs_z_gt_3_pct",
                "normal_share_abs_z_gt_3_pct",
                "Days with |z| > 3",
            ),
        ]
        for panel_index, (ax, observed_column, benchmark_column, title) in enumerate(panels):
            observed = plot_frame[observed_column].astype(float).to_numpy()
            benchmark = plot_frame[benchmark_column].astype(float).to_numpy()
            ax.hlines(
                positions,
                xmin=np.minimum(benchmark, observed),
                xmax=np.maximum(benchmark, observed),
                color="#CEC6B9",
                linewidth=2.0,
                zorder=2,
            )
            ax.scatter(
                benchmark,
                positions,
                color="#E6847A",
                s=42,
                zorder=3,
                label="Normal benchmark" if panel_index == 0 else None,
            )
            ax.scatter(
                observed,
                positions,
                color="#0F5499",
                s=46,
                zorder=4,
                label="Observed" if panel_index == 0 else None,
            )
            _apply_ft_bar_grid(ax)
            ax.set_yticks(positions)
            if panel_index == 0:
                ax.set_yticklabels(plot_frame["ticker"])
            else:
                ax.tick_params(axis="y", labelleft=False)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.1f}%"))
            ax.set_title(title, loc="left")
            ax.set_xlabel("Share of days")
            ax.set_ylabel("Ticker" if panel_index == 0 else "")
            panel_max = float(np.nanmax(np.concatenate([observed, benchmark])))
            ax.set_xlim(0.0, panel_max * 1.22)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncols=2,
            frameon=False,
        )
        fig.subplots_adjust(bottom=0.14, top=0.94)
    context = FigureContext(
        title="Observed crypto tail frequencies versus normal benchmarks",
        note=(
            "Each panel compares the observed share of unusually large standardized daily moves "
            "with the share a normal distribution would imply. A normal would deliver about "
            "4.55% of days with |z| > 2 and about 0.27% with |z| > 3, so dots to the right "
            "of the benchmark show fatter tails than normal."
        ),
        source=provider_source_note(),
        sample=None,
        units="Share of days (%).",
    )
    return export_stage2_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage2_tail_share_comparison",
        context,
        spec="landscape_wide",
    )


def make_cross_sectional_dispersion_figure(
    feature_panel: pd.DataFrame,
    *,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot cross-sectional volatility and interdecile spread through time."""

    dispersion = _build_cross_sectional_dispersion_frame(feature_panel)

    with figure_style("word_a4", style="ft"), plt.rc_context(
        {"figure.constrained_layout.use": False}
    ):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(9.7, 4.8),
            sharex=True,
            layout=None,
            gridspec_kw={"wspace": 0.12},
        )
        panels = [
            (axes[0], "cross_sectional_vol", "Cross-sectional volatility"),
            (axes[1], "interdecile_spread", "Interdecile return spread"),
        ]
        colors = _categorical_colors("ft", len(panels))
        for index, (ax, column, title) in enumerate(panels):
            ax.plot(
                dispersion["date"],
                dispersion[column],
                color=colors[index],
                linewidth=1.45,
                alpha=0.92,
            )
            add_nber_recession_shading(
                ax,
                data_start=dispersion["date"].min(),
                data_end=dispersion["date"].max(),
                style="ft",
            )
            _apply_ft_time_series_grid(ax)
            _format_date_axis(
                ax,
                date_start=dispersion["date"].min(),
                date_end=dispersion["date"].max(),
                max_ticks=6,
            )
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0%}"))
            ax.set_title(title, loc="left")
            ax.set_xlabel("Date")
            ax.set_ylabel("Return spread" if index == 0 else "")
        fig.subplots_adjust(bottom=0.12, top=0.95)
    context = FigureContext(
        title="Cross-sectional crypto dispersion",
        note=(
            "The left panel shows the daily standard deviation of returns across the full "
            "crypto panel. The right panel shows the gap between the 90th and 10th percentiles."
        ),
        source=provider_source_note(),
        sample=sample_label(feature_panel),
        units="Daily return spread (%).",
    )
    return export_stage2_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage2_cross_sectional_dispersion",
        context,
        spec="landscape_wide",
    )


def make_rolling_volatility_headline_figure(
    feature_panel: pd.DataFrame,
    *,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot headline rolling volatility small multiples."""

    frame = _ordered_ticker_frame(feature_panel.dropna(subset=["rolling_6m_vol"]), HEADLINE_FIVE)
    wide = frame.pivot(index="date", columns="ticker", values="rolling_6m_vol").sort_index()
    wide = wide.rename(columns={ticker: _display_ticker(ticker) for ticker in wide.columns})
    fig, axes = small_multiples(
        wide,
        list(wide.columns),
        title="Headline rolling volatility",
        ylabel="Annualized volatility",
        shade_recessions=True,
        profile="word_a4",
        style="ft",
    )
    for ax in np.atleast_1d(axes)[: len(wide.columns)]:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0%}"))
    context = FigureContext(
        title="Headline rolling volatility",
        note=(
            "Each panel shows the 180-day rolling annualized volatility from simple daily "
            "returns, scaled with 365 days."
        ),
        source=provider_source_note(),
        sample=sample_label(frame),
        units="Annualized volatility (%).",
    )
    return export_stage2_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage2_rolling_volatility_headline",
        context,
        spec="portrait_full",
    )


def make_rolling_sharpe_headline_figure(
    feature_panel: pd.DataFrame,
    *,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot headline rolling Sharpe small multiples."""

    frame = _ordered_ticker_frame(feature_panel.dropna(subset=["rolling_6m_sharpe"]), HEADLINE_FIVE)
    wide = frame.pivot(index="date", columns="ticker", values="rolling_6m_sharpe").sort_index()
    wide = wide.rename(columns={ticker: _display_ticker(ticker) for ticker in wide.columns})
    fig, axes = small_multiples(
        wide,
        list(wide.columns),
        title="Headline rolling Sharpe ratios",
        ylabel="Rolling Sharpe",
        shade_recessions=True,
        profile="word_a4",
        style="ft",
    )
    for ax in np.atleast_1d(axes)[: len(wide.columns)]:
        ax.axhline(0.0, color="#4D4D4D", linewidth=0.8, zorder=1)
    context = FigureContext(
        title="Headline rolling Sharpe ratios",
        note=(
            "Each panel shows the 180-day rolling annualized Sharpe ratio based on daily "
            "excess returns after the French risk-free merge."
        ),
        source=provider_source_note(),
        sample=sample_label(frame),
        units="Annualized Sharpe ratio.",
    )
    return export_stage2_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage2_rolling_sharpe_headline",
        context,
        spec="portrait_full",
    )


def make_max_drawdown_ranking_figure(
    feature_panel: pd.DataFrame,
    *,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot a max drawdown ranking for the crypto panel."""

    summary = _build_max_drawdown_summary(feature_panel)
    summary["display_ticker"] = summary["ticker"].map(_display_ticker)
    highlight = [ticker for ticker in summary["display_ticker"] if f"{ticker}-USD" in HEADLINE_FIVE]
    fig, ax = lollipop_plot(
        summary,
        category="display_ticker",
        value="max_drawdown_pct",
        title="Max drawdown ranking",
        xlabel="Max drawdown (%)",
        ylabel="Ticker",
        highlight=highlight,
        profile="word_a4",
        style="ft",
    )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0f}%"))
    context = FigureContext(
        title="Max drawdown ranking",
        note=(
            "Max drawdown is the deepest peak-to-trough decline implied by compounded simple "
            "daily returns for each coin."
        ),
        source=provider_source_note(),
        sample=sample_label(feature_panel),
        units="Max drawdown (%).",
    )
    return export_stage2_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage2_max_drawdown_ranking",
        context,
        spec="portrait_full",
    )


def make_risk_return_scatter_figure(
    summary: pd.DataFrame,
    *,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot the risk-return cross-section with headline coins highlighted."""

    plot_frame = summary.copy()
    plot_frame["display_ticker"] = plot_frame["ticker"].map(_display_ticker)
    headline_set = set(HEADLINE_FIVE)
    plot_frame["is_headline"] = plot_frame["ticker"].isin(headline_set)
    headline = plot_frame.loc[plot_frame["is_headline"]].copy()
    others = plot_frame.loc[~plot_frame["is_headline"]].copy()
    headline_colors = _categorical_colors("ft", len(HEADLINE_FIVE))
    headline_palette = {
        ticker: color
        for ticker, color in zip(HEADLINE_FIVE, headline_colors, strict=True)
    }

    with figure_style("word_a4", style="ft"), plt.rc_context(
        {"figure.constrained_layout.use": False}
    ):
        fig, ax = plt.subplots(figsize=(7.4, 5.2), layout=None)
        if not others.empty:
            ax.scatter(
                others["ann_volatility_pct"],
                others["ann_return_pct"],
                color="#B3B3B3",
                alpha=0.55,
                s=34,
                zorder=2,
                label="Other coins",
            )
        for _, row in headline.iterrows():
            ticker = row["ticker"]
            color = headline_palette[ticker]
            ax.scatter(
                row["ann_volatility_pct"],
                row["ann_return_pct"],
                color=color,
                s=58,
                alpha=0.92,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
            ax.annotate(
                row["display_ticker"],
                (row["ann_volatility_pct"], row["ann_return_pct"]),
                textcoords="offset points",
                xytext=(6, 6),
                ha="left",
                fontsize=8.4,
                color="#4D4D4D",
            )
        ax.set_axisbelow(True)
        ax.grid(True, axis="both", color="#E6E2DC", linewidth=0.7, alpha=0.8)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0f}%"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0f}%"))
        ax.set_title("Risk-return map of the crypto cross-section", loc="left")
        ax.set_xlabel("Annualized volatility (%)")
        ax.set_ylabel("Annualized return (%)")
        if not others.empty:
            ax.legend(loc="upper left", frameon=False)
        fig.subplots_adjust(bottom=0.12, top=0.95)
    context = FigureContext(
        title="Risk-return map of the crypto cross-section",
        note=(
            "The figure compares annualized return and annualized volatility across the crypto "
            "panel. Headline coins are highlighted while the rest of the market is muted."
        ),
        source=provider_source_note(),
        sample=None,
        units="Annualized return and volatility (%).",
    )
    return export_stage2_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage2_risk_return_scatter",
        context,
        spec="full_width",
    )


def make_headline_dollar_volume_small_multiples(
    stage1_panel: pd.DataFrame,
    *,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot headline crypto dollar volume through time in a 2x2 grid."""

    volume_wide = _build_headline_dollar_volume_wide(stage1_panel) / 1_000_000_000.0
    dates = pd.to_datetime(volume_wide.index)

    with figure_style("word_a4", style="ft"), plt.rc_context(
        {"figure.constrained_layout.use": False}
    ):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(9.7, 6.0),
            sharex=True,
            layout=None,
            gridspec_kw={"hspace": 0.14, "wspace": 0.14},
        )
        palette = _categorical_colors("ft", len(volume_wide.columns))
        axes_array = axes.flatten()
        for index, (ax, column, color) in enumerate(
            zip(axes_array, volume_wide.columns, palette, strict=True)
        ):
            ax.plot(
                dates,
                volume_wide[column],
                color=color,
                linewidth=1.45,
                alpha=0.92,
            )
            add_nber_recession_shading(
                ax,
                data_start=dates.min(),
                data_end=dates.max(),
                style="ft",
            )
            _apply_ft_time_series_grid(ax)
            _format_date_axis(
                ax,
                date_start=dates.min(),
                date_end=dates.max(),
                max_ticks=6,
            )
            ax.yaxis.set_major_formatter(FuncFormatter(_format_usd_billions))
            ax.set_title(column, loc="left")
            row_index, col_index = divmod(index, 2)
            ax.set_xlabel("Date" if row_index == 1 else "")
            ax.set_ylabel("Dollar volume (USD bn)" if col_index == 0 else "")
            if row_index == 0:
                ax.tick_params(axis="x", labelbottom=False)
        fig.subplots_adjust(bottom=0.11, top=0.95)
    context = FigureContext(
        title="Headline crypto dollar volume",
        note=(
            "The figure uses Yahoo-reported daily `volume` directly for USD pairs. Week 5 "
            "treats this field as dollar volume and does not multiply by price again."
        ),
        source=provider_source_note(),
        sample=sample_label(stage1_panel),
        units="Dollar volume (USD bn).",
    )
    return export_stage2_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage2_headline_dollar_volume",
        context,
        spec="landscape_wide",
    )


def make_daily_volume_concentration_figure(
    stage1_panel: pd.DataFrame,
    *,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot daily dollar-volume shares across the crypto panel."""

    share_wide = _build_daily_volume_share_wide(stage1_panel)
    dates = pd.to_datetime(share_wide.index)
    x_values = mdates.date2num(dates.to_pydatetime())
    share_columns = list(share_wide.columns)
    stack_colors = [*_categorical_colors("ft", len(HEADLINE_FIVE)), "#CEC6B9"]

    with figure_style("word_a4", style="ft"), plt.rc_context(
        {"figure.constrained_layout.use": False}
    ):
        fig, ax = plt.subplots(figsize=(9.7, 5.45), layout=None)
        lower = np.zeros(len(share_wide), dtype=float)
        for column, color in zip(share_columns, stack_colors, strict=True):
            upper = lower + share_wide[column].to_numpy(dtype=float)
            ax.fill_between(
                x_values,
                lower,
                upper,
                color=color,
                alpha=0.9,
                linewidth=0.0,
                label=column,
                zorder=2,
            )
            lower = upper
        ax.xaxis_date()
        ax.set_axisbelow(True)
        ax.grid(False, axis="x")
        ax.grid(True, axis="y", color="#E6E2DC", linewidth=0.7, alpha=0.8)
        _format_date_axis(
            ax,
            date_start=dates.min(),
            date_end=dates.max(),
            max_ticks=6,
        )
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.0%}"))
        ax.set_ylim(0.0, 1.0)
        ax.set_title("How daily crypto dollar volume is split across the data", loc="left")
        ax.set_xlabel("Date")
        ax.set_ylabel("Share of daily total")
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            ncols=3,
            frameon=False,
        )
        fig.subplots_adjust(bottom=0.18, top=0.95)
    context = FigureContext(
        title="Daily crypto dollar-volume concentration",
        note=(
            "Each day sums to 100% across the full 20-coin dataset. The colored layers show the "
            "daily dollar-volume shares of the headline five, while the gray layer collects the "
            "rest of the universe into `Other`."
        ),
        source=provider_source_note(),
        sample=sample_label(stage1_panel),
        units="Share of total daily dollar volume (%).",
    )
    return export_stage2_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage2_dollar_volume_concentration",
        context,
        spec="landscape_wide",
    )


def make_dollar_volume_ranking_figure(
    stage1_panel: pd.DataFrame,
    *,
    output_dir: Path,
) -> dict[str, Path]:
    """Plot trailing dollar-volume rankings with a log-scale overview and zoom."""

    ranking = _build_trailing_dollar_volume_frame(stage1_panel)
    as_of_date = pd.Timestamp(ranking["window_end_date"].max()).strftime("%Y-%m-%d")
    figure_title = f"Latest 30-day median daily dollar volume, as of {as_of_date}"
    ranking["display_ticker"] = ranking["ticker"].map(_display_ticker)
    headline_palette = {
        ticker: color
        for ticker, color in zip(
            HEADLINE_FIVE,
            _categorical_colors("ft", len(HEADLINE_FIVE)),
            strict=True,
        )
    }
    ranking["point_color"] = ranking["ticker"].map(headline_palette).fillna("#D9D3C7")

    zoom_exclusions = ranking.head(2)["ticker"].tolist()
    zoom_exclusion_labels = [_display_ticker(ticker) for ticker in zoom_exclusions]
    altcoin_zoom = (
        ranking.loc[~ranking["ticker"].isin(zoom_exclusions)]
        .copy()
        .reset_index(drop=True)
    )
    left_positions = np.arange(len(ranking))
    right_positions = np.arange(len(altcoin_zoom))
    left_floor = float(
        10.0 ** np.floor(np.log10(ranking["trailing_median_dollar_volume_bn"].min()))
    )

    with figure_style("word_a4", style="ft"), plt.rc_context(
        {"figure.constrained_layout.use": False}
    ):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(9.7, 6.2),
            sharey=False,
            layout=None,
            gridspec_kw={"wspace": 0.18},
        )

        left_ax, right_ax = axes
        left_values = ranking["trailing_median_dollar_volume_bn"].to_numpy(dtype=float)
        right_values = altcoin_zoom["trailing_median_dollar_volume_bn"].to_numpy(dtype=float)

        left_ax.hlines(
            left_positions,
            xmin=np.repeat(left_floor, len(ranking)),
            xmax=left_values,
            color="#CEC6B9",
            linewidth=2.0,
            zorder=2,
        )
        left_ax.scatter(
            left_values,
            left_positions,
            color=ranking["point_color"],
            s=50,
            zorder=3,
        )
        left_ax.set_xscale("log")
        left_ax.set_xlim(left_floor * 0.85, float(left_values.max()) * 1.25)
        left_ax.xaxis.set_major_formatter(FuncFormatter(_format_usd_billions))
        left_ax.minorticks_off()
        _apply_ft_bar_grid(left_ax)
        left_ax.set_yticks(left_positions)
        left_ax.set_yticklabels(ranking["display_ticker"])
        left_ax.tick_params(axis="y", labelsize=9)
        left_ax.invert_yaxis()
        left_ax.set_title("All coins, log scale", loc="left")
        left_ax.set_xlabel("Trailing median dollar volume")
        left_ax.set_ylabel("Coin")

        right_ax.hlines(
            right_positions,
            xmin=0.0,
            xmax=right_values,
            color="#CEC6B9",
            linewidth=2.0,
            zorder=2,
        )
        right_ax.scatter(
            right_values,
            right_positions,
            color=altcoin_zoom["point_color"],
            s=50,
            zorder=3,
        )
        right_ax.set_xlim(0.0, float(right_values.max()) * 1.12)
        right_ax.xaxis.set_major_formatter(FuncFormatter(_format_usd_billions))
        _apply_ft_bar_grid(right_ax)
        right_ax.set_yticks(right_positions)
        right_ax.set_yticklabels(altcoin_zoom["display_ticker"])
        right_ax.tick_params(axis="y", labelsize=9)
        right_ax.invert_yaxis()
        right_ax.set_title(
            f"Zoom without {' and '.join(zoom_exclusion_labels)}, linear scale",
            loc="left",
        )
        right_ax.set_xlabel("Trailing median dollar volume")
        right_ax.set_ylabel("")

        fig.suptitle(figure_title, x=0.08, ha="left")
        fig.subplots_adjust(bottom=0.12, top=0.90)
    context = FigureContext(
        title=figure_title,
        note=(
            "For Yahoo `*-USD` crypto pairs, the Week 5 interpretation treats `volume` as a "
            "USD or dollar-volume field rather than base-asset units. The ranking therefore "
            "uses the trailing 30-day median of Yahoo-reported volume directly and does not "
            "multiply by price. The left panel uses a log scale because the largest coins "
            "dominate the market by orders of magnitude. "
            f"The right panel excludes {' and '.join(zoom_exclusion_labels)} so the remaining "
            "cross-section is visible on a linear scale."
        ),
        source=provider_source_note(),
        sample=sample_label(stage1_panel),
        units="Median dollar volume (USD bn).",
    )
    return export_stage2_figure(
        fig,
        output_dir,
        "yahoo_crypto_stage2_dollar_volume_ranking",
        context,
        spec="landscape_wide",
    )


def make_stage2_figure_pack(
    stage1_panel: pd.DataFrame,
    feature_panel: pd.DataFrame,
    *,
    output_dir: Path,
    summary: pd.DataFrame | None = None,
    include_appendix: bool = False,
) -> dict[str, dict[str, Path]]:
    """Export the full Stage 2 diagnostic figure pack."""

    summary_frame = summary if summary is not None else summarize_stage2_metrics(feature_panel)
    outputs: dict[str, dict[str, Path]] = {}
    outputs["headline_prices"] = make_headline_price_small_multiples(
        stage1_panel,
        output_dir=output_dir,
    )
    outputs["growth_of_one_dollar"] = make_growth_of_one_dollar_figure(
        feature_panel,
        output_dir=output_dir,
    )
    outputs["headline_drawdowns"] = make_headline_drawdown_figure(
        feature_panel,
        output_dir=output_dir,
    )
    outputs["relative_to_btc"] = make_relative_to_btc_figure(
        feature_panel,
        output_dir=output_dir,
    )
    outputs["rolling_corr_to_btc"] = make_rolling_corr_to_btc_figure(
        feature_panel,
        output_dir=output_dir,
    )
    outputs["extreme_moves"] = make_extreme_moves_figure(feature_panel, output_dir=output_dir)
    outputs["correlation_matrix"] = make_return_correlation_matrix_figure(
        feature_panel,
        output_dir=output_dir,
    )
    outputs["volatility_ranking"] = make_volatility_ranking_figure(
        summary_frame,
        output_dir=output_dir,
    )
    outputs["annualized_return_ranking"] = make_annualized_return_ranking_figure(
        summary_frame,
        output_dir=output_dir,
    )
    outputs["sharpe_ranking"] = make_sharpe_ranking_figure(summary_frame, output_dir=output_dir)
    outputs["sortino_ranking"] = make_sortino_ranking_figure(
        summary_frame,
        output_dir=output_dir,
    )
    outputs["distribution_vs_normal"] = make_distribution_vs_normal_figure(
        feature_panel,
        output_dir=output_dir,
    )
    outputs["tail_share_comparison"] = make_tail_share_comparison_figure(
        summary_frame,
        output_dir=output_dir,
    )
    if include_appendix:
        outputs["cross_sectional_dispersion"] = make_cross_sectional_dispersion_figure(
            feature_panel,
            output_dir=output_dir,
        )
        outputs["rolling_volatility_headline"] = make_rolling_volatility_headline_figure(
            feature_panel,
            output_dir=output_dir,
        )
        outputs["rolling_sharpe_headline"] = make_rolling_sharpe_headline_figure(
            feature_panel,
            output_dir=output_dir,
        )
        outputs["max_drawdown_ranking"] = make_max_drawdown_ranking_figure(
            feature_panel,
            output_dir=output_dir,
        )
        outputs["risk_return_scatter"] = make_risk_return_scatter_figure(
            summary_frame,
            output_dir=output_dir,
        )
        outputs["headline_dollar_volume"] = make_headline_dollar_volume_small_multiples(
            stage1_panel,
            output_dir=output_dir,
        )
        outputs["dollar_volume_concentration"] = make_daily_volume_concentration_figure(
            stage1_panel,
            output_dir=output_dir,
        )
        outputs["dollar_volume_ranking"] = make_dollar_volume_ranking_figure(
            stage1_panel,
            output_dir=output_dir,
        )
    return outputs
