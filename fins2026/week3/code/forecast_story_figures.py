# ruff: noqa
"""FT-style story figures for the Week 3 forecasting lecture."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fins2026.week3.code.beginner_forecasting import (
    DEFAULT_DATA_DIR,
    DEFAULT_TRAIN_END,
    DEFAULT_US_ARX_COLUMNS,
    DEFAULT_US_ARX_PLUS_COLUMNS,
    DEFAULT_US_DATA_DIR,
    DEFAULT_VARIANT_VALIDATION_PERIODS,
    align_backtests_to_common_dates,
    ar_one_step_forecast,
    arma_one_step_forecast,
    arx_one_step_forecast,
    beginner_model_one_step_forecasts,
    build_beginner_horse_race_bundle,
    build_us_beginner_horse_race_bundle,
    build_wide_backtest_forecast_table,
    ensure_beginner_source_tables,
    ensure_us_beginner_source_tables,
    equal_weight_ensemble_backtest,
    load_saved_beginner_macro_panel,
    metrics_table,
    naive_one_step_forecast,
    next_one_step_exogenous_features,
    resolve_repo_path,
    split_dates,
    suppress_expected_statsmodels_warnings,
)
from fintools.figures import (
    FT_COLORS,
    FigureContext,
    add_source_note,
    export_figure_bundle,
    export_word_figure,
    figure_style,
    validate_image_not_blank,
)

DEFAULT_STORY_FIGURES_DIR = Path("fins2026/week3/results/figures/beginner_forecast_story")
DEFAULT_STORY_TABLES_DIR = Path("fins2026/week3/results/tables/beginner_forecast_story")
DEFAULT_US_STORY_FIGURES_DIR = Path("fins2026/week3/results/figures/us_beginner_forecast_story")
DEFAULT_US_STORY_TABLES_DIR = Path("fins2026/week3/results/tables/us_beginner_forecast_story")
DEFAULT_RECENT_MONTHS = 24
MODEL_COLOR_SEQUENCE = [
    FT_COLORS["maroon"],
    FT_COLORS["blue"],
    FT_COLORS["teal"],
    FT_COLORS["orange"],
    FT_COLORS["purple"],
    FT_COLORS["gold"],
    FT_COLORS["green"],
    FT_COLORS["brown"],
    FT_COLORS["slate"],
    FT_COLORS["pink"],
]
OBSOLETE_STORY_STEMS = [
    "week3_beginner_story_cumulative_gain",
    "week3_us_beginner_story_cumulative_gain",
]


def _clean_series(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if not isinstance(clean.index, pd.DatetimeIndex):
        clean.index = pd.to_datetime(clean.index)
    return clean.sort_index()


def _format_month_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def _sample_label(index: pd.Index) -> str:
    dates = pd.to_datetime(index).dropna().sort_values()
    return f"{dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}"


def _finalize_layout(fig: plt.Figure, *, rect: tuple[float, float, float, float]) -> None:
    if hasattr(fig, "set_layout_engine"):
        fig.set_layout_engine(None)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The figure layout has changed to tight")
        fig.tight_layout(rect=rect)


def _series_color(label: str, *, winner_label: str, secondary_label: str | None = None) -> str:
    clean = str(label)
    if clean in {"Actual", "Observed level"}:
        return FT_COLORS["charcoal"]
    if clean == "Naive":
        return FT_COLORS["muted"]
    if clean == winner_label:
        return FT_COLORS["maroon"]
    if secondary_label and clean == secondary_label:
        return FT_COLORS["blue"]
    if clean.startswith("ARMAX"):
        return FT_COLORS["green"]
    if clean.startswith("ARX"):
        return FT_COLORS["teal"]
    if clean.startswith("ARMA"):
        return FT_COLORS["gold"]
    if clean.startswith("AR("):
        return FT_COLORS["slate"]
    if clean == "OLS":
        return FT_COLORS["purple"]
    if clean == "ENet":
        return FT_COLORS["orange"]
    index = sum(ord(character) for character in clean) % len(MODEL_COLOR_SEQUENCE)
    return MODEL_COLOR_SEQUENCE[index]


def _display_model_label(label: str) -> str:
    clean = str(label)
    replacements = {
        "ARMAX(1, 0) cash": "ARMAX cash",
        "ARMAX(1, 0) cash+commodity": "ARMAX cash+commodity",
        "ARX(1) cash": "ARX cash",
        "ARX(1) cash+commodity": "ARX cash+commodity",
        "ARX(1) fedfunds": "ARX fed funds",
        "ARX(1) fedfunds+spread": "ARX fed funds + spread",
        "Equal-weight ensemble": "Ensemble",
    }
    return replacements.get(clean, clean)


def _remove_obsolete_story_files(
    figures_dir: str | Path,
    tables_dir: str | Path,
    *,
    repo_root: Path | None = None,
) -> None:
    figures_root = resolve_repo_path(figures_dir, repo_root)
    tables_root = resolve_repo_path(tables_dir, repo_root)
    for stem in OBSOLETE_STORY_STEMS:
        for suffix in [".png", ".pdf", ".caption.md"]:
            path = figures_root / f"{stem}{suffix}"
            if path.exists():
                path.unlink()
        table_path = tables_root / f"{stem}.csv"
        if table_path.exists():
            table_path.unlink()


def emit_story_figure(
    fig: plt.Figure,
    output_dir: str | Path,
    stem: str,
    *,
    context: FigureContext,
    spec: str = "full_width",
    repo_root: Path | None = None,
) -> dict[str, Path]:
    """Export a story figure as Word-ready PNG, PDF, and caption sidecar."""

    output_root = resolve_repo_path(output_dir, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)
    base_paths = export_word_figure(fig, output_root, stem, context=context, spec=spec)
    base_paths.update(export_figure_bundle(fig, output_root, stem, formats=("pdf",)))
    image_issues = validate_image_not_blank(base_paths["png"])
    if image_issues:
        details = "; ".join(issue.message for issue in image_issues)
        plt.close(fig)
        raise RuntimeError(f"{stem} failed image validation: {details}")
    plt.close(fig)
    return {f"{stem}_{key}": path for key, path in base_paths.items()}


def build_level_change_story_figure(
    level: pd.Series,
    change: pd.Series,
    *,
    holdout_start: pd.Timestamp,
    title_prefix: str,
    target_label: str,
) -> plt.Figure:
    """Build an FT-style level-versus-change figure for the unit-root story."""

    level = _clean_series(level)
    change = _clean_series(change)
    with figure_style("word_a4", style="ft", ft_background=False):
        fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.8), sharex=True)
        axes[0].plot(level.index, level, color=FT_COLORS["charcoal"], linewidth=2.1)
        axes[0].axvspan(
            holdout_start,
            level.index.max(),
            color=FT_COLORS["grid"],
            alpha=0.65,
            zorder=0,
        )
        axes[0].set_title("Unemployment level (%)", loc="left", fontweight="bold")
        axes[0].set_ylabel("Level (%)")
        axes[0].grid(True, axis="y", alpha=0.9)
        axes[0].grid(False, axis="x")
        axes[0].text(
            0.01,
            0.08,
            "Visual clue: persistence in the level.",
            transform=axes[0].transAxes,
            fontsize=9.3,
            color=FT_COLORS["muted"],
        )

        axes[1].plot(change.index, change, color=FT_COLORS["blue"], linewidth=1.7)
        axes[1].axvspan(
            holdout_start,
            change.index.max(),
            color=FT_COLORS["grid"],
            alpha=0.65,
            zorder=0,
        )
        axes[1].axhline(0.0, color=FT_COLORS["axis"], linewidth=1.0, linestyle="--")
        axes[1].set_title("Change in unemployment (%)", loc="left", fontweight="bold")
        axes[1].set_ylabel("Change (%)")
        axes[1].set_xlabel("Date")
        axes[1].grid(True, axis="y", alpha=0.9)
        axes[1].grid(False, axis="x")
        axes[1].text(
            0.01,
            0.08,
            "This is the lecture target.",
            transform=axes[1].transAxes,
            fontsize=9.3,
            color=FT_COLORS["muted"],
        )

        for axis in axes:
            _format_month_axis(axis)
        add_source_note(
            fig,
            (
                "Out-of-sample evaluation window shaded | "
                f"Sample: {_sample_label(level.index)} | Week 3 fixture data"
            ),
            style="ft",
        )
        _finalize_layout(fig, rect=(0.02, 0.03, 1.0, 1.0))
        return fig


def build_model_scorecard_figure(
    metrics: pd.DataFrame,
    *,
    title: str,
    winner_label: str,
) -> plt.Figure:
    """Build a multi-metric model-evaluation scorecard."""

    ordered = metrics.sort_values(["target_rmse", "model"]).reset_index(drop=True)
    labels = ordered["model"].tolist()
    display_labels = [_display_model_label(label) for label in labels]
    y_positions = np.arange(len(ordered))
    colors = [
        _series_color(label, winner_label=winner_label) if label != "Naive" else FT_COLORS["muted"]
        for label in labels
    ]

    with figure_style("word_a4", style="ft", ft_background=False):
        fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.8), sharey=False)
        metric_specs = [
            ("target_rmse", "RMSE (%) ↓", "lower"),
            ("target_mae", "MAE (%) ↓", "lower"),
            ("target_mase", "MASE ↓", "lower"),
            ("target_oos_r2_vs_naive", "OOS $R^2$ vs naive ↑", "higher"),
        ]
        for index, (column, label, direction) in enumerate(metric_specs):
            ax = axes.flat[index]
            values = ordered[column].astype(float)
            ax.barh(y_positions, values, color=colors, alpha=0.95)
            ax.set_title(label, loc="left", fontweight="bold", fontsize=11, pad=10)
            ax.grid(True, axis="x", alpha=0.9)
            ax.grid(False, axis="y")
            ax.axvline(0.0, color=FT_COLORS["axis"], linewidth=1.0, linestyle="--", alpha=0.8)
            ax.set_yticks(y_positions)
            ax.set_yticklabels(display_labels)
            ax.tick_params(axis="y", labelleft=index in {0, 2})
            ax.invert_yaxis()
            if column == "target_oos_r2_vs_naive":
                upper = max(float(values.max()), 0.0)
                ax.set_xlim(left=min(-0.05, float(values.min()) - 0.02), right=upper + 0.08)
            else:
                upper = float(values.max())
                ax.set_xlim(left=0.0, right=upper * 1.18 if upper > 0 else 1.0)
        add_source_note(
            fig,
            (
                "Lower is better for RMSE, MAE, and MASE. Higher OOS $R^2$ means "
                "a larger forecast-error reduction versus naive."
            ),
            y=0.006,
            fontsize=7.8,
            style="ft",
        )
        fig.suptitle(title, x=0.05, ha="left", fontweight="bold", fontsize=16)
        _finalize_layout(fig, rect=(0.08, 0.10, 1.0, 0.94))
        fig.subplots_adjust(bottom=0.16, hspace=0.48, wspace=0.22)
        return fig


def build_holdout_backtest_story_figure(
    backtests: dict[str, pd.DataFrame],
    *,
    title: str,
    winner_label: str,
    secondary_label: str | None = None,
) -> plt.Figure:
    """Build a target-space backtest chart for the selected comparison models."""

    aligned, common_index = align_backtests_to_common_dates(backtests)
    if common_index.empty:
        raise ValueError("no common backtest dates are available")
    first = next(iter(aligned.values()))
    with figure_style("word_a4", style="ft", ft_background=False):
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        ax.plot(
            common_index,
            first["actual"],
            color=FT_COLORS["charcoal"],
            linewidth=2.1,
            label="Actual",
        )
        for label, frame in aligned.items():
            ax.plot(
                frame.index,
                frame["forecast"],
                linewidth=1.7,
                color=_series_color(
                    label,
                    winner_label=winner_label,
                    secondary_label=secondary_label,
                ),
                label=label,
            )
        ax.axhline(0.0, color=FT_COLORS["axis"], linewidth=1.0, linestyle="--")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel("Monthly change (%)")
        ax.set_xlabel("Date")
        ax.grid(True, axis="y", alpha=0.9)
        ax.grid(False, axis="x")
        _format_month_axis(ax)
        ax.legend(loc="lower right", frameon=False, ncol=1)
        add_source_note(
            fig,
            f"Out-of-sample only | Common evaluation dates: {len(common_index)} months",
            style="ft",
        )
        _finalize_layout(fig, rect=(0.0, 0.03, 1.0, 1.0))
        return fig


def absolute_error_frame(backtests: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return aligned absolute forecast errors for the comparison models."""

    aligned, common_index = align_backtests_to_common_dates(backtests)
    if common_index.empty:
        raise ValueError("no common backtest dates are available")
    output = pd.DataFrame(index=common_index)
    for label, frame in aligned.items():
        output[label] = frame["absolute_error"].astype(float)
    return output


def build_absolute_error_figure(
    error_frame: pd.DataFrame,
    *,
    title: str,
    winner_label: str,
    secondary_label: str | None = None,
) -> plt.Figure:
    """Build a standard out-of-sample absolute-error comparison figure."""

    with figure_style("word_a4", style="ft", ft_background=False):
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        for column in error_frame.columns:
            ax.plot(
                error_frame.index,
                error_frame[column],
                linewidth=1.9,
                color=_series_color(
                    column,
                    winner_label=winner_label,
                    secondary_label=secondary_label,
                ),
                label=column,
            )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel("Absolute error (%)")
        ax.set_xlabel("Date")
        ax.grid(True, axis="y", alpha=0.9)
        ax.grid(False, axis="x")
        _format_month_axis(ax)
        ax.legend(loc="upper right", frameon=False)
        add_source_note(
            fig,
            "Lower lines mean smaller misses in the out-of-sample evaluation window",
            style="ft",
        )
        _finalize_layout(fig, rect=(0.06, 0.03, 1.0, 1.0))
        return fig


def build_latest_forecast_story_figure(
    level_series: pd.Series,
    latest_forecasts: pd.DataFrame,
    *,
    title_prefix: str,
    winner_label: str,
    secondary_label: str | None = None,
    recent_months: int = DEFAULT_RECENT_MONTHS,
) -> plt.Figure:
    """Build a latest-forecast figure with implied level path and model spread."""

    level = _clean_series(level_series)
    recent_level = level.tail(recent_months)
    if recent_level.empty:
        raise ValueError("level series is empty")
    forecast_table = latest_forecasts.copy().sort_values("target_forecast").reset_index(drop=True)
    if forecast_table.empty:
        raise ValueError("latest forecasts table is empty")
    forecast_date = pd.to_datetime(forecast_table["forecast_date"]).max()
    last_date = recent_level.index[-1]
    last_level = float(recent_level.iloc[-1])
    winner_row = forecast_table.loc[forecast_table["model"] == winner_label]
    if winner_row.empty:
        raise ValueError(f"winner label missing from latest forecasts: {winner_label}")
    winner_target_forecast = float(winner_row["target_forecast"].iloc[0])
    winner_forecast_level = last_level + winner_target_forecast
    spread_labels = [_display_model_label(label) for label in forecast_table["model"]]
    x_positions = np.arange(len(forecast_table))

    with figure_style("word_a4", style="ft", ft_background=False):
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(7.2, 5.4),
            gridspec_kw={"height_ratios": [1.0, 1.15]},
        )
        axes[0].plot(
            recent_level.index,
            recent_level,
            color=FT_COLORS["charcoal"],
            linewidth=2.1,
        )
        axes[0].plot(
            [last_date, forecast_date],
            [last_level, winner_forecast_level],
            color=_series_color(
                winner_label,
                winner_label=winner_label,
                secondary_label=secondary_label,
            ),
            linewidth=2.1,
            linestyle="--",
        )
        axes[0].scatter(
            [forecast_date],
            [winner_forecast_level],
            color=_series_color(
                winner_label,
                winner_label=winner_label,
                secondary_label=secondary_label,
            ),
            s=34,
            zorder=4,
        )
        axes[0].annotate(
            "Winner",
            xy=(forecast_date, winner_forecast_level),
            xytext=(8, -12),
            textcoords="offset points",
            color=_series_color(
                winner_label,
                winner_label=winner_label,
                secondary_label=secondary_label,
            ),
            fontsize=9.2,
            ha="left",
            va="top",
        )
        axes[0].set_title(
            "Latest unemployment level and winner forecast",
            loc="left",
            fontweight="bold",
        )
        axes[0].set_ylabel("Level (%)")
        axes[0].grid(True, axis="y", alpha=0.9)
        axes[0].grid(False, axis="x")
        _format_month_axis(axes[0])
        axes[0].set_xlim(recent_level.index.min(), forecast_date + pd.Timedelta(days=20))

        bar_colors = [
            _series_color(
                str(label),
                winner_label=winner_label,
                secondary_label=secondary_label,
            )
            for label in forecast_table["model"]
        ]
        axes[1].axhline(0.0, color=FT_COLORS["axis"], linewidth=1.0, linestyle="--")
        axes[1].bar(
            x_positions,
            forecast_table["target_forecast"],
            color=bar_colors,
            alpha=0.92,
            width=0.68,
            zorder=3,
        )
        axes[1].set_xticks(x_positions)
        axes[1].set_xticklabels(spread_labels, rotation=28, ha="right")
        axes[1].set_title("Next-month forecast spread", loc="left", fontweight="bold")
        axes[1].set_ylabel("Forecast monthly change (%)")
        axes[1].grid(True, axis="y", alpha=0.9)
        axes[1].grid(False, axis="x")

        add_source_note(
            fig,
            "Top panel is for interpretation only | Model selection remains in target space",
            style="ft",
        )
        _finalize_layout(fig, rect=(0.05, 0.03, 1.0, 1.0))
        return fig


def australia_story_bundle(
    panel: pd.DataFrame,
    *,
    train_end: str | pd.Timestamp = DEFAULT_TRAIN_END,
    test_periods: int | None = None,
    variant_validation_periods: int = DEFAULT_VARIANT_VALIDATION_PERIODS,
) -> dict[str, object]:
    """Return the Australia story-pack inputs built from the beginner ladder."""

    bundle = build_beginner_horse_race_bundle(
        panel,
        train_end=train_end,
        test_periods=test_periods,
        variant_validation_periods=variant_validation_periods,
    )
    common_backtests = dict(bundle["common_backtests"])
    non_naive = {
        label: frame for label, frame in common_backtests.items() if label != "Naive"
    }
    ensemble_backtest = equal_weight_ensemble_backtest(
        non_naive,
        level_series=bundle["level_series"],
    )
    winner_label = str(bundle["winner_model"])
    comparison_backtests: dict[str, pd.DataFrame] = {
        "Naive": common_backtests["Naive"],
        winner_label: common_backtests[winner_label],
        "Equal-weight ensemble": ensemble_backtest,
    }
    comparison_metrics = metrics_table(
        comparison_backtests,
        target_series=bundle["target_series"],
        train_end=train_end,
        test_periods=test_periods,
        naive_label="Naive",
        align_common_dates=True,
    ).sort_values(["target_rmse", "model"]).reset_index(drop=True)
    latest_forecasts = beginner_model_one_step_forecasts(panel, bundle)
    ensemble_row = pd.DataFrame(
        {
            "forecast_date": [latest_forecasts["forecast_date"].iloc[0]],
            "model": ["Equal-weight ensemble"],
            "target_forecast": [
                float(
                    latest_forecasts.loc[
                        latest_forecasts["model"] != "Naive",
                        "target_forecast",
                    ].mean()
                )
            ],
        }
    )
    latest_forecasts = pd.concat([latest_forecasts, ensemble_row], ignore_index=True)
    latest_forecasts["forecast_level"] = (
        float(_clean_series(bundle["level_series"]).iloc[-1]) + latest_forecasts["target_forecast"]
    )
    absolute_errors = absolute_error_frame(comparison_backtests)
    comparison_backtest_wide = build_wide_backtest_forecast_table(comparison_backtests)
    return {
        **bundle,
        "winner_label": winner_label,
        "secondary_label": "Equal-weight ensemble",
        "comparison_backtests": comparison_backtests,
        "comparison_metrics": comparison_metrics,
        "latest_forecasts": latest_forecasts,
        "absolute_errors": absolute_errors,
        "comparison_backtest_wide": comparison_backtest_wide,
    }


def us_latest_forecasts(
    panel: pd.DataFrame,
    bundle: dict[str, object],
) -> pd.DataFrame:
    """Return the latest one-step U.S. model forecasts for the story pack."""

    unemployment_change = _clean_series(panel["unemployment_change_pp"])
    selected_lag = int(bundle["selected_lag"])
    selected_order = tuple(bundle["selected_order"])
    selected_arx_variant = str(bundle["selected_arx_variant"])

    fedfunds_future = next_one_step_exogenous_features(panel, DEFAULT_US_ARX_COLUMNS)
    fedfunds_spread_future = next_one_step_exogenous_features(panel, DEFAULT_US_ARX_PLUS_COLUMNS)

    fedfunds_train_exog = pd.DataFrame(bundle["fedfunds_only_exog"]).dropna()
    fedfunds_target = unemployment_change.reindex(fedfunds_train_exog.index).dropna()
    fedfunds_train_exog = fedfunds_train_exog.reindex(fedfunds_target.index)

    fedfunds_spread_train_exog = pd.DataFrame(bundle["fedfunds_spread_exog"]).dropna()
    fedfunds_spread_target = unemployment_change.reindex(fedfunds_spread_train_exog.index).dropna()
    fedfunds_spread_train_exog = fedfunds_spread_train_exog.reindex(fedfunds_spread_target.index)

    forecasts = {
        "Naive": naive_one_step_forecast(unemployment_change),
        f"AR({selected_lag})": ar_one_step_forecast(unemployment_change, lag=selected_lag),
        f"ARMA{selected_order}": arma_one_step_forecast(unemployment_change, order=selected_order),
        f"ARX({selected_lag}) {selected_arx_variant}": arx_one_step_forecast(
            fedfunds_target if selected_arx_variant == "fedfunds" else fedfunds_spread_target,
            fedfunds_train_exog
            if selected_arx_variant == "fedfunds"
            else fedfunds_spread_train_exog,
            fedfunds_future if selected_arx_variant == "fedfunds" else fedfunds_spread_future,
            lag=selected_lag,
        ),
    }
    forecast_date = unemployment_change.index[-1] + pd.offsets.MonthEnd(1)
    output = pd.DataFrame(
        {
            "forecast_date": [forecast_date] * len(forecasts),
            "model": list(forecasts.keys()),
            "target_forecast": [float(value) for value in forecasts.values()],
        }
    )
    output["forecast_level"] = (
        float(_clean_series(bundle["level_series"]).iloc[-1]) + output["target_forecast"]
    )
    return output


def us_story_bundle(
    panel: pd.DataFrame,
    *,
    train_end: str | pd.Timestamp = DEFAULT_TRAIN_END,
    test_periods: int | None = None,
    variant_validation_periods: int = DEFAULT_VARIANT_VALIDATION_PERIODS,
) -> dict[str, object]:
    """Return the U.S. story-pack inputs built from the mirror extension."""

    bundle = build_us_beginner_horse_race_bundle(
        panel,
        train_end=train_end,
        test_periods=test_periods,
        variant_validation_periods=variant_validation_periods,
    )
    winner_label = str(bundle["winner_model"])
    comparison_backtests: dict[str, pd.DataFrame] = {
        "Naive": bundle["common_backtests"]["Naive"],
        winner_label: bundle["common_backtests"][winner_label],
    }
    comparison_metrics = metrics_table(
        comparison_backtests,
        target_series=bundle["target_series"],
        train_end=train_end,
        test_periods=test_periods,
        naive_label="Naive",
        align_common_dates=True,
    ).sort_values(["target_rmse", "model"]).reset_index(drop=True)
    latest_forecasts = us_latest_forecasts(panel, bundle)
    absolute_errors = absolute_error_frame(comparison_backtests)
    comparison_backtest_wide = build_wide_backtest_forecast_table(comparison_backtests)
    return {
        **bundle,
        "winner_label": winner_label,
        "secondary_label": None,
        "comparison_backtests": comparison_backtests,
        "comparison_metrics": comparison_metrics,
        "latest_forecasts": latest_forecasts,
        "absolute_errors": absolute_errors,
        "comparison_backtest_wide": comparison_backtest_wide,
    }


def _build_story_with_suppressed_statsmodels_warnings(
    build_fn,
    panel: pd.DataFrame,
    *,
    train_end: str | pd.Timestamp,
    test_periods: int | None,
    variant_validation_periods: int,
) -> dict[str, object]:
    with suppress_expected_statsmodels_warnings():
        return build_fn(
            panel,
            train_end=train_end,
            test_periods=test_periods,
            variant_validation_periods=variant_validation_periods,
        )


def build_and_save_australia_story_pack(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    figures_dir: str | Path = DEFAULT_STORY_FIGURES_DIR,
    tables_dir: str | Path = DEFAULT_STORY_TABLES_DIR,
    train_end: str | pd.Timestamp = DEFAULT_TRAIN_END,
    test_periods: int | None = None,
    variant_validation_periods: int = DEFAULT_VARIANT_VALIDATION_PERIODS,
    repo_root: Path | None = None,
) -> dict[str, Path]:
    """Build and save the Australia forecast story pack."""

    ensure_beginner_source_tables(data_dir, repo_root=repo_root)
    panel = load_saved_beginner_macro_panel(data_dir, repo_root=repo_root)
    story = _build_story_with_suppressed_statsmodels_warnings(
        australia_story_bundle,
        panel,
        train_end=train_end,
        test_periods=test_periods,
        variant_validation_periods=variant_validation_periods,
    )
    level = _clean_series(panel["unemployment_rate"])
    change = _clean_series(panel["unemployment_change_pp"])
    _, holdout_start = split_dates(change, train_end=train_end, test_periods=test_periods)

    tables_output = resolve_repo_path(tables_dir, repo_root)
    tables_output.mkdir(parents=True, exist_ok=True)
    _remove_obsolete_story_files(figures_dir, tables_dir, repo_root=repo_root)
    table_paths = {
        "story_scorecard": tables_output / "week3_beginner_story_scorecard_metrics.csv",
        "story_comparison": tables_output / "week3_beginner_story_comparison_metrics.csv",
        "story_absolute_errors": tables_output / "week3_beginner_story_absolute_errors.csv",
        "story_latest_forecasts": tables_output / "week3_beginner_story_latest_forecasts.csv",
        "story_backtest_wide": tables_output / "week3_beginner_story_backtest_wide.csv",
    }
    story["metrics"].to_csv(table_paths["story_scorecard"], index=False)
    story["comparison_metrics"].to_csv(table_paths["story_comparison"], index=False)
    story["absolute_errors"].to_csv(table_paths["story_absolute_errors"])
    story["latest_forecasts"].to_csv(table_paths["story_latest_forecasts"], index=False)
    story["comparison_backtest_wide"].to_csv(table_paths["story_backtest_wide"])

    sample = _sample_label(level.index)
    figure_paths: dict[str, Path] = {}
    figure_paths.update(
        emit_story_figure(
            build_level_change_story_figure(
                level,
                change,
                holdout_start=holdout_start,
                title_prefix="Australia unemployment",
                target_label="Monthly change in unemployment",
            ),
            figures_dir,
            "week3_beginner_story_level_change",
            context=FigureContext(
                title="Australia unemployment rate: level versus monthly change",
                note=(
                    "The level is persistent and slow-moving, while the monthly change is the "
                    "cleaner forecasting target for the Week 3 lecture."
                ),
                source="Week 3 Australia fixture, release-lag-aware observable panel.",
                sample=sample,
                units="Top panel: unemployment rate (%). Bottom panel: monthly change (%).",
            ),
            spec="two_panel",
            repo_root=repo_root,
        )
    )
    figure_paths.update(
        emit_story_figure(
            build_model_scorecard_figure(
                story["metrics"],
                title="Model evaluation scorecard",
                winner_label=story["winner_label"],
            ),
            figures_dir,
            "week3_beginner_story_scorecard",
            context=FigureContext(
                title="Australia unemployment forecast race: model evaluation scorecard",
                note=(
                    "The four panels show the metrics used in the Week 3 horse race. "
                    "RMSE, MAE, and MASE are miss-size measures, so lower is better. "
                    "OOS R² is measured against naive, so higher is better."
                ),
                source="Week 3 beginner forecast horse race.",
                sample=_sample_label(pd.Index(story["common_index"])),
                units="Forecast errors are measured on the monthly unemployment-change target (%).",
            ),
            spec="full_width",
            repo_root=repo_root,
        )
    )
    figure_paths.update(
        emit_story_figure(
            build_holdout_backtest_story_figure(
                story["comparison_backtests"],
                title="Backtest forecast comparison",
                winner_label=story["winner_label"],
                secondary_label=story["secondary_label"],
            ),
            figures_dir,
            "week3_beginner_story_backtest",
            context=FigureContext(
                title=(
                    "Australia unemployment: out-of-sample backtest for the "
                    "main comparison models"
                ),
                note=(
                    "The chart compares actual monthly unemployment changes with the "
                    "naive benchmark, the horse-race winner, and the "
                    "equal-weight ensemble."
                ),
                source="Week 3 beginner forecast backtests.",
                sample=_sample_label(pd.Index(story["comparison_backtest_wide"].index)),
                units="Monthly unemployment change (%).",
            ),
            spec="full_width",
            repo_root=repo_root,
        )
    )
    figure_paths.update(
        emit_story_figure(
            build_absolute_error_figure(
                story["absolute_errors"],
                title="Out-of-sample absolute errors",
                winner_label=story["winner_label"],
                secondary_label=story["secondary_label"],
            ),
            figures_dir,
            "week3_beginner_story_absolute_errors",
            context=FigureContext(
                title=(
                    "Australia unemployment: absolute forecast errors in the "
                    "out-of-sample window"
                ),
                note=(
                    "This is a standard forecast-diagnostic chart: lower lines mean smaller misses "
                    "during the out-of-sample evaluation window."
                ),
                source="Week 3 beginner forecast backtests.",
                sample=_sample_label(story["absolute_errors"].index),
                units="Absolute forecast error in unemployment changes (%).",
            ),
            spec="full_width",
            repo_root=repo_root,
        )
    )
    figure_paths.update(
        emit_story_figure(
            build_latest_forecast_story_figure(
                story["level_series"],
                story["latest_forecasts"],
                title_prefix="Australia unemployment",
                winner_label=story["winner_label"],
                secondary_label=story["secondary_label"],
            ),
            figures_dir,
            "week3_beginner_story_latest_forecast",
            context=FigureContext(
                title="Australia unemployment: latest forecast and model spread",
                note=(
                    "The top panel maps the winner's target forecast back into "
                    "an implied next-month unemployment level for interpretation "
                    "only. The bottom panel shows the full model spread on the "
                    "target."
                ),
                source="Week 3 beginner forecast latest one-step outputs.",
                sample=sample,
                units=(
                    "Top panel: unemployment rate (%). Bottom panel: next-month "
                    "forecast of unemployment change (%)."
                ),
            ),
            spec="two_panel",
            repo_root=repo_root,
        )
    )

    return {**table_paths, **figure_paths}


def build_and_save_us_story_pack(
    *,
    data_dir: str | Path = DEFAULT_US_DATA_DIR,
    figures_dir: str | Path = DEFAULT_US_STORY_FIGURES_DIR,
    tables_dir: str | Path = DEFAULT_US_STORY_TABLES_DIR,
    train_end: str | pd.Timestamp = DEFAULT_TRAIN_END,
    test_periods: int | None = None,
    variant_validation_periods: int = DEFAULT_VARIANT_VALIDATION_PERIODS,
    repo_root: Path | None = None,
) -> dict[str, Path]:
    """Build and save the U.S. forecast story pack."""

    ensure_us_beginner_source_tables(data_dir, repo_root=repo_root)
    panel = load_saved_beginner_macro_panel(data_dir, repo_root=repo_root)
    story = _build_story_with_suppressed_statsmodels_warnings(
        us_story_bundle,
        panel,
        train_end=train_end,
        test_periods=test_periods,
        variant_validation_periods=variant_validation_periods,
    )
    level = _clean_series(panel["unemployment_rate"])
    change = _clean_series(panel["unemployment_change_pp"])
    _, holdout_start = split_dates(change, train_end=train_end, test_periods=test_periods)

    tables_output = resolve_repo_path(tables_dir, repo_root)
    tables_output.mkdir(parents=True, exist_ok=True)
    _remove_obsolete_story_files(figures_dir, tables_dir, repo_root=repo_root)
    table_paths = {
        "story_scorecard": tables_output / "week3_us_beginner_story_scorecard_metrics.csv",
        "story_comparison": tables_output / "week3_us_beginner_story_comparison_metrics.csv",
        "story_absolute_errors": tables_output / "week3_us_beginner_story_absolute_errors.csv",
        "story_latest_forecasts": tables_output / "week3_us_beginner_story_latest_forecasts.csv",
        "story_backtest_wide": tables_output / "week3_us_beginner_story_backtest_wide.csv",
    }
    story["metrics"].to_csv(table_paths["story_scorecard"], index=False)
    story["comparison_metrics"].to_csv(table_paths["story_comparison"], index=False)
    story["absolute_errors"].to_csv(table_paths["story_absolute_errors"])
    story["latest_forecasts"].to_csv(table_paths["story_latest_forecasts"], index=False)
    story["comparison_backtest_wide"].to_csv(table_paths["story_backtest_wide"])

    sample = _sample_label(level.index)
    figure_paths: dict[str, Path] = {}
    figure_paths.update(
        emit_story_figure(
            build_level_change_story_figure(
                level,
                change,
                holdout_start=holdout_start,
                title_prefix="U.S. unemployment",
                target_label="Monthly change in unemployment",
            ),
            figures_dir,
            "week3_us_beginner_story_level_change",
            context=FigureContext(
                title="U.S. unemployment rate: level versus monthly change",
                note=(
                    "The same visual logic carries over from Australia: the level is persistent, "
                    "while the monthly change is the cleaner forecasting target."
                ),
                source="Week 3 U.S. fixture, month-end macro panel.",
                sample=sample,
                units="Top panel: unemployment rate (%). Bottom panel: monthly change (%).",
            ),
            spec="two_panel",
            repo_root=repo_root,
        )
    )
    figure_paths.update(
        emit_story_figure(
            build_model_scorecard_figure(
                story["metrics"],
                title="Model evaluation scorecard",
                winner_label=story["winner_label"],
            ),
            figures_dir,
            "week3_us_beginner_story_scorecard",
            context=FigureContext(
                title="U.S. unemployment forecast race: model evaluation scorecard",
                note=(
                    "The U.S. extension uses the same four evaluation metrics as Australia. "
                    "RMSE, MAE, and MASE are miss-size measures, while OOS R² shows the gain "
                    "against naive."
                ),
                source="Week 3 U.S. beginner forecast horse race.",
                sample=_sample_label(pd.Index(story["common_index"])),
                units="Forecast errors are measured on the monthly unemployment-change target (%).",
            ),
            spec="full_width",
            repo_root=repo_root,
        )
    )
    figure_paths.update(
        emit_story_figure(
            build_holdout_backtest_story_figure(
                story["comparison_backtests"],
                title="Backtest forecast comparison",
                winner_label=story["winner_label"],
            ),
            figures_dir,
            "week3_us_beginner_story_backtest",
            context=FigureContext(
                title="U.S. unemployment: out-of-sample backtest for the main comparison models",
                note=(
                    "The chart compares actual monthly unemployment changes with "
                    "the naive benchmark and the U.S. horse-race winner."
                ),
                source="Week 3 U.S. beginner forecast backtests.",
                sample=_sample_label(pd.Index(story["comparison_backtest_wide"].index)),
                units="Monthly unemployment change (%).",
            ),
            spec="full_width",
            repo_root=repo_root,
        )
    )
    figure_paths.update(
        emit_story_figure(
            build_absolute_error_figure(
                story["absolute_errors"],
                title="Out-of-sample absolute errors",
                winner_label=story["winner_label"],
            ),
            figures_dir,
            "week3_us_beginner_story_absolute_errors",
            context=FigureContext(
                title="U.S. unemployment: absolute forecast errors in the out-of-sample window",
                note=(
                    "This is a standard forecast-diagnostic chart: lower lines mean smaller misses "
                    "during the out-of-sample evaluation window."
                ),
                source="Week 3 U.S. beginner forecast backtests.",
                sample=_sample_label(story["absolute_errors"].index),
                units="Absolute forecast error in unemployment changes (%).",
            ),
            spec="full_width",
            repo_root=repo_root,
        )
    )
    figure_paths.update(
        emit_story_figure(
            build_latest_forecast_story_figure(
                story["level_series"],
                story["latest_forecasts"],
                title_prefix="U.S. unemployment",
                winner_label=story["winner_label"],
            ),
            figures_dir,
            "week3_us_beginner_story_latest_forecast",
            context=FigureContext(
                title="U.S. unemployment: latest forecast and model spread",
                note=(
                    "The top panel maps the winner's target forecast back into the next implied "
                    "unemployment-rate level. The bottom panel shows the current model spread on "
                    "the target."
                ),
                source="Week 3 U.S. beginner forecast latest one-step outputs.",
                sample=sample,
                units=(
                    "Top panel: unemployment rate (%). Bottom panel: forecast "
                    "monthly change (%)."
                ),
            ),
            spec="two_panel",
            repo_root=repo_root,
        )
    )

    return {**table_paths, **figure_paths}


