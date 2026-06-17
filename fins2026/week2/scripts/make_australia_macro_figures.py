# ruff: noqa
"""Create the Week 2 Australia macro figure pack."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_DATA = Path("fins2026/week2/results/data/australia_macro_stage1_long.csv")
DEFAULT_FIXTURE = Path("fins2026/week2/data/australia_macro_stage1_long.csv")
DEFAULT_OUTPUT = Path("fins2026/week2/results/figures/australia_macro_story")
DOCX_NAME = "week2_australia_macro_story_ft.docx"


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repo root from this script location."""

    current = (start or Path(__file__)).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "tools" / "workflow_lib.py"
        ).is_file():
            return candidate
    raise RuntimeError("Could not find the fins-agent repo root.")


REPO_ROOT = find_repo_root()
SCRIPT_DIR = Path(__file__).resolve().parent
for path in [REPO_ROOT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fins2026.week2.code.australia_macro_panel import (  # noqa: E402
    build_beveridge_frame,
    build_common_endpoint_snapshot,
    build_external_relationship_frame,
    build_feature_panel,
    build_observable_panel,
    build_okun_frame,
    build_reference_panel,
    build_release_lag_frame,
    build_wage_phillips_frame,
    load_fixture_long_table,
)
from fins2026.week2.code.australia_macro_specs import (  # noqa: E402
    AUSTRALIA_MACRO_SPECS,
    CLASSROOM_INFORMATION_SET_MONTH_END,
    CLASSROOM_REFERENCE_ENDPOINT,
    CORE_SMALL_MULTIPLE_SERIES,
    LABOUR_TIGHTNESS_SERIES,
    TIGHTENING_EPISODE_SERIES,
)
from fintools.figures import (  # noqa: E402
    FigureContext,
    WordFigureEntry,
    dumbbell_plot,
    export_figure_bundle,
    export_word_figure,
    insert_figures_docx,
    scatter_plot,
    small_multiples,
    time_series_plot,
    validate_docx_images_fit_page,
    validate_equal_subplot_widths,
    validate_image_not_blank,
)
from fintools.figures.plots import (  # noqa: E402
    _apply_horizontal_grid,
    _format_date_axis,
    _line_color,
)
from fintools.figures.theme import figure_style  # noqa: E402


def resolve_repo_path(path: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve a repo-relative or absolute path."""

    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    return output_path


def load_stage_bundle(
    data_path: str | Path = DEFAULT_DATA,
    *,
    use_fixture_if_missing: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load the Stage 1 Australia data and rebuild the Stage 2 panels."""

    resolved = resolve_repo_path(data_path)
    if resolved.exists():
        stage1 = load_fixture_long_table(resolved)
    elif use_fixture_if_missing:
        stage1 = load_fixture_long_table(resolve_repo_path(DEFAULT_FIXTURE))
    else:
        raise FileNotFoundError(
            f"Australia macro data not found: {resolved}. Run pull_australia_macro_data.py first."
        )
    reference_panel = build_reference_panel(stage1)
    observable_panel = build_observable_panel(stage1)
    feature_panel = build_feature_panel(reference_panel)
    return {
        "stage1": stage1,
        "reference_panel": reference_panel,
        "observable_panel": observable_panel,
        "feature_panel": feature_panel,
    }


def sample_label_from_index(index: pd.Index) -> str:
    """Return a caption-ready sample period from a datetime index."""

    dates = pd.to_datetime(index).dropna()
    return f"{dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}"


def style_small_multiple_layout(
    fig,
    axes,
    data: pd.DataFrame,
    columns: list[str],
    *,
    ylabel_fontsize: float = 7.4,
    tick_fontsize: float = 8.0,
    x_tick_fontsize: float | None = None,
    date_tick_max: int = 6,
    title_fontsize: float = 7.5,
    suptitle_fontsize: float = 11.0,
    wspace: float = 0.44,
    hspace: float = 0.52,
    left: float = 0.14,
    right: float = 0.97,
    top: float = 0.87,
    bottom: float = 0.08,
) -> None:
    """Apply Word-sized layout settings to Week 2 small multiples."""

    effective_x_tick_fontsize = tick_fontsize if x_tick_fontsize is None else x_tick_fontsize
    axes_array = list(axes[: len(columns)])
    for ax, column in zip(axes_array, columns, strict=False):
        ax.yaxis.set_label_position("left")
        ax.yaxis.tick_left()
        ax.spines["right"].set_visible(False)
        ax.yaxis.label.set_fontsize(ylabel_fontsize)
        ax.tick_params(axis="y", labelsize=tick_fontsize)
        ax.tick_params(axis="x", labelsize=effective_x_tick_fontsize)
        ax.title.set_fontsize(title_fontsize)
        series = data[column].dropna()
        if not series.empty:
            _format_date_axis(
                ax,
                date_start=series.index.min(),
                date_end=series.index.max(),
                max_ticks=date_tick_max,
            )
    if hasattr(fig, "set_layout_engine"):
        fig.set_layout_engine(None)
    fig.subplots_adjust(
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        wspace=wspace,
        hspace=hspace,
    )
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_fontsize(suptitle_fontsize)


def emit_figure(
    fig,
    output_dir: Path,
    stem: str,
    context: FigureContext,
    entries: list[WordFigureEntry],
    *,
    spec: str = "full_width",
) -> dict[str, Path]:
    """Export one figure and register it for the Word proof pack."""

    bundle = export_word_figure(fig, output_dir, stem, context=context, spec=spec)
    bundle.update(export_figure_bundle(fig, output_dir, stem, formats=("pdf",)))
    image_issues = validate_image_not_blank(bundle["png"])
    if image_issues:
        details = "; ".join(issue.message for issue in image_issues)
        raise RuntimeError(f"{stem} failed image validation: {details}")
    subplot_issues = validate_equal_subplot_widths(fig)
    if subplot_issues:
        details = "; ".join(issue.message for issue in subplot_issues)
        raise RuntimeError(f"{stem} failed subplot validation: {details}")
    entries.append(WordFigureEntry(bundle["png"], context=context, spec=spec))
    plt.close(fig)
    return bundle


def display_series_names(series_ids: list[str]) -> list[str]:
    """Map ordered series ids to display names."""

    return [AUSTRALIA_MACRO_SPECS[series_id].display_name for series_id in series_ids]


def series_units_label(series_id: str) -> str:
    """Return a reader-facing y-axis label for one display series."""

    labels = {
        "GLFSURSA": "Unemployment rate (%)",
        "GCPIAGYP": "Year-ended headline\nCPI inflation (%)",
        "GCPIOCPMTMYP": "Year-ended trimmed\nmean inflation (%)",
        "GWPIYP": "Year-ended\nWPI growth (%)",
        "GGDPCVGDPY": "Year-ended real\nGDP growth (%)",
        "FIRMMCRT": "Cash rate target (%)",
        "FCMYGBAG10": "10Y government bond\nyield (%)",
        "FXRTWI": "Trade-weighted\nindex level",
        "GLFSPRSA": "Participation rate (%)",
        "GLFSEPTPOP": "Employment-to-population\nratio (%)",
        "GLFOSVT": "Job vacancies ('000,\npoint-in-time quarterly)",
    }
    return labels[series_id]


def compact_units_label(series_name: str) -> str:
    """Return a short unit label for the endpoint comparison mini-panels."""

    mapping = {
        "Unemployment rate": "%",
        "Headline CPI inflation": "%",
        "Wage Price Index growth": "%",
        "Real GDP growth": "%",
        "Cash rate target": "%",
        "10Y government bond yield": "%",
        "Trade-weighted index": "Index",
        "Commodity price index (A$)": "Index",
    }
    return mapping.get(series_name, "")


def value_label(series_name: str, value: float) -> str:
    """Format endpoint labels with appropriate units."""

    unit = compact_units_label(series_name)
    if unit == "%":
        return f"{value:.1f}%"
    return f"{value:.1f}"


def build_endpoint_comparison_figure(snapshot: pd.DataFrame) -> tuple[plt.Figure, list[plt.Axes]]:
    """Build a mixed-unit December 2000 versus December 2025 comparison grid."""

    with figure_style("word_a4", style="ft", ft_background=False):
        fig, axes = plt.subplots(
            4,
            2,
            figsize=(6.27, 7.10),
            constrained_layout=False,
            layout=None,
        )
        axes_array = list(axes.ravel())
        for ax, (_, row) in zip(axes_array, snapshot.iterrows(), strict=False):
            start_value = float(row["2000-12"])
            end_value = float(row["2025-12"])
            highlight = _line_color("ft", "primary")
            ax.plot(
                [0, 1],
                [start_value, end_value],
                color=highlight,
                linewidth=1.7,
                alpha=0.90,
            )
            ax.scatter([0, 1], [start_value, end_value], color=highlight, s=34, zorder=3)
            _apply_horizontal_grid(ax, style="ft")
            ax.set_xticks([0, 1], ["2000-12", "2025-12"])
            ax.set_title(str(row["Series"]), loc="left", fontsize=7.8)
            ax.set_ylabel(compact_units_label(str(row["Series"])), fontsize=7.8)
            ax.tick_params(axis="x", labelsize=7.8)
            ax.tick_params(axis="y", labelsize=7.8)
            y_padding = max(
                abs(end_value - start_value) * 0.12,
                max(abs(start_value), 1.0) * 0.05,
            )
            ax.set_ylim(
                min(start_value, end_value) - y_padding,
                max(start_value, end_value) + y_padding,
            )
            ax.annotate(
                value_label(str(row["Series"]), start_value),
                (0, start_value),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=7.5,
                color=_line_color("ft", "text"),
            )
            ax.annotate(
                value_label(str(row["Series"]), end_value),
                (1, end_value),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=7.5,
                color=_line_color("ft", "text"),
            )
        for ax in axes_array[len(snapshot) :]:
            ax.set_visible(False)
        if hasattr(fig, "set_constrained_layout"):
            fig.set_constrained_layout(False)
        if hasattr(fig, "set_tight_layout"):
            fig.set_tight_layout(False)
        if hasattr(fig, "set_layout_engine"):
            fig.set_layout_engine(None)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="This figure was using a layout engine that is incompatible.*",
                category=UserWarning,
            )
            fig.subplots_adjust(
                left=0.10,
                right=0.97,
                top=0.90,
                bottom=0.08,
                wspace=0.34,
                hspace=0.58,
            )
        fig.suptitle(
            "December 2000 Versus December 2025",
            x=0.01,
            ha="left",
            fontsize=11.0,
            weight="bold",
        )
        return fig, axes_array


def build_australia_macro_figures(
    data_path: str | Path = DEFAULT_DATA,
    output: str | Path = DEFAULT_OUTPUT,
    *,
    use_fixture_if_missing: bool = True,
) -> Path:
    """Build the curated Week 2 Australia macro story pack."""

    bundle = load_stage_bundle(data_path, use_fixture_if_missing=use_fixture_if_missing)
    reference_panel = bundle["reference_panel"]
    feature_panel = bundle["feature_panel"]
    output_dir = resolve_repo_path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries: list[WordFigureEntry] = []
    reference_sample = sample_label_from_index(reference_panel.index)
    lag_frame = build_release_lag_frame()
    fig, lag_ax = dumbbell_plot(
        lag_frame,
        "Series",
        "Reference month-end",
        "First observable month-end",
        title="Reference Date Versus Observable Month-End",
        xlabel="Month-end lag",
        ylabel="Core classroom series",
        start_label="Reference date",
        end_label="Observable month-end",
        sort=False,
        profile="word_a4",
        style="ft",
    )
    lag_ax.set_title("Reference Date Versus Observable Month-End", loc="left", fontsize=10.8)
    emit_figure(
        fig,
        output_dir,
        "week2_australia_macro_story_release_lags",
        FigureContext(
            title="Week 2 Australia Common-Endpoint Observability Map",
            note=(
                "The classroom reference endpoint is December 2025, but the shared information "
                "set only becomes complete by March 2026 because GDP is the slowest major release. "
                "This is a Stage 1 authenticity check before any figure merges the series."
            ),
            source="Reserve Bank of Australia statistical tables, classroom lag mapping.",
            sample=(
                "Reference endpoint 2025-12-31; common information set "
                "month-end 2026-03-31"
            ),
            units=(
                "Month-end lag from the reference period to the first "
                "classroom-observable month-end"
            ),
        ),
        entries,
    )

    core_columns = display_series_names(CORE_SMALL_MULTIPLE_SERIES)
    fig, axes = small_multiples(
        reference_panel,
        core_columns,
        title="Australia: 2000 To December 2025",
        ylabel=[series_units_label(series_id) for series_id in CORE_SMALL_MULTIPLE_SERIES],
        profile="word_a4",
        style="ft",
        title_fontsize=7.4,
    )
    style_small_multiple_layout(
        fig,
        axes,
        reference_panel,
        core_columns,
        ylabel_fontsize=7.4,
        tick_fontsize=7.8,
        x_tick_fontsize=7.0,
        date_tick_max=6,
        title_fontsize=7.2,
        suptitle_fontsize=10.8,
        wspace=0.48,
        hspace=0.56,
        top=0.88,
        left=0.15,
    )
    emit_figure(
        fig,
        output_dir,
        "week2_australia_macro_story_core_small_multiples",
        FigureContext(
            title="Week 2 Australia Long-Run Macro Small Multiples",
            note=(
                "This is the Station 2 storytelling panel. It keeps each series in the unit that "
                "answers the economic question cleanly rather than forcing rates, "
                "inflation, output, and exchange-rate levels onto one shared axis."
            ),
            source="Reserve Bank of Australia statistical tables.",
            sample=reference_sample,
            units="Mixed units by panel; see each y-axis label",
        ),
        entries,
        spec="portrait_full",
    )

    snapshot = build_common_endpoint_snapshot(reference_panel)
    fig, _ = build_endpoint_comparison_figure(snapshot)
    emit_figure(
        fig,
        output_dir,
        "week2_australia_macro_story_dec2000_vs_dec2025",
        FigureContext(
            title="Week 2 Australia December 2000 Versus December 2025 Comparison",
            note=(
                "Mixed-frequency macro series should not be compared at arbitrary latest dates. "
                "This grid uses the common December 2000 and December 2025 reference "
                "points so the comparison stays aligned across labour, inflation, wages, "
                "output, policy, and external series."
            ),
            source="Reserve Bank of Australia statistical tables.",
            sample="2000-12-31 and 2025-12-31 common reference dates",
            units="Panel-specific units",
        ),
        entries,
        spec="portrait_full",
    )

    episode_columns = display_series_names(TIGHTENING_EPISODE_SERIES)
    episode_panel = reference_panel.loc["2019-01-31":"2025-12-31", episode_columns].copy()
    fig, axes = small_multiples(
        episode_panel,
        episode_columns,
        title="Pandemic To Tightening In Australia",
        ylabel=[series_units_label(series_id) for series_id in TIGHTENING_EPISODE_SERIES],
        profile="word_a4",
        style="ft",
        title_fontsize=7.4,
    )
    style_small_multiple_layout(
        fig,
        axes,
        episode_panel,
        episode_columns,
        ylabel_fontsize=7.6,
        tick_fontsize=7.8,
        x_tick_fontsize=7.0,
        date_tick_max=6,
        title_fontsize=7.3,
        suptitle_fontsize=10.8,
        wspace=0.42,
        hspace=0.50,
        left=0.14,
    )
    emit_figure(
        fig,
        output_dir,
        "week2_australia_macro_story_tightening_episode",
        FigureContext(
            title="Week 2 Australia Pandemic-To-Tightening Episode",
            note=(
                "This matched panel asks students to narrate the recent Australian macro cycle "
                "without collapsing it into one unreadable connected scatter. The reference window "
                "starts before the pandemic and ends at the frozen December 2025 common endpoint."
            ),
            source="Reserve Bank of Australia statistical tables.",
            sample="2019-01-31 to 2025-12-31",
            units="Percent",
        ),
        entries,
    )

    labour_columns = display_series_names(LABOUR_TIGHTNESS_SERIES)
    fig, axes = small_multiples(
        reference_panel,
        labour_columns,
        title="Labour Tightness Needs Mixed Frequencies",
        ylabel=[series_units_label(series_id) for series_id in LABOUR_TIGHTNESS_SERIES],
        profile="word_a4",
        style="ft",
        title_fontsize=7.4,
    )
    style_small_multiple_layout(
        fig,
        axes,
        reference_panel,
        labour_columns,
        ylabel_fontsize=7.3,
        tick_fontsize=7.8,
        x_tick_fontsize=7.0,
        date_tick_max=6,
        title_fontsize=7.2,
        suptitle_fontsize=10.8,
        wspace=0.46,
        hspace=0.52,
        left=0.15,
    )
    emit_figure(
        fig,
        output_dir,
        "week2_australia_macro_story_labour_tightness",
        FigureContext(
            title="Week 2 Australia Labour Tightness Panel",
            note=(
                "Unemployment, participation, and the employment-to-population ratio are monthly, "
                "but job vacancies are a point-in-time quarterly survey. This is exactly "
                "the kind of Stage 1 frequency map students need to check before merging "
                "labour indicators."
            ),
            source="Reserve Bank of Australia statistical tables.",
            sample=reference_sample,
            units="Mixed labour-market units by panel",
        ),
        entries,
    )

    wage_phillips = build_wage_phillips_frame(feature_panel)
    fig, _ = scatter_plot(
        wage_phillips,
        "Unemployment rate",
        "Wage Price Index growth",
        fit=True,
        annotate=True,
        label_outliers=3,
        stats_location="upper right",
        title="Wage Phillips Curve",
        xlabel="Unemployment rate (%)",
        ylabel="Year-ended WPI growth (%)",
        profile="word_a4",
        style="ft",
    )
    emit_figure(
        fig,
        output_dir,
        "week2_australia_macro_story_wage_phillips_curve",
        FigureContext(
            title="Week 2 Australia Wage Phillips Curve",
            note=(
                "The pairing is timing-aware: quarterly WPI observations are matched with the "
                "same-quarter unemployment rate rather than with a falsely synchronized "
                "monthly wage series."
            ),
            source="Reserve Bank of Australia statistical tables.",
            sample=sample_label_from_index(wage_phillips.index),
            units="Percent",
        ),
        entries,
    )

    okun_frame = build_okun_frame(feature_panel)
    fig, _ = scatter_plot(
        okun_frame,
        "Real GDP quarterly log growth (%)",
        "Change in quarterly average unemployment rate (pp)",
        fit=True,
        annotate=True,
        label_outliers=3,
        stats_location="upper right",
        title="Okun's Law",
        xlabel="Quarterly real GDP log growth (%)",
        ylabel="Quarterly average\nunemployment change (%)",
        profile="word_a4",
        style="ft",
    )
    emit_figure(
        fig,
        output_dir,
        "week2_australia_macro_story_okuns_law",
        FigureContext(
            title="Week 2 Australia Okun's Law Scatter",
            note=(
                "This figure intentionally mixes a quarterly GDP transform with a "
                "quarterly-average labour-market transform built from monthly "
                "unemployment data. Students should see the aggregation step rather "
                "than treating the merge as automatic."
            ),
            source="Reserve Bank of Australia statistical tables.",
            sample=sample_label_from_index(okun_frame.index),
            units="Quarterly real GDP log growth (%) and unemployment-rate change (%)",
        ),
        entries,
    )

    beveridge = build_beveridge_frame(reference_panel)
    fig, _ = scatter_plot(
        beveridge,
        "Unemployment rate",
        "Vacancies to labour force ratio",
        fit=True,
        annotate=True,
        label_outliers=3,
        stats_location="upper right",
        title="Beveridge Curve",
        xlabel="Unemployment rate (%)",
        ylabel="Vacancies to labour force ratio (%)",
        profile="word_a4",
        style="ft",
    )
    emit_figure(
        fig,
        output_dir,
        "week2_australia_macro_story_beveridge_curve",
        FigureContext(
            title="Week 2 Australia Beveridge Curve",
            note=(
                "Vacancies are not a monthly flow series here. The figure pairs the point-in-time "
                "vacancies-to-labour-force ratio with the same reference-date "
                "unemployment rate so the relationship stays honest about sampling "
                "frequency."
            ),
            source="Reserve Bank of Australia statistical tables.",
            sample=sample_label_from_index(beveridge.index),
            units="Percent",
        ),
        entries,
    )

    external = build_external_relationship_frame(feature_panel)
    fig, _ = scatter_plot(
        external,
        "Commodity price index (A$) log change (%)",
        "Trade-weighted index log change (%)",
        fit=True,
        annotate=True,
        label_outliers=3,
        stats_location="upper right",
        title="Commodity Prices And The TWI",
        xlabel="Monthly commodity-price log change (%)",
        ylabel="Monthly TWI log change (%)",
        profile="word_a4",
        style="ft",
    )
    emit_figure(
        fig,
        output_dir,
        "week2_australia_macro_story_commodity_twi_scatter",
        FigureContext(
            title="Week 2 Australia Commodity Prices Versus The Trade-Weighted Index",
            note=(
                "This adds an Australia-specific external-sector relationship to the lecture. "
                "Both series are monthly, so the pairing is simpler than the labour "
                "or GDP relationships."
            ),
            source="Reserve Bank of Australia statistical tables.",
            sample=sample_label_from_index(external.index),
            units="Monthly log changes in percent",
        ),
        entries,
    )

    inflation_transition = reference_panel[
        ["Headline CPI inflation", "Trimmed mean inflation"]
    ].dropna(how="all")
    fig, _ = time_series_plot(
        inflation_transition,
        ["Headline CPI inflation", "Trimmed mean inflation"],
        title="Quarterly Inflation Measures Over Time",
        ylabel="Year-ended inflation (%)",
        legend=True,
        profile="word_a4",
        style="ft",
        direct_labels=False,
    )
    emit_figure(
        fig,
        output_dir,
        "week2_australia_macro_story_inflation_transition",
        FigureContext(
            title="Week 2 Australia Quarterly Inflation Measures Over Time",
            note=(
                "The monthly CPI history is too short for the core lecture pack, so this "
                "figure keeps the quarterly headline and trimmed-mean inflation series "
                "over the full classroom reference sample."
            ),
            source="Reserve Bank of Australia statistical tables.",
            sample=sample_label_from_index(inflation_transition.index),
            units="Year-ended inflation (%)",
        ),
        entries,
    )

    docx_path = insert_figures_docx(
        entries,
        output_dir / DOCX_NAME,
        title="Week 2 Australia Macro Story Figure Pack",
    )
    docx_issues = validate_docx_images_fit_page(docx_path)
    if docx_issues:
        details = "; ".join(issue.message for issue in docx_issues)
        raise RuntimeError(f"Word proof pack failed validation: {details}")
    return docx_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Build the Week 2 Australia macro figures.")
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
        help="Repo-relative or absolute Stage 1 long-table CSV path.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Repo-relative or absolute output folder for the Australia figure pack.",
    )
    parser.add_argument(
        "--no-fixture-fallback",
        action="store_true",
        help="Do not fall back to the committed Stage 1 fixture if results/data is missing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the Week 2 Australia figure build."""

    args = parse_args(argv)
    docx_path = build_australia_macro_figures(
        args.data,
        args.output,
        use_fixture_if_missing=not args.no_fixture_fallback,
    )
    print(f"Wrote Australia figure pack: {docx_path}")
    print(
        "Frozen timing contract: "
        f"reference endpoint {CLASSROOM_REFERENCE_ENDPOINT:%Y-%m-%d}; "
        f"information set month-end {CLASSROOM_INFORMATION_SET_MONTH_END:%Y-%m-%d}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
