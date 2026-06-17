# ruff: noqa
"""Create FT-style figures from the Week 2 FRED market-and-macro panel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.transforms as mtransforms

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_DATA = Path("fins2026/week2/results/data/fred_market_macro.csv")
DEFAULT_OUTPUT = Path("fins2026/week2/results/figures/market_macro_story")
DOCX_NAME = "week2_market_macro_story_ft.docx"


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

from pull_fred_market_data import (  # noqa: E402
    FRED_SOURCE,
    clean_fred_market_data,
    load_fixture_data,
)

from fins2026.week2.code.market_panel import (  # noqa: E402
    DAILY_MARKET_COLUMNS,
    MONTHLY_MACRO_COLUMNS,
    build_month_end_panel,
)
from fintools.figures import (  # noqa: E402
    FigureContext,
    WordFigureEntry,
    area_balance_plot,
    create_figure_suite,
    export_figure_bundle,
    export_word_figure,
    indexed_time_series_plot,
    insert_figures_docx,
    scatter_plot,
    small_multiples,
    time_series_plot,
    validate_docx_images_fit_page,
    validate_equal_subplot_widths,
    validate_image_not_blank,
)


def resolve_repo_path(path: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve a repo-relative or absolute path."""

    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    return output_path


def load_fred_market_frame(
    data_path: str | Path = DEFAULT_DATA,
    *,
    use_fixture_if_missing: bool = True,
) -> pd.DataFrame:
    """Load the live exercise data, falling back to fixtures when requested."""

    resolved = resolve_repo_path(data_path)
    if resolved.exists():
        return clean_fred_market_data(pd.read_csv(resolved))
    if use_fixture_if_missing:
        return load_fixture_data()
    raise FileNotFoundError(
        f"FRED exercise data not found: {resolved}. Run pull_fred_market_data.py first."
    )


def sample_label(frame: pd.DataFrame) -> str:
    """Return a caption-ready sample period."""

    dates = pd.to_datetime(frame["date"]).dropna()
    return f"{dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}"


def style_small_multiple_layout(
    fig,
    axes,
    columns: list[str],
    *,
    right_column_labels: bool = True,
    ylabel_fontsize: float = 8.6,
    tick_fontsize: float = 8.4,
    suptitle_fontsize: float = 12.0,
    wspace: float = 0.44,
    hspace: float = 0.50,
    left: float = 0.10,
    right: float = 0.96,
    top: float = 0.90,
    bottom: float = 0.09,
) -> None:
    """Adjust small-multiple spacing and label placement for Word exports."""

    axes_array = list(axes[: len(columns)])
    for index, ax in enumerate(axes_array):
        ax.yaxis.label.set_fontsize(ylabel_fontsize)
        ax.tick_params(axis="y", labelsize=tick_fontsize)
        ax.tick_params(axis="x", labelsize=tick_fontsize)
        if right_column_labels and index % 2 == 1:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.spines["right"].set_visible(True)
        else:
            ax.yaxis.set_label_position("left")
            ax.yaxis.tick_left()
            ax.spines["right"].set_visible(False)
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


def wealth_ticks(ymin: float, ymax: float) -> list[float]:
    """Return compact dollar ticks for a log-scale wealth chart."""

    candidates = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0]
    return [value for value in candidates if ymin <= value <= ymax]


def add_endpoint_wealth_label(ax, *, label: str, wealth: float) -> None:
    """Add the FT-style right-edge endpoint label for a wealth path."""

    color = ax.get_lines()[0].get_color()
    blend = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(
        1.02,
        wealth,
        f"{label}, ${wealth:.2f}",
        transform=blend,
        fontsize=8.0,
        color=color,
        ha="left",
        va="center",
        clip_on=False,
    )


def month_end_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the shared month-end market-and-macro panel for Week 2 figures."""

    daily = frame.set_index("date").sort_index()
    panel = build_month_end_panel(
        daily[DAILY_MARKET_COLUMNS],
        daily[MONTHLY_MACRO_COLUMNS],
    )
    return panel.reset_index()


def emit_figure(
    fig,
    output_dir: Path,
    stem: str,
    context: FigureContext,
    entries: list[WordFigureEntry],
) -> dict[str, Path]:
    """Export one figure and register it for the Word proof pack."""

    bundle = export_word_figure(fig, output_dir, stem, context=context)
    bundle.update(export_figure_bundle(fig, output_dir, stem, formats=("pdf",)))
    image_issues = validate_image_not_blank(bundle["png"])
    if image_issues:
        details = "; ".join(issue.message for issue in image_issues)
        raise RuntimeError(f"{stem} failed image validation: {details}")
    subplot_issues = validate_equal_subplot_widths(fig)
    if subplot_issues:
        details = "; ".join(issue.message for issue in subplot_issues)
        raise RuntimeError(f"{stem} failed subplot validation: {details}")
    entries.append(WordFigureEntry(bundle["png"], context=context))
    plt.close(fig)
    return bundle


def build_auto_suite(frame: pd.DataFrame, output_dir: Path, source: str) -> Path | None:
    """Run the dataframe-to-figure-suite exercise for students to inspect."""

    result = create_figure_suite(
        frame,
        output=output_dir,
        style="ft",
        docx=True,
        source=source,
        title_prefix="Week 2 Market Macro",
        max_figures=5,
        narrative=True,
        formats=("png", "pdf"),
    )
    if result.skipped:
        print("Automatic suite skipped:")
        for item in result.skipped:
            print(f"- {item}")
    if result.issues:
        print("Automatic suite validation notes:")
        for issue in result.issues:
            print(f"- {issue.code}: {issue.message}")
    return result.docx_path


def build_curated_figures(
    frame: pd.DataFrame,
    output: str | Path = DEFAULT_OUTPUT,
    *,
    source: str = FRED_SOURCE,
) -> Path:
    """Build the curated Week 2 market-and-macro story pack."""

    output_dir = resolve_repo_path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[WordFigureEntry] = []
    sample = sample_label(frame)
    month_end = month_end_panel(frame)
    month_end_sample = sample_label(month_end)

    treasury_frame = frame.rename(
        columns={
            "DGS10": "10Y Treasury Yield",
            "DGS2": "2Y Treasury Yield",
            "DTB3": "3M Treasury Bill Yield",
        }
    )
    fig, _ = time_series_plot(
        treasury_frame,
        ["10Y Treasury Yield", "2Y Treasury Yield", "3M Treasury Bill Yield"],
        date="date",
        title="Treasury Yields",
        ylabel="Yield (%)",
        profile="word_a4",
        style="ft",
        direct_labels=True,
    )
    emit_figure(
        fig,
        output_dir,
        "week2_market_macro_story_treasury_rates",
        FigureContext(
            title="Week 2 Treasury Yields Across Maturities",
            note=(
                "Start with levels when the question is the shape of the term structure. "
                "This figure keeps the daily sample and compares the 10-year, 2-year, "
                "and 3-month Treasury yields directly."
            ),
            source=source,
            sample=sample,
            units="Percent",
        ),
        entries,
    )

    yield_curve_frame = frame.rename(
        columns={"ten_year_minus_three_month": "Yield Curve Slope"}
    )
    fig, _ = area_balance_plot(
        yield_curve_frame,
        "Yield Curve Slope",
        date="date",
        reference=0.0,
        title="Yield-Curve Inversions",
        series_label="Yield curve slope (10Y minus 3M)",
        positive_label="Positive slope",
        negative_label="Inversion below zero",
        ylabel="Yield curve slope (percentage points)",
        profile="word_a4",
        style="ft",
    )
    emit_figure(
        fig,
        output_dir,
        "week2_market_macro_story_yield_curve_inversions",
        FigureContext(
            title="Week 2 Yield-Curve Inversions Against Zero",
            note=(
                "Deviation charts are useful when the interpretation hinges on "
                "a threshold. Here the 10-year minus 3-month yield-curve slope "
                "is shaded above and below zero to "
                "highlight inversion episodes."
            ),
            source=source,
            sample=sample,
            units="Percentage points",
        ),
        entries,
    )

    level_panel = month_end.rename(
        columns={
            "DGS10": "10Y Treasury Yield",
            "T10Y2Y": "10Y-2Y Yield Spread",
            "VIXCLS": "VIX",
            "SP500": "S&P 500 Index",
        }
    )
    level_columns = [
        "10Y Treasury Yield",
        "10Y-2Y Yield Spread",
        "VIX",
        "S&P 500 Index",
    ]
    fig, axes = small_multiples(
        level_panel,
        level_columns,
        date="date",
        title="Month-End Levels By Series",
        ylabel=[
            "10Y Treasury yield (%)",
            "10Y-2Y spread\n(percentage points)",
            "VIX level",
            "S&P 500\nIndex level",
        ],
        profile="word_a4",
        style="ft",
        title_fontsize=7.8,
    )
    style_small_multiple_layout(
        fig,
        axes,
        level_columns,
        right_column_labels=False,
        ylabel_fontsize=7.8,
        tick_fontsize=8.3,
        suptitle_fontsize=11.8,
        wspace=0.60,
        hspace=0.44,
        left=0.11,
        right=0.96,
        top=0.85,
    )
    emit_figure(
        fig,
        output_dir,
        "week2_market_macro_story_level_small_multiples",
        FigureContext(
            title="Week 2 Month-End Level Small Multiples",
            note=(
                "Small multiples solve the mixed-unit problem cleanly. The month-end panel "
                "shows rates, spreads, volatility, and equity levels without forcing them "
                "onto one shared y-axis."
            ),
            source=source,
            sample=month_end_sample,
            units="Mixed units by panel",
        ),
        entries,
    )

    transform_panel = month_end.rename(
        columns={
            "DGS10_CHANGE_BP": "10Y Treasury Yield Change",
            "FEDFUNDS_CHANGE_BP": "Federal Funds Rate Change",
            "UNRATE_CHANGE_PP": "Unemployment Rate Change",
            "INDPRO_LOG_GROWTH_PCT": "Industrial Production Log Growth",
            "PAYEMS_LOG_GROWTH_PCT": "Payroll Employment Log Growth",
            "SP500_LOG_RETURN_PCT": "S&P 500 Log Return",
        }
    )
    transform_columns = [
        "10Y Treasury Yield Change",
        "Federal Funds Rate Change",
        "Unemployment Rate Change",
        "Industrial Production Log Growth",
        "Payroll Employment Log Growth",
        "S&P 500 Log Return",
    ]
    fig, axes = small_multiples(
        transform_panel,
        transform_columns,
        date="date",
        title="Month-End Transformations By Series",
        ylabel=[
            "Basis points",
            "Basis points",
            "Percentage points",
            "Log growth (%)",
            "Log growth (%)",
            "Log return (%)",
        ],
        profile="word_a4",
        style="ft",
        title_fontsize=7.4,
    )
    style_small_multiple_layout(
        fig,
        axes,
        transform_columns,
        right_column_labels=False,
        ylabel_fontsize=7.8,
        tick_fontsize=8.1,
        suptitle_fontsize=11.8,
        wspace=0.62,
        hspace=0.50,
        left=0.10,
        right=0.96,
        top=0.85,
    )
    emit_figure(
        fig,
        output_dir,
        "week2_market_macro_story_transform_small_multiples",
        FigureContext(
            title="Week 2 Month-End Transformation Small Multiples",
            note=(
                "Week 2 should not only plot levels. Rate series are often interpreted in "
                "basis-point changes, the unemployment rate in percentage-point changes, "
                "and activity or equity series in returns or log growth."
            ),
            source=source,
            sample=month_end_sample,
            units="Mixed transformed units by panel",
        ),
        entries,
    )

    fig, _ = indexed_time_series_plot(
        month_end,
        ["SP500", "INDPRO", "PAYEMS"],
        date="date",
        title="Indexed Market And Activity Levels",
        ylabel="Index (first month-end = 100)",
        profile="word_a4",
        style="ft",
        direct_labels=True,
    )
    emit_figure(
        fig,
        output_dir,
        "week2_market_macro_story_market_macro_indexed",
        FigureContext(
            title="Week 2 Indexed Market And Activity Levels",
            note=(
                "Indexing lets students compare level paths across series with different "
                "units. The S&P 500 is resampled from daily to month-end before it is "
                "merged with industrial production and payroll employment."
            ),
            source=source,
            sample=month_end_sample,
            units="Index, first observation = 100",
        ),
        entries,
    )

    scatter_frame = month_end.set_index("date").rename(
        columns={
            "FEDFUNDS": "Federal funds rate (%)",
            "UNRATE": "Unemployment rate (%)",
        }
    )
    fig, _ = scatter_plot(
        scatter_frame,
        "Federal funds rate (%)",
        "Unemployment rate (%)",
        fit=True,
        annotate=True,
        label_outliers=3,
        stats_location="upper right",
        title="Policy Rate And Unemployment",
        profile="word_a4",
        style="ft",
    )
    emit_figure(
        fig,
        output_dir,
        "week2_market_macro_story_policy_scatter_fit",
        FigureContext(
            title="Week 2 Policy Rate Versus Unemployment",
            note=(
                "Scatter plots answer a different question from time-series charts. This one "
                "asks whether tighter policy tends to coexist with a stronger labor market "
                "in the 10-year month-end sample."
            ),
            source=source,
            sample=month_end_sample,
            units="Percent",
        ),
        entries,
    )

    episode = month_end[
        month_end["date"].between("2019-01-31", str(month_end["date"].max().date()))
    ].dropna(subset=["FEDFUNDS", "UNRATE"]).copy()
    episode_panel = episode.rename(
        columns={
            "FEDFUNDS": "Federal Funds Rate",
            "UNRATE": "Unemployment Rate",
        }
    )
    fig, axes = small_multiples(
        episode_panel,
        ["Federal Funds Rate", "Unemployment Rate"],
        date="date",
        title="Pandemic-To-Tightening Episode",
        ylabel=["Federal funds rate (%)", "Unemployment rate (%)"],
        profile="word_a4",
        style="ft",
        title_fontsize=7.8,
    )
    style_small_multiple_layout(
        fig,
        axes,
        ["Federal Funds Rate", "Unemployment Rate"],
        ylabel_fontsize=8.6,
        tick_fontsize=8.3,
        suptitle_fontsize=11.8,
        wspace=0.36,
        hspace=0.40,
        left=0.10,
        right=0.95,
        top=0.85,
    )
    emit_figure(
        fig,
        output_dir,
        "week2_market_macro_story_policy_episode",
        FigureContext(
            title="Week 2 Pandemic-To-Tightening Policy Episode",
            note=(
                "The earlier connected scatter collapsed into a hard-to-read path once the "
                "pandemic spike and later rate plateau were both included. For this episode, "
                "matched time-series panels make the sequence and turning points legible."
            ),
            source=source,
            sample=f"{episode['date'].min():%Y-%m-%d} to {episode['date'].max():%Y-%m-%d}",
            units="Percent",
        ),
        entries,
    )

    vix_return_frame = month_end.set_index("date").rename(
        columns={
            "VIX_MONTHLY_VOL_CHANGE_PP": "Change in monthly VIX (pp)",
            "SP500_RETURN_PCT": "S&P 500 month-end return (%)",
        }
    )
    fig, _ = scatter_plot(
        vix_return_frame,
        "Change in monthly VIX (pp)",
        "S&P 500 month-end return (%)",
        fit=True,
        annotate=True,
        label_outliers=3,
        stats_location="upper right",
        title="VIX Shocks And Equity Returns",
        xlabel="Change in monthly-equivalent VIX (percentage points)",
        ylabel="S&P 500 month-end return (%)",
        profile="word_a4",
        style="ft",
    )
    emit_figure(
        fig,
        output_dir,
        "week2_market_macro_story_vix_return_scatter",
        FigureContext(
            title="Week 2 Monthly VIX Change Versus S&P 500 Return",
            note=(
                "VIX is annualized, so this figure first converts month-end VIX to a monthly "
                "equivalent by dividing by sqrt(12), then differences that monthly series "
                "before relating it to S&P 500 month-end returns."
            ),
            source=source,
            sample=month_end_sample,
            units="Monthly VIX change (pp) and S&P 500 return (%)",
        ),
        entries,
    )

    growth_series = month_end.set_index("date")["SP500"].dropna().astype(float)
    growth_series = growth_series / float(growth_series.iloc[0])
    growth_frame = growth_series.rename("S&P 500").to_frame()
    fig, ax = time_series_plot(
        growth_frame,
        ["S&P 500"],
        title="S&P 500 Growth Of One Dollar",
        ylabel="Cumulative wealth ($1, log scale)",
        legend=False,
        profile="word_a4",
        style="ft",
        direct_labels=False,
        line_width=1.6,
        line_alpha=0.95,
    )
    ax.yaxis.label.set_fontsize(10.2)
    ax.tick_params(axis="y", labelsize=8.6)
    if hasattr(fig, "set_layout_engine"):
        fig.set_layout_engine(None)
    fig.subplots_adjust(left=0.12, right=0.86)
    ax.set_yscale("log")
    ymin = 0.95
    ymax = float(growth_series.max()) * 1.05
    ax.set_ylim(ymin, ymax)
    ticks = wealth_ticks(ymin, ymax)
    if ticks:
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"${tick:g}" for tick in ticks])
    ax.minorticks_off()
    add_endpoint_wealth_label(ax, label="S&P 500", wealth=float(growth_series.iloc[-1]))
    emit_figure(
        fig,
        output_dir,
        "week2_market_macro_story_sp500_cumulative_return",
        FigureContext(
            title="Week 2 S&P 500 Growth Of One Dollar",
            note=(
                "This figure uses a cumulative wealth path rather than a plain cumulative-"
                "return axis. Starting from a $1 investment, the month-end S&P 500 path is "
                "shown on a log scale with dollar ticks so compounding stays readable."
            ),
            source=source,
            sample=month_end_sample,
            units="Cumulative wealth, log scale ($1 investment)",
        ),
        entries,
    )

    docx_path = insert_figures_docx(
        entries,
        output_dir / DOCX_NAME,
        title="Week 2 Market And Macro Story Figure Pack",
    )
    docx_issues = validate_docx_images_fit_page(docx_path)
    if docx_issues:
        details = "; ".join(issue.message for issue in docx_issues)
        raise RuntimeError(f"Word proof pack failed validation: {details}")
    return docx_path


def build_fred_market_figures(
    data_path: str | Path = DEFAULT_DATA,
    output: str | Path = DEFAULT_OUTPUT,
    *,
    use_fixture_if_missing: bool = True,
    auto_suite: bool = False,
) -> dict[str, Path | None]:
    """Build automatic and curated FRED market-and-macro figure outputs."""

    frame = load_fred_market_frame(
        data_path,
        use_fixture_if_missing=use_fixture_if_missing,
    )
    output_dir = resolve_repo_path(output)
    auto_docx = build_auto_suite(frame, output_dir, FRED_SOURCE) if auto_suite else None
    curated_docx = build_curated_figures(frame, output_dir, source=FRED_SOURCE)
    return {"auto_docx": auto_docx, "curated_docx": curated_docx}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Build Week 2 FT-style figures from FRED market-and-macro data.",
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Cleaned FRED CSV path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output figure folder.")
    parser.add_argument(
        "--no-fixture-fallback",
        action="store_true",
        help="Fail if the cleaned FRED CSV does not exist.",
    )
    parser.add_argument(
        "--auto-suite",
        action="store_true",
        help="Also build the automatic create_figure_suite demonstration output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the FRED market-and-macro figure exercise."""

    args = parse_args(argv)
    paths = build_fred_market_figures(
        args.data,
        args.output,
        use_fixture_if_missing=not args.no_fixture_fallback,
        auto_suite=args.auto_suite,
    )
    for label, path in paths.items():
        if path is not None:
            print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
