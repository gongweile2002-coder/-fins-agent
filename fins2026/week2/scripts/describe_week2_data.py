# ruff: noqa
"""Describe the datasets used in the Week 2 figure and app lab."""

from __future__ import annotations

from fintools.datasets import available_validation_datasets, load_validation_dataset


def sample_for_index(index) -> str:
    """Return a compact sample label for a dataframe index."""

    if len(index) == 0:
        return "empty"
    if hasattr(index, "min") and hasattr(index, "max"):
        start = index.min()
        end = index.max()
        if hasattr(start, "strftime"):
            return f"{start:%Y-%m-%d} to {end:%Y-%m-%d}"
        return f"{start} to {end}"
    return "unknown"


def validation_dataset_lines() -> list[str]:
    """Return one-line descriptions of the frozen validation datasets."""

    lines = ["Frozen validation datasets:"]
    for name in available_validation_datasets():
        dataset = load_validation_dataset(name)
        frame = dataset.data
        lines.append(
            f"- {name}: {frame.shape[0]:,} rows x {frame.shape[1]:,} columns; "
            f"sample {sample_for_index(frame.index)}; source: {dataset.source}"
        )
    return lines


def describe_week2_data() -> str:
    """Return a complete student-facing Week 2 data description."""

    lines = [
        "Week 2 data overview",
        "",
        *validation_dataset_lines(),
        "",
        "Week 2 beginner workflow:",
        (
            "- make_beginner_synthetic_prices.py writes a deterministic business-day "
            "CSV with date, ticker, and price columns"
        ),
        (
            "- make_beginner_simple_time_series.py turns that CSV into one plain "
            "Matplotlib line chart with explicit labels"
        ),
        (
            "- make_beginner_ft_time_series.py rebuilds the same one-series story "
            "with the Week 2 FT-style helper and caption sidecar"
        ),
        (
            "- FIGURE_RUBRIC.md gives the Week 2 chart judgment contract: four "
            "standards, one question per figure, and sentence-title discipline"
        ),
        "",
        "Week 2 Australia workflow:",
        (
            "- make_australia_stage1_walkthrough.py is the student-facing Station 1 "
            "script for pulling, timing, and saving the long table"
        ),
        (
            "- pull_australia_macro_data.py builds a Stage 1 long table from official "
            "RBA statistical tables, then writes monthly, quarterly, reference-date, "
            "observable-date, and feature panels"
        ),
        (
            "- the Australia pack freezes the common classroom reference endpoint at "
            "2025-12-31 and the fully observed classroom information set at 2026-03-31"
        ),
        (
            "- students should separate reference dates from observable dates before "
            "they resample or merge mixed-frequency macro series"
        ),
        (
            "- monthly labour data, quarterly GDP and WPI, point-in-time vacancies, "
            "and market series all enter the story with different lag rules"
        ),
        (
            "- the curated Australia pack teaches DDF Station 1 authenticity first, "
            "then DDF Station 2 communication-ready transforms and visuals"
        ),
        "- rerun with --use-fixture when offline",
        "",
        "Week 2 FRED workflow:",
        (
            "- make_fred_stage1_walkthrough.py is the student-facing Station 1 "
            "script for typing and timing the U.S. FRED panel before plotting"
        ),
        (
            "- pull_fred_market_data.py downloads fresh FRED graph CSV data for "
            "Treasury, volatility, macro, and S&P 500 series"
        ),
        (
            "- by default the lecture-facing FRED path is frozen to 2015-01-01 "
            "through 2025-12-31 so lecture results stay reproducible"
        ),
        (
            "- students can opt into the latest rolling 10-year FRED panel with "
            "--rolling-window"
        ),
        (
            "- the offline fixture mirrors that with frozen FRED daily rates, "
            "VIX, SP500, and monthly macro data"
        ),
        "- the Week 2 app and figure pack resample daily market series to month-end",
        "- monthly macro observations are aligned to month-end before merging",
        (
            "- Week 2 stores classroom month-end release proxies rather than exact "
            "agency publication timestamps"
        ),
        (
            "- levels are not always the right plot: Week 2 also teaches bp "
            "changes, percentage-point changes, log growth, and returns"
        ),
        (
            "- the Week 2 app remains a rolling latest-10-year surface even "
            "though the lecture figure scripts are now frozen by default"
        ),
        "- style-gallery outputs include GDP, world-bank, and other FT-style examples",
        (
            "- the curated market-macro pack uses subplots, scatter, indexed "
            "lines, and cumulative-return views"
        ),
        "",
        "Week 2 figure surfaces:",
        "- results/figures/beginner_plotting/ for the synthetic one-series beginner ladder",
        "- results/figures/style_gallery/ for the broad FT-style validation gallery",
        "- results/figures/market_macro_story/ for the U.S. FRED market-and-macro pack",
        "- results/figures/australia_macro_story/ for the Australia macro lecture extension",
        "",
        "Week 2 narrative contract:",
        "- NARRATIVE_BRIEF.md defines the six-paragraph macro-story structure",
    ]
    return "\n".join(lines)


def main() -> int:
    """Print the Week 2 data overview."""

    print(describe_week2_data())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
