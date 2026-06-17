# ruff: noqa
"""Summarize the data sources used in Week 2."""

from __future__ import annotations

from pathlib import Path

WEEK_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = WEEK_ROOT / "data"
RESULTS_DATA_DIR = WEEK_ROOT / "results" / "data"


def visible_files(directory: Path) -> list[Path]:
    """Return non-placeholder files inside a directory tree."""

    if not directory.exists():
        return []
    return sorted(
        path for path in directory.rglob("*") if path.is_file() and path.name != ".gitkeep"
    )


def describe_directory(label: str, directory: Path) -> list[str]:
    """Return a short inventory for one week data directory."""

    files = visible_files(directory)
    lines = [f"{label}: {directory.relative_to(WEEK_ROOT).as_posix()}"]
    if not files:
        lines.append("- no files yet")
        return lines
    for path in files:
        rel = path.relative_to(WEEK_ROOT).as_posix()
        lines.append(f"- {rel} ({path.stat().st_size} bytes)")
    return lines


def describe_week_data() -> str:
    """Return a plain-text summary of source and generated datasets."""

    lines = ["Week 2 data inventory", ""]
    lines.extend(describe_directory("Source data", DATA_DIR))
    lines.append("")
    lines.append("App-backed data sources:")
    lines.append("- frozen FRED validation fixtures are loaded from fintools.datasets")
    lines.append(
        "- live app mode requests a no-key FRED graph CSV for Treasury, volatility, "
        "macro, and S&P 500 series"
    )
    lines.append(
        "- the beginner plotting ladder starts with a deterministic synthetic "
        "business-day price CSV under results/data/"
    )
    lines.append("- offline macro data comes from fred_macro_monthly")
    lines.append("- offline S&P 500 data comes from fred_sp500_daily")
    lines.append("- the Week 2 app resamples daily market data to month-end before merging")
    lines.append(
        "- the Week 2 story contrasts levels, bp changes, pp changes, "
        "log growth, and returns"
    )
    lines.append(
        "- FIGURE_RUBRIC.md defines the Week 2 figure judgment contract: "
        "caption, units, source, sample window, plus one question per figure"
    )
    lines.append(
        "- the Australia extension adds an official-source DDF pipeline with "
        "explicit release lags"
    )
    lines.append(
        "- students can walk through the U.S. Stage 1 path with "
        "scripts/make_fred_stage1_walkthrough.py"
    )
    lines.append(
        "- students can walk through Australia Station 1 with "
        "scripts/make_australia_stage1_walkthrough.py"
    )
    lines.append(
        "- the Australia common reference endpoint is 2025-12-31 and the classroom "
        "information-set month-end is 2026-03-31"
    )
    lines.append(
        "- Week 2 stores classroom month-end release proxies rather than exact "
        "agency publication timestamps"
    )
    lines.append(
        "- the lecture-facing FRED scripts now default to the frozen 2015-01-01 "
        "through 2025-12-31 sample"
    )
    lines.append(
        "- students can switch to fresher U.S. data with --rolling-window on the "
        "FRED Stage 1 and FRED pull scripts"
    )
    lines.append(
        "- beginner plotting outputs are written under "
        "results/figures/beginner_plotting/"
    )
    lines.append(
        "- market-macro narrative figures are written under "
        "results/figures/market_macro_story/"
    )
    lines.append(
        "- Australia macro narrative figures are written under "
        "results/figures/australia_macro_story/"
    )
    lines.append(
        "- broad style-gallery outputs are written under "
        "results/figures/style_gallery/"
    )
    lines.append(
        "- NARRATIVE_BRIEF.md defines the six-paragraph structure for the Week 2 "
        "five-figure story"
    )
    lines.append("")
    lines.extend(describe_directory("Generated data", RESULTS_DATA_DIR))
    return "\n".join(lines)


def main() -> None:
    print(describe_week_data())


if __name__ == "__main__":
    main()
