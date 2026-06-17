# ruff: noqa
"""Print the canonical Week 2 workflow and create output folders."""

from __future__ import annotations

from pathlib import Path

from describe_data import describe_week_data

WEEK_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIRS = [
    WEEK_ROOT / "results" / "data",
    WEEK_ROOT / "results" / "figures",
    WEEK_ROOT / "results" / "figures" / "beginner_plotting",
    WEEK_ROOT / "results" / "tables",
    WEEK_ROOT / "results" / "app",
]


def main() -> None:
    """Print the week inventory and confirm standard output paths."""

    for directory in RESULTS_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
    print("Week 2")
    print()
    print(describe_week_data())
    print()
    print("Next steps:")
    print("- start with scripts/make_beginner_synthetic_prices.py")
    print("- then build one plain line chart with scripts/make_beginner_simple_time_series.py")
    print("- then rebuild it with scripts/make_beginner_ft_time_series.py")
    print("- check the output against FIGURE_RUBRIC.md before opening larger figure packs")
    print("- walk through U.S. Stage 1 with scripts/make_fred_stage1_walkthrough.py")
    print(
        "- use --rolling-window on the FRED scripts only when you explicitly "
        "want fresher U.S. data"
    )
    print("- walk through Australia Station 1 with scripts/make_australia_stage1_walkthrough.py")
    print("- build the figure outputs with scripts/make_all_week2_figures.py")
    print("- pull live FRED data with scripts/pull_fred_market_data.py when internet is available")
    print(
        "- pull official Australia macro data with scripts/pull_australia_macro_data.py "
        "to build the frozen Stage 1 and Stage 2 lecture panels"
    )
    print("- launch the intro app with streamlit run fins2026/week2/app/streamlit_app.py")
    print("- keep generated datasets, figures, and tables under results/")
    print("- use NARRATIVE_BRIEF.md when you turn the five figures into prose")
    print(
        "- rehearse deployment with SUBMISSION_CHECKLIST.md and "
        "docs/apps/streamlit/student-quickstart.md"
    )
    print(
        "- refresh guidance/ with python tools/workflow.py "
        "build-week-context --target fins2026/week2"
    )


if __name__ == "__main__":
    main()
