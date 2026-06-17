# ruff: noqa
"""Local smoke test for the standard weekly scaffold."""

from __future__ import annotations

from pathlib import Path


def test_week_scaffold_smoke() -> None:
    week_root = Path(__file__).resolve().parents[1]
    for relative in [
        "README.md",
        "WORKSHOP.md",
        "FIGURE_RUBRIC.md",
        "BEGINNER_PLOTTING.md",
        "FRED_STAGE1_DDF.md",
        "STAGE1_DDF.md",
        "DATA_GUIDE.md",
        "FIGURE_GALLERY.md",
        "NARRATIVE_BRIEF.md",
        "APP_LAB.md",
        "APP_AUDIT.md",
        "SUBMISSION_CHECKLIST.md",
        "AGENTS.md",
        "guidance/week-context.md",
        "guidance/data-context.md",
        "guidance/output-context.md",
        "scripts/make_beginner_synthetic_prices.py",
        "scripts/make_beginner_simple_time_series.py",
        "scripts/make_beginner_ft_time_series.py",
        "scripts/make_fred_stage1_walkthrough.py",
        "scripts/make_australia_stage1_walkthrough.py",
        "scripts/run_week.py",
        "scripts/describe_data.py",
        "data/README.md",
        "scratch/README.md",
        "app/README.md",
        "app/streamlit_app.py",
        "app/app_config.py",
        "app/app_data.py",
        "app/tests/test_app_smoke.py",
        "prompts/streamlit_intro_prompt.md",
    ]:
        assert (week_root / relative).exists(), relative

