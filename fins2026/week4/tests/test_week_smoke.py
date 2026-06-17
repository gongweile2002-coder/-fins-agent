# ruff: noqa
"""Local smoke test for the standard weekly scaffold."""

from __future__ import annotations

from pathlib import Path


def test_week_scaffold_smoke() -> None:
    week_root = Path(__file__).resolve().parents[1]
    for relative in [
        'README.md',
        'WORKSHOP.md',
        'DATA_GUIDE.md',
        'SUBMISSION_CHECKLIST.md',
        'AGENTS.md',
        'guidance/week-context.md',
        'guidance/data-context.md',
        'guidance/output-context.md',
        'scripts/run_week.py',
        'scripts/_bootstrap.py',
        'scripts/describe_data.py',
        'scripts/run_beginner_ecb_api.py',
        'scripts/run_beginner_french_rfr.py',
        'scripts/run_beginner_tiingo_small_panel.py',
        'scripts/run_beginner_tiingo_wide_panel.py',
        'scripts/run_beginner_tiingo_famous_50.py',
        'scripts/run_beginner_yahoo_small_panel.py',
        'scripts/run_beginner_yahoo_famous_50.py',
        'scripts/run_beginner_stage2_returns_wide.py',
        'scripts/run_beginner_stage2_returns_long.py',
        'scripts/run_beginner_stage2_features_long.py',
        'scripts/make_stage2_return_check_figures.py',
        'scripts/run_beginner_stage3_portfolios.py',
        'scripts/make_stage3_portfolio_figures.py',
        'scripts/build_week4_app_fixture.py',
        'data/README.md',
        'data/tiingo_intro_3.txt',
        'data/tiingo_famous_50_pre2000.txt',
        'data/yahoo_intro_3.txt',
        'data/yahoo_famous_50.txt',
        'data/yahoo_app_10.txt',
        'scratch/README.md',
        'app/README.md',
        'app/app_config.py',
        'app/app_data.py',
        'app/app_insights.py',
        'app/app_views.py',
        'app/streamlit_app.py',
        'app/fixtures/README.md',
        'app/fixtures/yahoo_app_10_long.parquet',
        'app/fixtures/french_daily_rfr.parquet',
        'app/tests/test_app_smoke.py',
        'code/__init__.py',
        'code/api_intro_ecb.py',
        'code/risk_free_rate_french.py',
        'code/equity_api_tiingo.py',
        'code/equity_api_yahoo.py',
        'code/stage2_equity_returns.py',
        'code/stage2_return_figures.py',
        'code/stage3_portfolios.py',
        'code/stage3_portfolio_figures.py',
        'code/stage4_app.py',
        'tests/test_api_intro.py',
        'tests/test_stage4_app.py',
    ]:
        assert (week_root / relative).exists(), relative

