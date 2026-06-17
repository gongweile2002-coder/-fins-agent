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
        'scripts/_bootstrap.py',
        'scripts/run_week.py',
        'scripts/describe_data.py',
        'scripts/run_beginner_french_rfr.py',
        'scripts/run_beginner_yahoo_crypto_intro_5.py',
        'scripts/run_beginner_yahoo_crypto_20_since_2019.py',
        'scripts/run_beginner_stage2_returns_wide.py',
        'scripts/run_beginner_stage2_returns_long.py',
        'scripts/run_beginner_stage2_features_long.py',
        'scripts/run_beginner_stage3_oos_weights.py',
        'scripts/make_stage2_crypto_figures.py',
        'scripts/make_stage3_portfolio_figures.py',
        'scripts/build_week5_app_fixture.py',
        'data/README.md',
        'data/yahoo_crypto_intro_5.txt',
        'data/yahoo_crypto_20_since_2019.txt',
        'code/__init__.py',
        'code/README.md',
        'code/crypto_api_yahoo.py',
        'code/risk_free_rate_french.py',
        'code/stage2_crypto_returns.py',
        'code/stage2_crypto_figures.py',
        'code/stage3_oos_portfolios.py',
        'code/stage3_portfolio_figures.py',
        'code/stage4_app.py',
        'app/README.md',
        'app/app_config.py',
        'app/app_data.py',
        'app/app_insights.py',
        'app/app_views.py',
        'app/streamlit_app.py',
        'app/fixtures/README.md',
        'app/fixtures/week5_app_features_long.parquet',
        'app/tests/test_app_smoke.py',
        'scratch/README.md',
        'results/figures/.gitkeep',
        'results/data/stage3/.gitkeep',
        'results/tables/stage3/.gitkeep',
        'results/app/.gitkeep',
    ]:
        assert (week_root / relative).exists(), relative

