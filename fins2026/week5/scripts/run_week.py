# ruff: noqa
"""Print the canonical Week 5 workflow."""

from __future__ import annotations

from pathlib import Path

from describe_data import describe_week_data

WEEK_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIRS = [
    WEEK_ROOT / 'results' / 'data',
    WEEK_ROOT / 'results' / 'figures',
    WEEK_ROOT / 'results' / 'tables',
    WEEK_ROOT / 'results' / 'app',
]


def main() -> None:
    """Print the week inventory and the canonical Week 5 run order."""

    for directory in RESULTS_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
    print('Week 5: crypto data, return diagnostics, and out-of-sample portfolio checks')
    print()
    print(describe_week_data())
    print()
    print('Stage 1: source connection')
    print('- python fins2026/week5/scripts/run_beginner_french_rfr.py')
    print('- python fins2026/week5/scripts/run_beginner_yahoo_crypto_intro_5.py')
    print('- python fins2026/week5/scripts/run_beginner_yahoo_crypto_20_since_2019.py')
    print()
    print('Stage 2: returns, merge logic, and diagnostics')
    print('- python fins2026/week5/scripts/run_beginner_stage2_returns_wide.py')
    print('- python fins2026/week5/scripts/run_beginner_stage2_returns_long.py')
    print('- python fins2026/week5/scripts/run_beginner_stage2_features_long.py')
    print('- python fins2026/week5/scripts/make_stage2_crypto_figures.py')
    print('- python fins2026/week5/scripts/make_stage2_crypto_figures.py --include-appendix')
    print()
    print('Stage 3: out-of-sample portfolio weights, returns, and FT-style figures')
    print('- python fins2026/week5/scripts/run_beginner_stage3_oos_weights.py')
    print('- python fins2026/week5/scripts/make_stage3_portfolio_figures.py')
    print()
    print('Stage 4: client-facing app surface')
    print('- python fins2026/week5/scripts/build_week5_app_fixture.py')
    print('- streamlit run fins2026/week5/app/streamlit_app.py')
    print(
        '- python tools/workflow.py check-app-submission --target fins2026/week5 '
        '--entrypoint fins2026/week5/app/streamlit_app.py'
    )
    print()
    print('Key Week 5 rules:')
    print('- Yahoo is the canonical Week 5 crypto dataset')
    print('- Kenneth French daily RF is merged onto the 24/7 crypto panel by date')
    print('- forward-fill RF across weekends, holidays, and tail dates')
    print('- keep long panel data as the canonical saved dataset')
    print('- use adjusted prices for Stage 2 returns')
    print('- compare wide and long return calculations')
    print('- use 365-day annualization for crypto risk metrics')
    print('- Stage 3 weights are estimated on decision dates and saved on later holding dates')
    print('- Stage 3 returns keep return_date first and formation_date second')
    print('- Stage 3 now uses the long-only portfolio family only')
    print('- Stage 3 figure exports include both research views and app-style factsheet views')
    print('- Stage 4 turns the same long-only published fund shelf into a client-facing app')
    print(
        '- refresh guidance/ with python tools/workflow.py '
        'build-week-context --target fins2026/week5'
    )


if __name__ == '__main__':
    main()

