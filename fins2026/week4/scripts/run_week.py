# ruff: noqa
"""Print the canonical Week 4 workflow."""

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
    """Print the week inventory and the canonical Week 4 run order."""

    for directory in RESULTS_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
    print('Week 4: APIs, returns, portfolio construction, and the first app surface')
    print()
    print(describe_week_data())
    print()
    print('Stage 1: source connection')
    print('- python fins2026/week4/scripts/run_beginner_ecb_api.py')
    print('- python fins2026/week4/scripts/run_beginner_french_rfr.py')
    print('- before any Tiingo step, check or set TIINGO_API_KEY in the current shell')
    print('- python fins2026/week4/scripts/run_beginner_tiingo_small_panel.py')
    print('- python fins2026/week4/scripts/run_beginner_tiingo_wide_panel.py')
    print('- python fins2026/week4/scripts/run_beginner_yahoo_small_panel.py')
    print('- python fins2026/week4/scripts/run_beginner_yahoo_famous_50.py')
    print('- python fins2026/week4/scripts/run_beginner_tiingo_famous_50.py')
    print('  optional keyed extension')
    print()
    print('Stage 2: returns and diagnostics')
    print('- python fins2026/week4/scripts/run_beginner_stage2_returns_wide.py')
    print('- python fins2026/week4/scripts/run_beginner_stage2_returns_long.py')
    print('- python fins2026/week4/scripts/run_beginner_stage2_features_long.py')
    print('- python fins2026/week4/scripts/make_stage2_return_check_figures.py')
    print('  all four Stage 2 scripts default to Yahoo')
    print()
    print('Stage 3: in-sample portfolios')
    print('- python fins2026/week4/scripts/run_beginner_stage3_portfolios.py')
    print('- python fins2026/week4/scripts/make_stage3_portfolio_figures.py')
    print()
    print('Stage 4: client-facing app')
    print('- python fins2026/week4/scripts/build_week4_app_fixture.py')
    print('- streamlit run fins2026/week4/app/streamlit_app.py')
    print('  the Week 4 app uses the 10-stock Yahoo app universe')
    print()
    print('Key Week 4 rules:')
    print('- Yahoo is the canonical Week 4 equity dataset')
    print('- Tiingo is the keyed API/authentication teaching example')
    print('- keep long panel data as the canonical saved dataset')
    print('- use adjusted prices for Stage 2 returns')
    print('- compare wide and long return calculations')
    print('- flag extreme moves instead of silently deleting them')
    print('- Stage 3 is in-sample only; out-of-sample re-estimation is discussed but not coded yet')
    print('- Stage 4 keeps that in-sample scope and turns it into a client-facing app')
    print(
        '- refresh guidance/ with python tools/workflow.py '
        'build-week-context --target fins2026/week4'
    )


if __name__ == '__main__':
    main()

