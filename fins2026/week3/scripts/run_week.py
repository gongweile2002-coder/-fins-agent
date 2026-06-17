# ruff: noqa
"""Print the canonical Week 3 workflow and create output folders."""

from __future__ import annotations

from pathlib import Path

from describe_data import describe_week_data
from describe_forecast_data import describe_forecast_data

WEEK_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIRS = [
    WEEK_ROOT / "results" / "data",
    WEEK_ROOT / "results" / "data" / "beginner_forecasting",
    WEEK_ROOT / "results" / "figures",
    WEEK_ROOT / "results" / "figures" / "beginner_forecasting",
    WEEK_ROOT / "results" / "figures" / "beginner_forecast_story",
    WEEK_ROOT / "results" / "figures" / "us_beginner_forecasting",
    WEEK_ROOT / "results" / "figures" / "us_beginner_forecast_story",
    WEEK_ROOT / "results" / "tables",
    WEEK_ROOT / "results" / "tables" / "beginner_forecasting",
    WEEK_ROOT / "results" / "tables" / "beginner_forecast_story",
    WEEK_ROOT / "results" / "tables" / "us_beginner_forecasting",
    WEEK_ROOT / "results" / "tables" / "us_beginner_forecast_story",
    WEEK_ROOT / "results" / "data" / "us_beginner_forecasting",
    WEEK_ROOT / "results" / "app",
    WEEK_ROOT / "results" / "forecasts",
]


def main() -> None:
    """Print the week inventory and confirm standard output paths."""

    for directory in RESULTS_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
    print("Week 3")
    print()
    print(describe_week_data())
    print()
    print(describe_forecast_data())
    print()
    print("Next steps:")
    print("- core in-class path:")
    print("  - scripts/make_beginner_forecasting_series.py")
    print("  - scripts/run_beginner_unit_root_check.py")
    print("  - scripts/run_beginner_naive_forecast.py")
    print("  - scripts/run_beginner_ar_forecast.py")
    print("  - scripts/run_beginner_arma_forecast.py")
    print("  - scripts/run_beginner_arx_forecast.py")
    print("  - scripts/run_beginner_model_horse_race.py")
    print("  - scripts/make_beginner_forecast_story_figures.py")
    print("  - then open the Australia app")
    print("- extension path:")
    print("  - scripts/run_beginner_armax_forecast.py")
    print("  - scripts/run_beginner_ols_forecast.py")
    print("  - scripts/run_beginner_enet_forecast.py")
    print("  - scripts/run_beginner_ensemble_forecast.py")
    print("- U.S. mirror extension:")
    print("  - scripts/make_us_beginner_forecasting_series.py")
    print("  - scripts/run_us_beginner_unit_root_check.py")
    print("  - scripts/run_us_beginner_naive_forecast.py")
    print("  - scripts/run_us_beginner_ar_forecast.py")
    print("  - scripts/run_us_beginner_arma_forecast.py")
    print("  - scripts/run_us_beginner_arx_forecast.py")
    print("  - scripts/run_us_beginner_model_horse_race.py")
    print("  - scripts/make_us_beginner_forecast_story_figures.py")
    print(
        "- after the lecture ladder, build the reusable Week 3 input bundle with "
        "python fins2026/week3/scripts/build_forecast_inputs.py --use-fixture"
    )
    print(
        "- benchmark the approved models with python "
        "fins2026/week3/scripts/run_forecast_benchmarks.py --use-fixture"
    )
    print("- launch the full app with streamlit run fins2026/week3/app/streamlit_app.py")
    print(
        "- use BEGINNER_FORECASTING.md to explain the naive -> AR -> ARMA -> ARX "
        "-> ARMAX -> OLS -> ENet -> horse race -> ensemble lecture path"
    )
    print(
        "- use APP_LAB.md and APP_AUDIT.md to explain the Australia-first extension "
        "from Week 2"
    )
    print("- keep generated datasets, figures, and tables under results/")
    print(
        "- refresh guidance/ with python tools/workflow.py "
        "build-week-context --target fins2026/week3"
    )


if __name__ == "__main__":
    main()
