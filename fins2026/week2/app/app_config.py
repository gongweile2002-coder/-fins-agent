# ruff: noqa
"""Configuration constants for the Week 2 Streamlit intro app."""

from __future__ import annotations

from fins2026.week2.code.market_window import WEEK2_FRED_WINDOW_YEARS

APP_TITLE = "U.S. Month-End Market and Macro Monitor"
FRED_SERIES = (
    "DGS10",
    "DGS2",
    "DTB3",
    "T10Y2Y",
    "VIXCLS",
    "UNRATE",
    "INDPRO",
    "PAYEMS",
    "FEDFUNDS",
    "SP500",
)
SAMPLE_PERIODS = {
    "10Y": 10,
    "5Y": 5,
    "2Y": 2,
    "1Y": 1,
}
DEFAULT_SAMPLE_PERIOD = "10Y"
DATA_MODE_OPTIONS = ["Fixture", "Live FRED"]
SERIES_LABELS = {
    "DGS10": "10-Year Treasury (%)",
    "DGS2": "2-Year Treasury (%)",
    "DTB3": "3-Month Treasury Bill (%)",
    "T10Y2Y": "10Y-2Y Treasury Spread (%)",
    "VIXCLS": "VIX (%)",
    "UNRATE": "Unemployment rate (%)",
    "INDPRO": "Industrial production index (2017=100)",
    "PAYEMS": "Payroll employment (thousands)",
    "FEDFUNDS": "Federal funds rate (%)",
    "SP500": "S&P 500 index level",
    "DGS10_CHANGE_BP": "10-Year Treasury monthly change (bp)",
    "DGS2_CHANGE_BP": "2-Year Treasury monthly change (bp)",
    "DTB3_CHANGE_BP": "3-Month Treasury monthly change (bp)",
    "T10Y2Y_CHANGE_BP": "10Y-2Y spread monthly change (bp)",
    "UNRATE_CHANGE_PP": "Unemployment monthly change (pp)",
    "INDPRO_LOG_GROWTH_PCT": "Industrial production monthly log growth (%)",
    "PAYEMS_LOG_GROWTH_PCT": "Payroll employment monthly log growth (%)",
    "FEDFUNDS_CHANGE_BP": "Federal funds monthly change (bp)",
    "VIX_CHANGE_PCT": "VIX monthly percent change (%)",
    "SP500_RETURN_PCT": "S&P 500 month-end return (%)",
    "SP500_LOG_RETURN_PCT": "S&P 500 month-end log return (%)",
    "SP500_CUMULATIVE_RETURN_PCT": "S&P 500 cumulative return (%)",
}
DAILY_RESAMPLE_NOTE = (
    "Daily market data are resampled to month-end before merging with "
    "monthly macro series."
)
WINDOW_NOTE = (
    f"The shared panel is standardized to the latest {WEEK2_FRED_WINDOW_YEARS} years "
    "so each series can be compared over one common market cycle window."
)
MONTHLY_ALIGN_NOTE = (
    "Monthly macro releases are aligned to month-end so they can be compared "
    "directly with the market series in the shared panel."
)
CHANGE_NOTE = (
    "Month-end changes emphasize movement between observations rather than the "
    "absolute level itself."
)
LOG_GROWTH_NOTE = (
    "Log growth is useful for activity series because it approximates a "
    "percentage growth rate and is easier to compare across levels."
)
RETURN_NOTE = (
    "Returns are computed after sampling the daily S&P 500 index at month-end."
)
SERIES_NOTES = {
    "DGS10": f"{DAILY_RESAMPLE_NOTE} {WINDOW_NOTE}",
    "DGS2": f"{DAILY_RESAMPLE_NOTE} {WINDOW_NOTE}",
    "DTB3": f"{DAILY_RESAMPLE_NOTE} {WINDOW_NOTE}",
    "T10Y2Y": f"{DAILY_RESAMPLE_NOTE} {WINDOW_NOTE}",
    "VIXCLS": f"{DAILY_RESAMPLE_NOTE} {WINDOW_NOTE}",
    "UNRATE": f"{MONTHLY_ALIGN_NOTE} {WINDOW_NOTE}",
    "INDPRO": f"{MONTHLY_ALIGN_NOTE} {WINDOW_NOTE}",
    "PAYEMS": f"{MONTHLY_ALIGN_NOTE} {WINDOW_NOTE}",
    "FEDFUNDS": f"{MONTHLY_ALIGN_NOTE} {WINDOW_NOTE}",
    "SP500": (
        "The live S&P 500 source is daily; it is sampled at month-end before "
        f"plotting and kept inside the latest {WEEK2_FRED_WINDOW_YEARS}-year panel."
    ),
    "DGS10_CHANGE_BP": f"{DAILY_RESAMPLE_NOTE} {CHANGE_NOTE}",
    "DGS2_CHANGE_BP": f"{DAILY_RESAMPLE_NOTE} {CHANGE_NOTE}",
    "DTB3_CHANGE_BP": f"{DAILY_RESAMPLE_NOTE} {CHANGE_NOTE}",
    "T10Y2Y_CHANGE_BP": f"{DAILY_RESAMPLE_NOTE} {CHANGE_NOTE}",
    "UNRATE_CHANGE_PP": f"{MONTHLY_ALIGN_NOTE} {CHANGE_NOTE}",
    "INDPRO_LOG_GROWTH_PCT": f"{MONTHLY_ALIGN_NOTE} {LOG_GROWTH_NOTE}",
    "PAYEMS_LOG_GROWTH_PCT": f"{MONTHLY_ALIGN_NOTE} {LOG_GROWTH_NOTE}",
    "FEDFUNDS_CHANGE_BP": f"{MONTHLY_ALIGN_NOTE} {CHANGE_NOTE}",
    "VIX_CHANGE_PCT": f"{DAILY_RESAMPLE_NOTE} {CHANGE_NOTE}",
    "SP500_RETURN_PCT": (
        f"{RETURN_NOTE} It is the simple month-end percentage return in the latest "
        f"{WEEK2_FRED_WINDOW_YEARS}-year panel."
    ),
    "SP500_LOG_RETURN_PCT": (
        f"{RETURN_NOTE} It is the log return, which adds cleanly through time."
    ),
    "SP500_CUMULATIVE_RETURN_PCT": (
        "Cumulative return is measured relative to the first observed month-end "
        f"S&P 500 level inside the shared {WEEK2_FRED_WINDOW_YEARS}-year panel."
    ),
}
DEFAULT_SERIES = "SP500"
