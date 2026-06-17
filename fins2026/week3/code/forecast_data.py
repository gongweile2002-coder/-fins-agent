# ruff: noqa
"""Australia-first data builders for forecasts and the app."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fins2026.week3.code.panel_support import (
    CLASSROOM_INFORMATION_SET_MONTH_END,
    DAILY_MARKET_COLUMNS,
    MONTHLY_MACRO_COLUMNS,
    build_month_end_panel,
    build_observable_panel,
    load_stage1_fixture_csv,
    log_change_percent,
)
from fintools.apps import clean_fred_graph_csv, fred_graph_url, read_fred_graph_csv
from fintools.datasets import load_validation_dataset

AUSTRALIA_STAGE1_FIXTURE = Path("fins2026/week3/data/australia_macro_stage1_long.csv")
LEGACY_AUSTRALIA_STAGE1_FIXTURE = Path("fins2026/week2/data/australia_macro_stage1_long.csv")
DEFAULT_RESULTS_DIR = Path("fins2026/week3/results/data")
US_FRED_SERIES = (
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


def _resolve(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return Path(__file__).resolve().parents[3] / resolved


def load_australia_stage1_fixture(
    path: str | Path = AUSTRALIA_STAGE1_FIXTURE,
) -> pd.DataFrame:
    """Load the committed Australia Stage 1 fixture."""

    candidates = [_resolve(path)]
    if Path(path) == AUSTRALIA_STAGE1_FIXTURE:
        candidates.append(_resolve(LEGACY_AUSTRALIA_STAGE1_FIXTURE))
    for candidate in candidates:
        if candidate.exists():
            return load_stage1_fixture_csv(str(candidate))
    missing = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find an Australia Stage 1 fixture at: {missing}")


def load_australia_stage1_live() -> pd.DataFrame:
    """Download live Australia source tables and rebuild the Stage 1 long table."""

    try:
        from fins2026.week2.code.australia_macro_panel import (
            build_stage1_long_table,
            download_live_rba_bundles,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Live Australia rebuild is unavailable in this standalone bundle "
            "because the optional RBA parser modules are not packaged here."
        ) from exc
    return build_stage1_long_table(download_live_rba_bundles())


def build_australia_monthly_panel(stage1: pd.DataFrame) -> pd.DataFrame:
    """Build the observable month-end panel used for monthly targets and context."""

    panel = build_observable_panel(stage1)
    panel = panel.loc[panel.index <= CLASSROOM_INFORMATION_SET_MONTH_END].copy()
    panel.index.name = "date"
    return panel


def build_australia_quarterly_panel(stage1: pd.DataFrame) -> pd.DataFrame:
    """Build the quarterly observable panel without post-release forward fills."""

    quarterly = stage1.loc[
        stage1["native_frequency"].isin(["quarterly", "point_in_time_quarterly"]),
        ["observable_month_end", "display_name", "value"],
    ].copy()
    panel = (
        quarterly.pivot_table(
            index="observable_month_end",
            columns="display_name",
            values="value",
            aggfunc="last",
        )
        .sort_index()
        .rename_axis(index="date")
    )
    return panel


def build_us_monthly_panel(*, use_fixture: bool) -> pd.DataFrame:
    """Build the U.S. month-end panel used as secondary context and exogenous input."""

    if use_fixture:
        rates = load_validation_dataset("fred_rates_daily").data[
            ["DGS10", "DGS2", "DTB3", "T10Y2Y"]
        ]
        stress = load_validation_dataset("fred_financial_stress_daily").data[["VIXCLS"]]
        sp500 = load_validation_dataset("fred_sp500_daily").data[["SP500"]]
        macro = load_validation_dataset("fred_macro_monthly").data[MONTHLY_MACRO_COLUMNS]
        daily_market = rates.join(stress, how="outer").join(sp500, how="outer")
    else:
        raw = clean_fred_graph_csv(read_fred_graph_csv(fred_graph_url(US_FRED_SERIES))).sort_index()
        daily_market = raw[DAILY_MARKET_COLUMNS]
        macro = raw[MONTHLY_MACRO_COLUMNS]
    panel = build_month_end_panel(daily_market, macro)
    panel.index.name = "date"
    return panel


def build_us_quarterly_panel(us_monthly: pd.DataFrame) -> pd.DataFrame:
    """Build a compact U.S. quarterly context panel from the month-end panel."""

    panel = pd.DataFrame(index=us_monthly.resample("QE").last().index)
    panel["DGS10"] = us_monthly["DGS10"].resample("QE").mean()
    panel["T10Y2Y"] = us_monthly["T10Y2Y"].resample("QE").mean()
    panel["FEDFUNDS"] = us_monthly["FEDFUNDS"].resample("QE").mean()
    panel["UNRATE"] = us_monthly["UNRATE"].resample("QE").mean()
    panel["VIXCLS"] = us_monthly["VIXCLS"].resample("QE").mean()
    panel["INDPRO"] = us_monthly["INDPRO"].resample("QE").last()
    panel["INDPRO_QUARTERLY_LOG_GROWTH_PCT"] = log_change_percent(panel["INDPRO"])
    panel["SP500"] = us_monthly["SP500"].resample("QE").last()
    panel["SP500_QUARTERLY_RETURN_PCT"] = panel["SP500"].pct_change() * 100.0
    panel.index.name = "date"
    return panel


def build_forecast_input_bundle(*, use_fixture: bool) -> dict[str, pd.DataFrame]:
    """Build the Australia and U.S. forecast input panels."""

    stage1 = load_australia_stage1_fixture() if use_fixture else load_australia_stage1_live()
    australia_monthly = build_australia_monthly_panel(stage1)
    australia_quarterly = build_australia_quarterly_panel(stage1)
    us_monthly = build_us_monthly_panel(use_fixture=use_fixture)
    us_quarterly = build_us_quarterly_panel(us_monthly)
    return {
        "australia_stage1": stage1,
        "australia_monthly": australia_monthly,
        "australia_quarterly": australia_quarterly,
        "us_monthly": us_monthly,
        "us_quarterly": us_quarterly,
    }


def load_forecast_input_bundle(
    *, use_fixture: bool, rebuild: bool = False
) -> dict[str, pd.DataFrame]:
    """Load saved forecast input panels when available, or rebuild them."""

    if rebuild:
        return build_forecast_input_bundle(use_fixture=use_fixture)
    results_dir = _resolve(DEFAULT_RESULTS_DIR)
    required = {
        "australia_monthly": results_dir / "australia_monthly_forecast_panel.csv",
        "australia_quarterly": results_dir / "australia_quarterly_forecast_panel.csv",
        "us_monthly": results_dir / "us_monthly_context_panel.csv",
        "us_quarterly": results_dir / "us_quarterly_context_panel.csv",
    }
    if not all(path.exists() for path in required.values()):
        return build_forecast_input_bundle(use_fixture=use_fixture)
    bundle = {
        "australia_stage1": load_australia_stage1_fixture(),
        "australia_monthly": pd.read_csv(
            required["australia_monthly"], parse_dates=["date"]
        ).set_index("date"),
        "australia_quarterly": pd.read_csv(
            required["australia_quarterly"], parse_dates=["date"]
        ).set_index("date"),
        "us_monthly": pd.read_csv(
            required["us_monthly"],
            parse_dates=["date"],
        ).set_index("date"),
        "us_quarterly": pd.read_csv(
            required["us_quarterly"],
            parse_dates=["date"],
        ).set_index("date"),
    }
    return bundle


def write_forecast_input_bundle(
    bundle: dict[str, pd.DataFrame],
    *,
    output_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> dict[str, Path]:
    """Write the forecast input bundle to results/data."""

    root = _resolve(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    stems = {
        "australia_monthly": "australia_monthly_forecast_panel",
        "australia_quarterly": "australia_quarterly_forecast_panel",
        "us_monthly": "us_monthly_context_panel",
        "us_quarterly": "us_quarterly_context_panel",
    }
    for key, stem in stems.items():
        path = root / f"{stem}.csv"
        frame = bundle[key].reset_index()
        frame.to_csv(path, index=False)
        outputs[key] = path
    stage1_path = root / "australia_stage1_observable_source.csv"
    bundle["australia_stage1"].to_csv(stage1_path, index=False)
    outputs["australia_stage1"] = stage1_path
    return outputs
