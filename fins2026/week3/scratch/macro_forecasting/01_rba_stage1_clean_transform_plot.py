"""Week 3, part 1: read Australian macro data, clean it, transform it, and plot it.

This script is the first half of the Week 3 forecasting walkthrough. It covers
Steps 1 to 4: read the RBA data, clean it and build two date columns, transform the
data and explain the unit-root idea, and draw a standard (ugly) plot next to a
Financial Times (FT) style plot. It saves a cleaned monthly panel that part 2
(02_forecasting_ar1_arima.py) reads in for the forecasting models.

PyCharm shortcut note:
Settings -> Keymap -> Search for -> Execute Selection in Python Console
Change it to the shortcut you want for running a line or a selected block. Then you
can run this file one numbered stage at a time and read the printed output as you go.
"""

import csv
import io
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
from statsmodels.tsa.stattools import adfuller

# Where to save outputs. This has to work two ways:
#   1. Running the whole file (the green Play button). Python defines the variable
#      __file__, so we save next to this script.
#   2. Sending a highlighted block to the Python Console (Alt+Shift+E). The console
#      does NOT define __file__, so we fall back to the current working directory
#      (the folder shown in the console prompt, normally the project root).
# Both are absolute paths, so neither one builds a nested, duplicated folder tree, and
# both behave the same on Mac and Windows.
# Tip: run a script the same way from start to finish, so part 1 and part 2 use the
# same output folder.
THIS_SCRIPT_FOLDER = Path("fins2026") / "week3" / "scratch" / "macro_forecasting"
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    # No __file__ (highlighted selection in the console). PyCharm starts the console at
    # the project root, so step into this script's folder when we can find it there.
    # That way the whole-file run and the console run use the SAME output folder.
    guess = Path.cwd() / THIS_SCRIPT_FOLDER
    BASE_DIR = guess if guess.is_dir() else Path.cwd()
OUTPUT_DIR = BASE_DIR / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
print(f"Saving outputs to: {OUTPUT_DIR}")


# -----------------------------------------------------------------------------
# 1. Read the RBA data (Stage 1 of the Data Factory Floor)
# -----------------------------------------------------------------------------

# The Reserve Bank of Australia (RBA) publishes its statistical tables as CSV files.
# Each table has a fixed web address. We download a few monthly tables here.
RBA_CSV_URLS = {
    "H5": "https://www.rba.gov.au/statistics/tables/csv/h5-data.csv",  # labour force
    "G4": "https://www.rba.gov.au/statistics/tables/csv/g4-data.csv",  # monthly CPI
    "I2": "https://www.rba.gov.au/statistics/tables/csv/i2-data.csv",  # commodity prices
    "F11": "https://www.rba.gov.au/statistics/tables/csv/f11-data.csv",  # exchange rate
}

# The series we want. Each row is:
#   (series_id, table_code, display_name, units, release_lag_months)
# release_lag_months is how many months after the period ends before the RBA
# publishes the number. Unemployment and monthly CPI arrive about one month late.
SERIES_TO_DOWNLOAD = [
    ("GLFSURSA", "H5", "Unemployment rate", "Per cent", 1),
    # Australia's monthly headline CPI is a brand-new series. The ABS only began the
    # full monthly CPI collection in 2024, so this year-ended rate starts in 2025 and
    # has only about a year of history. Quarterly CPI (RBA table G1) goes back decades,
    # but there is no long-history MONTHLY headline CPI for Australia yet.
    ("GCPIAGYPM", "G4", "Monthly headline CPI inflation", "Per cent", 1),
    ("GRCPAIAD", "I2", "Commodity price index (A$)", "Index", 0),
    ("FXRTWI", "F11", "Trade-weighted index", "Index", 0),
]


def download_rba_series(url, series_id):
    """Download one RBA CSV table and return a tidy date/value frame for one series.

    An RBA CSV starts with about ten rows of header information (title, units,
    frequency, and the series ID codes), then the actual numbers. We find the header
    row that names our series, read the column it sits in, and collect every
    (date, value) pair in the rows below it.
    """
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    rows = list(csv.reader(io.StringIO(response.text)))

    # Find the header row that contains our series ID, and which column it is in.
    header_index = next(i for i, row in enumerate(rows) if series_id in row)
    column = rows[header_index].index(series_id)

    records = []
    for row in rows[header_index + 1:]:
        if not row or not row[0].strip() or column >= len(row):
            continue
        date_text = row[0].strip()
        # RBA dates are written day-first when they use slashes (e.g. 30/06/2024).
        day_first = "/" in date_text
        date = pd.to_datetime(date_text, dayfirst=day_first, errors="coerce")
        value = pd.to_numeric(row[column], errors="coerce")
        records.append({"date": date, "value": value})

    frame = pd.DataFrame(records).dropna().sort_values("date").reset_index(drop=True)
    return frame


# Download each series and keep the raw frames in a dictionary keyed by series ID.
raw_series = {}
for series_id, table_code, display_name, units, lag in SERIES_TO_DOWNLOAD:
    raw_series[series_id] = download_rba_series(RBA_CSV_URLS[table_code], series_id)
    print(f"Downloaded {series_id} ({display_name}): {len(raw_series[series_id]):,} rows")

print("\nUnemployment rate, most recent rows")
print(raw_series["GLFSURSA"].tail())


# -----------------------------------------------------------------------------
# 2. Clean the data and build the date columns
# -----------------------------------------------------------------------------

# THIS IS A KEY IDEA. A macro number has more than one date, and mixing them up causes
# look-ahead bias (using information before you could really have known it). We keep
# three dates for every observation:
#   raw_date       = the date exactly as it appears in the RBA file.
#   reference_date = the month the number describes (the period the data is over),
#                    the raw date snapped to the end of its month.
#   release_date   = the month the public can first see the number (reference_date
#                    plus the publication lag in months).
# Example: the December unemployment rate describes December (reference_date), but the
# RBA only releases it in January (release_date). If you use the December number as if
# you knew it in December, you are using information you did not actually have yet.

long_frames = []
for series_id, table_code, display_name, units, lag in SERIES_TO_DOWNLOAD:
    frame = raw_series[series_id].copy()
    # Keep the original file date under a clear name.
    frame = frame.rename(columns={"date": "raw_date"})
    # Snap the raw date to the end of its month so every monthly series lines up.
    frame["reference_date"] = frame["raw_date"].dt.to_period("M").dt.to_timestamp("M")
    # The release date is the reference month-end shifted forward by the lag.
    frame["release_date"] = frame["reference_date"] + pd.offsets.MonthEnd(lag)
    frame["series_id"] = series_id
    frame["display_name"] = display_name
    frame["units"] = units
    long_frames.append(
        frame[["series_id", "display_name", "units",
               "raw_date", "reference_date", "release_date", "value"]]
    )

macro_long = pd.concat(long_frames, ignore_index=True)
macro_long = macro_long.sort_values(["series_id", "reference_date"]).reset_index(drop=True)

# Basic Stage 1 checks: how many duplicates and missing values do we have?
duplicate_rows = macro_long.duplicated(subset=["series_id", "reference_date"])
print("\nStage 1 checks")
print(f"Rows: {len(macro_long):,}")
print(f"Duplicate series-month rows: {duplicate_rows.sum():,}")
print("Missing values by column:")
print(macro_long.isna().sum())

# Look at the unemployment rate, where the one-month release lag is easy to see:
# reference_date is the month described, release_date is one month later.
print("\nLong table preview: note reference_date vs release_date (one month apart here)")
print(macro_long[macro_long["series_id"] == "GLFSURSA"].tail())

# Build a wide monthly panel: one row per month, one column per series. This is the
# convenient shape for plotting and for the forecasting models in part 2.
monthly_panel = macro_long.pivot_table(
    index="reference_date",
    columns="display_name",
    values="value",
    aggfunc="last",
).sort_index()
monthly_panel.index.name = "reference_date"

print("\nWide monthly panel preview (organised by reference_date)")
print(monthly_panel.tail())


# -----------------------------------------------------------------------------
# 3. Transform the data and the unit-root idea
# -----------------------------------------------------------------------------

# Why we do not forecast the level directly:
# Many macro series have a "unit root". A series with a unit root wanders up and down
# with no fixed mean to pull it back, so its level is very hard to forecast and
# regressions run on levels can show strong but fake relationships (spurious
# regression). The standard fix is to forecast the CHANGE from one month to the next
# (the first difference), which usually has a stable mean and is easier to model.

# The Augmented Dickey-Fuller (ADF) test checks for a unit root.
#   - A small p-value (below 0.05) means we reject the unit root: the series is
#     stationary and safe to model directly.
#   - A large p-value means we cannot reject the unit root: transform it first.

# We test every series, not just unemployment. For each one we run the ADF test on the
# level; if it has a unit root we transform it to make it stationary and test again to
# confirm the transform worked. The right transform depends on the series:
#   - a rate measured in per cent (unemployment, inflation) -> first difference, the
#     month-to-month change in percentage points;
#   - a price index (commodity prices, the exchange rate) -> monthly percentage change.
SERIES_ORDER = [display_name for _, _, display_name, _, _ in SERIES_TO_DOWNLOAD]
UNITS = {display_name: units for _, _, display_name, units, _ in SERIES_TO_DOWNLOAD}


def transform_series(name, series):
    """Return (transformed_series, short_label) using the right transform for a series."""
    if UNITS[name] == "Per cent":
        # A rate: take the month-to-month change in percentage points.
        return series.diff(), "change (pp)"
    # A price index: take the monthly percentage change.
    return series.pct_change() * 100, "monthly % change"


transformed = {}        # series name -> its stationary, transformed version
transform_labels = {}   # series name -> short label describing the transform
for name in SERIES_ORDER:
    level_series = monthly_panel[name].dropna()
    level_p = adfuller(level_series)[1]
    has_unit_root = level_p > 0.05
    print(f"\n{name}")
    # Warn when a series is too short for the test to mean much. Monthly CPI is the
    # case here: it only started in 2025, so treat its result as indicative only.
    if len(level_series) < 36:
        print(f"  NOTE: only {len(level_series)} months of data, this series started "
              f"recently, so read the test below as indicative only")
    print(f"  ADF on the level: p-value = {level_p:.3f} -> "
          f"{'unit root, needs transforming' if has_unit_root else 'already stationary'}")
    if has_unit_root:
        change_series, label = transform_series(name, level_series)
        change_series = change_series.dropna()
        change_p = adfuller(change_series)[1]
        print(f"  ADF on the {label}: p-value = {change_p:.3f} -> "
              f"{'stationary, safe to model' if change_p <= 0.05 else 'still not stationary'}")
        transformed[name] = change_series
        transform_labels[name] = label

# The unemployment level is used again for the hero chart below and in part 2.
unemployment_level = monthly_panel["Unemployment rate"].dropna()


# -----------------------------------------------------------------------------
# 4. A standard (ugly) plot next to a Financial Times (FT) style plot
# -----------------------------------------------------------------------------

# We plot all four series, not just one. The series are on very different scales
# (a rate in per cent, an index near 100), so a single shared chart would be unreadable.
# Small multiples solve this: a grid of small charts, one per series, each with its own
# y-axis. We draw the grid twice, first in the plain default style and then in the FT
# style, so the difference is easy to see side by side.

# First the plain default grid. It works, but it is busy: heavy spines, a boxed-in
# look, and no clear final value.
plt.rcParams.update(plt.rcParamsDefault)  # make sure we start from the plain defaults

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, name in zip(axes.flat, SERIES_ORDER):
    series = monthly_panel[name].dropna()
    ax.plot(series.index, series.values)
    ax.set_title(name)
    ax.set_ylabel(UNITS[name])
fig.suptitle("All four series, standard matplotlib defaults")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "01_all_series_plain.png", dpi=150)
plt.show()
plt.close()


def apply_ft_style():
    """Set a few rcParams so the next figure follows a clean FT-style look."""
    plt.rcParams.update({
        "figure.facecolor": "#FDF1E6",   # warm off-white background
        "axes.facecolor": "#FDF1E6",
        "axes.spines.top": False,        # drop the top and right border lines
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.edgecolor": "#66605C",
        "axes.grid": True,
        "grid.color": "#E2D8CF",         # faint horizontal gridlines only
        "axes.axisbelow": True,
        "font.family": "DejaVu Sans",
        "font.size": 11,
    })


FT_MAROON = "#990F3D"  # the Financial Times signature colour
# One colour per panel, so each series is easy to tell apart.
FT_COLOURS = ["#990F3D", "#0F5499", "#0F766E", "#D56F3E"]

# Now the same four series in the FT style: off-white background, no top/right spines,
# faint horizontal gridlines, a clear final value labelled on each line.
apply_ft_style()
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, name, colour in zip(axes.flat, SERIES_ORDER, FT_COLOURS):
    series = monthly_panel[name].dropna()
    ax.plot(series.index, series.values, color=colour, linewidth=1.8)
    ax.grid(axis="x", visible=False)  # keep horizontal gridlines only
    ax.set_title(name, loc="left", fontweight="bold", color="#262A33")
    # Label the final value directly on the line instead of using a legend.
    ax.annotate(f"  {series.iloc[-1]:.1f}", xy=(series.index[-1], series.iloc[-1]),
                color=colour, fontweight="bold", va="center", fontsize=10)
fig.suptitle("All four series, Financial Times style", x=0.012, ha="left",
             fontsize=15, fontweight="bold", color="#262A33")
fig.text(0.012, 0.005, "Source: RBA tables H5, G4, I2, F11", fontsize=8, color="#6B625C")
fig.tight_layout(rect=(0, 0.02, 1, 0.96))
fig.savefig(FIGURE_DIR / "02_all_series_ft.png", dpi=150)
plt.show()
plt.close()

# Now the transformed series for every variable that had a unit root. These are the
# stationary series we would actually model. Notice they wander around a flat level
# near zero instead of trending, which is what stationary means.
unit_root_names = list(transformed.keys())
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, name, colour in zip(axes.flat, unit_root_names, FT_COLOURS):
    series = transformed[name]
    ax.plot(series.index, series.values, color=colour, linewidth=1.2)
    ax.axhline(0.0, color="#66605C", linewidth=0.8)  # the flat level it varies around
    ax.grid(axis="x", visible=False)
    ax.set_title(f"{name} -- {transform_labels[name]}", loc="left",
                 fontweight="bold", color="#262A33", fontsize=11)
# Hide any unused panels if there are fewer than four unit-root series.
for ax in axes.flat[len(unit_root_names):]:
    ax.set_visible(False)
fig.suptitle("After transforming: the stationary series we forecast", x=0.012,
             ha="left", fontsize=15, fontweight="bold", color="#262A33")
fig.text(0.012, 0.005, "Source: RBA tables H5, G4, I2, F11", fontsize=8, color="#6B625C")
fig.tight_layout(rect=(0, 0.02, 1, 0.96))
fig.savefig(FIGURE_DIR / "03_all_series_transformed_ft.png", dpi=150)
plt.show()
plt.close()

# Finally a single FT "hero" chart of the unemployment rate, the series we forecast in
# part 2. A single series gets a full title block and a direct endpoint label.
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(unemployment_level.index, unemployment_level.values,
        color=FT_MAROON, linewidth=2.0)
ax.grid(axis="x", visible=False)
ax.annotate(f"  {unemployment_level.iloc[-1]:.1f}%",
            xy=(unemployment_level.index[-1], unemployment_level.iloc[-1]),
            color=FT_MAROON, fontweight="bold", va="center")
fig.text(0.012, 0.96, "Australia's unemployment rate", fontsize=15,
         fontweight="bold", color="#262A33")
fig.text(0.012, 0.91, "Seasonally adjusted, per cent", fontsize=11, color="#6B625C")
fig.text(0.012, 0.01,
         f"Source: RBA table H5 | Sample: {unemployment_level.index.min():%b %Y} "
         f"to {unemployment_level.index.max():%b %Y}",
         fontsize=8, color="#6B625C")
ax.set_xlabel("")
ax.set_ylabel("")
fig.subplots_adjust(top=0.86, bottom=0.10)
fig.savefig(FIGURE_DIR / "04_unemployment_ft.png", dpi=150)
plt.show()
plt.close()

plt.rcParams.update(plt.rcParamsDefault)  # reset so later runs start clean


# -----------------------------------------------------------------------------
# 5. Save the cleaned data for the forecasting script
# -----------------------------------------------------------------------------

# We save two files, because they answer two different questions.
#
# (1) The LONG table is the Stage 1 record. It keeps one row per series per month and,
#     importantly, BOTH the reference_date and the release_date for every observation.
#     This is where the timing of the data lives, so it is the file to look at when you
#     need to know when a number was actually available. A wide panel cannot store the
#     release dates, because each series has its own publication lag.
def save_both(frame, csv_path, parquet_path, **to_csv_kwargs):
    """Always save the CSV. Try the Parquet file too, but do not crash if the Parquet
    engine is missing on a minimal Python install -- the CSV is enough for part 2."""
    frame.to_csv(csv_path, **to_csv_kwargs)
    try:
        frame.to_parquet(parquet_path, index=to_csv_kwargs.get("index", True))
    except Exception as exc:  # e.g. no pyarrow/fastparquet installed
        print(f"  (skipped {parquet_path.name}: {exc}; the CSV is saved and is enough)")


LONG_CSV = OUTPUT_DIR / "au_macro_long.csv"
LONG_PARQUET = OUTPUT_DIR / "au_macro_long.parquet"
save_both(macro_long, LONG_CSV, LONG_PARQUET, index=False)

# (2) The wide PANEL is the convenient shape for plotting and for the forecasting
#     models in part 2: one row per reference month, one column per series. It is
#     organised by reference_date (its index). The changes are easy to recompute with
#     .diff() in part 2, so we save the levels only.
PANEL_CSV = OUTPUT_DIR / "au_monthly_panel.csv"
PANEL_PARQUET = OUTPUT_DIR / "au_monthly_panel.parquet"
save_both(monthly_panel, PANEL_CSV, PANEL_PARQUET)

print("\nSaved the Stage 1 long table (keeps both reference_date and release_date)")
print(LONG_CSV)
print(LONG_PARQUET)
print("\nSaved the wide monthly panel (organised by reference_date)")
print(PANEL_CSV)
print(PANEL_PARQUET)
print("\nSaved figures")
print(FIGURE_DIR / "01_all_series_plain.png")
print(FIGURE_DIR / "02_all_series_ft.png")
print(FIGURE_DIR / "03_all_series_transformed_ft.png")
print(FIGURE_DIR / "04_unemployment_ft.png")
