"""Week 3, part 2: basic forecasting with an AR(1) model and a basic ARIMA model.

This is the second half of the Week 3 walkthrough. It reads the cleaned monthly panel
that part 1 (01_rba_stage1_clean_transform_plot.py) saved, then builds two simple
forecasts of the Australian unemployment rate and checks them against a naive
benchmark. Run part 1 first so the panel file exists.

PyCharm shortcut note:
Settings -> Keymap -> Search for -> Execute Selection in Python Console
Change it to the shortcut you want, then run this file one numbered stage at a time.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ValueWarning

# Our dates are monthly. statsmodels can work this out on its own, but it prints a
# note every time it infers the monthly frequency. We hide that one harmless note so
# the printed output stays easy to read.
warnings.simplefilter("ignore", ValueWarning)

# Find the output folder the same way part 1 does, so this script works both when you
# run the whole file (the Play button defines __file__) and when you send a highlighted
# block to the Python Console (no __file__, so we fall back to the working directory).
# Both are absolute paths and never nest. Run part 1 first, the same way, so the panel
# file it saves is in the folder this script looks in.
THIS_SCRIPT_FOLDER = Path("fins2026") / "week3" / "scratch" / "macro_forecasting"
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    # No __file__ (highlighted selection in the console). PyCharm starts the console at
    # the project root, so step into this script's folder when we can find it there, so
    # part 1 and part 2 use the SAME folder whichever way you run them.
    guess = Path.cwd() / THIS_SCRIPT_FOLDER
    BASE_DIR = guess if guess.is_dir() else Path.cwd()
OUTPUT_DIR = BASE_DIR / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
print(f"Reading inputs from: {OUTPUT_DIR}")


# -----------------------------------------------------------------------------
# 1. Load the data, pick the target, and split into in-sample and out-of-sample
# -----------------------------------------------------------------------------

# Read the cleaned monthly panel saved by part 1. We try the Parquet file first and
# fall back to the CSV, so this still works on a minimal Python install without a
# Parquet engine. If neither file is there, part 1 has not been run (the same way),
# so we say so clearly instead of showing a confusing error.
panel_parquet = OUTPUT_DIR / "au_monthly_panel.parquet"
panel_csv = OUTPUT_DIR / "au_monthly_panel.csv"
try:
    panel = pd.read_parquet(panel_parquet)
except (FileNotFoundError, ImportError, ValueError):
    if panel_csv.exists():
        panel = pd.read_csv(panel_csv, index_col=0, parse_dates=True)
    else:
        raise SystemExit(
            f"Could not find the monthly panel in {OUTPUT_DIR}.\n"
            f"Run 01_rba_stage1_clean_transform_plot.py first, the SAME way you are "
            f"running this script (whole file, or console), then run this again."
        )

# The level is the unemployment rate itself. The target we forecast is the monthly
# CHANGE in the rate, because part 1 showed the level has a unit root and the change
# is stationary. Forecasting the change is the safe choice.
level = panel["Unemployment rate"].dropna()
change = level.diff().dropna()

# Split the data in time. We fit the models on data up to the end of 2019, then test
# them on everything from 2020 onward, which the models never saw while fitting. This
# out-of-sample test judges a forecast on data it has not seen.
SPLIT_DATE = "2019-12-31"
oos_dates = change.index[change.index > SPLIT_DATE]
first_oos_position = change.index.get_loc(oos_dates[0])

print(f"Target: monthly change in the unemployment rate")
print(f"In-sample:  {change.index.min():%b %Y} to {SPLIT_DATE}")
print(f"Out-of-sample: {oos_dates.min():%b %Y} to {oos_dates.max():%b %Y} "
      f"({len(oos_dates)} months)")

# The actual changes over the out-of-sample window. Everything is scored against these.
actual_change = change.loc[oos_dates]


def score(name, forecast_change):
    """Print and return MAE, RMSE, and the out-of-sample R-squared.

    The out-of-sample R-squared compares the model's squared errors with those of the
    naive benchmark, whose forecast is zero change. It is
        R2_oos = 1 - sum((actual - forecast)^2) / sum((actual - 0)^2).
    A positive value means the model beats the naive forecast; zero means it ties it;
    a negative value means it does worse than simply predicting no change.
    """
    error = actual_change - forecast_change
    mae = error.abs().mean()
    rmse = np.sqrt((error ** 2).mean())
    r2_oos = 1 - (error ** 2).sum() / (actual_change ** 2).sum()
    print(f"  {name:<18} MAE = {mae:.3f}   RMSE = {rmse:.3f}   R2_oos = {r2_oos:+.3f}")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2_oos": r2_oos}


# -----------------------------------------------------------------------------
# 2. The naive benchmark and an AR(1) model
# -----------------------------------------------------------------------------

# The naive benchmark is the forecast every model must beat: it says next month's
# change will be zero (the unemployment rate will be the same as this month). It is
# the random walk, and for macro data it is surprisingly hard to beat.
naive_change = pd.Series(0.0, index=oos_dates)

# An AR(1) model says this month's change depends on last month's change:
#   change_t = constant + phi * change_(t-1) + noise.
# In statsmodels an AR(1) on the change is ARIMA with order (1, 0, 0).
ar1_fit = ARIMA(change.loc[:SPLIT_DATE], order=(1, 0, 0)).fit()
print("\nAR(1) coefficient on last month's change:")
print(f"  phi = {ar1_fit.arparams[0]:.3f}")

# Apply the fitted AR(1) to the full change series and read off one-step-ahead
# forecasts for the out-of-sample months. One-step-ahead means each forecast uses the
# real data up to the month before, which is what a forecaster has in practice.
ar1_full = ar1_fit.apply(change)
ar1_change = ar1_full.get_prediction(start=first_oos_position).predicted_mean
ar1_change = ar1_change.loc[oos_dates]

print("\nForecast accuracy in target (change) space:")
results = [score("Naive", naive_change), score("AR(1)", ar1_change)]

# Plot the actual change against the naive and AR(1) forecasts.
plt.figure(figsize=(11, 6))
plt.axhline(0.0, color="#6B625C", linewidth=1.0, label="Naive (no change)")
plt.plot(actual_change.index, actual_change.values, color="#262A33",
         linewidth=1.6, label="Actual change")
plt.plot(ar1_change.index, ar1_change.values, color="#0F5499",
         linewidth=1.6, label="AR(1) forecast")
plt.title("Out-of-sample forecast of the monthly change in unemployment")
plt.xlabel("Date")
plt.ylabel("Change in unemployment rate (percentage points)")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "05_ar1_change_forecast.png", dpi=150)
plt.show()
plt.close()


# -----------------------------------------------------------------------------
# 3. A basic ARIMA model with a forecast fan
# -----------------------------------------------------------------------------

# ARIMA has three numbers, written ARIMA(p, d, q):
#   p = how many past values of the series the model uses (autoregressive terms)
#   d = how many times we difference the series to make it stationary
#   q = how many past forecast errors the model uses (moving-average terms)
# We fit ARIMA(1, 1, 1) on the LEVEL. The d = 1 does the differencing for us, which
# matches the unit-root finding from part 1, so the model works on the change inside.
arima_fit = ARIMA(level.loc[:SPLIT_DATE], order=(1, 1, 1)).fit()

# One-step-ahead forecasts of the level over the out-of-sample window, with a
# confidence interval we can draw as a shaded fan.
arima_full = arima_fit.apply(level)
level_oos_dates = level.index[level.index > SPLIT_DATE]
arima_level_pred = arima_full.get_prediction(start=level.index.get_loc(level_oos_dates[0]))
level_forecast = arima_level_pred.predicted_mean.loc[level_oos_dates]
level_ci = arima_level_pred.conf_int().loc[level_oos_dates]

# Mean absolute percentage error (MAPE) of the LEVEL forecast. MAPE only makes sense for
# a series that stays well away from zero, like the unemployment rate (around 4-7%), so
# we report it here for the level, not for the change (which crosses zero).
level_actual = level.loc[level_oos_dates]
arima_level_mape = (((level_actual - level_forecast).abs()) / level_actual.abs()).mean() * 100
print(f"\nARIMA one-step-ahead LEVEL forecast: MAPE = {arima_level_mape:.2f}%")

# Pretty FT-style fan plot: actual level, ARIMA forecast, and the uncertainty band.
plt.rcParams.update({
    "figure.facecolor": "#FDF1E6", "axes.facecolor": "#FDF1E6",
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.edgecolor": "#66605C", "axes.grid": True, "grid.color": "#E2D8CF",
    "axes.axisbelow": True, "font.family": "DejaVu Sans", "font.size": 12,
})
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(level.loc["2017":].index, level.loc["2017":].values,
        color="#262A33", linewidth=1.8, label="Actual")
ax.plot(level_forecast.index, level_forecast.values,
        color="#990F3D", linewidth=1.8, label="ARIMA(1,1,1) forecast")
ax.fill_between(level_ci.index, level_ci.iloc[:, 0], level_ci.iloc[:, 1],
                color="#990F3D", alpha=0.15, label="95% interval")
ax.grid(axis="x", visible=False)
fig.text(0.012, 0.96, "ARIMA one-step-ahead forecast of unemployment",
         fontsize=15, fontweight="bold", color="#262A33")
fig.text(0.012, 0.91, "Per cent, out-of-sample from 2020", fontsize=11, color="#6B625C")
fig.text(0.012, 0.01, "Source: RBA table H5", fontsize=8, color="#6B625C")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86))
fig.subplots_adjust(top=0.86, bottom=0.10)
fig.savefig(FIGURE_DIR / "06_arima_level_forecast.png", dpi=150)
plt.show()
plt.close()
plt.rcParams.update(plt.rcParamsDefault)

# Convert the ARIMA level forecast into a change forecast so we can score it in the
# same target space as the other models: forecast change = forecast level minus the
# actual level the month before.
arima_change = level_forecast - level.shift(1).loc[level_oos_dates]
arima_change = arima_change.loc[oos_dates]


# -----------------------------------------------------------------------------
# 4. Score every model and pick a winner
# -----------------------------------------------------------------------------

print("\nFinal leaderboard in target (change) space, lower is better:")
results = [
    score("Naive", naive_change),
    score("AR(1)", ar1_change),
    score("ARIMA(1,1,1)", arima_change),
]
leaderboard = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
print("\n" + leaderboard.to_string(index=False))

best = leaderboard.iloc[0]["model"]
naive_rmse = leaderboard.set_index("model").loc["Naive", "RMSE"]
best_rmse = leaderboard.iloc[0]["RMSE"]
if best == "Naive":
    print("\nThe naive benchmark wins: the simple models do not beat a random walk here.")
else:
    improvement = (1 - best_rmse / naive_rmse) * 100
    print(f"\n{best} has the lowest RMSE, about {improvement:.0f}% below the naive benchmark.")

# The bigger lesson: all three models are close together. The forecast line in the
# figure hugs the actual line because a one-step-ahead forecast starts from last
# month's real value, but the leaderboard shows it barely beats "no change at all".
# The out-of-sample R-squared makes this precise: it is near zero for every model.
print("\nLesson: a forecast that looks close to the actual line can still barely beat")
print("the naive benchmark. A small positive out-of-sample R-squared (a few per cent)")
print("is a real but tiny improvement; a negative one means the model does worse than")
print("predicting no change at all.")


# -----------------------------------------------------------------------------
# 5. Run the same forecast for the other series
# -----------------------------------------------------------------------------

# The workflow is not specific to unemployment. We run the same ARIMA(1,1,1)
# one-step-ahead forecast for every series and show them together. Some series cannot
# use this split: monthly CPI only starts in 2025, so it has no in-sample data before
# the end of 2019. We label those panels instead of forcing a fit on too few points.

def one_step_arima(level_series):
    """Fit ARIMA(1,1,1) on the in-sample part and forecast the out-of-sample months.

    Returns (forecast, lower, upper) over the out-of-sample window, or None when there
    is not enough data on each side of the split to fit and test the model.
    """
    level_series = level_series.dropna()
    in_sample = level_series.loc[:SPLIT_DATE]
    oos = level_series.index[level_series.index > SPLIT_DATE]
    if len(in_sample) < 60 or len(oos) < 12:
        return None
    fit = ARIMA(in_sample, order=(1, 1, 1)).fit()
    prediction = fit.apply(level_series).get_prediction(
        start=level_series.index.get_loc(oos[0]))
    band = prediction.conf_int().loc[oos]
    return prediction.predicted_mean.loc[oos], band.iloc[:, 0], band.iloc[:, 1]


# Use the same FT style as part 1 for the grid of forecasts.
plt.rcParams.update({
    "figure.facecolor": "#FDF1E6", "axes.facecolor": "#FDF1E6",
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.edgecolor": "#66605C", "axes.grid": True, "grid.color": "#E2D8CF",
    "axes.axisbelow": True, "font.family": "DejaVu Sans", "font.size": 11,
})
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, name in zip(axes.flat, panel.columns):
    actual = panel[name].dropna().loc["2016":]  # show recent history only
    ax.plot(actual.index, actual.values, color="#262A33", linewidth=1.6, label="Actual")
    ax.set_title(name, loc="left", fontweight="bold", color="#262A33", fontsize=11)
    ax.grid(axis="x", visible=False)
    result = one_step_arima(panel[name])
    if result is None:
        ax.text(0.5, 0.5, "history too short\nto forecast", transform=ax.transAxes,
                ha="center", va="center", color="#6B625C")
        continue
    forecast, lower, upper = result
    ax.plot(forecast.index, forecast.values, color="#990F3D", linewidth=1.6,
            label="ARIMA forecast")
    ax.fill_between(forecast.index, lower, upper, color="#990F3D", alpha=0.15)
axes.flat[0].legend(loc="upper left", fontsize=9)
fig.suptitle("ARIMA(1,1,1) one-step-ahead forecasts, all series", x=0.012, ha="left",
             fontsize=15, fontweight="bold", color="#262A33")
fig.text(0.012, 0.005, "Source: RBA tables H5, G4, I2, F11 | out-of-sample from 2020",
         fontsize=8, color="#6B625C")
fig.tight_layout(rect=(0, 0.02, 1, 0.96))
fig.savefig(FIGURE_DIR / "07_all_series_arima_forecasts_ft.png", dpi=150)
plt.show()
plt.close()
plt.rcParams.update(plt.rcParamsDefault)

print("\nSaved figures")
print(FIGURE_DIR / "05_ar1_change_forecast.png")
print(FIGURE_DIR / "06_arima_level_forecast.png")
print(FIGURE_DIR / "07_all_series_arima_forecasts_ft.png")
