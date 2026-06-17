# ruff: noqa
"""Week 3, even simpler: forecast the Bitcoin daily return with AR(1) and ARIMA.

This is the second of two scripts. It reads the cleaned Bitcoin data that part 1
(01_btc_price_returns_plot.py) saved, then tries to forecast the daily return and checks
the result against a naive benchmark. Run part 1 first, the same way you run this script.

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

# Our data is daily. statsmodels prints a note when it infers the daily frequency; we
# hide that one harmless note so the printed output stays easy to read.
warnings.simplefilter("ignore", ValueWarning)

# Same output folder logic as part 1, so this works whether you run the whole file (the
# Play button defines __file__) or send a highlighted block to the Python Console (no
# __file__, so we fall back to the working directory). Both are absolute and never nest.
THIS_SCRIPT_FOLDER = Path("fins2026") / "week3" / "scratch" / "crypto_forecasting"
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:  # no __file__ when running a highlighted selection in the console
    guess = Path.cwd() / THIS_SCRIPT_FOLDER
    BASE_DIR = guess if guess.is_dir() else Path.cwd()
OUTPUT_DIR = BASE_DIR / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
print(f"Reading inputs from: {OUTPUT_DIR}")


# -----------------------------------------------------------------------------
# 1. Load the data, pick the target, and split into in-sample and out-of-sample
# -----------------------------------------------------------------------------

# Read the cleaned Bitcoin data saved by part 1. We try the Parquet file first and fall
# back to the CSV, so this still works on a minimal Python install without a Parquet
# engine. If neither file is there, part 1 has not been run (the same way), so we say so.
btc_parquet = OUTPUT_DIR / "btc_daily.parquet"
btc_csv = OUTPUT_DIR / "btc_daily.csv"
try:
    btc = pd.read_parquet(btc_parquet)
except (FileNotFoundError, ImportError, ValueError):
    if btc_csv.exists():
        btc = pd.read_csv(btc_csv, index_col=0, parse_dates=True)
    else:
        raise SystemExit(
            f"Could not find btc_daily in {OUTPUT_DIR}.\n"
            f"Run 01_btc_price_returns_plot.py first, the SAME way you are running this "
            f"script (whole file, or console), then run this again."
        )

# The price has a unit root, so we forecast the daily return, which is stationary.
price = btc["close"].dropna()
daily_return = btc["return"].dropna()

# Split the data in time. We fit on data up to the end of 2023, then test on 2024 onward,
# which the models never saw while fitting. This out-of-sample test is how we judge a
# forecast on data it has not seen.
SPLIT_DATE = "2023-12-31"
oos_dates = daily_return.index[daily_return.index > SPLIT_DATE]
first_oos_position = daily_return.index.get_loc(oos_dates[0])

print(f"Target: daily return of {price.name}")
print(f"In-sample:  {daily_return.index.min():%Y-%m-%d} to {SPLIT_DATE}")
print(f"Out-of-sample: {oos_dates.min():%Y-%m-%d} to {oos_dates.max():%Y-%m-%d} "
      f"({len(oos_dates)} days)")

actual_return = daily_return.loc[oos_dates]


def score(name, forecast_return):
    """Print and return MAE, RMSE, and the out-of-sample R-squared.

    The out-of-sample R-squared compares the model's squared errors with those of the
    naive benchmark, whose forecast is a zero return. It is
        R2_oos = 1 - sum((actual - forecast)^2) / sum((actual - 0)^2).
    A positive value means the model beats the naive forecast; zero means it ties it;
    a negative value means it does worse than simply predicting a zero return.
    """
    error = actual_return - forecast_return
    mae = error.abs().mean()
    rmse = np.sqrt((error ** 2).mean())
    r2_oos = 1 - (error ** 2).sum() / (actual_return ** 2).sum()
    print(f"  {name:<18} MAE = {mae:.3f}   RMSE = {rmse:.3f}   R2_oos = {r2_oos:+.4f}")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2_oos": r2_oos}


# -----------------------------------------------------------------------------
# 2. The naive benchmark and an AR(1) model
# -----------------------------------------------------------------------------

# The naive benchmark is the forecast every model must beat: it says tomorrow's return
# will be zero (the price will be the same tomorrow). For a market price this is very
# hard to beat.
naive_return = pd.Series(0.0, index=oos_dates)

# An AR(1) model says today's return depends on yesterday's return. On the return series,
# AR(1) is ARIMA with order (1, 0, 0).
ar1_fit = ARIMA(daily_return.loc[:SPLIT_DATE], order=(1, 0, 0)).fit()
print("\nAR(1) coefficient on yesterday's return:")
print(f"  phi = {ar1_fit.arparams[0]:.3f}")

# Apply the fitted AR(1) to the full return series and read off one-step-ahead forecasts
# for the out-of-sample days. One-step-ahead means each forecast uses the real data up to
# the day before.
ar1_full = ar1_fit.apply(daily_return)
ar1_return = ar1_full.get_prediction(start=first_oos_position).predicted_mean.loc[oos_dates]

print("\nForecast accuracy on the daily return:")
score("Naive", naive_return)
score("AR(1)", ar1_return)

# Plot the actual return against the naive and AR(1) forecasts.
plt.figure(figsize=(11, 6))
plt.axhline(0.0, color="#6B625C", linewidth=1.0, label="Naive (zero return)")
plt.plot(actual_return.index, actual_return.values, color="#262A33",
         linewidth=0.7, label="Actual return")
plt.plot(ar1_return.index, ar1_return.values, color="#0F5499",
         linewidth=1.2, label="AR(1) forecast")
plt.title("Out-of-sample forecast of the Bitcoin daily return")
plt.xlabel("Date")
plt.ylabel("Daily return (%)")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "04_btc_ar1_returns_forecast.png", dpi=150)
plt.show()
plt.close()


# -----------------------------------------------------------------------------
# 3. A basic ARIMA model with a forecast fan
# -----------------------------------------------------------------------------

# ARIMA has three numbers, written ARIMA(p, d, q):
#   p = how many past values the model uses (autoregressive terms)
#   d = how many times we difference the series to make it stationary
#   q = how many past forecast errors the model uses (moving-average terms)
# We fit ARIMA(1, 1, 1) on the PRICE. The d = 1 differences the price for us, which
# matches the unit-root finding from part 1, so the model works on the change inside.
arima_fit = ARIMA(price.loc[:SPLIT_DATE], order=(1, 1, 1)).fit()

arima_full = arima_fit.apply(price)
price_oos_dates = price.index[price.index > SPLIT_DATE]
arima_pred = arima_full.get_prediction(start=price.index.get_loc(price_oos_dates[0]))
price_forecast = arima_pred.predicted_mean.loc[price_oos_dates]
price_ci = arima_pred.conf_int().loc[price_oos_dates]

# Mean absolute percentage error (MAPE) of the PRICE forecast. MAPE only makes sense for
# a series that stays well away from zero, like a price, so we report it here for the
# price, not for the return (which crosses zero).
price_actual = price.loc[price_oos_dates]
arima_price_mape = (((price_actual - price_forecast).abs()) / price_actual.abs()).mean() * 100
print(f"\nARIMA one-step-ahead PRICE forecast: MAPE = {arima_price_mape:.2f}%")

# Pretty FT-style fan plot: actual price, ARIMA forecast, and the uncertainty band.
plt.rcParams.update({
    "figure.facecolor": "#FDF1E6", "axes.facecolor": "#FDF1E6",
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.edgecolor": "#66605C", "axes.grid": True, "grid.color": "#E2D8CF",
    "axes.axisbelow": True, "font.family": "DejaVu Sans", "font.size": 12,
})
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(price.loc["2023":].index, price.loc["2023":].values,
        color="#262A33", linewidth=1.4, label="Actual")
ax.plot(price_forecast.index, price_forecast.values,
        color="#990F3D", linewidth=1.4, label="ARIMA(1,1,1) forecast")
ax.fill_between(price_ci.index, price_ci.iloc[:, 0], price_ci.iloc[:, 1],
                color="#990F3D", alpha=0.15, label="95% interval")
ax.grid(axis="x", visible=False)
fig.text(0.012, 0.96, "ARIMA one-step-ahead forecast of the Bitcoin price",
         fontsize=15, fontweight="bold", color="#262A33")
fig.text(0.012, 0.91, "US dollars, out-of-sample from 2024", fontsize=11, color="#6B625C")
fig.text(0.012, 0.01, "Source: Yahoo Finance", fontsize=8, color="#6B625C")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86))
fig.subplots_adjust(top=0.86, bottom=0.10)
fig.savefig(FIGURE_DIR / "05_btc_arima_price_forecast.png", dpi=150)
plt.show()
plt.close()

# A zoomed-in view of the last 30 days. The full chart hides what the forecast actually
# does day to day. Up close you can see the forecast is just yesterday's price nudged a
# little, so it always lags one day behind the real move.
last_30 = price_oos_dates[price_oos_dates >= price_oos_dates[-1] - pd.Timedelta(days=30)]
plt.rcParams.update({
    "figure.facecolor": "#FDF1E6", "axes.facecolor": "#FDF1E6",
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.edgecolor": "#66605C", "axes.grid": True, "grid.color": "#E2D8CF",
    "axes.axisbelow": True, "font.family": "DejaVu Sans", "font.size": 12,
})
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(last_30, price_actual.loc[last_30].values, color="#262A33", linewidth=2.0,
        marker="o", markersize=4, label="Actual")
ax.plot(last_30, price_forecast.loc[last_30].values, color="#990F3D", linewidth=2.0,
        marker="o", markersize=4, label="ARIMA(1,1,1) forecast")
ax.fill_between(last_30, price_ci.loc[last_30].iloc[:, 0], price_ci.loc[last_30].iloc[:, 1],
                color="#990F3D", alpha=0.15, label="95% interval")
ax.grid(axis="x", visible=False)
fig.text(0.012, 0.96, "The forecast lags one day behind", fontsize=15,
         fontweight="bold", color="#262A33")
fig.text(0.012, 0.91, "Last 30 days, BTC-USD price and one-step-ahead forecast",
         fontsize=11, color="#6B625C")
fig.text(0.012, 0.01, "Source: Yahoo Finance", fontsize=8, color="#6B625C")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86))
fig.subplots_adjust(top=0.86, bottom=0.10)
fig.savefig(FIGURE_DIR / "06_btc_arima_last_month.png", dpi=150)
plt.show()
plt.close()
plt.rcParams.update(plt.rcParamsDefault)

# Convert the ARIMA price forecast into a return forecast so we can score it in the same
# space as the other models: forecast return = (forecast price / yesterday's price - 1).
arima_return = (price_forecast / price.shift(1).loc[price_oos_dates] - 1) * 100
arima_return = arima_return.loc[oos_dates]


# -----------------------------------------------------------------------------
# 4. Score every model and pick a winner
# -----------------------------------------------------------------------------

print("\nFinal leaderboard on the daily return, lower is better:")
results = [
    score("Naive", naive_return),
    score("AR(1)", ar1_return),
    score("ARIMA(1,1,1)", arima_return),
]
leaderboard = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
print("\n" + leaderboard.to_string(index=False))

best = leaderboard.iloc[0]["model"]
print(f"\nLowest RMSE: {best}.")
# The expected result for a market price: the models sit right on top of the naive
# benchmark, because a daily return is close to unforecastable noise. The out-of-sample
# R-squared makes this precise: it is within a fraction of a per cent of zero.
print("\nLesson: a daily market return is close to pure noise. Every out-of-sample")
print("R-squared sits within a fraction of a per cent of zero, so none of the models")
print("reliably beats the naive zero-return benchmark.")


# -----------------------------------------------------------------------------
# 5. Does the forecast make money? A simple long/short trading strategy
# -----------------------------------------------------------------------------

# A forecast is only useful if it leads to a better decision. We turn the ARIMA return
# forecast into a trading rule:
#   forecast return > 0  ->  go long  (position = +1, we hold Bitcoin tomorrow)
#   forecast return < 0  ->  go short (position = -1, we bet the price falls)
# The position for day t uses only the forecast made with data up to day t-1, so there
# is no look-ahead. The strategy return on day t is the position times the actual return.
position = np.sign(arima_return)
strategy_gross = position * actual_return            # percent per day, before costs

# Trading is not free. Each time we change position we pay a cost (spread plus fees).
# We charge COST_BPS per unit of position change, so a full flip from short to long
# (a change of 2) costs 2 * COST_BPS. We use 10 basis points (0.10%) per unit.
COST_BPS = 0.10
position_change = position.diff().abs().fillna(position.abs())
strategy_net = strategy_gross - COST_BPS * position_change
buy_and_hold = actual_return                          # always long, the benchmark

# Hit rate: how often the forecast gets the direction right.
hit_rate = (np.sign(arima_return) == np.sign(actual_return)).mean() * 100
long_days = (position > 0).sum()
short_days = (position < 0).sum()
print(f"\nTrading strategy: long {long_days} days, short {short_days} days; "
      f"direction correct {hit_rate:.1f}% of days")


def growth_of_one(daily_pct):
    """Turn a daily percent-return series into the growth of one dollar."""
    return (1 + daily_pct / 100).cumprod()


def annualised_sharpe(daily_pct):
    """Annualise the daily Sharpe ratio (crypto trades 365 days a year, rf = 0)."""
    daily = daily_pct / 100
    if daily.std() == 0:
        return float("nan")
    return daily.mean() / daily.std() * np.sqrt(365)


for label, series in [("Buy and hold", buy_and_hold),
                      ("Strategy (gross)", strategy_gross),
                      ("Strategy (net of costs)", strategy_net)]:
    total = (growth_of_one(series).iloc[-1] - 1) * 100
    print(f"  {label:<26} total return = {total:7.1f}%   Sharpe = {annualised_sharpe(series):5.2f}")

# Plot the growth of one dollar for each, on a log scale so equal percentage moves look
# the same size anywhere on the chart.
plt.rcParams.update({
    "figure.facecolor": "#FDF1E6", "axes.facecolor": "#FDF1E6",
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.edgecolor": "#66605C", "axes.grid": True, "grid.color": "#E2D8CF",
    "axes.axisbelow": True, "font.family": "DejaVu Sans", "font.size": 12,
})
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(actual_return.index, growth_of_one(buy_and_hold).values,
        color="#262A33", linewidth=2.0, label="Buy and hold")
ax.plot(actual_return.index, growth_of_one(strategy_gross).values,
        color="#0F5499", linewidth=2.0, label="Strategy (gross)")
ax.plot(actual_return.index, growth_of_one(strategy_net).values,
        color="#990F3D", linewidth=2.0, label="Strategy (net of costs)")
ax.axhline(1.0, color="#66605C", linewidth=0.8)
ax.set_yscale("log")
ax.grid(axis="x", visible=False)
fig.text(0.012, 0.96, "Does the forecast make money?", fontsize=15,
         fontweight="bold", color="#262A33")
fig.text(0.012, 0.91, "Growth of one dollar, out-of-sample from 2024 (log scale)",
         fontsize=11, color="#6B625C")
fig.text(0.012, 0.01, "Source: Yahoo Finance | 10 bp cost per position change",
         fontsize=8, color="#6B625C")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86))
fig.subplots_adjust(top=0.86, bottom=0.10)
fig.savefig(FIGURE_DIR / "07_btc_strategy_growth.png", dpi=150)
plt.show()
plt.close()
plt.rcParams.update(plt.rcParamsDefault)


# A gentler version: long when the forecast is positive, otherwise out of the market
# (in cash, earning zero). This never takes the short side, so the worst a wrong signal
# can do is leave us in cash and miss a move, rather than bet the wrong way.
position_long_flat = (arima_return > 0).astype(float)   # +1 when long, 0 when in cash
lf_change = position_long_flat.diff().abs().fillna(position_long_flat.abs())
long_flat_gross = position_long_flat * actual_return
long_flat_net = long_flat_gross - COST_BPS * lf_change
time_in_market = position_long_flat.mean() * 100
print(f"\nLong/flat strategy: in the market {time_in_market:.0f}% of days "
      f"(in cash the rest)")

for label, series in [("Long/flat (gross)", long_flat_gross),
                      ("Long/flat (net of costs)", long_flat_net)]:
    total = (growth_of_one(series).iloc[-1] - 1) * 100
    print(f"  {label:<26} total return = {total:7.1f}%   Sharpe = {annualised_sharpe(series):5.2f}")

# Compare the two strategy designs against buy and hold, all net of costs.
plt.rcParams.update({
    "figure.facecolor": "#FDF1E6", "axes.facecolor": "#FDF1E6",
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.edgecolor": "#66605C", "axes.grid": True, "grid.color": "#E2D8CF",
    "axes.axisbelow": True, "font.family": "DejaVu Sans", "font.size": 12,
})
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(actual_return.index, growth_of_one(buy_and_hold).values,
        color="#262A33", linewidth=2.0, label="Buy and hold")
ax.plot(actual_return.index, growth_of_one(long_flat_net).values,
        color="#0F766E", linewidth=2.0, label="Long / flat (net)")
ax.plot(actual_return.index, growth_of_one(strategy_net).values,
        color="#990F3D", linewidth=2.0, label="Long / short (net)")
ax.axhline(1.0, color="#66605C", linewidth=0.8)
ax.set_yscale("log")
ax.grid(axis="x", visible=False)
fig.text(0.012, 0.96, "Long/flat avoids the short side, but still trails",
         fontsize=15, fontweight="bold", color="#262A33")
fig.text(0.012, 0.91, "Growth of one dollar, out-of-sample from 2024 (log scale)",
         fontsize=11, color="#6B625C")
fig.text(0.012, 0.01, "Source: Yahoo Finance | 10 bp cost per position change",
         fontsize=8, color="#6B625C")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86))
fig.subplots_adjust(top=0.86, bottom=0.10)
fig.savefig(FIGURE_DIR / "08_btc_strategy_long_flat.png", dpi=150)
plt.show()
plt.close()
plt.rcParams.update(plt.rcParamsDefault)

print("\nSaved figures")
print(FIGURE_DIR / "04_btc_ar1_returns_forecast.png")
print(FIGURE_DIR / "05_btc_arima_price_forecast.png")
print(FIGURE_DIR / "06_btc_arima_last_month.png")
print(FIGURE_DIR / "07_btc_strategy_growth.png")
print(FIGURE_DIR / "08_btc_strategy_long_flat.png")
