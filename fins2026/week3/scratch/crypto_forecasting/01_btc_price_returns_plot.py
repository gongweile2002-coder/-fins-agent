# ruff: noqa
"""Week 3, even simpler: download the Bitcoin price, plot it, and compute returns.

This is the first of two scripts. It downloads the daily BTC-USD price from Yahoo
Finance, plots it, tests it for a unit root, then computes daily returns and plots them.
Part 2 (02_forecasting_btc_returns.py) reads the file this script saves and forecasts.

A market price is simpler than the macro data we used before: it has only ONE date,
the day it traded. There is no reference date and no release date to keep apart.

PyCharm shortcut note:
Settings -> Keymap -> Search for -> Execute Selection in Python Console
Change it to the shortcut you want, then run this file one numbered stage at a time.
"""

from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
from statsmodels.tsa.stattools import adfuller

# Where to save outputs. This works two ways: running the whole file (the green Play
# button defines __file__) and sending a highlighted block to the Python Console (no
# __file__, so we fall back to the working directory and step into this script's folder
# when we can find it). Both are absolute paths, so neither nests, on Mac or Windows.
THIS_SCRIPT_FOLDER = Path("fins2026") / "week3" / "scratch" / "crypto_forecasting"
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:  # no __file__ when running a highlighted selection in the console
    guess = Path.cwd() / THIS_SCRIPT_FOLDER
    BASE_DIR = guess if guess.is_dir() else Path.cwd()
OUTPUT_DIR = BASE_DIR / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
print(f"Saving outputs to: {OUTPUT_DIR}")


# -----------------------------------------------------------------------------
# 1. Download the Bitcoin price from Yahoo Finance
# -----------------------------------------------------------------------------

TICKER = "BTC-USD"
START_DATE = date(2018, 1, 1)
END_DATE = date.today()

# Yahoo Finance serves daily prices from a web address called the chart API. We send a
# normal browser User-Agent so Yahoo does not reject us, and we try a second address if
# the first one fails.
YAHOO_ENDPOINTS = [
    "https://query2.finance.yahoo.com/v8/finance/chart/{ticker}",
    "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
]


def download_yahoo_daily(ticker, start, end):
    """Download one ticker's daily adjusted close from Yahoo and return date/close."""
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    params = {
        "period1": int(datetime.combine(start, time.min, tzinfo=UTC).timestamp()),
        "period2": int(datetime.combine(end + timedelta(days=1), time.min, tzinfo=UTC).timestamp()),
        "interval": "1d",
        "includeAdjustedClose": "true",
    }
    payload = None
    for endpoint in YAHOO_ENDPOINTS:
        try:
            response = session.get(endpoint.format(ticker=ticker), params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            break
        except requests.RequestException:
            continue
    if payload is None:
        raise SystemExit("Could not reach Yahoo Finance. Check your internet connection.")

    item = payload["chart"]["result"][0]
    dates = pd.to_datetime(item["timestamp"], unit="s", utc=True).tz_localize(None).normalize()
    close = item["indicators"]["adjclose"][0]["adjclose"]
    return pd.DataFrame({"date": dates, "close": close})


btc = download_yahoo_daily(TICKER, START_DATE, END_DATE)

print(f"Downloaded {TICKER}: {len(btc):,} daily rows")
print("\nFirst rows")
print(btc.head())
print("\nLast rows")
print(btc.tail())


# -----------------------------------------------------------------------------
# 2. Clean the data (note the single date)
# -----------------------------------------------------------------------------

# Unlike the macro data, a traded price has only ONE date: the day it traded. There is
# no release lag, because the price is known the moment the market sets it. So there is
# only one date column to look after here.
btc["date"] = pd.to_datetime(btc["date"])
btc["close"] = pd.to_numeric(btc["close"], errors="coerce")
btc = btc.dropna().drop_duplicates(subset="date").sort_values("date")
btc = btc.set_index("date")

print("\nStage 1 checks")
print(f"Rows: {len(btc):,}")
print(f"First date: {btc.index.min():%Y-%m-%d},  last date: {btc.index.max():%Y-%m-%d}")
print(f"Missing close values: {btc['close'].isna().sum()}")

price = btc["close"]


# -----------------------------------------------------------------------------
# 3. Plot the price: a plain plot, then a Financial Times (FT) style plot
# -----------------------------------------------------------------------------

# First the plain default plot. It works, but it is busy and has no clear final value.
plt.rcParams.update(plt.rcParamsDefault)  # start from the plain defaults

plt.figure(figsize=(10, 6))
price.plot(ax=plt.gca())
plt.title("Bitcoin price (BTC-USD)")
plt.xlabel("Date")
plt.ylabel("US dollars")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "01_btc_price_plain.png", dpi=150)
plt.show()
plt.close()


def apply_ft_style():
    """Set a few rcParams so the next figure follows a clean FT-style look."""
    plt.rcParams.update({
        "figure.facecolor": "#FDF1E6", "axes.facecolor": "#FDF1E6",
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.edgecolor": "#66605C", "axes.grid": True, "grid.color": "#E2D8CF",
        "axes.axisbelow": True, "font.family": "DejaVu Sans", "font.size": 12,
    })


FT_MAROON = "#990F3D"  # the Financial Times signature colour

apply_ft_style()
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(price.index, price.values, color=FT_MAROON, linewidth=1.8)
ax.grid(axis="x", visible=False)
# Label the final value directly on the line instead of using a legend.
ax.annotate(f"  ${price.iloc[-1]:,.0f}", xy=(price.index[-1], price.iloc[-1]),
            color=FT_MAROON, fontweight="bold", va="center")
fig.text(0.012, 0.96, "The Bitcoin price", fontsize=15, fontweight="bold", color="#262A33")
fig.text(0.012, 0.91, "BTC-USD, daily close", fontsize=11, color="#6B625C")
fig.text(0.012, 0.01,
         f"Source: Yahoo Finance | {price.index.min():%b %Y} to {price.index.max():%b %Y}",
         fontsize=8, color="#6B625C")
ax.set_xlabel("")
ax.set_ylabel("")
fig.subplots_adjust(top=0.86, bottom=0.10)
fig.savefig(FIGURE_DIR / "02_btc_price_ft.png", dpi=150)
plt.show()
plt.close()
plt.rcParams.update(plt.rcParamsDefault)


# -----------------------------------------------------------------------------
# 4. Unit-root test on the price
# -----------------------------------------------------------------------------

# The price trends and wanders with no fixed mean to return to. That is a "unit root",
# and it means the level of the price is close to unforecastable. The Augmented
# Dickey-Fuller (ADF) test checks for it: a large p-value means we cannot reject a unit
# root, so we should not forecast the price level directly.
adf_price = adfuller(price)
print("\nADF test on the PRICE level")
print(f"  test statistic: {adf_price[0]:.3f},  p-value: {adf_price[1]:.3f}")
print("  large p-value -> unit root -> do not forecast the price level")


# -----------------------------------------------------------------------------
# 5. Compute returns and plot them
# -----------------------------------------------------------------------------

# The daily return is the percentage change in the price from one day to the next. This
# is the series we forecast, because the price has a unit root and the return does not.
btc["return"] = btc["close"].pct_change() * 100
daily_return = btc["return"].dropna()

adf_return = adfuller(daily_return)
print("\nADF test on the daily RETURN")
print(f"  test statistic: {adf_return[0]:.3f},  p-value: {adf_return[1]:.3f}")
print("  small p-value -> stationary -> this is the series we forecast")

apply_ft_style()
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(daily_return.index, daily_return.values, color="#0F5499", linewidth=0.8)
ax.axhline(0.0, color="#66605C", linewidth=0.8)  # the flat level it varies around
ax.grid(axis="x", visible=False)
fig.text(0.012, 0.96, "Bitcoin daily returns", fontsize=15, fontweight="bold", color="#262A33")
fig.text(0.012, 0.91, "Daily percentage change in BTC-USD", fontsize=11, color="#6B625C")
fig.text(0.012, 0.01, "Source: Yahoo Finance", fontsize=8, color="#6B625C")
ax.set_xlabel("")
ax.set_ylabel("Per cent")
fig.subplots_adjust(top=0.86, bottom=0.10)
fig.savefig(FIGURE_DIR / "03_btc_returns_ft.png", dpi=150)
plt.show()
plt.close()
plt.rcParams.update(plt.rcParamsDefault)


# -----------------------------------------------------------------------------
# 6. Save the cleaned data for the forecasting script
# -----------------------------------------------------------------------------

def save_both(frame, csv_path, parquet_path):
    """Always save the CSV. Try the Parquet file too, but do not crash if the Parquet
    engine is missing on a minimal Python install -- the CSV is enough for part 2."""
    frame.to_csv(csv_path)
    try:
        frame.to_parquet(parquet_path)
    except Exception as exc:  # e.g. no pyarrow/fastparquet installed
        print(f"  (skipped {parquet_path.name}: {exc}; the CSV is saved and is enough)")


BTC_CSV = OUTPUT_DIR / "btc_daily.csv"
BTC_PARQUET = OUTPUT_DIR / "btc_daily.parquet"
save_both(btc[["close", "return"]], BTC_CSV, BTC_PARQUET)

print("\nSaved cleaned Bitcoin data")
print(BTC_CSV)
print(BTC_PARQUET)
print("\nSaved figures")
print(FIGURE_DIR / "01_btc_price_plain.png")
print(FIGURE_DIR / "02_btc_price_ft.png")
print(FIGURE_DIR / "03_btc_returns_ft.png")
