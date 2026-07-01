# Week 5 Exercise Data --- Bitcoin versus the S&P 500

This download has the data for your Week 5 exercise. Read this first --- it tells you
exactly where to put the files and what each one is.

## Step 1 --- Create this folder in your repository

In your **fins-agent** repository, create this exact folder path:

```
fins2026/week5/scratch/crypto_vs_stocks/provided_data/
```

## Step 2 --- Put the downloaded files inside that folder

Unzip this download and move **all** of these files into the folder from Step 1:

- `btc_usd_daily.csv` and `btc_usd_daily.parquet`
- `sp500_tr_index_daily.csv` and `sp500_tr_index_daily.parquet`
- `french_daily_rfr.csv` and `french_daily_rfr.parquet`
- `README.md` (this file)

Use this exact path. Your code reads the data from `provided_data/`, so saving the
files anywhere else will break your file paths.

## Step 3 --- Save your own work separately

- Write your own Python code in the `crypto_vs_stocks/` folder (one level up from
  `provided_data/`).
- Save any figures and tables you make into `crypto_vs_stocks/output/` --- create that
  `output/` folder yourself.
- Do **not** edit the files in `provided_data/`. They are the fixed inputs everyone
  uses, so that your numbers line up with the worked answer released after the exercise.

## What the data is

Three daily datasets, all in US dollars. Each is saved as both CSV and Parquet --- the
two file types hold the same data, so use whichever your code prefers.

| File | Rows | Columns | Coverage |
|------|------|---------|----------|
| `btc_usd_daily` | 5,826 | `date`, `adjClose` | Bitcoin price in US dollars, 2010-07-18 to 2026-06-29 |
| `sp500_tr_index_daily` | 26,300 | `date`, `sp500_tr_index` | S&P 500 total-return index, 1926-01-02 to 2025-12-31 |
| `french_daily_rfr` | 26,233 | `date`, `rfr` | US daily risk-free rate, 1926-07-01 to 2026-04-30 |

## What each column means

- **`adjClose`** --- Bitcoin's daily US-dollar price. Turn it into a daily return with
  `pct_change()`. Prices before 2014-09-17 come from CoinMetrics community data; from
  then on they are Yahoo Finance. The early market was very thin, so treat the
  2010--2014 numbers with care.
- **`sp500_tr_index`** --- the S&P 500 with dividends reinvested, an index that starts
  at 100 and grows with the market. Turn it into a daily return with `pct_change()`.
- **`rfr`** --- the daily risk-free rate as a decimal (for example `0.0002` means 0.02%
  per day). It is already divided by 100, so you do not need to rescale it.

## Two things to watch

- Bitcoin has a row for **every calendar day**, weekends included. The S&P 500 only has
  rows for days the US stock market was open, so the two calendars do not line up. Work
  out each asset's returns on its own dates first, then line them up.
- The risk-free rate is a business-day series. To attach it to Bitcoin's weekend days,
  carry the last value forward with `.ffill()`.

## Where this data came from

- **Bitcoin** --- Yahoo Finance (`BTC-USD`) from 2014-09-17, spliced onto CoinMetrics
  community data (`PriceUSD`) for 2010--2014.
- **S&P 500 total-return index** --- CRSP daily gross S&P 500 index.
- **Risk-free rate** --- Kenneth R. French Data Library, daily research factors.
