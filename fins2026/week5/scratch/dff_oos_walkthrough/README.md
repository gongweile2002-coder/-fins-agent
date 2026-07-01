# Week 5 Lecture Scripts --- Out-of-Sample Data Factory Floor (DFF)

These are the scripts we run together in class. Read this first --- it tells you exactly
where to put the files and how to run them.

## Step 1 --- Create this folder in your repository

In your **fins-agent** repository, create this exact folder path:

```
fins2026/week5/scratch/dff_oos_walkthrough/
```

## Step 2 --- Put the downloaded files inside that folder

Unzip this download and move **all** of these files into the folder from Step 1:

- `data_access.py` --- loads the stock and crypto data (provided, do not edit)
- `dff_oos_helpers.py` --- the figure helpers, portfolio maths, and backtest engine
- `01_stage1_2_inputs.py` --- Stages 1--2: daily returns and the two model inputs
  (average return `mu`, covariance matrix `Sigma`)
- `02_stage3_oos.py` --- Stage 3: the out-of-sample test on the 50 stocks, run
  long-short (like Week 4) and long-only (the fair test)
- `03_universes.py` --- the same test on stocks, crypto, and the two combined
- `04_risk_parity_vol_target.py` --- extension: risk parity and volatility targeting
- `05_blend.py` --- extension: blending 1/N with minimum-variance
- `README.md` (this file)

You do **not** add any data file. `data_access.py` downloads the datasets (50 US stocks
plus 10 cryptocurrencies, daily) automatically the first time you run a script, so you
only need an internet connection.

## Step 3 --- Run them in order, as we reach each part of the lecture

```
python 01_stage1_2_inputs.py
python 02_stage3_oos.py
python 03_universes.py
python 04_risk_parity_vol_target.py
python 05_blend.py
```

Run from inside the `dff_oos_walkthrough/` folder, or open each file in PyCharm and
click the green **Run** button. Run them **in order**: `02_stage3_oos.py` reads a table
that `01_stage1_2_inputs.py` saves. The figures and tables each script makes appear in a
new `output/` folder next to the scripts.

## What this lecture does

This is a strict extension of the Week 4 Data Factory Floor. Week 4 chose portfolio
weights and judged them on the **same** data, so the optimised portfolios looked
spectacular (the tangency Sharpe ratio was 2.82). This week we run the out-of-sample
test: each month we choose weights using **only past data**, hold them for the next
month, and roll forward. We then run the same machine on three universes (50 US stocks,
10 cryptocurrencies, and the two combined) and add risk parity and volatility targeting.

## What you should see

- Stage 3 long-short on the 50 stocks **wipes out** out-of-sample --- the tangency
  portfolio takes huge leveraged bets that lose almost everything --- while long-only
  1/N does best.
- Across universes, 1/N is hard to beat. Minimum-variance (which ignores the noisy
  average-return estimate) is the only optimised rule that sometimes wins, and only for
  crypto. The tangency portfolio trails everywhere.
- Risk parity matches 1/N's Sharpe ratio at lower risk. Volatility targeting helps the
  crypto-heavy combined book but hurts equities --- a real but sample-dependent edge,
  not a free lunch.
- Blending 1/N with minimum-variance trades a little Sharpe for a large cut in risk. On
  the combined book a 50/50 blend cuts volatility about a quarter (21.7% to 15.7%) and
  the drawdown from 27% to 20%, giving up only 0.05 of Sharpe.

## If something does not run

- `No module named pandas` (or numpy / matplotlib / scipy) --- your course Python
  environment is not active. Activate it, or ask your AI assistant to install the
  missing package. (The long-only and risk-parity solvers use `scipy`.)
- A download error --- check your internet connection and run the script again. The data
  is cached after the first successful download, so later runs are fast.
