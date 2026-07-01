# Week 5 Practice: Bitcoin versus the S&P 500

## Executive conclusion

On the shared 2010-07-18 to 2025-12-31 sample, Bitcoin produced vastly more total wealth, but it also exposed investors to much higher volatility, heavier daily tail losses, and a far deeper maximum drawdown. On the full-sample Sharpe-ratio comparison, **Bitcoin** delivered the stronger risk-adjusted result. This is a historical result, not a forecast.

## Evidence

- Bitcoin CAGR: **144.8%**; S&P 500 CAGR: **14.9%**.
- Bitcoin annualized volatility: **90.1%**; S&P 500: **17.2%**.
- Sharpe ratios: **1.42** for Bitcoin and **0.82** for the S&P 500.
- Sortino ratios: **2.23** for Bitcoin and **1.15** for the S&P 500.
- Maximum drawdowns: **-92.7%** for Bitcoin and **-33.5%** for the S&P 500.
- Daily 95% Expected Shortfall: **10.4%** for Bitcoin and **2.6%** for the S&P 500.
- Same-date daily-return correlation: **0.14**. The low-to-moderate average suggests diversification potential, but the rolling chart shows that correlation changes materially over time.

## Interpretation

Bitcoin won decisively on raw wealth creation, yet that statement alone hides the cost of the journey. Its much larger volatility and drawdowns mean that an investor needed both a high risk tolerance and the ability to remain invested through repeated crashes. The S&P 500 offered a smoother path and shallower tail losses. Which investment was 'better' therefore depends on the objective: maximum historical growth favoured Bitcoin; capital stability favoured the S&P 500; the Sharpe and Sortino ratios provide the explicit risk-adjusted comparison.

## Method and caveats

Returns are computed on each asset's native calendar before comparison. Bitcoin is annualized with 365 observations and the S&P 500 with 252. The business-day risk-free rate is forward-filled across Bitcoin weekends and holidays. For shared-timeline figures, S&P 500 wealth is carried flat on market-closed days. Bitcoin's 2010-2014 market was small and illiquid, so early returns may not represent an investable institutional experience. Results are sensitive to the starting date and should not be treated as evidence that past performance will repeat.
