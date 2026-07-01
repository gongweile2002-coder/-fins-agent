# Week 5 Practice - Bitcoin versus the S&P 500

This folder contains a complete, reproducible answer to the Week 5 practice.

## Run

From the repository root:

```bash
python fins2026/week5/scratch/crypto_vs_stocks/analyze_crypto_vs_stocks.py
```

The script:

- computes returns on each asset's native calendar before alignment;
- annualizes Bitcoin with 365 days and the S&P 500 with 252 days;
- forward-fills the business-day risk-free rate onto Bitcoin weekends;
- calculates return, volatility, Sharpe, Sortino, drawdown, VaR, Expected Shortfall, skewness, and kurtosis;
- writes tables and an evidence-based narrative to `output/`;
- creates four FT-style comparison figures.

The fixed course inputs are stored under `provided_data/` and are not modified.
