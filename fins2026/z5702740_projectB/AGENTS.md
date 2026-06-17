# Agent Instructions for FINS5545 Project B

Student: z5702740

This folder is for FINS5545 Project B only. The goal is to build systematic investment funds, a sentiment analytics layer, and a deployed Streamlit app.

The agent should help with:
- reading and following PROJECT_BRIEF.md exactly;
- reusing the Part A data foundation without committing raw source data;
- building out-of-sample fund returns and weights with no look-ahead bias;
- creating at least two optimisation methods for a combined equity-plus-crypto fund;
- generating app-readable CSV files in results/data/ using the exact required filenames;
- generating report tables and figures in results/tables/ and results/figures/;
- building a lightweight Streamlit app that reads precomputed results instead of recomputing heavy models;
- drafting clear report notes supported by saved outputs.

Rules:
- Do not invent data, fund results, or performance metrics.
- Do not use future data when forming portfolio weights.
- Lag sentiment signals by at least one trading day before using them in any trading or fund logic.
- Keep raw .parquet/source data out of the submitted folder and GitHub commit.
- Use the exact required output filenames from PROJECT_BRIEF.md:
  - results/data/fund_returns.csv
  - results/data/fund_weights.csv
  - results/data/sector_sentiment_index.csv
  - results/tables/performance_metrics.csv
- The deployed app must load precomputed artifacts from results/ and should not recompute backtests or sentiment scoring on the free Streamlit tier.
- Written interpretation must be checked and rewritten by the student.
