# AGENTS.md - FINS5545 Part A Working Instructions

This folder builds the Part A data foundation for the FINS5545 project. Raw data
must be loaded only through `src/data_access.py`; do not commit raw parquet files
or full cleaned raw datasets.

Rules for AI/code assistance:

- Part A is data assembly only. Do not create a sentiment index and do not use
  sentiment scores in Part A.
- Deduplicate price data by `ticker` and `date`; deduplicate news by `ticker`,
  `date`, and `title`.
- Compute returns within each asset panel before any cross-asset calendar
  alignment. Use adjusted close.
- Keep genuine return outliers in the dataset and describe them rather than
  deleting them.
- Save required outputs using the exact filenames in the brief:
  `results/tables/dataset_inventory.csv` and
  `results/tables/descriptive_stats_returns.csv`.
- Save only small derived samples under `results/data`.
- AI-generated report wording is a draft and must be reviewed and rewritten by
  the student before submission.

Verification steps:

1. Run `python scripts/run_part_a.py`.
2. Run `python scripts/check_handin.py`.
3. Inspect the figures, tables, and `report/report.pdf` before zipping.
