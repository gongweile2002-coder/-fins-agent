# ruff: noqa
"""Station 1 - your ETL: load and clean the data.

Load raw data through src.data_access (see context/DATA_GUIDE.md). Add your own
integrity checks. Do not commit data files.
"""
from src import data_access


def load_clean_equities():
    """Load equity prices and run your Station 1 integrity checks.

    TODO: missing-date audit, duplicate ticker-date check, outlier/extreme-value
    screen on returns. Return the clean frame and record what you found.
    """
    df = data_access.load_equity_prices()
    return df


def load_clean_crypto():
    """Load crypto prices (365-day calendar). TODO: your checks."""
    return data_access.load_crypto_prices()
