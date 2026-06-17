"""Station 3 (extension) - fuse sentiment into the funds.

Tilt or factor: combine your sentiment signal with the portfolio weights,
look-ahead safe, then test whether it adds value. An honest negative result,
explained, is good work.
"""
import pandas as pd


def apply_sentiment(weights: pd.DataFrame, sentiment: pd.DataFrame):
    """TODO: your fusion rule (for example tilt weights toward high-sentiment names)."""
    raise NotImplementedError
