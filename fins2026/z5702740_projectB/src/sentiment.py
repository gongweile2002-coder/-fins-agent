# ruff: noqa
"""Station 3 - your sentiment model and index from news headlines.

This is the model step: score each headline, aggregate to a daily per-ticker score,
then to an equal-weight sector index. Headlines are a noisy proxy, so lag to avoid
look-ahead.
"""
import pandas as pd


def score_headlines(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply a sentiment model (VADER or another) to the assembled headlines.

    TODO: return a per-headline or per-ticker-day sentiment score. VADER uses
    casing, punctuation, and negation, so do not strip them. VADER also needs a
    one-time nltk.download('vader_lexicon') before it scores (a build step, not the
    deployed app).
    """
    raise NotImplementedError


def sector_sentiment_index(scores: pd.DataFrame) -> pd.DataFrame:
    """TODO: build a daily sentiment index per sector (equal-weight across names)."""
    raise NotImplementedError
