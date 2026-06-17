"""Station 3 - your funds: optimal portfolios + out-of-sample backtest.

Build at least a combined equity-plus-crypto fund with two optimisation methods.
Backtest rules: walk-forward, no look-ahead, weights from past data only, annualise
with 252 (equity) or 365 (crypto). See the brief, Part B.
"""
import pandas as pd


def oos_backtest(returns: pd.DataFrame, method: str = "min_variance"):
    """Your walk-forward out-of-sample backtest.

    TODO: return at least the daily portfolio returns, the weights over time,
    growth of $1, and metrics (annualised return, volatility, Sharpe, max drawdown).
    """
    raise NotImplementedError


def performance_metrics(daily_returns: pd.Series, periods_per_year: int = 252) -> dict:
    """TODO: annualised return, annualised volatility, Sharpe, and max drawdown."""
    raise NotImplementedError
