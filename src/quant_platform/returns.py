"""
Return transforms.
==================

Timing convention -- the single most important thing in this repository
-----------------------------------------------------------------------
``prices.loc[t]`` is the **closing price at date t**.

Therefore::

    returns.loc[t] = prices.loc[t] / prices.loc[t-1] - 1

is the return **earned over the interval (t-1, t]**. It is not knowable until
the close of ``t``.

A signal computed from the close of ``t`` cannot capture ``returns.loc[t]`` --
that return already happened. It can only capture ``returns.loc[t+1]`` onwards.
That is why the engine holds

    executed_weights = target_weights.shift(1)

so that ``executed_weights.loc[t]`` -- the weights actually held over
``(t-1, t]`` -- equals ``target_weights.loc[t-1]``, which was decided at the
close of ``t-1`` using only data up to ``t-1``.

That single shift is what separates a backtest from a fantasy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "simple_returns",
    "log_returns",
    "cumulative_returns",
    "rolling_volatility",
    "to_equity_curve",
    "annualisation_factor",
]


def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Arithmetic (simple) returns: ``p_t / p_{t-1} - 1``.

    Simple returns are what the portfolio engine uses, because a portfolio
    return is the weighted **arithmetic** mean of its constituents' arithmetic
    returns. Log returns do not aggregate that way across assets.

    The first row is NaN by construction -- there is no prior price.
    """
    return prices.pct_change(fill_method=None)


def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Continuously compounded returns: ``ln(p_t / p_{t-1})``.

    Useful for statistical work (they aggregate additively **through time**) but
    never for cross-sectional portfolio aggregation. See :func:`simple_returns`.
    """
    return np.log(prices / prices.shift(1))


def cumulative_returns(returns: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Cumulative growth of 1 unit: ``cumprod(1 + r) - 1``.

    NaNs are treated as zero-return periods so a single missing observation does
    not annihilate the whole path.
    """
    return (1.0 + returns.fillna(0.0)).cumprod() - 1.0


def to_equity_curve(
    returns: pd.DataFrame | pd.Series,
    initial_capital: float = 1.0,
) -> pd.DataFrame | pd.Series:
    """Compound returns into a capital path starting at ``initial_capital``."""
    return initial_capital * (1.0 + returns.fillna(0.0)).cumprod()


def rolling_volatility(
    returns: pd.DataFrame | pd.Series,
    window: int = 63,
    periods_per_year: int = 252,
    min_periods: int | None = None,
) -> pd.DataFrame | pd.Series:
    """Annualised rolling standard deviation of returns.

    Uses the sample standard deviation (``ddof=1``) and scales by
    ``sqrt(periods_per_year)``. Default window of 63 trading days is ~one quarter.
    """
    if min_periods is None:
        min_periods = max(2, window // 2)
    return returns.rolling(window, min_periods=min_periods).std(ddof=1) * np.sqrt(
        periods_per_year
    )


def annualisation_factor(freq: str) -> int:
    """Periods per year for a rebalance/observation frequency label."""
    table = {"daily": 252, "weekly": 52, "monthly": 12, "quarterly": 4, "annual": 1}
    key = freq.lower()
    if key not in table:
        raise ValueError(f"Unknown frequency '{freq}'; expected one of {sorted(table)}")
    return table[key]
