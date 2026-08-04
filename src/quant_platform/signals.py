"""
Signal generation and target-weight construction.
=================================================

This module answers "what do I *want* to hold?". It never answers "what did I
actually hold?" -- that is the portfolio engine's job, and keeping the two apart
is what makes the execution lag auditable.

Pipeline stage boundaries::

    prices --[trailing_momentum]--> scores
    scores --[rebalance_dates + top_n_equal_weight + apply_max_weight]--> target weights
    target weights --[portfolio.run_backtest]--> executed weights -> returns
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "trailing_momentum",
    "top_n_equal_weight",
    "apply_max_weight",
    "rebalance_dates",
    "build_target_weights",
    "random_selection_weights",
    "equal_weight_all",
]


# --------------------------------------------------------------------------- #
# Scores
# --------------------------------------------------------------------------- #
def trailing_momentum(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
) -> pd.DataFrame:
    """Classic "12-1" cross-sectional momentum score.

    The score at date ``t`` is the total return from ``t - lookback`` to
    ``t - skip``::

        score_t = p_{t-skip} / p_{t-lookback} - 1

    Parameters
    ----------
    lookback
        Total formation window in trading days (252 ~ 12 months).
    skip
        Most recent days to exclude (21 ~ 1 month). Skipping the last month is
        standard practice: short-horizon returns exhibit *reversal* rather than
        continuation, so including them dilutes the momentum signal. It also
        buys a buffer against microstructure noise and stale prices.

    Notes
    -----
    Implemented as ``prices.shift(skip).pct_change(lookback - skip)``. Because of
    the ``shift(skip)``, the value at ``t`` depends only on prices at or before
    ``t - skip``: the signal is strictly backward-looking even before the
    engine's execution lag is applied. ``tests/test_no_lookahead.py`` verifies
    this by perturbing future prices and asserting history is unchanged.
    """
    if lookback <= skip:
        raise ValueError(f"lookback ({lookback}) must exceed skip ({skip})")
    if skip < 0:
        raise ValueError(f"skip must be non-negative, got {skip}")
    return prices.shift(skip).pct_change(lookback - skip, fill_method=None)


# --------------------------------------------------------------------------- #
# Scores -> weights
# --------------------------------------------------------------------------- #
def top_n_equal_weight(scores: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Select the ``n`` highest-scoring assets per row and weight them equally.

    Rows where fewer than ``n`` assets have a score produce zero weights (no
    position) rather than a partially-filled portfolio -- during the initial
    warm-up the signal simply does not exist yet, and holding a 1-name
    "portfolio" there would be an artefact of the warm-up, not a decision.

    Ties are broken by ``rank(method="first")``, i.e. by column order, which is
    deterministic and therefore reproducible.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    valid = scores.notna().sum(axis=1)
    ranks = scores.rank(axis=1, ascending=False, method="first", na_option="bottom")
    selected = (ranks <= n) & scores.notna()
    selected = selected.where(valid.ge(n), other=False)

    counts = selected.sum(axis=1)
    weights = selected.astype(float).div(counts.replace(0, np.nan), axis=0)
    return weights.fillna(0.0)


def apply_max_weight(weights: pd.DataFrame, max_weight: float | None) -> pd.DataFrame:
    """Cap each position at ``max_weight``, redistributing the excess.

    Excess weight from capped names is pushed into the remaining uncapped names
    (iteratively, since redistribution can push another name over the cap). If
    *every* held name is at the cap, the residual stays **uninvested in cash**
    and the row sums to less than 1.

    That last case matters for the sensitivity analysis: with ``max_weight=0.40``
    and ``holdings=1``, the portfolio is only 40% invested and the remaining 60%
    earns the cash rate. This is a deliberate modelling choice, not an oversight
    -- the alternative (renormalising back to 1.0) would silently ignore the
    concentration limit it was asked to enforce. The reported exposure series
    makes the under-investment visible.
    """
    if max_weight is None:
        return weights
    if not 0 < max_weight <= 1:
        raise ValueError(f"max_weight must be in (0, 1], got {max_weight}")

    capped = weights.clip(upper=max_weight)

    # Iteratively hand the clipped-off weight to names with headroom.
    for _ in range(10):
        excess = weights.sum(axis=1) - capped.sum(axis=1)
        if (excess.abs() < 1e-12).all():
            break
        headroom = (max_weight - capped).where(capped > 0, 0.0).clip(lower=0.0)
        total_headroom = headroom.sum(axis=1)
        share = headroom.div(total_headroom.replace(0, np.nan), axis=0).fillna(0.0)
        capped = (capped + share.mul(excess, axis=0)).clip(upper=max_weight)

    return capped


# --------------------------------------------------------------------------- #
# Rebalance schedule
# --------------------------------------------------------------------------- #
_FREQ_ALIASES = {
    "daily": ("D", "D"),
    "weekly": ("W", "W"),
    "monthly": ("ME", "M"),
    "quarterly": ("QE", "Q"),
    "annual": ("YE", "A"),
}


def rebalance_dates(index: pd.DatetimeIndex, freq: str = "monthly") -> pd.DatetimeIndex:
    """The last **actual trading day** of each period in ``index``.

    Why not ``scores.resample("ME").last()`` directly? Because ``resample``
    stamps its result on the *calendar* period end (e.g. 2010-01-31), which is
    frequently not a trading day. Reindexing such a label onto a trading-day
    index silently drops it, and a forward-fill then carries the *previous*
    month's weights -- a missed rebalance that no assertion would catch.

    Resampling a Series *of the index itself* and taking ``.last()`` returns
    genuine trading days, so the result is always a subset of ``index`` and
    reindex/ffill is exact.
    """
    if freq.lower() == "daily":
        return pd.DatetimeIndex(index)

    primary, fallback = _resolve_freq(freq)
    marker = pd.Series(index, index=index)
    try:
        picked = marker.resample(primary).last()
    except ValueError:  # pragma: no cover - older pandas
        picked = marker.resample(fallback).last()

    dates = pd.DatetimeIndex(picked.dropna().values)
    return dates[dates.isin(index)]


def _resolve_freq(freq: str) -> tuple[str, str]:
    key = freq.lower()
    if key not in _FREQ_ALIASES:
        raise ValueError(f"Unknown rebalance frequency '{freq}'; expected {sorted(_FREQ_ALIASES)}")
    return _FREQ_ALIASES[key]


# --------------------------------------------------------------------------- #
# End-to-end target weights
# --------------------------------------------------------------------------- #
def build_target_weights(
    prices: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    holdings: int = 3,
    rebalance: str = "monthly",
    max_weight: float | None = 0.40,
) -> pd.DataFrame:
    """Full signal -> target-weight pipeline, on the daily trading calendar.

    Returns a daily date x symbol frame of **desired** weights. Weights are
    decided on rebalance dates from that day's close and held constant until the
    next rebalance date. They are *not* yet lagged -- the engine applies the
    execution delay, so this frame is "what I want at the close of t".
    """
    scores = trailing_momentum(prices, lookback=lookback, skip=skip)
    reb = rebalance_dates(prices.index, rebalance)

    target_on_reb = top_n_equal_weight(scores.loc[reb], n=holdings)
    target_on_reb = apply_max_weight(target_on_reb, max_weight)

    # Every rebalance date is a real trading day, so reindex+ffill is lossless.
    daily = target_on_reb.reindex(prices.index).ffill().fillna(0.0)
    return daily.reindex(columns=prices.columns, fill_value=0.0)


# --------------------------------------------------------------------------- #
# Baseline weight schemes
# --------------------------------------------------------------------------- #
def equal_weight_all(prices: pd.DataFrame, rebalance: str = "monthly") -> pd.DataFrame:
    """Equal weight across every asset with a price, rebalanced on schedule."""
    reb = rebalance_dates(prices.index, rebalance)
    available = prices.loc[reb].notna()
    weights = available.astype(float).div(
        available.sum(axis=1).replace(0, np.nan), axis=0
    )
    return weights.reindex(prices.index).ffill().fillna(0.0)


def random_selection_weights(
    prices: pd.DataFrame,
    holdings: int = 3,
    rebalance: str = "monthly",
    max_weight: float | None = 0.40,
    seed: int = 42,
    warmup: int = 252,
) -> pd.DataFrame:
    """Pick ``holdings`` assets uniformly at random each rebalance date.

    The control for "is the momentum *ranking* doing anything, or is any
    concentrated 3-of-9 rotation in this universe good enough?". Seeded, so it
    reproduces exactly. ``warmup`` keeps it flat over the same initial window the
    momentum strategy needs to form its first signal, so the two are comparable.
    """
    rng = np.random.default_rng(seed)
    reb = rebalance_dates(prices.index, rebalance)
    reb = reb[reb >= prices.index[min(warmup, len(prices.index) - 1)]]

    rows = pd.DataFrame(0.0, index=reb, columns=prices.columns)
    for date in reb:
        eligible = prices.columns[prices.loc[date].notna()]
        if len(eligible) == 0:
            continue
        picks = rng.choice(eligible, size=min(holdings, len(eligible)), replace=False)
        rows.loc[date, picks] = 1.0 / len(picks)

    rows = apply_max_weight(rows, max_weight)
    return rows.reindex(prices.index).ffill().fillna(0.0)
