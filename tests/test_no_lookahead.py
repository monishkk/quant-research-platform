"""
Look-ahead bias tests.
======================

The most dangerous bug in a backtest is one that makes the result *better*.
Look-ahead bias does exactly that and produces no error, no warning, and a
beautiful equity curve.

The tests here use the perturbation method: change a price in the future,
recompute, and assert that nothing in the past moved. Any dependence of history
on the future -- however indirect, through a rolling window, a resample, an
interpolation, or a fillna -- shows up as a diff.

This is stronger than reading the code, because it catches leakage introduced by
pandas semantics that no one intended, which is where these bugs actually come
from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.portfolio import run_backtest
from quant_platform.returns import simple_returns
from quant_platform.signals import (
    build_target_weights,
    rebalance_dates,
    top_n_equal_weight,
    trailing_momentum,
)


# --------------------------------------------------------------------------- #
# Signal level
# --------------------------------------------------------------------------- #
def test_signal_does_not_use_future_data(realistic_prices):
    """The canonical test from the spec: perturb the last price by 100x."""
    original = trailing_momentum(realistic_prices, 252, 21)

    modified = realistic_prices.copy()
    modified.iloc[-1] *= 100

    changed = trailing_momentum(modified, 252, 21)

    pd.testing.assert_frame_equal(original.iloc[:-1], changed.iloc[:-1])


@pytest.mark.parametrize("perturb_at", [-1, -5, -21, -100])
def test_signal_immune_to_perturbation_at_various_horizons(realistic_prices, perturb_at):
    """History must be invariant to a future shock, wherever the shock lands."""
    original = trailing_momentum(realistic_prices, 252, 21)

    modified = realistic_prices.copy()
    modified.iloc[perturb_at] *= 50

    changed = trailing_momentum(modified, 252, 21)

    pd.testing.assert_frame_equal(original.iloc[:perturb_at], changed.iloc[:perturb_at])


def test_signal_skip_window_excludes_recent_prices(realistic_prices):
    """With skip=21, the score must not react to the most recent 21 prices.

    This is a sharper claim than 'no future data': it pins the *skip* semantics.
    Perturbing the last 21 prices must leave the final score untouched, because
    the formation window ends 21 days before it.
    """
    original = trailing_momentum(realistic_prices, 252, 21)

    modified = realistic_prices.copy()
    modified.iloc[-21:] *= 3.0

    changed = trailing_momentum(modified, 252, 21)

    # The final score depends on prices up to t-21 only.
    pd.testing.assert_series_equal(original.iloc[-1], changed.iloc[-1])


def test_signal_is_not_forward_filled_across_a_gap(realistic_prices):
    """A NaN in the middle must not be filled with a later value."""
    holed = realistic_prices.copy()
    holed.iloc[500:505, 0] = np.nan

    scores = trailing_momentum(holed, 252, 21)

    # The affected column must show NaN scores in the windows that need those
    # prices, rather than silently borrowing a later observation.
    assert scores.iloc[500 + 21 : 505 + 21, 0].isna().any()


# --------------------------------------------------------------------------- #
# Weight level
# --------------------------------------------------------------------------- #
def test_target_weights_do_not_use_future_data(realistic_prices):
    """Full signal -> weight pipeline must be causal, including the resampling."""
    original = build_target_weights(realistic_prices, 252, 21, 3, "monthly", 0.40)

    modified = realistic_prices.copy()
    modified.iloc[-1] *= 100

    changed = build_target_weights(modified, 252, 21, 3, "monthly", 0.40)

    pd.testing.assert_frame_equal(original.iloc[:-1], changed.iloc[:-1])


def test_monthly_resample_does_not_leak_backwards(realistic_prices):
    """Perturbing mid-month must not change *earlier* months' weights.

    Resampling is a classic leakage vector: `resample().last()` labelled on the
    period end, then reindexed and back-filled instead of forward-filled, would
    hand a month its own future. This asserts it does not.
    """
    original = build_target_weights(realistic_prices, 252, 21, 3, "monthly", 0.40)

    cut = len(realistic_prices) // 2
    modified = realistic_prices.copy()
    modified.iloc[cut:] *= 1.5

    changed = build_target_weights(modified, 252, 21, 3, "monthly", 0.40)

    # Scores at t use prices up to t-21, so weights are unaffected until the
    # perturbation enters the formation window 21 days later.
    pd.testing.assert_frame_equal(original.iloc[: cut + 20], changed.iloc[: cut + 20])


def test_rebalance_dates_are_real_trading_days(realistic_prices):
    """Rebalance dates must be a subset of the price index.

    If they were calendar month-ends, some would not exist in the trading
    calendar; reindexing would drop them and forward-fill would silently carry
    stale weights -- a missed rebalance with no error.
    """
    for freq in ("weekly", "monthly", "quarterly"):
        reb = rebalance_dates(realistic_prices.index, freq)
        assert len(reb) > 0
        assert reb.isin(realistic_prices.index).all(), f"{freq} produced non-trading dates"
        assert reb.is_monotonic_increasing
        assert not reb.duplicated().any()


def test_monthly_rebalance_count_is_plausible(realistic_prices):
    """One rebalance per month, no more and no fewer."""
    reb = rebalance_dates(realistic_prices.index, "monthly")
    months = realistic_prices.index.to_period("M").nunique()
    assert len(reb) == months


# --------------------------------------------------------------------------- #
# Engine level
# --------------------------------------------------------------------------- #
def test_backtest_returns_do_not_use_future_data(realistic_prices):
    """End to end: perturb the final price, assert every prior return is identical."""
    returns = simple_returns(realistic_prices)
    weights = build_target_weights(realistic_prices, 252, 21, 3, "monthly", 0.40)
    original = run_backtest(returns, weights, 100_000, 2, 3, execution_lag=1)

    modified_prices = realistic_prices.copy()
    modified_prices.iloc[-1] *= 100
    modified = run_backtest(
        simple_returns(modified_prices),
        build_target_weights(modified_prices, 252, 21, 3, "monthly", 0.40),
        100_000, 2, 3, execution_lag=1,
    )

    pd.testing.assert_series_equal(
        original.net_returns.iloc[:-1], modified.net_returns.iloc[:-1]
    )
    pd.testing.assert_series_equal(original.equity.iloc[:-1], modified.equity.iloc[:-1])


@pytest.mark.parametrize("lag", [1, 2, 5])
def test_executed_weights_lag_target_weights(realistic_prices, lag):
    """executed[t] must equal target[t-lag] -- the shift, asserted directly.

    Checked with ``warmup_trim=False`` so the frames still share their origin.
    Trimming drops the leading flat rows from both frames, which necessarily
    breaks the row-wise identity at the new first row: the position held on the
    first active day was decided on the last trimmed-away day, so the value that
    ``shift`` would need has been cut off. The relationship still holds on every
    row that survives, which the companion test below checks.
    """
    returns = simple_returns(realistic_prices)
    weights = build_target_weights(realistic_prices, 252, 21, 3, "monthly", 0.40)
    result = run_backtest(returns, weights, execution_lag=lag, warmup_trim=False)

    expected = result.target_weights.shift(lag).fillna(0.0)
    pd.testing.assert_frame_equal(result.executed_weights, expected)


def test_trimmed_executed_weights_still_lag_the_untrimmed_target(realistic_prices):
    """After trimming, executed[t] must still equal the *original* target[t-1]."""
    returns = simple_returns(realistic_prices)
    weights = build_target_weights(realistic_prices, 252, 21, 3, "monthly", 0.40)
    result = run_backtest(returns, weights, execution_lag=1, warmup_trim=True)

    expected = weights.shift(1).fillna(0.0).loc[result.index]
    pd.testing.assert_frame_equal(result.executed_weights, expected)

    # And the first held position must be non-zero -- that is what "trim to the
    # first active day" means.
    assert result.executed_weights.iloc[0].sum() > 0


def test_zero_lag_produces_a_better_result_than_one_lag(realistic_prices):
    """Look-ahead should *help* -- which is exactly why it is dangerous.

    This test documents the bias rather than forbidding it. If removing the lag
    did not improve the result, the lag would not be doing anything and the
    causality machinery would be decorative.
    """
    returns = simple_returns(realistic_prices)
    weights = build_target_weights(realistic_prices, 252, 21, 3, "monthly", 0.40)

    honest = run_backtest(returns, weights, execution_lag=1)
    with pytest.warns(UserWarning):
        cheating = run_backtest(returns, weights, execution_lag=0)

    # Not a strict inequality on every path, but the cheat must not be *worse*
    # on average across a long sample -- it sees one extra day of information.
    assert cheating.net_returns.mean() >= honest.net_returns.mean() - 1e-6


def test_first_position_is_not_held_before_it_is_known(realistic_prices):
    """The engine must hold nothing until the signal has actually formed."""
    returns = simple_returns(realistic_prices)
    weights = build_target_weights(realistic_prices, 252, 21, 3, "monthly", 0.40)
    result = run_backtest(returns, weights, execution_lag=1, warmup_trim=False)

    first_target = weights[weights.sum(axis=1) > 0].index[0]
    held_before = result.executed_weights.loc[:first_target].iloc[:-1]
    assert (held_before.abs().sum(axis=1) == 0).all(), "position held before signal existed"


def test_trimming_does_not_change_realised_returns(realistic_prices):
    """warmup_trim is presentational: it must not alter any realised return."""
    returns = simple_returns(realistic_prices)
    weights = build_target_weights(realistic_prices, 252, 21, 3, "monthly", 0.40)

    trimmed = run_backtest(returns, weights, execution_lag=1, warmup_trim=True)
    full = run_backtest(returns, weights, execution_lag=1, warmup_trim=False)

    overlap = trimmed.index.intersection(full.index)
    pd.testing.assert_series_equal(
        trimmed.net_returns.loc[overlap], full.net_returns.loc[overlap]
    )


# --------------------------------------------------------------------------- #
# Ranking sanity
# --------------------------------------------------------------------------- #
def test_ranking_picks_the_actual_winners(growing_prices):
    """On deterministic data the ranking has one correct answer; check it."""
    scores = trailing_momentum(growing_prices, 252, 21).dropna(how="all")
    weights = top_n_equal_weight(scores, n=1)
    live = weights[weights.sum(axis=1) > 0]

    # AAA compounds fastest, so it must always be the single pick.
    assert (live["AAA"] == 1.0).all()


def test_no_position_before_enough_assets_have_scores(realistic_prices):
    """During warm-up the target must be flat, not a partially-filled portfolio."""
    scores = trailing_momentum(realistic_prices, 252, 21)
    weights = top_n_equal_weight(scores, n=3)

    warmup = weights.loc[scores.notna().sum(axis=1) < 3]
    assert (warmup.sum(axis=1) == 0).all()
