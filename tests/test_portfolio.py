"""
Portfolio-engine accounting tests.
==================================

These are the tests that establish the engine can be trusted. Each one pins a
property that has an unambiguous right answer, so a failure localises the bug.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.costs import CostModel, compute_turnover
from quant_platform.portfolio import buy_and_hold, run_backtest
from quant_platform.returns import simple_returns
from quant_platform.signals import apply_max_weight, build_target_weights, top_n_equal_weight


# --------------------------------------------------------------------------- #
# Constant prices
# --------------------------------------------------------------------------- #
def test_constant_prices_give_zero_return(constant_prices):
    """Prices never change -> every strategy return must be exactly zero."""
    returns = simple_returns(constant_prices)
    weights = pd.DataFrame(0.25, index=constant_prices.index, columns=constant_prices.columns)

    result = run_backtest(returns, weights, initial_capital=100_000,
                          commission_bps=0, slippage_bps=0)

    assert np.allclose(result.gross_returns.fillna(0), 0.0), "flat prices produced non-zero P&L"
    # Equity is flat apart from the one-off entry, which is free at zero cost.
    assert result.equity.iloc[-1] == pytest.approx(100_000, rel=1e-12)


def test_constant_prices_with_costs_only_lose_money(constant_prices):
    """With flat prices, any cost must make the portfolio strictly poorer."""
    returns = simple_returns(constant_prices)
    weights = pd.DataFrame(0.25, index=constant_prices.index, columns=constant_prices.columns)

    result = run_backtest(returns, weights, initial_capital=100_000,
                          commission_bps=10, slippage_bps=10)

    assert result.equity.iloc[-1] < 100_000
    # The only trade is the initial entry: turnover 1.0 at 20 bps.
    assert result.turnover.sum() == pytest.approx(1.0, abs=1e-12)
    assert result.equity.iloc[-1] == pytest.approx(100_000 * (1 - 0.0020), rel=1e-9)


# --------------------------------------------------------------------------- #
# Buy and hold
# --------------------------------------------------------------------------- #
def test_buy_and_hold_matches_asset_return(growing_prices):
    """After entry, a 100% position must track the asset exactly."""
    returns = simple_returns(growing_prices)
    result = buy_and_hold(returns, "AAA", initial_capital=1.0,
                          commission_bps=0, slippage_bps=0, execution_lag=1)

    # Engine growth from its own start, versus the asset over the same window.
    engine_growth = result.equity.iloc[-1] / result.initial_capital
    asset = growing_prices["AAA"]
    asset_growth = asset.loc[result.index[-1]] / asset.loc[result.index[0] - pd.offsets.BDay(1)]

    assert engine_growth == pytest.approx(asset_growth, rel=1e-9)


def test_buy_and_hold_has_no_turnover_after_entry(growing_prices):
    """A static position trades once and then never again."""
    returns = simple_returns(growing_prices)
    result = buy_and_hold(returns, "AAA", commission_bps=1, slippage_bps=1)

    assert result.turnover.iloc[0] == pytest.approx(1.0)
    assert result.turnover.iloc[1:].sum() == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Cost monotonicity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bps", [0, 1, 5, 10, 25, 100])
def test_higher_costs_never_improve_returns(realistic_prices, bps):
    """Net return must be monotonically non-increasing in the cost rate."""
    returns = simple_returns(realistic_prices)
    weights = build_target_weights(realistic_prices, 252, 21, 3, "monthly", 0.40)

    free = run_backtest(returns, weights, commission_bps=0, slippage_bps=0)
    charged = run_backtest(returns, weights, commission_bps=bps, slippage_bps=0)

    assert charged.net_returns.sum() <= free.net_returns.sum() + 1e-12
    assert charged.equity.iloc[-1] <= free.equity.iloc[-1] + 1e-9


def test_cost_monotonic_across_a_ladder(realistic_prices):
    """Final equity must decrease monotonically as costs rise."""
    returns = simple_returns(realistic_prices)
    weights = build_target_weights(realistic_prices, 252, 21, 3, "monthly", 0.40)

    finals = [
        run_backtest(returns, weights, commission_bps=b, slippage_bps=0).equity.iloc[-1]
        for b in (0, 2, 5, 10, 20, 50)
    ]
    assert all(a >= b for a, b in zip(finals, finals[1:])), f"not monotonic: {finals}"


def test_costs_equal_turnover_times_rate(realistic_prices):
    """The cost series must be exactly turnover x rate -- no hidden fudge."""
    returns = simple_returns(realistic_prices)
    weights = build_target_weights(realistic_prices, 252, 21, 3, "monthly", 0.40)
    result = run_backtest(returns, weights, commission_bps=2, slippage_bps=3)

    expected = result.turnover * 5 / 10_000
    pd.testing.assert_series_equal(result.costs, expected, check_names=False)


def test_net_equals_gross_minus_costs(realistic_prices):
    """The fundamental accounting identity."""
    returns = simple_returns(realistic_prices)
    weights = build_target_weights(realistic_prices, 252, 21, 3, "monthly", 0.40)
    r = run_backtest(returns, weights, commission_bps=2, slippage_bps=3)

    pd.testing.assert_series_equal(r.net_returns, r.gross_returns - r.costs, check_names=False)


# --------------------------------------------------------------------------- #
# Hand-computed toy example
# --------------------------------------------------------------------------- #
def test_toy_example_matches_manual_arithmetic(toy_prices):
    """A six-day, two-asset case worked out by hand.

    Day:        0      1      2      3      4       5
    AAA:      100    110    121    121    133.1   133.1
    return:    --   +10%   +10%     0%   +10%      0%

    Target is 100% AAA on every day, including day 0. With ``execution_lag=1``
    the weight decided at the close of day 0 is held over (day 0, day 1], so the
    portfolio *does* earn day 1's +10%. Warm-up trim then starts the reported
    series on day 1, the first day a position is actually held.

    Captured returns are therefore days 1-5: [+10%, +10%, 0%, +10%, 0%].
    """
    returns = simple_returns(toy_prices)
    weights = pd.DataFrame({"AAA": [1.0] * 6, "BBB": [0.0] * 6}, index=toy_prices.index)

    result = run_backtest(returns, weights, initial_capital=1000.0,
                          commission_bps=0, slippage_bps=0, execution_lag=1,
                          warmup_trim=True)

    assert result.net_returns.tolist() == pytest.approx([0.10, 0.10, 0.0, 0.10, 0.0], abs=1e-12)

    # 1000 * 1.1 * 1.1 * 1.0 * 1.1 * 1.0 = 1331
    assert result.equity.iloc[-1] == pytest.approx(1331.0, rel=1e-12)
    # Equivalently: AAA itself went 100 -> 133.1, i.e. +33.1%.
    assert result.equity.iloc[-1] / 1000.0 == pytest.approx(133.1 / 100.0, rel=1e-12)


def test_toy_example_execution_lag_shifts_the_entry(toy_prices):
    """A longer lag must skip the earliest returns, never gain extra ones."""
    returns = simple_returns(toy_prices)
    weights = pd.DataFrame({"AAA": [1.0] * 6, "BBB": [0.0] * 6}, index=toy_prices.index)

    lag1 = run_backtest(returns, weights, initial_capital=1000.0, commission_bps=0,
                        slippage_bps=0, execution_lag=1)
    lag3 = run_backtest(returns, weights, initial_capital=1000.0, commission_bps=0,
                        slippage_bps=0, execution_lag=3)

    # Entering two days later misses days 1 and 2 (+10% each): 1000 * 1.1 = 1100.
    assert lag3.net_returns.tolist() == pytest.approx([0.0, 0.10, 0.0], abs=1e-12)
    assert lag3.equity.iloc[-1] == pytest.approx(1100.0, rel=1e-12)
    assert lag3.equity.iloc[-1] < lag1.equity.iloc[-1]


def test_toy_turnover_is_hand_checkable(toy_prices):
    """Switching a full position between two assets is 2.0 of two-way turnover."""
    idx = toy_prices.index
    weights = pd.DataFrame(
        {"AAA": [1, 1, 1, 0, 0, 0], "BBB": [0, 0, 0, 1, 1, 1]},
        index=idx, dtype=float,
    )
    turnover = compute_turnover(weights)

    assert turnover.iloc[0] == pytest.approx(1.0)  # entry from cash
    assert turnover.iloc[3] == pytest.approx(2.0)  # sell 1.0 + buy 1.0
    assert turnover.sum() == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# Weight constraints
# --------------------------------------------------------------------------- #
def test_weights_never_exceed_capital(realistic_prices):
    """The constraint from the spec: gross weight <= 100% of capital."""
    weights = build_target_weights(realistic_prices, 252, 21, 3, "monthly", 0.40)
    assert (weights.abs().sum(axis=1) <= 1.000001).all()


def test_long_only_weights_are_non_negative(realistic_prices):
    weights = build_target_weights(realistic_prices, 252, 21, 3, "monthly", 0.40)
    assert (weights >= -1e-12).all().all()


def test_top_n_selects_exactly_n(growing_prices):
    """With distinct scores, exactly n names must be held, equally weighted."""
    from quant_platform.signals import trailing_momentum

    scores = trailing_momentum(growing_prices, 252, 21).dropna(how="all")
    weights = top_n_equal_weight(scores, n=2)
    live = weights[weights.sum(axis=1) > 0]

    assert ((live > 0).sum(axis=1) == 2).all()
    assert np.allclose(live.sum(axis=1), 1.0)
    # AAA and BBB have the highest growth rates, so they must be the picks.
    assert (live["AAA"] > 0).all() and (live["BBB"] > 0).all()
    assert np.allclose(live[["CCC", "DDD"]].to_numpy(), 0.0)


def test_max_weight_cap_is_enforced():
    """A cap below 1/n leaves the residual in cash rather than silently renormalising."""
    idx = pd.bdate_range("2020-01-01", periods=3)
    weights = pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [0.0, 0.0, 0.0]}, index=idx)

    capped = apply_max_weight(weights, 0.40)

    assert np.allclose(capped["A"], 0.40)
    assert np.allclose(capped.sum(axis=1), 0.40), "excess should stay in cash"


def test_max_weight_redistributes_to_uncapped_names():
    """Excess from a capped name flows to names with headroom, not to cash."""
    idx = pd.bdate_range("2020-01-01", periods=1)
    weights = pd.DataFrame({"A": [0.8], "B": [0.1], "C": [0.1]}, index=idx)

    capped = apply_max_weight(weights, 0.50)

    assert capped.loc[idx[0], "A"] == pytest.approx(0.50)
    assert capped.sum(axis=1).iloc[0] == pytest.approx(1.0), "total should be preserved"
    assert capped.loc[idx[0], "B"] > 0.1 and capped.loc[idx[0], "C"] > 0.1


def test_uninvested_residual_earns_cash_rate():
    """When a cap binds, the cash residual must actually be credited."""
    idx = pd.bdate_range("2020-01-01", periods=253)
    returns = pd.DataFrame(0.0, index=idx, columns=["A"])
    weights = pd.DataFrame(0.40, index=idx, columns=["A"])

    result = run_backtest(returns, weights, initial_capital=1.0, commission_bps=0,
                          slippage_bps=0, cash_rate=0.10, periods_per_year=252)

    # 60% in cash at 10% annualised, over the periods actually held.
    n = len(result.net_returns)
    assert result.equity.iloc[-1] == pytest.approx((1 + 0.60 * 0.10 / 252) ** n, rel=1e-9)


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def test_identical_config_produces_identical_results(realistic_prices):
    """Running the same configuration twice must be bit-identical."""
    returns = simple_returns(realistic_prices)

    def once():
        w = build_target_weights(realistic_prices, 252, 21, 3, "monthly", 0.40)
        return run_backtest(returns, w, 100_000, 2, 3, execution_lag=1)

    a, b = once(), once()

    pd.testing.assert_series_equal(a.equity, b.equity)
    pd.testing.assert_series_equal(a.net_returns, b.net_returns)
    pd.testing.assert_series_equal(a.turnover, b.turnover)
    pd.testing.assert_frame_equal(a.executed_weights, b.executed_weights)


def test_random_baseline_is_seeded(realistic_prices):
    """The random comparator must be a fixed benchmark, not a moving target."""
    from quant_platform.signals import random_selection_weights

    a = random_selection_weights(realistic_prices, 3, "monthly", 0.40, seed=42)
    b = random_selection_weights(realistic_prices, 3, "monthly", 0.40, seed=42)
    c = random_selection_weights(realistic_prices, 3, "monthly", 0.40, seed=43)

    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c), "different seeds should give different draws"


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def test_engine_rejects_negative_lag(realistic_prices):
    returns = simple_returns(realistic_prices)
    weights = build_target_weights(realistic_prices, 252, 21, 3, "monthly", 0.40)
    with pytest.raises(ValueError, match="execution_lag"):
        run_backtest(returns, weights, execution_lag=-1)


def test_engine_warns_on_zero_lag(realistic_prices):
    """Zero lag is look-ahead; it must be loud, not silent."""
    returns = simple_returns(realistic_prices)
    weights = build_target_weights(realistic_prices, 252, 21, 3, "monthly", 0.40)
    with pytest.warns(UserWarning, match="look-ahead"):
        run_backtest(returns, weights, execution_lag=0)


def test_engine_rejects_duplicate_dates(realistic_prices):
    returns = simple_returns(realistic_prices)
    weights = build_target_weights(realistic_prices, 252, 21, 3, "monthly", 0.40)
    dupe = pd.concat([returns, returns.iloc[[0]]]).sort_index()
    with pytest.raises(ValueError, match="duplicate"):
        run_backtest(dupe, weights)


def test_cost_model_total_is_additive():
    assert CostModel(2, 3).total_bps == pytest.approx(5.0)
    assert CostModel(0, 0).total_bps == pytest.approx(0.0)
