"""
Research-validation tests.
==========================

Sample splitting, baselines, sensitivity, and the end-to-end reproducibility
guarantee. These tests protect the research discipline itself: a bug that lets
the test period leak into parameter selection would not break any calculation,
it would just quietly invalidate the conclusion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform import validation
from quant_platform.portfolio import run_backtest
from quant_platform.returns import simple_returns
from quant_platform.signals import build_target_weights


@pytest.fixture
def backtest(realistic_prices):
    returns = simple_returns(realistic_prices)
    weights = build_target_weights(realistic_prices, 252, 21, 3, "monthly", 0.40)
    return run_backtest(returns, weights, 100_000, 2, 3, name="momentum")


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def test_splits_are_contiguous_and_ordered(backtest):
    splits = validation.make_splits(backtest.index, "2015-12-31", "2017-12-31", "2020-12-31")
    named = {s.name: s for s in splits}

    assert named["training"].end < named["validation"].start
    assert named["validation"].end < named["test"].start
    assert named["full"].start == named["training"].start


def test_splits_do_not_overlap(backtest):
    splits = validation.make_splits(backtest.index, "2015-12-31", "2017-12-31", "2020-12-31")
    idx = backtest.index

    train = set(idx[splits[0].mask(idx)])
    valid = set(idx[splits[1].mask(idx)])
    test = set(idx[splits[2].mask(idx)])

    assert not (train & valid), "training and validation overlap"
    assert not (valid & test), "validation and test overlap"
    assert not (train & test), "training and test overlap"


def test_splits_partition_the_sample(backtest):
    """Every date belongs to exactly one of the three windows."""
    splits = validation.make_splits(backtest.index, "2015-12-31", "2017-12-31", "2020-12-31")
    idx = backtest.index
    covered = sum(int(s.mask(idx).sum()) for s in splits[:3])
    assert covered == len(idx)


def test_splits_reject_out_of_order_boundaries(backtest):
    with pytest.raises(ValueError, match="must increase"):
        validation.make_splits(backtest.index, "2018-12-31", "2015-12-31", "2020-12-31")


def test_slice_rebases_equity_to_initial_capital(backtest):
    sub = backtest.slice("2016-01-01", "2017-12-31")
    first = sub.net_returns.iloc[0]
    assert sub.equity.iloc[0] == pytest.approx(backtest.initial_capital * (1 + first), rel=1e-9)


def test_slice_preserves_returns(backtest):
    """Slicing must not alter any realised return -- only the equity base."""
    sub = backtest.slice("2016-01-01", "2017-12-31")
    original = backtest.net_returns.loc["2016-01-01":"2017-12-31"]
    pd.testing.assert_series_equal(sub.net_returns, original)


def test_evaluate_splits_produces_a_column_per_window(backtest):
    splits = validation.make_splits(backtest.index, "2015-12-31", "2017-12-31", "2020-12-31")
    table = validation.evaluate_splits(backtest, splits)

    assert {"training", "validation", "full"} <= set(table.columns)
    assert "sharpe" in table.index


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
def test_align_results_puts_everything_on_one_window(realistic_prices):
    """Unaligned curves would credit the benchmark with extra compounding."""
    returns = simple_returns(realistic_prices)
    results = validation.build_baselines(realistic_prices, returns, benchmark_symbol="SPY")
    aligned = validation.align_results(results)

    starts = {r.index[0] for r in aligned.values()}
    ends = {r.index[-1] for r in aligned.values()}
    assert len(starts) == 1, f"misaligned starts: {starts}"
    assert len(ends) == 1, f"misaligned ends: {ends}"


def test_align_results_rebases_all_to_same_capital(realistic_prices):
    returns = simple_returns(realistic_prices)
    results = validation.build_baselines(realistic_prices, returns, benchmark_symbol="SPY")
    aligned = validation.align_results(results)

    for r in aligned.values():
        implied = r.equity.iloc[0] / (1 + r.net_returns.iloc[0])
        assert implied == pytest.approx(r.initial_capital, rel=1e-9)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def test_baselines_include_the_required_comparators(realistic_prices):
    returns = simple_returns(realistic_prices)
    baselines = validation.build_baselines(realistic_prices, returns, benchmark_symbol="SPY")

    assert {"SPY buy & hold", "Equal weight (all)", "Random selection",
            "Momentum, zero cost", "Momentum, T+6 exec"} <= set(baselines)


def test_zero_cost_baseline_beats_the_costed_strategy(realistic_prices):
    """Removing costs cannot make a strategy worse."""
    returns = simple_returns(realistic_prices)
    baselines = validation.build_baselines(realistic_prices, returns, benchmark_symbol="SPY")

    weights = build_target_weights(realistic_prices, 252, 21, 3, "monthly", 0.40)
    costed = run_backtest(returns, weights, 100_000, 2, 3)

    assert baselines["Momentum, zero cost"].equity.iloc[-1] >= costed.equity.iloc[-1] - 1e-6


def test_equal_weight_baseline_holds_everything(realistic_prices):
    from quant_platform.signals import equal_weight_all

    weights = equal_weight_all(realistic_prices, "monthly")
    live = weights[weights.sum(axis=1) > 0]

    assert np.allclose(live.sum(axis=1), 1.0)
    assert (live > 0).sum(axis=1).eq(realistic_prices.shape[1]).all()


def test_buy_and_hold_baseline_has_zero_ongoing_turnover(realistic_prices):
    returns = simple_returns(realistic_prices)
    baselines = validation.build_baselines(realistic_prices, returns, benchmark_symbol="SPY")
    spy = baselines["SPY buy & hold"]
    assert spy.turnover.iloc[1:].sum() == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Sensitivity
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def sensitivity(realistic_prices):
    returns = simple_returns(realistic_prices)
    return validation.run_sensitivity(
        realistic_prices, returns,
        lookback_months=[6, 12], holdings_grid=[2, 3],
        rebalance_grid=["monthly"], cost_grid=[0, 10],
        eval_start=pd.Timestamp("2012-01-01"), eval_end=pd.Timestamp("2018-12-31"),
    )


def test_sensitivity_covers_the_full_grid(sensitivity):
    assert len(sensitivity) == 2 * 2 * 1 * 2
    assert set(sensitivity["lookback_months"]) == {6, 12}
    assert set(sensitivity["holdings"]) == {2, 3}


def test_sensitivity_higher_cost_lowers_sharpe(sensitivity):
    """Within every parameter cell, more cost must not raise Sharpe."""
    for (lb, h, reb), grp in sensitivity.groupby(["lookback_months", "holdings", "rebalance"]):
        ordered = grp.sort_values("cost_bps")
        sharpes = ordered["sharpe"].to_numpy()
        assert np.all(np.diff(sharpes) <= 1e-9), f"cost raised Sharpe at {lb}/{h}/{reb}"


def test_sensitivity_reports_exposure(sensitivity):
    """Needed to interpret capped low-holdings cells honestly."""
    assert "avg_exposure" in sensitivity.columns
    assert (sensitivity["avg_exposure"] > 0).all()


def test_every_marginal_dimension_reaches_the_report(sensitivity):
    """Each grid dimension must produce a table the report actually renders.

    The report used to name four table keys explicitly while the marginals were
    keyed by the raw column names, so ``lookback_months`` and ``cost_bps`` never
    matched and two of the four tables silently rendered as "No data." Nothing
    failed -- the report simply omitted a third of its own sensitivity analysis.
    """
    marginals = validation.sensitivity_marginals(sensitivity, "sharpe")
    assert set(marginals) == {"lookback_months", "holdings", "rebalance", "cost_bps"}

    # run.py prefixes each key; the report renders every key with that prefix.
    tables = {f"sens_{k}": df for k, df in marginals.items()}
    rendered = [k for k in tables if k.startswith("sens_")]
    assert len(rendered) == len(marginals)
    for key in rendered:
        assert not tables[key].empty, f"{key} would render as 'No data.'"


def test_sensitivity_cells_share_one_evaluation_window(realistic_prices):
    """Every grid cell must be measured over the same dates.

    Cells go live at different times -- a 3-month lookback forms a signal long
    before a 12-month one, and a quarterly rebalance waits for the next quarter
    end. If each were scored on its own start date, the grid would compare
    periods as well as parameters, and the earliest months of this sample are
    exactly the ones that dominate the result.
    """
    returns = simple_returns(realistic_prices)
    sens = validation.run_sensitivity(
        realistic_prices, returns,
        lookback_months=[3, 12],            # very different warm-up lengths
        holdings_grid=[3],
        rebalance_grid=["monthly", "quarterly"],  # different first rebalance dates
        cost_grid=[0, 10],
        eval_start=pd.Timestamp("2011-01-01"), eval_end=pd.Timestamp("2018-12-31"),
    )

    assert not sens.empty
    assert sens["n_periods"].nunique() == 1, (
        f"grid cells span different windows: {sorted(sens['n_periods'].unique())}"
    )


def test_sensitivity_is_reproducible(realistic_prices):
    returns = simple_returns(realistic_prices)
    kwargs = dict(
        lookback_months=[12], holdings_grid=[3], rebalance_grid=["monthly"], cost_grid=[5],
        eval_start=pd.Timestamp("2012-01-01"), eval_end=pd.Timestamp("2018-12-31"),
    )
    a = validation.run_sensitivity(realistic_prices, returns, **kwargs)
    b = validation.run_sensitivity(realistic_prices, returns, **kwargs)
    pd.testing.assert_frame_equal(a, b)


def test_marginals_summarise_each_dimension(sensitivity):
    marginals = validation.sensitivity_marginals(sensitivity, "sharpe")
    assert {"lookback_months", "holdings", "cost_bps"} <= set(marginals)
    for df in marginals.values():
        assert {"mean", "std", "min", "max"} <= set(df.columns)


# --------------------------------------------------------------------------- #
# Regimes
# --------------------------------------------------------------------------- #
def test_regime_analysis_produces_expected_rows(backtest, realistic_prices):
    bench = simple_returns(realistic_prices)["SPY"]
    regimes = validation.regime_analysis(backtest.net_returns, bench)

    assert not regimes.empty
    assert any("Vol:" in str(i) for i in regimes.index)
    assert "strategy_sharpe" in regimes.columns


def test_regime_percentages_are_sane(backtest, realistic_prices):
    bench = simple_returns(realistic_prices)["SPY"]
    regimes = validation.regime_analysis(backtest.net_returns, bench)
    assert (regimes["pct_of_sample"] > 0).all()
    assert (regimes["pct_of_sample"] <= 1.0).all()


def test_leave_one_year_out_covers_every_full_year(backtest, realistic_prices):
    bench = simple_returns(realistic_prices)["SPY"]
    loyo = validation.leave_one_year_out(backtest.net_returns, bench)

    assert not loyo.empty
    assert {"sharpe_edge_ex", "edge_change", "strategy_return_in_year"} <= set(loyo.columns)
    assert "full_edge" in loyo.attrs and "sharpe_se" in loyo.attrs


def test_leave_one_year_out_edge_change_is_consistent(backtest, realistic_prices):
    """edge_change must equal (edge without the year) - (full-sample edge)."""
    bench = simple_returns(realistic_prices)["SPY"]
    loyo = validation.leave_one_year_out(backtest.net_returns, bench)

    np.testing.assert_allclose(
        loyo["edge_change"].to_numpy(),
        (loyo["sharpe_edge_ex"] - loyo.attrs["full_edge"]).to_numpy(),
        rtol=1e-12,
    )


def test_leave_one_year_out_detects_a_planted_outlier_year():
    """A strategy whose entire edge is one year must be flagged as such."""
    idx = pd.bdate_range("2010-01-01", periods=252 * 6)
    rng = np.random.default_rng(21)

    bench = pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)
    strat = bench.copy()
    # Give 2012 alone a large, steady advantage; every other year is identical.
    strat[strat.index.year == 2012] += 0.004

    loyo = validation.leave_one_year_out(strat, bench)

    assert loyo["edge_change"].idxmin() == 2012, "should identify the planted year"
    # Removing 2012 should collapse the edge to roughly nothing.
    assert abs(float(loyo.loc[2012, "sharpe_edge_ex"])) < 0.05
    assert loyo.attrs["full_edge"] > 0.2


def test_sharpe_standard_error_shrinks_with_sample_length():
    from quant_platform.metrics import sharpe_standard_error

    short = sharpe_standard_error(1.0, 252 * 2)
    long = sharpe_standard_error(1.0, 252 * 20)

    assert short > long
    # SE(S) = sqrt((1 + S^2/2)/years); at S=1, 2 years -> sqrt(1.5/2)
    assert short == pytest.approx(np.sqrt(1.5 / 2), rel=1e-9)


def test_calendar_year_table_excess_is_the_difference(backtest, realistic_prices):
    bench = simple_returns(realistic_prices)["SPY"]
    years = validation.calendar_year_table(backtest.net_returns, bench)

    assert "excess" in years.columns
    np.testing.assert_allclose(
        years["excess"].to_numpy(),
        (years["strategy"] - years["benchmark"]).to_numpy(),
        rtol=1e-12,
    )


# --------------------------------------------------------------------------- #
# End-to-end reproducibility
# --------------------------------------------------------------------------- #
def test_full_pipeline_is_reproducible(realistic_prices):
    """The spec's requirement: the same config twice gives identical results."""
    returns = simple_returns(realistic_prices)

    def pipeline():
        weights = build_target_weights(realistic_prices, 252, 21, 3, "monthly", 0.40)
        result = run_backtest(returns, weights, 100_000, 2, 3, execution_lag=1)
        splits = validation.make_splits(result.index, "2015-12-31", "2017-12-31", "2020-12-31")
        return result, validation.evaluate_splits(result, splits)

    (r1, m1), (r2, m2) = pipeline(), pipeline()

    pd.testing.assert_series_equal(r1.equity, r2.equity)
    pd.testing.assert_frame_equal(m1, m2)
    assert r1.equity.iloc[-1] == r2.equity.iloc[-1]
