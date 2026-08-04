"""
Metric tests.
=============

Each metric is checked against a case where the answer is known analytically,
rather than against a snapshot of its own output. A snapshot test only proves the
code still does what it did; these prove it does the right thing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform import metrics


@pytest.fixture
def flat_returns():
    idx = pd.bdate_range("2020-01-01", periods=504)
    return pd.Series(0.0, index=idx)


@pytest.fixture
def constant_growth():
    """Exactly 0.05% per day for two years -- CAGR is computable by hand."""
    idx = pd.bdate_range("2020-01-01", periods=504)
    return pd.Series(0.0005, index=idx)


@pytest.fixture
def noisy_returns():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=2520)
    return pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)


# --------------------------------------------------------------------------- #
# Return / volatility / Sharpe
# --------------------------------------------------------------------------- #
def test_annualized_return_of_constant_growth(constant_growth):
    """CAGR must invert the compounding exactly."""
    expected = (1.0005) ** 252 - 1
    assert metrics.annualized_return(constant_growth) == pytest.approx(expected, rel=1e-9)


def test_annualized_return_of_flat_is_zero(flat_returns):
    assert metrics.annualized_return(flat_returns) == pytest.approx(0.0, abs=1e-12)


def test_annualized_volatility_scales_by_sqrt_time():
    idx = pd.bdate_range("2020-01-01", periods=1000)
    r = pd.Series(np.tile([0.01, -0.01], 500), index=idx)
    # Sample std of an alternating +/-1% series is ~0.01 (ddof=1).
    assert metrics.annualized_volatility(r) == pytest.approx(0.01 * np.sqrt(252), rel=1e-3)


def test_zero_volatility_gives_nan_sharpe(constant_growth):
    """A constant return has no risk; Sharpe must be NaN, not infinity."""
    assert np.isnan(metrics.annualized_sharpe(constant_growth))


def test_sharpe_matches_manual_formula(noisy_returns):
    manual = np.sqrt(252) * noisy_returns.mean() / noisy_returns.std(ddof=1)
    assert metrics.annualized_sharpe(noisy_returns) == pytest.approx(manual, rel=1e-12)


def test_sharpe_respects_risk_free_rate(noisy_returns):
    """A higher hurdle must lower the Sharpe ratio."""
    zero = metrics.annualized_sharpe(noisy_returns, risk_free_rate=0.0)
    high = metrics.annualized_sharpe(noisy_returns, risk_free_rate=0.05)
    assert high < zero


def test_sharpe_is_scale_invariant_in_the_right_way(noisy_returns):
    """Doubling returns roughly doubles Sharpe when rf = 0."""
    base = metrics.annualized_sharpe(noisy_returns)
    doubled = metrics.annualized_sharpe(noisy_returns * 2)
    assert doubled == pytest.approx(base, rel=1e-9)  # mean and std both double


def test_short_series_returns_nan():
    idx = pd.bdate_range("2020-01-01", periods=1)
    assert np.isnan(metrics.annualized_sharpe(pd.Series([0.01], index=idx)))


def test_sortino_exceeds_sharpe_for_right_skewed_returns():
    """With upside outliers, downside deviation < total deviation.

    The series needs genuine negative returns for downside deviation to be
    defined at all; large *positive* outliers then inflate the Sharpe denominator
    without touching the Sortino one, so Sortino must come out higher.
    """
    idx = pd.bdate_range("2020-01-01", periods=504)
    rng = np.random.default_rng(11)
    values = rng.normal(0.0002, 0.008, 504)
    values[::50] = 0.06  # periodic upside shocks
    r = pd.Series(values, index=idx)

    assert (r < 0).any(), "fixture must contain losses for downside deviation to exist"
    assert metrics.sortino_ratio(r) > metrics.annualized_sharpe(r)


def test_sortino_ignores_upside_volatility():
    """A series with no negative returns has zero downside deviation -> NaN."""
    idx = pd.bdate_range("2020-01-01", periods=100)
    r = pd.Series(np.linspace(0.001, 0.02, 100), index=idx)
    assert np.isnan(metrics.sortino_ratio(r))


# --------------------------------------------------------------------------- #
# Drawdown
# --------------------------------------------------------------------------- #
def test_drawdown_of_monotonic_curve_is_zero():
    idx = pd.bdate_range("2020-01-01", periods=100)
    equity = pd.Series(np.linspace(100, 200, 100), index=idx)
    assert metrics.maximum_drawdown(equity) == pytest.approx(0.0, abs=1e-12)


def test_maximum_drawdown_hand_computed():
    """100 -> 120 -> 60 -> 90: worst decline is 60/120 - 1 = -50%."""
    idx = pd.bdate_range("2020-01-01", periods=4)
    equity = pd.Series([100.0, 120.0, 60.0, 90.0], index=idx)
    assert metrics.maximum_drawdown(equity) == pytest.approx(-0.5, rel=1e-12)


def test_drawdown_series_is_never_positive():
    idx = pd.bdate_range("2020-01-01", periods=500)
    rng = np.random.default_rng(1)
    equity = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, 500)), index=idx)
    assert (metrics.drawdown_series(equity) <= 1e-12).all()


def test_drawdown_details_locates_peak_and_trough():
    idx = pd.bdate_range("2020-01-01", periods=5)
    equity = pd.Series([100.0, 150.0, 75.0, 120.0, 160.0], index=idx)
    d = metrics.drawdown_details(equity)

    assert d["max_drawdown"] == pytest.approx(-0.5)
    assert d["peak_date"] == idx[1]
    assert d["trough_date"] == idx[2]
    assert d["recovery_date"] == idx[4]
    assert d["still_underwater"] is False


def test_drawdown_details_flags_unrecovered():
    idx = pd.bdate_range("2020-01-01", periods=3)
    equity = pd.Series([100.0, 200.0, 150.0], index=idx)
    assert metrics.drawdown_details(equity)["still_underwater"] is True


def test_calmar_is_cagr_over_drawdown():
    idx = pd.bdate_range("2020-01-01", periods=504)
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.0005, 0.01, 504), index=idx)
    equity = (1 + r).cumprod()

    expected = metrics.annualized_return(r) / abs(metrics.maximum_drawdown(equity))
    assert metrics.calmar_ratio(r, equity) == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------- #
# Benchmark-relative
# --------------------------------------------------------------------------- #
def test_beta_against_self_is_one(noisy_returns):
    assert metrics.beta(noisy_returns, noisy_returns) == pytest.approx(1.0, rel=1e-9)


def test_beta_of_scaled_series(noisy_returns):
    """A 2x-levered clone must have beta 2."""
    assert metrics.beta(noisy_returns * 2, noisy_returns) == pytest.approx(2.0, rel=1e-9)


def test_beta_of_independent_series_is_near_zero():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2015-01-01", periods=5000)
    a = pd.Series(rng.normal(0, 0.01, 5000), index=idx)
    b = pd.Series(rng.normal(0, 0.01, 5000), index=idx)
    assert abs(metrics.beta(a, b)) < 0.06


def test_alpha_against_self_is_zero(noisy_returns):
    assert metrics.alpha(noisy_returns, noisy_returns) == pytest.approx(0.0, abs=1e-9)


def test_alpha_detects_a_constant_edge(noisy_returns):
    """Adding a fixed 2 bps/day must show up as ~5% annualised alpha."""
    enhanced = noisy_returns + 0.0002
    assert metrics.alpha(enhanced, noisy_returns) == pytest.approx(0.0002 * 252, rel=1e-6)


def test_information_ratio_against_self_is_nan(noisy_returns):
    """Zero tracking error must give NaN, not a division blow-up."""
    assert np.isnan(metrics.information_ratio(noisy_returns, noisy_returns))


def test_information_ratio_positive_when_outperforming(noisy_returns):
    """Positive mean active return with non-zero tracking error gives IR > 0.

    The active return must actually vary: adding a *constant* to a series gives
    zero tracking error, and IR is then correctly NaN rather than positive (see
    the test above).
    """
    rng = np.random.default_rng(12)
    active = pd.Series(rng.normal(0.0002, 0.002, len(noisy_returns)), index=noisy_returns.index)

    ir = metrics.information_ratio(noisy_returns + active, noisy_returns)

    assert ir > 0
    assert ir == pytest.approx(np.sqrt(252) * active.mean() / active.std(ddof=1), rel=1e-9)


def test_benchmark_metrics_align_on_dates():
    """Misaligned indices must be inner-joined, not silently mis-paired."""
    idx_a = pd.bdate_range("2020-01-01", periods=100)
    idx_b = pd.bdate_range("2020-02-01", periods=100)
    rng = np.random.default_rng(4)
    a = pd.Series(rng.normal(0, 0.01, 100), index=idx_a)
    b = pd.Series(rng.normal(0, 0.01, 100), index=idx_b)
    assert not np.isnan(metrics.beta(a, b))


# --------------------------------------------------------------------------- #
# Tail risk
# --------------------------------------------------------------------------- #
def test_value_at_risk_is_the_empirical_quantile(noisy_returns):
    assert metrics.value_at_risk(noisy_returns, 0.05) == pytest.approx(
        np.quantile(noisy_returns, 0.05), rel=1e-12
    )


def test_expected_shortfall_is_worse_than_var(noisy_returns):
    """CVaR conditions on the tail, so it must be at least as bad as VaR."""
    assert metrics.expected_shortfall(noisy_returns) <= metrics.value_at_risk(noisy_returns)


def test_var_of_known_distribution():
    """Uniform returns on [-0.10, 0]: the 5% quantile is -0.095."""
    idx = pd.bdate_range("2020-01-01", periods=1001)
    r = pd.Series(np.linspace(-0.10, 0.0, 1001), index=idx)
    assert metrics.value_at_risk(r, 0.05) == pytest.approx(-0.095, abs=1e-3)


# --------------------------------------------------------------------------- #
# Activity
# --------------------------------------------------------------------------- #
def test_win_rate_counts_positive_periods():
    idx = pd.bdate_range("2020-01-01", periods=10)
    r = pd.Series([0.01] * 6 + [-0.01] * 4, index=idx)
    assert metrics.win_rate(r) == pytest.approx(0.6)


def test_win_rate_excludes_zeros():
    """Zero is not a win."""
    idx = pd.bdate_range("2020-01-01", periods=4)
    r = pd.Series([0.01, 0.0, 0.0, -0.01], index=idx)
    assert metrics.win_rate(r) == pytest.approx(0.25)


def test_annual_turnover_scales_to_years():
    """1.0 of turnover per day for one year is 252x annual turnover."""
    idx = pd.bdate_range("2020-01-01", periods=252)
    t = pd.Series(1.0, index=idx)
    assert metrics.annual_turnover(t) == pytest.approx(252.0, rel=1e-9)


# --------------------------------------------------------------------------- #
# Rolling and tabular
# --------------------------------------------------------------------------- #
def test_rolling_sharpe_converges_to_full_sample(noisy_returns):
    """A rolling window covering the whole sample equals the full-sample value."""
    rs = metrics.rolling_sharpe(noisy_returns, window=len(noisy_returns))
    assert rs.iloc[-1] == pytest.approx(metrics.annualized_sharpe(noisy_returns), rel=1e-9)


def test_monthly_table_rows_compound_to_annual():
    """Each row's monthly returns must compound to that row's Year figure."""
    idx = pd.bdate_range("2020-01-01", "2021-12-31")
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0.0004, 0.008, len(idx)), index=idx)

    table = metrics.monthly_return_table(r)
    for year in table.index:
        months = table.loc[year].drop("Year").dropna()
        assert (1 + months).prod() - 1 == pytest.approx(table.loc[year, "Year"], rel=1e-9)


def test_summary_metrics_has_no_missing_headline_fields(noisy_returns):
    rng = np.random.default_rng(6)
    bench = pd.Series(rng.normal(0.0003, 0.011, len(noisy_returns)), index=noisy_returns.index)
    turnover = pd.Series(0.02, index=noisy_returns.index)

    s = metrics.summary_metrics(noisy_returns, benchmark=bench, turnover=turnover)

    for field in ("cagr", "ann_volatility", "sharpe", "sortino", "max_drawdown",
                  "calmar", "win_rate", "beta", "alpha", "information_ratio",
                  "annual_turnover", "var_95", "cvar_95"):
        assert field in s.index, f"missing {field}"
        assert not (isinstance(s[field], float) and np.isnan(s[field])), f"{field} is NaN"


def test_metrics_ignore_nans(noisy_returns):
    """A few NaNs must not poison the estimates."""
    holed = noisy_returns.copy()
    holed.iloc[[10, 20, 30]] = np.nan

    assert not np.isnan(metrics.annualized_sharpe(holed))
    assert metrics.annualized_sharpe(holed) == pytest.approx(
        metrics.annualized_sharpe(noisy_returns.drop(noisy_returns.index[[10, 20, 30]])),
        rel=1e-12,
    )


def test_empty_input_returns_nan_not_exception():
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    assert np.isnan(metrics.annualized_sharpe(empty))
    assert np.isnan(metrics.annualized_return(empty))
    assert np.isnan(metrics.win_rate(empty))
