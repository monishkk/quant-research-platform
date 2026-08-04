"""Tests for the significance module.

The bootstrap is stochastic, so these tests pin the things that must hold
regardless of the draw: shape and range invariants, reproducibility under a
fixed seed, and monotone responses to inputs whose effect is known a priori
(more trials must raise the hurdle; fatter tails must lower confidence).

Where a closed form exists -- the iid bootstrap degenerate case, PSR at zero
edge -- it is checked against that form rather than against a recorded number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform import significance as sig


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _series_with_exact_sharpe(
    seed: int, daily_sharpe: float, n: int = 1500, vol: float = 0.01
) -> pd.Series:
    """Noise rescaled so its realised per-period Sharpe is *exactly* the target.

    Drawing from ``normal(mu, sigma)`` and assuming the sample Sharpe equals
    ``mu / sigma`` is wrong at these sample sizes: over 1,500 days a genuinely
    zero-mean series routinely realises an annualised Sharpe of +/-0.25, and a
    fixture built that way makes a test's meaning depend on the luck of its
    seed. Standardising first removes the sampling error from the *fixture*, so
    a failure can only come from the code under test.
    """
    idx = pd.bdate_range("2010-01-01", periods=n)
    raw = np.random.default_rng(seed).standard_normal(n)
    z = (raw - raw.mean()) / raw.std(ddof=1)  # exactly mean 0, sd 1
    return pd.Series(vol * (z + daily_sharpe), index=idx)


@pytest.fixture
def flat_returns() -> pd.Series:
    """A strategy with *exactly* no edge: realised Sharpe is 0 to machine precision."""
    return _series_with_exact_sharpe(seed=0, daily_sharpe=0.0)


@pytest.fixture
def edged_returns() -> pd.Series:
    """A large, unambiguous edge: exactly 0.1 daily, ~1.59 annualised."""
    return _series_with_exact_sharpe(seed=1, daily_sharpe=0.1)


def test_fixtures_have_the_sharpe_they_claim(flat_returns, edged_returns):
    """Guard the guard -- if this drifts, several tests below lose their meaning."""
    from quant_platform import metrics
    assert metrics.annualized_sharpe(flat_returns) == pytest.approx(0.0, abs=1e-12)
    assert metrics.annualized_sharpe(edged_returns) == pytest.approx(
        0.1 * np.sqrt(252), rel=1e-9
    )


# --------------------------------------------------------------------------- #
# Bootstrap index construction
# --------------------------------------------------------------------------- #
def test_bootstrap_indices_have_right_shape_and_range():
    idx = sig.stationary_bootstrap_indices(100, n_boot=50, block_length=10,
                                           rng=np.random.default_rng(0))
    assert idx.shape == (50, 100)
    assert idx.min() >= 0 and idx.max() < 100


def test_block_length_one_degenerates_to_iid_bootstrap():
    """With block_length=1 every step starts a new block, so draws are iid.

    The check is that consecutive indices are almost never consecutive values,
    which is what distinguishes an iid resample from a block resample.
    """
    idx = sig.stationary_bootstrap_indices(500, n_boot=40, block_length=1,
                                           rng=np.random.default_rng(3))
    consecutive = (np.diff(idx, axis=1) == 1).mean()
    assert consecutive < 0.05


def test_long_blocks_preserve_contiguity():
    """A long mean block length must produce mostly-contiguous runs."""
    idx = sig.stationary_bootstrap_indices(500, n_boot=40, block_length=50,
                                           rng=np.random.default_rng(4))
    consecutive = (np.diff(idx, axis=1) == 1).mean()
    assert consecutive > 0.90


def test_bootstrap_is_reproducible_under_a_fixed_seed():
    a = sig.stationary_bootstrap_indices(200, 20, 10, np.random.default_rng(7))
    b = sig.stationary_bootstrap_indices(200, 20, 10, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_bootstrap_rejects_degenerate_inputs():
    with pytest.raises(ValueError, match="at least 2 observations"):
        sig.stationary_bootstrap_indices(1, 10)
    with pytest.raises(ValueError, match="n_boot"):
        sig.stationary_bootstrap_indices(100, 0)
    with pytest.raises(ValueError, match="block_length"):
        sig.stationary_bootstrap_indices(100, 10, block_length=0.5)


# --------------------------------------------------------------------------- #
# Sharpe confidence intervals
# --------------------------------------------------------------------------- #
def test_ci_brackets_the_point_estimate(edged_returns):
    out = sig.bootstrap_sharpe(edged_returns, n_boot=400, seed=1)
    assert out["ci_low"] < out["sharpe"] < out["ci_high"]
    assert out["bootstrap_se"] > 0


def test_no_edge_interval_contains_zero(flat_returns):
    out = sig.bootstrap_sharpe(flat_returns, n_boot=600, seed=2)
    assert out["ci_low"] < 0 < out["ci_high"]


def test_real_edge_interval_excludes_zero(edged_returns):
    out = sig.bootstrap_sharpe(edged_returns, n_boot=600, seed=3)
    assert out["ci_low"] > 0


def test_wider_confidence_gives_wider_interval(edged_returns):
    narrow = sig.bootstrap_sharpe(edged_returns, n_boot=600, confidence=0.80, seed=4)
    wide = sig.bootstrap_sharpe(edged_returns, n_boot=600, confidence=0.99, seed=4)
    assert (wide["ci_high"] - wide["ci_low"]) > (narrow["ci_high"] - narrow["ci_low"])


def test_bootstrap_sharpe_is_reproducible(edged_returns):
    a = sig.bootstrap_sharpe(edged_returns, n_boot=300, seed=11)
    b = sig.bootstrap_sharpe(edged_returns, n_boot=300, seed=11)
    assert a == b


# --------------------------------------------------------------------------- #
# Sharpe differences
# --------------------------------------------------------------------------- #
def test_identical_series_have_zero_difference_and_no_significance(edged_returns):
    out = sig.bootstrap_sharpe_difference(edged_returns, edged_returns,
                                          n_boot=300, seed=5)
    assert out["difference"] == pytest.approx(0.0, abs=1e-12)
    assert out["p_value"] > 0.5


def test_large_genuine_difference_is_detected(flat_returns, edged_returns):
    out = sig.bootstrap_sharpe_difference(edged_returns, flat_returns,
                                          n_boot=800, seed=6)
    assert out["difference"] > 0
    assert out["ci_low"] > 0
    assert out["p_value"] < 0.05


def test_paired_resampling_beats_independent_for_correlated_series():
    """Correlated series must be resampled together, not separately.

    Two series that move together have a *far* more stable difference than
    either has a level. If the pairing were broken the interval would inflate,
    so this asserts the paired interval is materially tighter than the naive
    independent-difference interval.
    """
    idx = pd.bdate_range("2010-01-01", periods=1200)
    rng = np.random.default_rng(9)
    common = rng.normal(0.0004, 0.01, len(idx))
    a = pd.Series(common + rng.normal(0.0002, 0.001, len(idx)), index=idx)
    b = pd.Series(common, index=idx)

    paired = sig.bootstrap_sharpe_difference(a, b, n_boot=600, seed=8)
    paired_width = paired["ci_high"] - paired["ci_low"]

    sa = sig.bootstrap_sharpe(a, n_boot=600, seed=8)
    sb = sig.bootstrap_sharpe(b, n_boot=600, seed=99)
    independent_width = np.hypot(
        sa["ci_high"] - sa["ci_low"], sb["ci_high"] - sb["ci_low"]
    )
    assert paired_width < independent_width * 0.5


# --------------------------------------------------------------------------- #
# Probabilistic Sharpe
# --------------------------------------------------------------------------- #
def test_psr_of_exactly_zero_edge_is_exactly_one_half(flat_returns):
    """Closed form: a realised Sharpe of 0 tested against 0 gives z = 0.

    Independent of skew and kurtosis, since both enter only through terms
    multiplied by the Sharpe itself.
    """
    psr = sig.probabilistic_sharpe_ratio(flat_returns, benchmark_sharpe=0.0)
    assert psr == pytest.approx(0.5, abs=1e-12)


def test_psr_of_strong_edge_is_near_certainty(edged_returns):
    assert sig.probabilistic_sharpe_ratio(edged_returns, benchmark_sharpe=0.0) > 0.99


def test_psr_falls_as_the_threshold_rises(edged_returns):
    levels = [sig.probabilistic_sharpe_ratio(edged_returns, benchmark_sharpe=t)
              for t in (0.0, 1.0, 2.0, 3.0)]
    assert levels == sorted(levels, reverse=True)


def test_negative_skew_is_penalised():
    """Two series, same mean and volatility, different skew.

    The negatively skewed one is the more dangerous strategy and must receive
    the lower confidence -- this is the correction the plain standard error
    cannot make.
    """
    rng = np.random.default_rng(12)
    idx = pd.bdate_range("2010-01-01", periods=2000)
    base = rng.standard_normal(len(idx))

    left_tail = -np.abs(rng.standard_normal(len(idx))) ** 2
    right_tail = np.abs(rng.standard_normal(len(idx))) ** 2

    def standardise(x):
        x = (x - x.mean()) / x.std(ddof=1)
        return pd.Series(0.0005 + 0.01 * x, index=idx)

    neg = standardise(base + 0.8 * left_tail)
    pos = standardise(base + 0.8 * right_tail)

    assert sig.probabilistic_sharpe_ratio(neg) < sig.probabilistic_sharpe_ratio(pos)


# --------------------------------------------------------------------------- #
# Deflation
# --------------------------------------------------------------------------- #
def test_hurdle_rises_with_the_number_of_trials():
    """The core claim: searching harder raises the bar you must clear."""
    hurdles = [
        sig.expected_max_sharpe(n_trials=n, sharpe_variance=0.25)
        for n in (2, 10, 100, 1000)
    ]
    assert hurdles == sorted(hurdles)
    assert all(h > 0 for h in hurdles)


def test_hurdle_rises_with_dispersion_across_trials():
    low = sig.expected_max_sharpe(n_trials=100, sharpe_variance=0.01)
    high = sig.expected_max_sharpe(n_trials=100, sharpe_variance=1.00)
    assert high > low


def test_hurdle_can_be_estimated_from_the_trials_themselves():
    trials = pd.Series([0.1, 0.4, 0.35, 0.6, -0.05, 0.2, 0.45, 0.3])
    from_series = sig.expected_max_sharpe(trial_sharpes=trials)
    from_variance = sig.expected_max_sharpe(
        n_trials=len(trials), sharpe_variance=float(trials.var(ddof=1))
    )
    assert from_series == pytest.approx(from_variance)


def test_deflated_sharpe_is_never_above_undeflated(edged_returns):
    """Deflation can only ever cost confidence, never add it."""
    out = sig.deflated_sharpe_ratio(edged_returns, n_trials=200, sharpe_variance=0.25)
    assert out["deflated_sharpe"] <= out["psr_vs_zero"]
    assert out["noise_hurdle_sharpe"] > 0


def test_more_trials_lower_the_deflated_sharpe(edged_returns):
    few = sig.deflated_sharpe_ratio(edged_returns, n_trials=5, sharpe_variance=0.25)
    many = sig.deflated_sharpe_ratio(edged_returns, n_trials=5000, sharpe_variance=0.25)
    assert many["deflated_sharpe"] < few["deflated_sharpe"]


def test_a_marginal_edge_fails_deflation_that_a_strong_one_survives(flat_returns,
                                                                    edged_returns):
    """Trial dispersion of 0.2 annualised puts the 200-trial hurdle near 0.55.

    The edged fixture (1.59) clears that comfortably; the flat one (0.00) cannot.
    """
    weak = sig.deflated_sharpe_ratio(flat_returns, n_trials=200, sharpe_variance=0.04)
    strong = sig.deflated_sharpe_ratio(edged_returns, n_trials=200, sharpe_variance=0.04)
    assert weak["deflated_sharpe"] < 0.5 < strong["deflated_sharpe"]
    assert 0.5 < weak["noise_hurdle_sharpe"] < 0.6


def test_deflation_rejects_impossible_trial_counts():
    with pytest.raises(ValueError, match="n_trials must be >= 2"):
        sig.expected_max_sharpe(n_trials=1, sharpe_variance=0.25)
    with pytest.raises(ValueError, match="either trial_sharpes or sharpe_variance"):
        sig.expected_max_sharpe()
