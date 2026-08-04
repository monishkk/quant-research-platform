"""Statistical significance for backtest results.

A backtest reports a number. This module reports how much of that number could
be luck. Two questions matter, and they are not the same question:

1. **Is this Sharpe distinguishable from zero?**
   The analytic standard error in :func:`metrics.sharpe_standard_error` assumes
   returns are iid and normal. Daily strategy returns are neither -- they are
   autocorrelated, skewed and fat-tailed. A **stationary block bootstrap**
   (Politis & Romano, 1994) resamples *blocks* of consecutive days rather than
   individual days, so each resampled path keeps the dependence structure of the
   original. Blocks of random (geometric) length are what make the resampled
   series stationary; fixed-length blocks do not have that property.

2. **Would this Sharpe look impressive even if the strategy were worthless?**
   Any search over N configurations produces a best result, and the expected
   maximum of N draws from a zero-mean distribution grows with N. The
   **Deflated Sharpe Ratio** (Bailey & Lopez de Prado, 2014) asks whether an
   observed Sharpe exceeds the maximum you would expect from that many trials of
   pure noise, while correcting for the skew and kurtosis of the returns.

The second question is the one this project most needs answered. A 192-cell
sensitivity grid *is* a search, and quoting its best cell without deflation
would be precisely the error the rest of the platform exists to avoid.

Both tools are descriptive, not decisive. A bootstrap cannot manufacture
information the sample does not contain, and the deflation correction assumes
the trials are independent -- ours are not, since neighbouring parameter cells
share most of their trades. Where that matters it is said plainly in the
docstrings below rather than buried.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Euler-Mascheroni constant, which appears in the expected maximum of N draws
# from a Gumbel-type extreme value distribution.
_EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------- #
# Stationary block bootstrap
# --------------------------------------------------------------------------- #
def stationary_bootstrap_indices(
    n: int,
    n_boot: int,
    block_length: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Row indices for ``n_boot`` stationary-bootstrap resamples of length ``n``.

    The construction (Politis & Romano, 1994): start at a uniformly random
    observation; at each subsequent step either continue to the next observation
    (wrapping at the end) with probability ``1 - p``, or jump to a fresh random
    observation with probability ``p = 1 / block_length``. Block lengths are
    therefore geometric with mean ``block_length``.

    Returns
    -------
    ndarray of shape ``(n_boot, n)``

    Notes
    -----
    The recurrence is sequential, but it vectorises: the positions where a new
    block starts partition each row, and within a block the index just counts
    upward from that block's random start. Computing the segment starts with a
    running maximum turns an O(n_boot * n) Python loop into a handful of array
    operations -- which matters, because a 2,000 x 4,500 resample is 9 million
    steps.
    """
    if n < 2:
        raise ValueError(f"need at least 2 observations to bootstrap, got {n}")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")

    if block_length is None:
        block_length = optimal_block_length(n)
    if block_length < 1:
        raise ValueError(f"block_length must be >= 1, got {block_length}")

    rng = np.random.default_rng() if rng is None else rng
    p = 1.0 / block_length

    starts = rng.integers(0, n, size=(n_boot, n))
    new_block = rng.random((n_boot, n)) < p
    new_block[:, 0] = True  # every path begins a block

    steps = np.arange(n)
    # Position of the most recent block start at or before each step.
    seg_pos = np.maximum.accumulate(np.where(new_block, steps, -1), axis=1)
    offset = steps[None, :] - seg_pos
    base = np.take_along_axis(starts, seg_pos, axis=1)
    return (base + offset) % n


def optimal_block_length(n: int) -> float:
    """Mean block length, as the ``n**(1/3)`` rate.

    This is the rate at which the optimal block length grows, not the
    data-driven constant of Politis & White (2004). It is used as a default
    because it is transparent and depends only on sample size; for ~4,500 daily
    observations it gives roughly 16 days, which is long enough to carry the
    short-horizon autocorrelation in daily strategy returns without producing so
    few distinct blocks that the resamples stop varying.

    Pass ``block_length`` explicitly to override it.
    """
    return max(2.0, float(round(n ** (1.0 / 3.0))))


def _sharpe_along_rows(
    samples: np.ndarray, periods_per_year: int, risk_free_rate: float
) -> np.ndarray:
    """Annualised Sharpe for each row, computed in numpy for speed."""
    excess = samples - risk_free_rate / periods_per_year
    mean = excess.mean(axis=1)
    std = excess.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.sqrt(periods_per_year) * mean / std
    return np.where(std > 0, out, np.nan)


def _resample_in_chunks(
    arrays: list[np.ndarray],
    n_boot: int,
    block_length: float | None,
    rng: np.random.Generator,
    fn,
    chunk_target: int = 2_000_000,
) -> list[np.ndarray]:
    """Apply ``fn`` to bootstrap resamples of several *time-aligned* arrays.

    All arrays are indexed with the *same* draws, which is what preserves the
    contemporaneous relationship between a strategy and its benchmark. Bootstrap
    them separately and any comparison between the two becomes meaningless.

    Chunked so peak memory stays near ``chunk_target`` elements rather than
    ``n_boot * n``, which for a full daily history would be hundreds of MB.
    """
    n = len(arrays[0])
    per_chunk = max(1, min(n_boot, chunk_target // max(n, 1)))
    results: list[list[np.ndarray]] = [[] for _ in arrays]

    done = 0
    while done < n_boot:
        size = min(per_chunk, n_boot - done)
        idx = stationary_bootstrap_indices(n, size, block_length, rng)
        for i, arr in enumerate(arrays):
            results[i].append(fn(arr[idx]))
        done += size

    return [np.concatenate(parts) for parts in results]


def bootstrap_sharpe(
    returns: pd.Series,
    n_boot: int = 2000,
    block_length: float | None = None,
    confidence: float = 0.95,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence interval for a single annualised Sharpe ratio."""
    r = returns.dropna()
    if len(r) < 3:
        raise ValueError(f"need at least 3 observations, got {len(r)}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    values = r.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    block_length = block_length or optimal_block_length(len(values))

    (draws,) = _resample_in_chunks(
        [values], n_boot, block_length, rng,
        lambda s: _sharpe_along_rows(s, periods_per_year, risk_free_rate),
    )
    draws = draws[np.isfinite(draws)]

    point = float(_sharpe_along_rows(values[None, :], periods_per_year, risk_free_rate)[0])
    tail = (1.0 - confidence) / 2.0

    return {
        "sharpe": point,
        "ci_low": float(np.quantile(draws, tail)),
        "ci_high": float(np.quantile(draws, 1.0 - tail)),
        "bootstrap_se": float(draws.std(ddof=1)),
        "n_boot": int(len(draws)),
        "block_length": float(block_length),
        "confidence": confidence,
    }


def bootstrap_sharpe_difference(
    strategy: pd.Series,
    benchmark: pd.Series,
    n_boot: int = 2000,
    block_length: float | None = None,
    confidence: float = 0.95,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    seed: int = 42,
) -> dict:
    """Bootstrap the *difference* in annualised Sharpe, strategy minus benchmark.

    This is the quantity the project's headline actually claims, so it is the
    one that deserves an interval. The two series are resampled with identical
    draws so their correlation survives -- the difference of two independently
    bootstrapped Sharpes would have far too wide a spread and would understate
    significance.

    The reported p-value is a two-sided bootstrap test: the resampled
    distribution is recentred on zero to impose the null of no difference, and
    the p-value is the share of recentred draws at least as extreme as the
    observed difference.
    """
    s, b = strategy.align(benchmark, join="inner")
    mask = s.notna() & b.notna()
    s, b = s[mask], b[mask]
    if len(s) < 3:
        raise ValueError(f"need at least 3 overlapping observations, got {len(s)}")

    sv, bv = s.to_numpy(dtype=float), b.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    block_length = block_length or optimal_block_length(len(sv))

    s_draws, b_draws = _resample_in_chunks(
        [sv, bv], n_boot, block_length, rng,
        lambda x: _sharpe_along_rows(x, periods_per_year, risk_free_rate),
    )
    diffs = s_draws - b_draws
    diffs = diffs[np.isfinite(diffs)]

    s_point = float(_sharpe_along_rows(sv[None, :], periods_per_year, risk_free_rate)[0])
    b_point = float(_sharpe_along_rows(bv[None, :], periods_per_year, risk_free_rate)[0])
    observed = s_point - b_point

    centred = diffs - diffs.mean()
    p_value = float(np.mean(np.abs(centred) >= abs(observed)))
    tail = (1.0 - confidence) / 2.0

    return {
        "strategy_sharpe": s_point,
        "benchmark_sharpe": b_point,
        "difference": observed,
        "ci_low": float(np.quantile(diffs, tail)),
        "ci_high": float(np.quantile(diffs, 1.0 - tail)),
        "bootstrap_se": float(diffs.std(ddof=1)),
        "p_value": p_value,
        "n_boot": int(len(diffs)),
        "block_length": float(block_length),
        "confidence": confidence,
        "draws": diffs,
    }


# --------------------------------------------------------------------------- #
# Probabilistic and Deflated Sharpe
# --------------------------------------------------------------------------- #
def probabilistic_sharpe_ratio(
    returns: pd.Series,
    benchmark_sharpe: float = 0.0,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """Probability that the true Sharpe exceeds ``benchmark_sharpe``.

    Bailey & Lopez de Prado (2012). Unlike the textbook standard error, this
    accounts for the third and fourth moments: negative skew and fat tails both
    make a given Sharpe less trustworthy, because they mean the sample mean is
    being driven by a distribution prone to rare large losses.

    ``benchmark_sharpe`` is annualised, as is the strategy's own.
    """
    r = returns.dropna()
    if len(r) < 4:
        return float("nan")

    values = r.to_numpy(dtype=float) - risk_free_rate / periods_per_year
    n = len(values)
    sd = values.std(ddof=1)
    if sd == 0:
        return float("nan")

    # The formula is defined on per-period Sharpe, not annualised.
    sr = values.mean() / sd
    sr_star = benchmark_sharpe / np.sqrt(periods_per_year)

    skew = float(stats.skew(values, bias=False))
    # scipy returns *excess* kurtosis; the formula needs the raw fourth moment.
    kurt = float(stats.kurtosis(values, fisher=True, bias=False)) + 3.0

    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2
    if denom <= 0:
        return float("nan")

    z = (sr - sr_star) * np.sqrt(n - 1) / np.sqrt(denom)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(
    trial_sharpes: np.ndarray | pd.Series | None = None,
    n_trials: int | None = None,
    sharpe_variance: float | None = None,
    periods_per_year: int = 252,
) -> float:
    """Expected **maximum** annualised Sharpe from ``n_trials`` of pure noise.

    If you try N configurations that all genuinely have zero edge, the best of
    them will still post a positive Sharpe. This is how positive, and it is the
    hurdle a searched-over result has to clear before it means anything.

    Uses the Gumbel approximation to the expected maximum of N independent
    normal draws with standard deviation ``sqrt(V)``, where ``V`` is the
    variance of the Sharpe ratios *across trials*.

    The independence assumption is generous to the strategy here: neighbouring
    cells of a parameter grid share most of their trades, so the effective
    number of independent trials is smaller than the nominal count, and the true
    hurdle is therefore somewhat lower than this returns. Treat it as an upper
    bound on the noise threshold, not a precise one.
    """
    if sharpe_variance is None:
        if trial_sharpes is None:
            raise ValueError("provide either trial_sharpes or sharpe_variance")
        arr = np.asarray(pd.Series(trial_sharpes).dropna(), dtype=float)
        if len(arr) < 2:
            raise ValueError("need at least 2 trials to estimate a variance")
        sharpe_variance = float(arr.var(ddof=1))
        if n_trials is None:
            n_trials = len(arr)

    if n_trials is None:
        raise ValueError("n_trials is required when passing sharpe_variance")
    if n_trials < 2:
        raise ValueError(f"n_trials must be >= 2, got {n_trials}")

    # Work in per-period units, then re-annualise the answer.
    sd = np.sqrt(sharpe_variance) / np.sqrt(periods_per_year)

    a = stats.norm.ppf(1.0 - 1.0 / n_trials)
    b = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    expected = sd * ((1.0 - _EULER_GAMMA) * a + _EULER_GAMMA * b)
    return float(expected * np.sqrt(periods_per_year))


def deflated_sharpe_ratio(
    returns: pd.Series,
    trial_sharpes: np.ndarray | pd.Series | None = None,
    n_trials: int | None = None,
    sharpe_variance: float | None = None,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> dict:
    """Probability the strategy's edge survives the size of the search.

    The Deflated Sharpe Ratio is the Probabilistic Sharpe Ratio measured against
    :func:`expected_max_sharpe` instead of against zero. A DSR of 0.95 means
    that, after accounting for how many configurations were tried and for the
    skew and kurtosis of the returns, there is a 95% probability the true Sharpe
    is above the noise hurdle.

    Returns the DSR alongside the hurdle it was measured against, because the
    hurdle is the interpretable part: it says what Sharpe this search could have
    produced from nothing.
    """
    hurdle = expected_max_sharpe(
        trial_sharpes=trial_sharpes,
        n_trials=n_trials,
        sharpe_variance=sharpe_variance,
        periods_per_year=periods_per_year,
    )
    dsr = probabilistic_sharpe_ratio(
        returns, benchmark_sharpe=hurdle,
        periods_per_year=periods_per_year, risk_free_rate=risk_free_rate,
    )
    psr = probabilistic_sharpe_ratio(
        returns, benchmark_sharpe=0.0,
        periods_per_year=periods_per_year, risk_free_rate=risk_free_rate,
    )

    n_used = n_trials
    if n_used is None and trial_sharpes is not None:
        n_used = int(pd.Series(trial_sharpes).notna().sum())

    return {
        "psr_vs_zero": psr,
        "deflated_sharpe": dsr,
        "noise_hurdle_sharpe": hurdle,
        "n_trials": n_used,
    }
