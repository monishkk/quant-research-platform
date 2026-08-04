"""
Research validation.
====================

This module is what separates a backtest from a research result. It provides:

1. **Sample splitting** into training / validation / test, with the test period
   treated as write-once.
2. **Baselines** -- the comparisons that make a Sharpe ratio interpretable.
3. **Parameter sensitivity** -- is there a broad stable region, or one lucky cell?
4. **Regime analysis** -- where does the strategy earn and where does it bleed?

The discipline being enforced
-----------------------------
Parameter search runs on the **training window only**. The validation window is
looked at after the parameter region is chosen. The test window is evaluated
**once**, at the end, and never used to revise the strategy.

This is a convention, not something code can enforce -- but
:func:`run_sensitivity` defaults its evaluation window to the training split
precisely so that the lazy path is also the correct one. Every function that can
touch the test period says so in its signature.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from quant_platform import metrics
from quant_platform.portfolio import BacktestResult, buy_and_hold, run_backtest
from quant_platform.signals import (
    build_target_weights,
    equal_weight_all,
    random_selection_weights,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SampleSplit",
    "make_splits",
    "evaluate_splits",
    "build_baselines",
    "align_results",
    "run_sensitivity",
    "sensitivity_marginals",
    "regime_analysis",
    "leave_one_year_out",
    "calendar_year_table",
    "TRADING_DAYS_PER_MONTH",
]

TRADING_DAYS_PER_MONTH = 21


# --------------------------------------------------------------------------- #
# Sample splitting
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SampleSplit:
    """One named, half-open-in-spirit date window (both ends inclusive)."""

    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    purpose: str

    def mask(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index >= self.start) & (index <= self.end)

    def __str__(self) -> str:
        return f"{self.name}: {self.start.date()} -> {self.end.date()} ({self.purpose})"


def make_splits(
    index: pd.DatetimeIndex,
    training_end: str,
    validation_end: str,
    test_end: str,
) -> list[SampleSplit]:
    """Build the training / validation / test windows from config boundaries.

    The first window starts at the first date the strategy is actually live
    (``index[0]`` after the engine's warm-up trim), not at the config's nominal
    start, so annualised statistics are not diluted by a flat warm-up.
    """
    start = pd.Timestamp(index[0])
    t_end = pd.Timestamp(training_end)
    v_end = pd.Timestamp(validation_end)
    x_end = min(pd.Timestamp(test_end), pd.Timestamp(index[-1]))

    if not start < t_end < v_end <= x_end:
        raise ValueError(
            f"Split boundaries must increase: start={start.date()}, "
            f"training_end={t_end.date()}, validation_end={v_end.date()}, test_end={x_end.date()}"
        )

    return [
        SampleSplit("training", start, t_end, "in-sample; parameters chosen here"),
        SampleSplit("validation", t_end + pd.Timedelta(days=1), v_end, "out-of-sample check"),
        SampleSplit("test", v_end + pd.Timedelta(days=1), x_end, "final; evaluated once"),
        SampleSplit("full", start, x_end, "entire sample"),
    ]


def evaluate_splits(
    result: BacktestResult,
    splits: list[SampleSplit],
    benchmark: pd.Series | None = None,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Metrics for one strategy across every sample split (columns = splits)."""
    out: dict[str, pd.Series] = {}
    for sp in splits:
        sub = result.slice(sp.start, sp.end)
        if len(sub.net_returns.dropna()) < 2:
            logger.warning("Split '%s' has too few observations; skipping", sp.name)
            continue
        bench = None
        if benchmark is not None:
            bench = benchmark.loc[sp.mask(benchmark.index)]
        out[sp.name] = metrics.summary_metrics(
            sub.net_returns,
            equity=sub.equity,
            benchmark=bench,
            turnover=sub.turnover,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
            name=sp.name,
        )
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def build_baselines(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    benchmark_symbol: str = "SPY",
    lookback: int = 252,
    skip: int = 21,
    holdings: int = 3,
    rebalance: str = "monthly",
    max_weight: float | None = 0.40,
    initial_capital: float = 100_000.0,
    commission_bps: float = 2.0,
    slippage_bps: float = 3.0,
    execution_lag: int = 1,
    seed: int = 42,
) -> dict[str, BacktestResult]:
    """Construct the comparison set.

    Each baseline isolates one claim the strategy implicitly makes:

    ``SPY buy & hold``
        Would a passive investor have done better? The honest hurdle.
    ``Equal weight (all 9)``
        Is the *selection* adding anything over simply owning the universe?
    ``Random selection (3 of 9)``
        Is the momentum **ranking** informative, or is any concentrated
        3-of-9 rotation in this universe roughly as good? Seeded, so it is a
        fixed comparator rather than a moving target.
    ``Momentum, zero cost``
        How much of the result does the cost assumption consume?
    ``Momentum, T+6 execution``
        Does the edge survive if the signal is acted on a week late? A signal
        that dies with a small delay is a microstructure artefact, not an
        investable edge.
    """
    common = dict(
        initial_capital=initial_capital,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        execution_lag=execution_lag,
    )
    momentum_weights = build_target_weights(
        prices, lookback, skip, holdings, rebalance, max_weight
    )

    baselines: dict[str, BacktestResult] = {}

    baselines["SPY buy & hold"] = buy_and_hold(
        returns, benchmark_symbol, name=f"{benchmark_symbol} buy & hold", **common
    )
    baselines["Equal weight (all)"] = run_backtest(
        returns,
        equal_weight_all(prices, rebalance),
        name="Equal weight (all)",
        **common,
    )
    baselines["Random selection"] = run_backtest(
        returns,
        random_selection_weights(prices, holdings, rebalance, max_weight, seed, lookback),
        name="Random selection",
        **common,
    )
    baselines["Momentum, zero cost"] = run_backtest(
        returns,
        momentum_weights,
        initial_capital=initial_capital,
        commission_bps=0.0,
        slippage_bps=0.0,
        execution_lag=execution_lag,
        name="Momentum, zero cost",
    )
    baselines["Momentum, T+6 exec"] = run_backtest(
        returns,
        momentum_weights,
        initial_capital=initial_capital,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        execution_lag=execution_lag + 5,
        name="Momentum, T+6 exec",
    )
    return baselines


def align_results(results: dict[str, BacktestResult]) -> dict[str, BacktestResult]:
    """Trim every result to the window all of them share, re-basing each equity curve.

    Different strategies become live on different dates: a buy-and-hold benchmark
    starts on day one, while momentum needs a 12-month formation window before it
    can hold anything. Comparing their equity curves without alignment credits the
    benchmark with an extra year of compounding and makes the comparison
    meaningless. This puts every curve on the same start date and the same
    starting capital.
    """
    if not results:
        return results
    start = max(r.index[0] for r in results.values())
    end = min(r.index[-1] for r in results.values())
    logger.info("Aligning %d results to %s -> %s", len(results), start.date(), end.date())
    return {name: r.slice(start, end) for name, r in results.items()}


# --------------------------------------------------------------------------- #
# Parameter sensitivity
# --------------------------------------------------------------------------- #
def run_sensitivity(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    lookback_months: list[int],
    holdings_grid: list[int],
    rebalance_grid: list[str],
    cost_grid: list[float],
    skip: int = 21,
    max_weight: float | None = 0.40,
    initial_capital: float = 100_000.0,
    execution_lag: int = 1,
    eval_start: pd.Timestamp | None = None,
    eval_end: pd.Timestamp | None = None,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Evaluate the full parameter cross-product on one window.

    ``eval_start``/``eval_end`` should be the **training** window. Searching a
    grid on the test period and then reporting the best cell is the single most
    common way a backtest becomes fiction: with 192 combinations, a Sharpe of 1.5
    somewhere in the grid is what you would expect from noise alone.

    Returns one row per parameter combination with performance and activity
    statistics. ``avg_exposure`` is included because a binding ``max_weight``
    leaves the portfolio partly in cash -- without it, low-``holdings`` rows look
    misleadingly defensive.
    """
    rows: list[dict] = []
    combos = list(product(lookback_months, holdings_grid, rebalance_grid, cost_grid))
    logger.info("Running sensitivity grid: %d combinations", len(combos))

    # Weight construction is the expensive part and does not depend on cost, so
    # cache it across the cost dimension.
    weight_cache: dict[tuple, pd.DataFrame] = {}
    for months, n_hold, reb, _ in combos:
        lookback = months * TRADING_DAYS_PER_MONTH
        if lookback <= skip:
            logger.debug("Skipping lookback=%d months (<= skip)", months)
            continue
        key = (lookback, n_hold, reb)
        if key not in weight_cache:
            weight_cache[key] = build_target_weights(
                prices, lookback, skip, n_hold, reb, max_weight
            )

    # Different parameter cells go live on different dates: a 3-month lookback
    # forms a signal months before a 12-month one, and a quarterly rebalance
    # waits for the next quarter end. Evaluating each on its own start date would
    # mean the grid compares different *periods* as well as different parameters
    # -- and in this sample the earliest months are crisis months that dominate
    # the result, so that confound is not small. Every cell is therefore
    # evaluated on the window all of them share.
    common_start = _common_live_date(weight_cache, execution_lag, eval_start)
    if common_start is not None and (eval_start is None or common_start > eval_start):
        logger.info("Sensitivity cells aligned to common start %s", common_start.date())
    eval_start = common_start if common_start is not None else eval_start

    for months, n_hold, reb, cost_bps in combos:
        lookback = months * TRADING_DAYS_PER_MONTH
        if lookback <= skip:
            continue
        key = (lookback, n_hold, reb)

        res = run_backtest(
            returns,
            weight_cache[key],
            initial_capital=initial_capital,
            commission_bps=cost_bps,
            slippage_bps=0.0,
            execution_lag=execution_lag,
            name=f"L{months}m_H{n_hold}_{reb}_{cost_bps}bps",
        )
        sub = res.slice(eval_start, eval_end)
        r = sub.net_returns.dropna()
        if len(r) < periods_per_year // 2:
            continue

        rows.append(
            {
                "lookback_months": months,
                "holdings": n_hold,
                "rebalance": reb,
                "cost_bps": cost_bps,
                "sharpe": metrics.annualized_sharpe(r, periods_per_year, risk_free_rate),
                "cagr": metrics.annualized_return(r, periods_per_year),
                "ann_volatility": metrics.annualized_volatility(r, periods_per_year),
                "max_drawdown": metrics.maximum_drawdown(sub.equity),
                "calmar": metrics.calmar_ratio(r, sub.equity, periods_per_year),
                "sortino": metrics.sortino_ratio(r, periods_per_year, risk_free_rate),
                "ann_turnover": metrics.annual_turnover(sub.turnover, periods_per_year),
                "avg_exposure": float(sub.exposure.mean()),
                "n_periods": len(r),
            }
        )

    return pd.DataFrame(rows)


def _common_live_date(
    weight_cache: dict[tuple, pd.DataFrame],
    execution_lag: int,
    eval_start: pd.Timestamp | None,
) -> pd.Timestamp | None:
    """Latest date on which *every* parameter cell already holds a position.

    Derived from the weights alone -- no backtest needed -- because a cell's
    first live date depends only on its signal and rebalance schedule, not on
    the cost assumption.
    """
    starts: list[pd.Timestamp] = []
    for weights in weight_cache.values():
        active = weights.shift(execution_lag).abs().sum(axis=1) > 0
        if not active.any():
            continue
        first = active.idxmax()
        starts.append(max(first, eval_start) if eval_start is not None else first)
    return max(starts) if starts else None


def sensitivity_marginals(sensitivity: pd.DataFrame, metric: str = "sharpe") -> dict[str, pd.DataFrame]:
    """Collapse the grid one dimension at a time.

    A parameter is *robust* if the spread of ``metric`` across its levels is
    small relative to the spread across the whole grid. ``std`` and the min/max
    columns are the point of this table -- not ``mean``.
    """
    out: dict[str, pd.DataFrame] = {}
    for dim in ("lookback_months", "holdings", "rebalance", "cost_bps"):
        if dim not in sensitivity.columns:
            continue
        out[dim] = (
            sensitivity.groupby(dim)[metric]
            .agg(["mean", "median", "std", "min", "max", "count"])
            .round(4)
        )
    return out


# --------------------------------------------------------------------------- #
# Regime and calendar analysis
# --------------------------------------------------------------------------- #
def regime_analysis(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
    trend_window: int = 126,
    vol_window: int = 63,
) -> pd.DataFrame:
    """Strategy performance conditional on the benchmark's state.

    Regimes are defined from a **trailing** benchmark window, so the
    classification at date ``t`` uses only data up to ``t``. This is descriptive
    attribution rather than a tradeable rule, but keeping it causal means the
    table can be read as "how the strategy behaved entering each environment"
    rather than "how it behaved in months we later labelled bad".

    Rows:
      * ``Benchmark up`` / ``Benchmark down`` -- sign of the trailing 6-month
        benchmark return.
      * ``Vol: low/mid/high`` -- terciles of trailing 3-month benchmark
        realised volatility.
    """
    s, b = strategy_returns.align(benchmark_returns, join="inner")
    s, b = s.dropna(), b.dropna()
    s, b = s.align(b, join="inner")
    if len(s) < trend_window:
        return pd.DataFrame()

    trend = (1 + b).rolling(trend_window).apply(np.prod, raw=True) - 1
    vol = b.rolling(vol_window).std(ddof=1) * np.sqrt(periods_per_year)

    regimes: dict[str, pd.Series] = {
        "Benchmark up": trend >= 0,
        "Benchmark down": trend < 0,
    }
    q1, q2 = vol.quantile(1 / 3), vol.quantile(2 / 3)
    regimes["Vol: low"] = vol <= q1
    regimes["Vol: mid"] = (vol > q1) & (vol <= q2)
    regimes["Vol: high"] = vol > q2

    rows = {}
    for label, mask in regimes.items():
        mask = mask.fillna(False)
        sr, br = s[mask], b[mask]
        if len(sr) < 20:
            continue
        rows[label] = pd.Series(
            {
                "n_days": len(sr),
                "pct_of_sample": len(sr) / len(s),
                "strategy_ann_return": metrics.annualized_return(sr, periods_per_year),
                "benchmark_ann_return": metrics.annualized_return(br, periods_per_year),
                "strategy_ann_vol": metrics.annualized_volatility(sr, periods_per_year),
                "strategy_sharpe": metrics.annualized_sharpe(sr, periods_per_year),
                "benchmark_sharpe": metrics.annualized_sharpe(br, periods_per_year),
                "excess_ann_return": metrics.annualized_return(sr, periods_per_year)
                - metrics.annualized_return(br, periods_per_year),
                "win_rate": metrics.win_rate(sr),
            }
        )
    return pd.DataFrame(rows).T


def leave_one_year_out(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Recompute the strategy-vs-benchmark Sharpe edge with each year removed.

    Why this matters more than almost any other diagnostic
    ------------------------------------------------------
    A strategy can post a respectable full-sample edge that is, on inspection,
    one good year attached to a decade of nothing. The full-sample number cannot
    distinguish that from a persistent edge -- but deleting one year at a time
    can. If removing a single year collapses the edge, the strategy has not been
    shown to work; it has been shown to have survived one episode.

    Returns one row per calendar year with the edge computed *excluding* that
    year, and ``edge_change`` = (edge without the year) - (full-sample edge). A
    large negative ``edge_change`` marks a year the result depends on.
    """
    s, b = strategy_returns.align(benchmark_returns, join="inner")
    s, b = s.dropna(), b.dropna()
    s, b = s.align(b, join="inner")
    if len(s) < periods_per_year:
        return pd.DataFrame()

    full_s = metrics.annualized_sharpe(s, periods_per_year, risk_free_rate)
    full_b = metrics.annualized_sharpe(b, periods_per_year, risk_free_rate)
    full_edge = full_s - full_b

    rows: dict[int, pd.Series] = {}
    for year in sorted(set(s.index.year)):
        mask = s.index.year != year
        if mask.sum() < periods_per_year:
            continue
        sub_s, sub_b = s[mask], b[mask]
        edge = metrics.annualized_sharpe(sub_s, periods_per_year, risk_free_rate) - \
            metrics.annualized_sharpe(sub_b, periods_per_year, risk_free_rate)
        rows[year] = pd.Series(
            {
                "strategy_sharpe_ex": metrics.annualized_sharpe(sub_s, periods_per_year, risk_free_rate),
                "benchmark_sharpe_ex": metrics.annualized_sharpe(sub_b, periods_per_year, risk_free_rate),
                "sharpe_edge_ex": edge,
                "edge_change": edge - full_edge,
                "strategy_return_in_year": float((1 + s[~mask]).prod() - 1),
                "benchmark_return_in_year": float((1 + b[~mask]).prod() - 1),
            }
        )

    table = pd.DataFrame(rows).T
    table.attrs["full_edge"] = full_edge
    table.attrs["full_strategy_sharpe"] = full_s
    table.attrs["full_benchmark_sharpe"] = full_b
    table.attrs["sharpe_se"] = metrics.sharpe_standard_error(full_s, len(s), periods_per_year)
    return table


def calendar_year_table(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Year-by-year total returns for strategy and benchmark, plus the difference."""
    s = strategy_returns.dropna()
    if s.empty:
        return pd.DataFrame()

    yearly = ((1 + s).resample("YE").prod() - 1).rename("strategy")
    table = yearly.to_frame()
    table.index = table.index.year

    if benchmark_returns is not None:
        b = benchmark_returns.dropna()
        by = ((1 + b).resample("YE").prod() - 1).rename("benchmark")
        by.index = by.index.year
        table = table.join(by, how="left")
        table["excess"] = table["strategy"] - table["benchmark"]

    return table
