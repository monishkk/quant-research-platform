"""
Portfolio engine.
=================

Takes **target** weights and asset returns, applies the execution delay, charges
transaction costs, and returns the realised path.

The accounting, in full
-----------------------
::

    executed_weights = target_weights.shift(execution_lag)
    exposure         = executed_weights.sum(axis=1)
    cash_weight      = 1 - exposure

    gross_returns    = (executed_weights * asset_returns).sum(axis=1)
                       + cash_weight * cash_rate_per_period

    turnover         = |executed_weights - executed_weights.shift(1)|.sum(axis=1)
    costs            = turnover * (commission_bps + slippage_bps) / 10_000

    net_returns      = gross_returns - costs
    equity           = initial_capital * (1 + net_returns).cumprod()

Why the shift is placed *here* and not in the signal
----------------------------------------------------
``target_weights.loc[t]`` means "the portfolio I want, decided at the close of
``t``". ``asset_returns.loc[t]`` is the return earned over ``(t-1, t]``. With
``execution_lag=1``::

    executed_weights.loc[t] == target_weights.loc[t-1]

so the weights multiplying ``asset_returns.loc[t]`` were fixed at the close of
``t-1``, strictly before that return was observable. Setting
``execution_lag=0`` would multiply a return by weights chosen with knowledge of
that same return -- the canonical look-ahead bug. The parameter is exposed
precisely so the bug can be *demonstrated* (see the report's execution-lag
sensitivity) rather than accidentally shipped.

v1 simplifications
------------------
Weights are assumed to be restored to target every period rather than drifting
with relative performance between rebalances. See ``costs`` module docstring for
the consequences. Long-only, unlevered, no borrow, no financing, no taxes.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_platform.costs import CostModel, compute_turnover

logger = logging.getLogger(__name__)

__all__ = ["run_backtest", "BacktestResult"]


@dataclass
class BacktestResult:
    """Everything the engine produced, plus the assumptions that produced it."""

    equity: pd.Series
    net_returns: pd.Series
    gross_returns: pd.Series
    costs: pd.Series
    turnover: pd.Series
    exposure: pd.Series
    executed_weights: pd.DataFrame
    target_weights: pd.DataFrame
    initial_capital: float
    cost_model: CostModel
    execution_lag: int
    name: str = "strategy"
    meta: dict = field(default_factory=dict)

    # -- convenience ------------------------------------------------------- #
    @property
    def index(self) -> pd.DatetimeIndex:
        return self.equity.index

    def slice(self, start=None, end=None) -> BacktestResult:
        """Restrict every series to a date window, re-basing equity to 1.0 x capital.

        Used for in-sample / out-of-sample reporting: the sub-period equity curve
        starts fresh at ``initial_capital`` so periods are visually comparable.
        """
        mask = pd.Series(True, index=self.index)
        if start is not None:
            mask &= self.index >= pd.Timestamp(start)
        if end is not None:
            mask &= self.index <= pd.Timestamp(end)

        net = self.net_returns[mask.values]
        return BacktestResult(
            equity=self.initial_capital * (1 + net.fillna(0)).cumprod(),
            net_returns=net,
            gross_returns=self.gross_returns[mask.values],
            costs=self.costs[mask.values],
            turnover=self.turnover[mask.values],
            exposure=self.exposure[mask.values],
            executed_weights=self.executed_weights[mask.values],
            target_weights=self.target_weights[mask.values],
            initial_capital=self.initial_capital,
            cost_model=self.cost_model,
            execution_lag=self.execution_lag,
            name=self.name,
            meta=dict(self.meta),
        )

    def summary(self, benchmark: pd.Series | None = None, **kwargs) -> pd.Series:
        """Headline performance metrics (delegates to :mod:`quant_platform.metrics`)."""
        from quant_platform import metrics

        return metrics.summary_metrics(
            self.net_returns,
            equity=self.equity,
            benchmark=benchmark,
            turnover=self.turnover,
            name=self.name,
            **kwargs,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"BacktestResult(name={self.name!r}, "
            f"{self.index[0].date()}..{self.index[-1].date()}, "
            f"n={len(self.equity)}, final_equity={self.equity.iloc[-1]:,.0f})"
        )


def run_backtest(
    returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    initial_capital: float = 100_000.0,
    commission_bps: float = 2.0,
    slippage_bps: float = 3.0,
    execution_lag: int = 1,
    cash_rate: float = 0.0,
    on_missing: str = "raise",
    allow_drift: bool = True,
    rebalance_on: pd.DatetimeIndex | None = None,
    periods_per_year: int = 252,
    name: str = "strategy",
    warmup_trim: bool = True,
) -> BacktestResult:
    """Simulate a long-only weight-following portfolio.

    Parameters
    ----------
    returns
        date x symbol simple returns. ``returns.loc[t]`` is the return over
        ``(t-1, t]``.
    target_weights
        date x symbol desired weights, decided at the close of each date.
    execution_lag
        Periods between deciding a weight and holding it. **Must be >= 1** for a
        causally valid backtest; 0 is permitted only so the resulting bias can be
        measured, and emits a warning.
    cash_rate
        Annualised return on the uninvested residual (``1 - sum(weights)``),
        which is non-zero whenever a max-weight cap binds.
    warmup_trim
        Drop the leading rows before the strategy first takes a position, so
        warm-up flat periods do not distort annualised statistics.

    Returns
    -------
    BacktestResult
    """
    if execution_lag < 0:
        raise ValueError(f"execution_lag must be >= 0, got {execution_lag}")
    if execution_lag == 0:
        warnings.warn(
            "execution_lag=0 lets a position capture the same period's return, "
            "which is look-ahead bias. Use it only to quantify that bias.",
            UserWarning,
            stacklevel=2,
        )
    if initial_capital <= 0:
        raise ValueError(f"initial_capital must be positive, got {initial_capital}")

    asset_returns, weights = _align(returns, target_weights)

    # ---- stage: desired position -> executed position -------------------- #
    executed_weights = weights.shift(execution_lag).fillna(0.0)

    if warmup_trim:
        active = executed_weights.abs().sum(axis=1) > 0
        if active.any():
            first = active.idxmax()
            keep = executed_weights.index >= first
            asset_returns = asset_returns[keep]
            weights = weights[keep]
            executed_weights = executed_weights[keep]

    _check_returns_available(executed_weights, asset_returns, on_missing)

    # ---- stage: executed position -> portfolio return -------------------- #
    held_weights, turnover = _simulate_holdings(
        executed_weights, asset_returns, cash_rate, periods_per_year, allow_drift,
        _executed_rebalance_positions(executed_weights.index, rebalance_on, execution_lag),
    )

    exposure = held_weights.sum(axis=1)
    cash_weight = 1.0 - exposure

    asset_pnl = (held_weights * asset_returns.fillna(0.0)).sum(axis=1)
    cash_pnl = cash_weight * (cash_rate / periods_per_year)
    gross_returns = asset_pnl + cash_pnl

    cost_model = CostModel(commission_bps=commission_bps, slippage_bps=slippage_bps)
    costs = cost_model.apply(turnover)

    net_returns = gross_returns - costs
    equity = initial_capital * (1.0 + net_returns).cumprod()

    for series, label in (
        (gross_returns, "gross_returns"),
        (net_returns, "net_returns"),
        (equity, "equity"),
    ):
        series.name = label
    turnover.name = "turnover"
    costs.name = "costs"
    exposure.name = "exposure"

    return BacktestResult(
        equity=equity,
        net_returns=net_returns,
        gross_returns=gross_returns,
        costs=costs,
        turnover=turnover,
        exposure=exposure,
        # What was actually held each period: equal to the target on rebalance
        # dates, and the drifted position in between.
        executed_weights=held_weights,
        target_weights=weights,
        initial_capital=float(initial_capital),
        cost_model=cost_model,
        execution_lag=int(execution_lag),
        name=name,
        meta={
            "cash_rate": cash_rate,
            "periods_per_year": periods_per_year,
            "warmup_trim": warmup_trim,
        },
    )


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _align(
    returns: pd.DataFrame, target_weights: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Put returns and weights on a common, sorted, unique index and column set."""
    if not isinstance(returns, pd.DataFrame) or not isinstance(target_weights, pd.DataFrame):
        raise TypeError("returns and target_weights must both be DataFrames")

    returns = returns.sort_index()
    target_weights = target_weights.sort_index()

    if returns.index.duplicated().any():
        raise ValueError("returns index contains duplicate dates")
    if target_weights.index.duplicated().any():
        raise ValueError("target_weights index contains duplicate dates")

    common_cols = returns.columns.intersection(target_weights.columns)
    if len(common_cols) == 0:
        raise ValueError("returns and target_weights share no common symbols")
    dropped = set(target_weights.columns) - set(common_cols)
    if dropped:
        logger.warning("Ignoring weights for symbols absent from returns: %s", sorted(dropped))

    common_idx = returns.index.intersection(target_weights.index)
    if len(common_idx) == 0:
        raise ValueError("returns and target_weights share no common dates")

    return (
        returns.loc[common_idx, common_cols],
        target_weights.loc[common_idx, common_cols].fillna(0.0),
    )


def _executed_rebalance_positions(
    index: pd.DatetimeIndex, rebalance_on: pd.DatetimeIndex | None, execution_lag: int
) -> set[int]:
    """Row positions at which the portfolio is restored to target.

    ``rebalance_on`` holds the dates on which the *target* is set. A target set
    at date R is executed ``execution_lag`` periods later, so the trade lands at
    position ``pos(R) + lag``. Passing the schedule matters whenever the target
    can repeat: an equal-weight mandate has a target that never changes, and a
    momentum portfolio that reselects the same three assets two months running
    has one that changes only sometimes. Without the schedule both would look
    like "no trade needed" and drift indefinitely, which is buy-and-hold wearing
    a rebalanced portfolio's name.
    """
    if rebalance_on is None or len(index) == 0:
        return set()
    positions = index.get_indexer(pd.DatetimeIndex(rebalance_on))
    return {int(p) + execution_lag for p in positions
            if p >= 0 and 0 <= p + execution_lag < len(index)}


def _simulate_holdings(
    executed_weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    cash_rate: float,
    periods_per_year: int,
    allow_drift: bool,
    rebalance_positions: set[int] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Track what is actually held, period by period, and what trading it costs.

    The distinction this exists to capture: a *target* weight is what you want,
    a *held* weight is what you have. Between rebalances they diverge, because
    positions grow and shrink with their own returns. Applying the target
    directly to returns -- the obvious implementation -- silently assumes the
    portfolio is restored to target every single period, and since the target
    has not changed, a turnover measured as ``target.diff()`` reports zero. The
    result is free daily rebalancing: the portfolio captures the rebalancing
    premium and is never billed for the trades that produce it.

    The effect on this strategy is small and, as it happens, favours the
    corrected version. It is not small for a constant-target mandate: an
    equal-weight portfolio has a target that *never* changes, so the old
    accounting charged it nothing at all while still crediting it with monthly
    rebalancing it never paid for.

    Holdings evolve self-financingly::

        earned_t   = h_{t-1} . r_t                      (on what was held)
        grown_i    = h_{t-1,i} (1 + r_{t,i})            (position values)
        drifted    = grown / (1 + earned_t)             (renormalised)

    and a trade happens only when the target changes, priced off the *drifted*
    holdings rather than off last period's target.

    ``allow_drift=False`` restores the older constant-weight behaviour, kept so
    the two conventions can be compared rather than argued about.
    """
    cols = list(executed_weights.columns)
    idx = executed_weights.index
    rebalance_positions = rebalance_positions or set()
    if not allow_drift:
        return executed_weights, compute_turnover(executed_weights)

    targets = executed_weights.to_numpy(dtype=float)
    rets = asset_returns.reindex(index=idx, columns=cols).fillna(0.0).to_numpy(dtype=float)

    n, k = targets.shape
    held = np.zeros((n, k), dtype=float)
    turnover = np.zeros(n, dtype=float)
    current = np.zeros(k, dtype=float)
    cash_per_period = cash_rate / periods_per_year

    for t in range(n):
        # Order matters: trade into the target FIRST, then earn the period's
        # return on what is actually held. Drifting before rebalancing would
        # delay every trade by a period and quietly reintroduce a timing error.
        want = targets[t]
        scheduled = t in rebalance_positions
        changed = t == 0 or not np.allclose(want, targets[t - 1], atol=1e-12, rtol=0.0)
        if scheduled or changed:
            turnover[t] = float(np.abs(want - current).sum())
            current = want.copy()

        held[t] = current

        # Self-financing update. Cash is the residual 1 - sum(weights) and grows
        # at the cash rate, so total value is 1 + earned and the renormalised
        # asset weights plus the residual cash weight still sum to one.
        earned = float(current @ rets[t]) + (1.0 - current.sum()) * cash_per_period
        grown = current * (1.0 + rets[t])
        denom = 1.0 + earned
        current = grown / denom if abs(denom) > 1e-12 else grown

    return pd.DataFrame(held, index=idx, columns=cols), pd.Series(turnover, index=idx)


def _check_returns_available(
    executed_weights: pd.DataFrame, asset_returns: pd.DataFrame, on_missing: str = "raise"
) -> None:
    """Decide what to do when capital sits in an asset with no return.

    A missing return is not the same thing as a flat day. Treating it as zero
    understates risk and quietly invents a price for an asset that may have been
    halted, delisted, or simply absent from the vendor's file -- and the
    resulting equity curve looks entirely normal, which is what makes it
    dangerous.

    The default is therefore to **raise**. ``on_missing="zero"`` restores the
    older behaviour (warn, then treat as flat) for callers who have looked at
    the gap and decided it is benign; ``"ignore"`` suppresses the check
    entirely. There is no correct universal policy here, so the choice is made
    explicit rather than buried in a fillna.
    """
    if on_missing not in {"raise", "zero", "ignore"}:
        raise ValueError(f"on_missing must be 'raise', 'zero' or 'ignore', got {on_missing!r}")
    if on_missing == "ignore":
        return

    held = executed_weights.abs() > 1e-12
    missing = held & asset_returns.isna()
    n = int(missing.to_numpy().sum())
    if not n:
        return

    first = asset_returns.index[missing.any(axis=1)][0]
    symbols = sorted(missing.columns[missing.any(axis=0)])
    detail = (
        f"{n} periods hold a non-zero weight in an asset with a missing return "
        f"(first at {first.date()}; symbols: {', '.join(symbols)})"
    )
    if on_missing == "raise":
        raise ValueError(
            detail + ". Refusing to treat a missing return as flat -- clean the panel, "
            "or pass on_missing='zero' if the gap is known to be benign."
        )
    warnings.warn(detail + "; those returns are treated as 0.0.", UserWarning, stacklevel=3)


def buy_and_hold(
    returns: pd.DataFrame,
    symbol: str,
    initial_capital: float = 100_000.0,
    commission_bps: float = 2.0,
    slippage_bps: float = 3.0,
    execution_lag: int = 1,
    name: str | None = None,
) -> BacktestResult:
    """Single-asset buy-and-hold run through the same engine.

    Deliberately not a shortcut computation: routing the benchmark through the
    identical code path means benchmark and strategy share every convention
    (execution lag, entry cost, warm-up handling), so the comparison is apples
    to apples.
    """
    if symbol not in returns.columns:
        raise KeyError(f"{symbol!r} not in returns columns: {list(returns.columns)}")
    weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    weights[symbol] = 1.0
    return run_backtest(
        returns=returns,
        target_weights=weights,
        initial_capital=initial_capital,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        execution_lag=execution_lag,
        name=name or f"{symbol} buy & hold",
    )
