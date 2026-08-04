"""
Transaction-cost model.
=======================

Version 1 is a linear cost on turnover::

    cost_t = turnover_t * (commission_bps + slippage_bps) / 10_000

Definition of turnover (state it precisely or the numbers mean nothing)
-----------------------------------------------------------------------
    turnover_t = sum_i | w_i,t - w_i,t-1 |

where ``w`` are the **executed** weights. This is the *two-way* (gross) sum of
absolute weight changes: a rotation out of one 33% position and into another
counts as 0.33 + 0.33 = 0.66, not 0.33.

Consequences to keep in mind when comparing against other backtests:

* Many papers quote *one-way* turnover, which is exactly half of this. If you
  compare a turnover figure from here against a published one, check the
  convention first -- a factor of two in turnover is a factor of two in costs.
* Charging the full two-way sum at the full bps rate is therefore the
  **conservative** choice: every share that changes hands is charged once,
  which is what a broker actually does.

What this model deliberately ignores (v1 simplifications)
---------------------------------------------------------
* **Drift.** Between rebalances, weights are held at the target rather than
  allowed to drift with relative performance. Real drift creates small
  additional turnover at each rebalance, so true costs are slightly higher than
  modelled. For a monthly-rebalanced 3-asset ETF sleeve the omission is small
  relative to the 5 bps assumption, but it is an omission.
* **Market impact.** Cost per unit of turnover is constant, independent of order
  size and ADV. Fine for nine ETFs that trade billions per day at any plausible
  AUM; wrong for small caps or size.
* **Bid-ask spread dynamics.** A flat slippage number stands in for a spread
  that in reality widens in exactly the stressed markets where momentum
  strategies tend to rebalance hardest. Costs are therefore understated in the
  tail.
* **Financing, borrow, and taxes.** Long-only and unlevered here, so no
  financing or borrow. Taxes are out of scope entirely.

Each of these makes the modelled result *better* than reality, never worse. The
cost sensitivity analysis (0/5/10/20 bps) exists to show how much that matters.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = ["CostModel", "compute_turnover"]


def compute_turnover(executed_weights: pd.DataFrame) -> pd.Series:
    """Two-way turnover per period: ``sum_i |w_i,t - w_i,t-1|``.

    The first period is treated as a move from all-cash into the initial
    portfolio, so the initial purchase is charged rather than being free.
    """
    weights = executed_weights.fillna(0.0)
    deltas = weights.diff()
    if len(weights) > 0:
        deltas.iloc[0] = weights.iloc[0]  # entry from cash costs money
    return deltas.abs().sum(axis=1)


@dataclass(frozen=True)
class CostModel:
    """Linear per-unit-of-turnover cost model.

    Parameters
    ----------
    commission_bps
        Broker commission in basis points of traded notional.
    slippage_bps
        Execution slippage / effective half-spread, also in basis points.
    """

    commission_bps: float = 2.0
    slippage_bps: float = 3.0

    @property
    def total_bps(self) -> float:
        """Combined cost charged per unit of turnover, in basis points."""
        return float(self.commission_bps) + float(self.slippage_bps)

    def apply(self, turnover: pd.Series) -> pd.Series:
        """Turnover series -> cost series, expressed as a drag on period return."""
        return turnover.fillna(0.0) * self.total_bps / 10_000.0

    def __str__(self) -> str:
        return (
            f"CostModel(commission={self.commission_bps:g}bps, "
            f"slippage={self.slippage_bps:g}bps, total={self.total_bps:g}bps "
            f"per unit of two-way turnover)"
        )
