"""
quant_platform
==============

A small, tested, reproducible research platform for cross-sectional ETF momentum.

Design principle
----------------
The pipeline is deliberately split into visibly separate stages:

    raw data -> signal -> desired position -> executed position -> portfolio return

Each stage lives in its own module and each boundary is testable. Nothing in this
package is allowed to use information that would not have been available at the
moment a decision is made; see ``portfolio.run_backtest`` and
``tests/test_no_lookahead.py`` for how that is enforced.
"""

__version__ = "0.1.0"

from quant_platform import metrics, reporting, validation
from quant_platform.costs import CostModel
from quant_platform.data import (
    download_prices,
    load_prices,
    to_long,
    to_wide,
    validate_price_panel,
)
from quant_platform.portfolio import BacktestResult, run_backtest
from quant_platform.returns import (
    cumulative_returns,
    log_returns,
    rolling_volatility,
    simple_returns,
)
from quant_platform.signals import (
    apply_max_weight,
    build_target_weights,
    rebalance_dates,
    top_n_equal_weight,
    trailing_momentum,
)

__all__ = [
    "__version__",
    # data
    "load_prices",
    "download_prices",
    "validate_price_panel",
    "to_wide",
    "to_long",
    # returns
    "simple_returns",
    "log_returns",
    "cumulative_returns",
    "rolling_volatility",
    # signals
    "trailing_momentum",
    "top_n_equal_weight",
    "apply_max_weight",
    "rebalance_dates",
    "build_target_weights",
    # engine
    "CostModel",
    "run_backtest",
    "BacktestResult",
    # modules
    "metrics",
    "validation",
    "reporting",
]
