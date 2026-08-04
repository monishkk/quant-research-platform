"""Shared fixtures.

The fixtures here are deliberately *analytically tractable* rather than
realistic: a constant price, a fixed-growth price, a two-asset toy. When a test
over these fails, the expected value can be computed by hand, so the test tells
you what is broken rather than merely that something is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.data import synthetic_prices, to_wide


@pytest.fixture(scope="session")
def dates() -> pd.DatetimeIndex:
    """Three years of business days -- enough for a 252-day signal to form."""
    return pd.bdate_range("2015-01-01", periods=756, name="date")


@pytest.fixture
def constant_prices(dates) -> pd.DataFrame:
    """Every price flat forever. All returns must be exactly zero."""
    return pd.DataFrame(100.0, index=dates, columns=["AAA", "BBB", "CCC", "DDD"])


@pytest.fixture
def growing_prices(dates) -> pd.DataFrame:
    """Deterministic compounding at known, distinct daily rates.

    Rates are ordered AAA > BBB > CCC > DDD, so the momentum ranking has a single
    unambiguous correct answer at every date.
    """
    rates = {"AAA": 0.0008, "BBB": 0.0005, "CCC": 0.0002, "DDD": -0.0001}
    n = len(dates)
    return pd.DataFrame(
        {sym: 100.0 * (1.0 + r) ** np.arange(n) for sym, r in rates.items()},
        index=dates,
    )


@pytest.fixture
def toy_prices() -> pd.DataFrame:
    """A six-day, two-asset panel small enough to verify by hand."""
    idx = pd.bdate_range("2020-01-01", periods=6, name="date")
    return pd.DataFrame(
        {
            "AAA": [100.0, 110.0, 121.0, 121.0, 133.1, 133.1],
            "BBB": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        },
        index=idx,
    )


@pytest.fixture(scope="session")
def realistic_prices() -> pd.DataFrame:
    """A seeded synthetic panel: noisy, but identical on every run."""
    long = synthetic_prices(
        ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "VNQ", "DBC"],
        "2010-01-01",
        "2020-12-31",
        seed=7,
    )
    return to_wide(long)
