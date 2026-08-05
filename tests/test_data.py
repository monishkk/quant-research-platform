"""
Data-pipeline tests.
====================

Everything downstream assumes the price panel is sorted, unique, and complete.
These tests hold that assumption up, and check that the validator actually
rejects the malformed inputs it claims to reject -- a validator that silently
passes bad data is worse than none, because it manufactures confidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.data import (
    SCHEMA,
    DataProvenance,
    load_prices,
    standardise,
    synthetic_prices,
    to_long,
    to_wide,
    validate_price_panel,
)
from quant_platform.returns import cumulative_returns, log_returns, simple_returns

SYMBOLS = ["SPY", "QQQ", "TLT"]


@pytest.fixture
def panel():
    return synthetic_prices(SYMBOLS, "2015-01-01", "2018-12-31", seed=11)


# --------------------------------------------------------------------------- #
# Schema and invariants
# --------------------------------------------------------------------------- #
def test_panel_has_canonical_schema(panel):
    assert list(panel.columns) == list(SCHEMA)


def test_panel_passes_validation(panel):
    report = validate_price_panel(panel)
    assert report["ok"]
    assert report["errors"] == []
    assert report["n_symbols"] == len(SYMBOLS)


def test_wide_invariants_hold(panel):
    """The three assertions the spec calls for."""
    prices = to_wide(panel)
    assert prices.index.is_monotonic_increasing
    assert not prices.index.duplicated().any()
    assert prices.columns.is_unique


def test_long_wide_roundtrip_preserves_values(panel):
    prices = to_wide(panel)
    back = to_long(prices)
    reshaped = back.pivot(index="date", columns="symbol", values="adjusted_close")
    reshaped.columns.name = None
    pd.testing.assert_frame_equal(prices, reshaped, check_freq=False)


def test_dates_are_timezone_naive(panel):
    assert panel["date"].dt.tz is None


def test_symbols_are_uppercase():
    long = synthetic_prices(["spy", "qqq"], "2020-01-01", "2020-06-30", seed=1)
    assert set(long["symbol"]) == {"SPY", "QQQ"}


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def test_standardise_drops_duplicate_rows(panel):
    dirty = pd.concat([panel, panel.iloc[:20]], ignore_index=True)
    cleaned = standardise(dirty)

    assert not cleaned.duplicated(subset=["date", "symbol"]).any()
    assert len(cleaned) == len(panel)


def test_standardise_sorts_unsorted_input(panel):
    shuffled = panel.sample(frac=1.0, random_state=0).reset_index(drop=True)
    cleaned = standardise(shuffled)

    for _, grp in cleaned.groupby("symbol"):
        assert grp["date"].is_monotonic_increasing


def test_standardise_rejects_missing_columns(panel):
    with pytest.raises(ValueError, match="missing required columns"):
        standardise(panel.drop(columns=["adjusted_close"]))


def test_standardise_accepts_adj_close_alias(panel):
    aliased = panel.rename(columns={"adjusted_close": "adj_close"})
    assert "adjusted_close" in standardise(aliased).columns


# --------------------------------------------------------------------------- #
# Validation catches bad data
# --------------------------------------------------------------------------- #
def test_validator_rejects_duplicates(panel):
    dirty = pd.concat([panel, panel.iloc[:5]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_price_panel(dirty, raise_on_error=True)


def test_validator_rejects_non_positive_prices(panel):
    dirty = panel.copy()
    dirty.loc[dirty.index[0], "adjusted_close"] = 0.0
    with pytest.raises(ValueError, match="non-positive"):
        validate_price_panel(dirty, raise_on_error=True)


def test_validator_rejects_inverted_high_low(panel):
    dirty = panel.copy()
    dirty.loc[dirty.index[0], "high"] = 1.0
    dirty.loc[dirty.index[0], "low"] = 999.0
    with pytest.raises(ValueError, match="high < low"):
        validate_price_panel(dirty, raise_on_error=True)


def test_validator_rejects_unsorted_dates(panel):
    dirty = panel.copy()
    dirty.iloc[0, dirty.columns.get_loc("date")] = pd.Timestamp("2099-01-01")
    with pytest.raises(ValueError, match="monotonically increasing"):
        validate_price_panel(dirty, raise_on_error=True)


def test_validator_can_report_without_raising(panel):
    dirty = pd.concat([panel, panel.iloc[:5]], ignore_index=True)
    report = validate_price_panel(dirty, raise_on_error=False)
    assert not report["ok"] and report["errors"]


# --------------------------------------------------------------------------- #
# Return calculations
# --------------------------------------------------------------------------- #
def test_simple_returns_first_row_is_nan(panel):
    r = simple_returns(to_wide(panel))
    assert r.iloc[0].isna().all(), "there is no return on the first observation"


def test_simple_returns_match_manual_ratio(panel):
    prices = to_wide(panel)
    r = simple_returns(prices)
    manual = prices.iloc[5, 0] / prices.iloc[4, 0] - 1
    assert r.iloc[5, 0] == pytest.approx(manual, rel=1e-12)


def test_log_and_simple_returns_are_consistent(panel):
    prices = to_wide(panel)
    simple, log = simple_returns(prices), log_returns(prices)
    np.testing.assert_allclose(
        np.log1p(simple.dropna().to_numpy()), log.dropna().to_numpy(), rtol=1e-10
    )


def test_log_returns_sum_to_total_growth(panel):
    """Log returns aggregate additively through time; that is their whole point."""
    prices = to_wide(panel)
    lr = log_returns(prices).dropna()
    total = np.exp(lr.sum())
    expected = prices.iloc[-1] / prices.iloc[0]
    np.testing.assert_allclose(total.to_numpy(), expected.to_numpy(), rtol=1e-10)


def test_cumulative_returns_reach_total_growth(panel):
    prices = to_wide(panel)
    cum = cumulative_returns(simple_returns(prices))
    expected = prices.iloc[-1] / prices.iloc[0] - 1
    np.testing.assert_allclose(cum.iloc[-1].to_numpy(), expected.to_numpy(), rtol=1e-10)


def test_adjusted_and_unadjusted_differ_conceptually(panel):
    """Both columns must exist so the comparison in notebook 01 is possible."""
    assert "close" in panel.columns and "adjusted_close" in panel.columns


# --------------------------------------------------------------------------- #
# Synthetic generator sanity
# --------------------------------------------------------------------------- #
def test_synthetic_data_is_deterministic():
    a = synthetic_prices(SYMBOLS, "2015-01-01", "2016-01-01", seed=99)
    b = synthetic_prices(SYMBOLS, "2015-01-01", "2016-01-01", seed=99)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_data_differs_by_seed():
    a = synthetic_prices(SYMBOLS, "2015-01-01", "2016-01-01", seed=1)
    b = synthetic_prices(SYMBOLS, "2015-01-01", "2016-01-01", seed=2)
    assert not a["adjusted_close"].equals(b["adjusted_close"])


def test_synthetic_prices_stay_in_a_plausible_range():
    """Guards the AR(1) drift term against the runaway-compounding bug.

    An earlier version added a persistent drift wobble with a ~0.9%/day standard
    deviation, which compounded to price paths 700,000x their starting value and
    a Sharpe of 6. Realistic bounds here would have caught that immediately.
    """
    long = synthetic_prices(
        ["A", "B", "C", "D", "E"], "2010-01-01", "2025-12-31", seed=3
    )
    wide = to_wide(long)
    total_growth = wide.iloc[-1] / wide.iloc[0]

    assert (total_growth > 0.2).all(), f"an asset lost ~everything: {total_growth.min():.4f}"
    assert (total_growth < 30).all(), f"runaway compounding: {total_growth.max():.1f}x"

    ann_vol = simple_returns(wide).std(ddof=1) * np.sqrt(252)
    assert (ann_vol > 0.03).all() and (ann_vol < 0.60).all(), f"implausible vol: {ann_vol.to_dict()}"

    ann_ret = total_growth ** (252 / len(wide)) - 1
    assert (ann_ret > -0.25).all() and (ann_ret < 0.40).all(), f"implausible drift: {ann_ret.to_dict()}"


def test_synthetic_assets_are_positively_correlated():
    """A market factor is in the generator; the panel should reflect it."""
    long = synthetic_prices(["A", "B", "C", "D"], "2010-01-01", "2020-12-31", seed=4)
    corr = simple_returns(to_wide(long)).corr()
    off_diagonal = corr.to_numpy()[~np.eye(len(corr), dtype=bool)]
    assert off_diagonal.mean() > 0.1


# --------------------------------------------------------------------------- #
# Caching and provenance
# --------------------------------------------------------------------------- #
def test_load_prices_caches_and_records_provenance(tmp_path):
    prices = load_prices(
        SYMBOLS, "2015-01-01", "2016-12-31",
        source="synthetic", data_dir=tmp_path, seed=5,
    )
    assert isinstance(prices, pd.DataFrame) and not prices.empty

    processed = list((tmp_path / "processed").glob("*.parquet"))
    raw = list((tmp_path / "raw").glob("*.parquet"))
    meta = list((tmp_path / "processed").glob("*.meta.json"))
    assert len(processed) == 1 and len(raw) == 1 and len(meta) == 1

    prov = DataProvenance.from_json(meta[0])
    assert prov.source == "synthetic"
    assert prov.n_rows > 0
    assert "NOT REAL MARKET DATA" in prov.note


def test_cached_load_returns_identical_data(tmp_path):
    kwargs = dict(source="synthetic", data_dir=tmp_path, seed=6)
    first = load_prices(SYMBOLS, "2015-01-01", "2016-12-31", **kwargs)
    second = load_prices(SYMBOLS, "2015-01-01", "2016-12-31", **kwargs)
    pd.testing.assert_frame_equal(first, second)


def test_load_prices_can_return_long_format(tmp_path):
    long = load_prices(
        SYMBOLS, "2015-01-01", "2016-12-31",
        source="synthetic", data_dir=tmp_path, seed=7, wide=False,
    )
    assert list(long.columns) == list(SCHEMA)


def test_parquet_roundtrip_preserves_dtypes(tmp_path, panel):
    path = tmp_path / "panel.parquet"
    panel.to_parquet(path, index=False)
    reloaded = standardise(pd.read_parquet(path))
    pd.testing.assert_frame_equal(panel, reloaded)


# --------------------------------------------------------------------------- #
# Cache integrity
# --------------------------------------------------------------------------- #
def test_cached_panel_without_provenance_is_refused(tmp_path):
    """A panel whose origin cannot be confirmed must not be handed back.

    Losing the sidecar is exactly the state in which synthetic data becomes
    indistinguishable from real data, so it fails loudly instead.
    """
    load_prices(SYMBOLS, "2015-01-01", "2016-12-31",
                source="synthetic", data_dir=tmp_path, seed=5)
    for meta in (tmp_path / "processed").glob("*.meta.json"):
        meta.unlink()

    with pytest.raises(FileNotFoundError, match="no provenance record"):
        load_prices(SYMBOLS, "2015-01-01", "2016-12-31",
                    source="synthetic", data_dir=tmp_path, seed=5)


def test_cache_refuses_to_serve_one_source_as_another(tmp_path):
    """Synthetic prices must never satisfy a request for real ones.

    This reproduces the contamination path directly: a synthetic panel is moved
    into the slot a yfinance request would look in. Without the guard it would
    load silently and every downstream number would describe simulated data.
    """
    load_prices(SYMBOLS, "2015-01-01", "2016-12-31",
                source="synthetic", data_dir=tmp_path, seed=5)
    proc = tmp_path / "processed"
    for path in list(proc.iterdir()):
        path.rename(proc / path.name.replace("synthetic_", "yfinance_", 1))

    with pytest.raises(ValueError, match="was produced by source"):
        load_prices(SYMBOLS, "2015-01-01", "2016-12-31",
                    source="yfinance", data_dir=tmp_path, use_cache=True)


def test_fallback_panel_is_stored_under_the_source_it_came_from(tmp_path):
    """A fallback must not occupy the real provider's cache slot."""
    from quant_platform import data as data_mod

    def boom(*_a, **_k):
        raise RuntimeError("network down")

    original = data_mod.download_prices
    data_mod.download_prices = boom
    try:
        data_mod.load_prices(SYMBOLS, "2015-01-01", "2016-12-31", source="yfinance",
                             data_dir=tmp_path, allow_synthetic_fallback=True, seed=5)
    finally:
        data_mod.download_prices = original

    written = {p.name.split("_")[0] for p in (tmp_path / "processed").glob("*.parquet")}
    assert written == {"synthetic"}, f"fallback poisoned the yfinance slot: {written}"


def test_provenance_records_code_and_data_identity(tmp_path):
    """A result is traceable only if both the code and the data are pinned."""
    load_prices(SYMBOLS, "2015-01-01", "2016-12-31",
                source="synthetic", data_dir=tmp_path, seed=5)
    meta = next((tmp_path / "processed").glob("*.meta.json"))
    prov = DataProvenance.from_json(meta)
    assert len(prov.data_sha256) == 64
    assert prov.git_commit == "" or len(prov.git_commit) == 40


def test_checksum_detects_a_changed_price(tmp_path):
    from quant_platform.data import panel_checksum

    long = load_prices(SYMBOLS, "2015-01-01", "2016-12-31", source="synthetic",
                       data_dir=tmp_path, seed=5, wide=False)
    before = panel_checksum(long)
    tweaked = long.copy()
    tweaked.loc[tweaked.index[10], "adjusted_close"] *= 1.0001
    assert panel_checksum(tweaked) != before
    assert panel_checksum(long.sample(frac=1.0, random_state=0)) == before  # order-independent
