"""
Data pipeline.
==============

Responsibilities
----------------
1. Download raw daily bars for a list of symbols.
2. Standardise them into one canonical long format.
3. Run integrity checks (duplicates, ordering, missing values, gaps).
4. Persist raw + processed data as Parquet, with provenance metadata.
5. Hand back either the long panel or a wide price matrix.

Canonical long format
---------------------
    date | symbol | open | high | low | close | adjusted_close | volume

``date`` is a tz-naive daily timestamp, ``symbol`` an uppercase ticker string.

Prices used for research
------------------------
Everything downstream uses ``adjusted_close`` -- split- and dividend-adjusted --
because a momentum signal computed on unadjusted closes would read a 2-for-1
split as a -50% return. ``notebooks/01_data_exploration.ipynb`` shows the
difference explicitly.

Survivorship bias
-----------------
This universe is a fixed, hand-picked list of ETFs that all exist today, so the
panel is survivorship-biased by construction. For a 9-ticker cross-asset ETF
sleeve the effect is small (none of these were ever at risk of delisting), but it
is a real limitation and is stated in the report. For single-name equities you
would need point-in-time index membership, delisting returns, and corporate
actions -- do not reuse this module for that without fixing those first.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# The canonical column order for the long panel.
SCHEMA: tuple[str, ...] = (
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
)

PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "adjusted_close")


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DataProvenance:
    """Where a processed dataset came from, so a result can be traced back."""

    source: str
    symbols: tuple[str, ...]
    start: str
    end: str
    downloaded_at_utc: str
    n_rows: int
    first_date: str
    last_date: str
    note: str = ""
    # Identity of the code and of the data itself. A result is only traceable if
    # you can pin down *both*: the same script against revised prices, or revised
    # code against the same prices, are different experiments.
    git_commit: str = ""
    data_sha256: str = ""

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> DataProvenance:
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["symbols"] = tuple(raw["symbols"])
        # Tolerate records written before these fields existed.
        raw.setdefault("git_commit", "")
        raw.setdefault("data_sha256", "")
        return cls(**raw)


def current_git_commit() -> str:
    """The repository's HEAD, or ``""`` outside a checkout.

    Recorded rather than assumed: 'reproducible' means nothing if the reader
    cannot tell which version of the code produced a number.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment dependent
        return ""


def panel_checksum(long: pd.DataFrame) -> str:
    """Content hash of a canonical long panel.

    Deliberately computed on sorted values rather than on the file bytes:
    Parquet encoding can differ run to run, but the *data* either changed or it
    did not. Yahoo back-adjusts its close series whenever a dividend is paid, so
    a silently revised history is a real possibility this makes detectable.
    """
    ordered = long.sort_values(["symbol", "date"]).reset_index(drop=True)
    numeric = ordered.select_dtypes("number").round(10).to_numpy(dtype="float64")
    h = hashlib.sha256()
    h.update(numeric.tobytes())
    h.update("|".join(ordered["symbol"].astype(str)).encode("utf-8"))
    # Normalise the datetime resolution before hashing. Parquet stores dates as
    # datetime64[ms] while the in-memory frame is datetime64[ns], so hashing the
    # raw int64 makes identical dates differ by a factor of 10**6 -- which made
    # the checksum mismatch on *every* cache read and cry wolf about corruption
    # that was not there. The unit is an encoding detail; the instant is not.
    stamps = pd.to_datetime(ordered["date"]).astype("datetime64[ns]").astype("int64")
    h.update(stamps.to_numpy().tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def download_prices(
    symbols: Sequence[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download daily bars from Yahoo Finance and return the canonical long panel.

    Raises
    ------
    RuntimeError
        If yfinance is unavailable or returns nothing usable. Callers that want a
        graceful degradation should catch this and fall back to
        :func:`synthetic_prices`.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("yfinance is not installed; `pip install yfinance`") from exc

    symbols = [s.upper() for s in symbols]
    logger.info("Downloading %d symbols from yfinance (%s -> %s)", len(symbols), start, end)

    raw = yf.download(
        tickers=symbols,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,  # keep close AND adj close so we can compare them
        actions=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if raw is None or len(raw) == 0:
        raise RuntimeError("yfinance returned an empty frame")

    frames: list[pd.DataFrame] = []
    for sym in symbols:
        sub = _extract_symbol_frame(raw, sym, single=len(symbols) == 1)
        if sub is None or sub.dropna(how="all").empty:
            logger.warning("No data returned for %s -- dropping it from the panel", sym)
            continue
        sub = sub.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adjusted_close",
                "Volume": "volume",
            }
        )
        if "adjusted_close" not in sub.columns:
            # Newer yfinance builds may already return adjusted closes only.
            sub["adjusted_close"] = sub["close"]
        sub = sub.reset_index().rename(columns={"Date": "date", "index": "date"})
        sub["symbol"] = sym
        frames.append(sub)

    if not frames:
        raise RuntimeError("yfinance returned no usable rows for any requested symbol")

    long = pd.concat(frames, ignore_index=True)
    return standardise(long)


def _extract_symbol_frame(raw: pd.DataFrame, symbol: str, single: bool) -> pd.DataFrame | None:
    """Pull one symbol out of a yfinance frame, tolerating its various layouts."""
    if not isinstance(raw.columns, pd.MultiIndex):
        return raw.copy() if single else None

    # group_by="ticker" -> level 0 is the ticker; some versions invert the levels.
    if symbol in raw.columns.get_level_values(0):
        return raw[symbol].copy()
    if symbol in raw.columns.get_level_values(-1):
        return raw.xs(symbol, axis=1, level=-1).copy()
    return None


# --------------------------------------------------------------------------- #
# Synthetic fallback
# --------------------------------------------------------------------------- #
def synthetic_prices(
    symbols: Sequence[str],
    start: str,
    end: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a deterministic synthetic panel with the canonical schema.

    This exists so the full pipeline -- backtest, metrics, report -- can be
    exercised and tested with no network access. It is a correlated geometric
    Brownian motion with a slow-moving drift component, which produces mild,
    non-degenerate cross-sectional momentum.

    A result produced from synthetic data is a software test, not a research
    finding. Every artefact generated this way is labelled as such.
    """
    symbols = [s.upper() for s in symbols]
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end, name="date")
    n_days, n_assets = len(dates), len(symbols)

    # Per-asset annualised drift and volatility, loosely in the range that
    # cross-asset ETFs actually occupy.
    drift = rng.uniform(0.01, 0.10, size=n_assets)
    vol = rng.uniform(0.09, 0.26, size=n_assets)

    # One market factor plus idiosyncratic noise => realistic positive correlation.
    # The combination is renormalised to unit variance so that realised vol
    # matches the target `vol` rather than drifting above it as beta rises.
    beta = rng.uniform(0.35, 0.95, size=n_assets)
    resid = np.sqrt(np.maximum(1 - beta**2, 0.05))
    market = rng.standard_normal(n_days)
    idio = rng.standard_normal((n_days, n_assets))
    shock = (beta * market[:, None] + resid * idio) / np.sqrt(beta**2 + resid**2)

    # A slow AR(1) tilt on the *annualised* drift, so relative strength persists
    # for months at a time -- otherwise there is no momentum for the strategy to
    # find. phi=0.99 gives a ~69-day half-life; the innovation is scaled so the
    # stationary standard deviation is TILT_SD (an annualised rate), keeping the
    # total drift inside a plausible band instead of compounding without bound.
    phi, tilt_sd = 0.99, 0.04
    innovation = tilt_sd * np.sqrt(1 - phi**2)
    tilt = np.zeros((n_days, n_assets))
    for t in range(1, n_days):
        tilt[t] = phi * tilt[t - 1] + rng.standard_normal(n_assets) * innovation

    dt = 1 / 252
    daily = (drift + tilt - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * shock
    prices = 100 * np.exp(np.cumsum(daily, axis=0))

    wide = pd.DataFrame(prices, index=dates, columns=symbols)
    long = wide.stack().rename("adjusted_close").reset_index()
    long.columns = ["date", "symbol", "adjusted_close"]
    long["close"] = long["adjusted_close"]
    long["open"] = long["adjusted_close"] * (1 + rng.normal(0, 0.001, len(long)))
    long["high"] = long[["open", "close"]].max(axis=1) * (1 + abs(rng.normal(0, 0.002, len(long))))
    long["low"] = long[["open", "close"]].min(axis=1) * (1 - abs(rng.normal(0, 0.002, len(long))))
    long["volume"] = rng.integers(1_000_000, 90_000_000, len(long))
    return standardise(long)


# --------------------------------------------------------------------------- #
# Standardise + validate
# --------------------------------------------------------------------------- #
def standardise(long: pd.DataFrame) -> pd.DataFrame:
    """Coerce a long frame to the canonical schema, dtypes, and ordering."""
    df = long.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"adj_close": "adjusted_close", "adjclose": "adjusted_close"})

    missing = [c for c in SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"Long frame is missing required columns: {missing}")

    df = df[list(SCHEMA)]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    for col in (*PRICE_COLUMNS, "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with no usable price, de-duplicate, and sort.
    df = df.dropna(subset=["adjusted_close"])
    before = len(df)
    df = df.drop_duplicates(subset=["date", "symbol"], keep="last")
    if len(df) < before:
        logger.warning("Dropped %d duplicate (date, symbol) rows", before - len(df))

    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def validate_price_panel(long: pd.DataFrame, *, raise_on_error: bool = True) -> dict:
    """Run integrity checks on the long panel and return a report dictionary.

    Checks
    ------
    - schema and dtypes are as expected
    - no duplicate ``(date, symbol)`` pairs
    - dates are sorted ascending within each symbol
    - prices are strictly positive
    - ``high >= low``
    - missing-value counts per symbol
    - per-symbol coverage (first/last date, row count)
    """
    errors: list[str] = []
    warnings: list[str] = []

    missing_cols = [c for c in SCHEMA if c not in long.columns]
    if missing_cols:
        errors.append(f"missing columns: {missing_cols}")
        if raise_on_error:
            raise ValueError(f"Price panel failed validation: {errors}")
        return {"ok": False, "errors": errors, "warnings": warnings}

    dupes = long.duplicated(subset=["date", "symbol"]).sum()
    if dupes:
        errors.append(f"{dupes} duplicate (date, symbol) rows")

    for sym, grp in long.groupby("symbol"):
        if not grp["date"].is_monotonic_increasing:
            errors.append(f"{sym}: dates are not monotonically increasing")

    non_positive = int((long["adjusted_close"] <= 0).sum())
    if non_positive:
        errors.append(f"{non_positive} rows with non-positive adjusted_close")

    both = long[["high", "low"]].dropna()
    inverted = int((both["high"] < both["low"]).sum())
    if inverted:
        errors.append(f"{inverted} rows where high < low")

    nulls = long[list(PRICE_COLUMNS)].isna().sum()
    for col, n in nulls.items():
        if n:
            warnings.append(f"{n} missing values in '{col}'")

    coverage = (
        long.groupby("symbol")["date"]
        .agg(first="min", last="max", rows="count")
        .sort_values("first")
    )
    # Symbols that start late shrink the usable common history -- worth flagging.
    latest_start = coverage["first"].max()
    for sym, row in coverage.iterrows():
        if row["first"] > coverage["first"].min():
            warnings.append(f"{sym} starts at {row['first'].date()} (panel start is later)")

    report = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "n_rows": len(long),
        "n_symbols": long["symbol"].nunique(),
        "date_min": str(long["date"].min().date()),
        "date_max": str(long["date"].max().date()),
        "common_history_starts": str(latest_start.date()),
        "coverage": coverage,
    }

    if errors and raise_on_error:
        raise ValueError(f"Price panel failed validation: {errors}")
    return report


# --------------------------------------------------------------------------- #
# Reshaping
# --------------------------------------------------------------------------- #
def to_wide(long: pd.DataFrame, field: str = "adjusted_close") -> pd.DataFrame:
    """Long panel -> wide matrix indexed by date with one column per symbol."""
    if field not in long.columns:
        raise KeyError(f"'{field}' not in panel; available: {list(long.columns)}")
    wide = long.pivot(index="date", columns="symbol", values=field).sort_index()
    wide.columns.name = None
    _assert_wide_invariants(wide)
    return wide


def to_long(wide: pd.DataFrame, field: str = "adjusted_close") -> pd.DataFrame:
    """Wide matrix -> long panel (inverse of :func:`to_wide` for a single field)."""
    long = wide.stack().rename(field).reset_index()
    long.columns = ["date", "symbol", field]
    return long


def _assert_wide_invariants(wide: pd.DataFrame) -> None:
    """The three invariants every downstream function is allowed to assume.

    Raised rather than asserted: the whole platform relies on these holding, and
    `python -O` removes assert statements. A guarantee that evaporates under an
    optimisation flag is worse than no guarantee, because it is still advertised.
    """
    if not wide.index.is_monotonic_increasing:
        raise ValueError("price index is not sorted ascending")
    if wide.index.duplicated().any():
        dupes = wide.index[wide.index.duplicated()].unique()[:3]
        raise ValueError(f"price index contains duplicate dates (e.g. {list(dupes)})")
    if not wide.columns.is_unique:
        raise ValueError("price columns contain duplicate symbols")


# --------------------------------------------------------------------------- #
# Persistence + top-level loader
# --------------------------------------------------------------------------- #
def _cache_key(symbols: Iterable[str], start: str, end: str, source: str) -> str:
    syms = "-".join(sorted(s.upper() for s in symbols))
    return f"{source}_{syms}_{start}_{end}".replace(":", "")


def load_prices(
    symbols: Sequence[str],
    start: str = "2007-01-01",
    end: str = "2025-12-31",
    *,
    source: str = "yfinance",
    data_dir: str | Path = "data",
    use_cache: bool = True,
    allow_synthetic_fallback: bool = False,
    seed: int = 42,
    wide: bool = True,
    field: str = "adjusted_close",
) -> pd.DataFrame:
    """Load prices, downloading and caching to Parquet on first use.

    ``allow_synthetic_fallback`` defaults to **False**: a research pipeline
    should fail loudly when its data source is unavailable rather than quietly
    substituting simulated prices. Enable it only for smoke tests.

    Parameters
    ----------
    wide
        ``True`` (default) returns a date x symbol matrix of ``field``.
        ``False`` returns the canonical long panel.

    Returns
    -------
    pandas.DataFrame
        Guaranteed, for the wide form, to satisfy::

            prices.index.is_monotonic_increasing
            not prices.index.duplicated().any()
            prices.columns.is_unique
    """
    return load_prices_with_provenance(
        symbols, start, end, source=source, data_dir=data_dir, use_cache=use_cache,
        allow_synthetic_fallback=allow_synthetic_fallback, seed=seed, wide=wide, field=field,
    )[0]


def load_prices_with_provenance(
    symbols: Sequence[str],
    start: str = "2007-01-01",
    end: str = "2025-12-31",
    *,
    source: str = "yfinance",
    data_dir: str | Path = "data",
    use_cache: bool = True,
    allow_synthetic_fallback: bool = False,
    seed: int = 42,
    wide: bool = True,
    field: str = "adjusted_close",
) -> tuple[pd.DataFrame, DataProvenance]:
    """As :func:`load_prices`, but also returns the provenance of what was loaded.

    Callers that report results should use this rather than looking the sidecar
    up separately by cache key. Deriving provenance from the same call that
    produced the data makes it impossible for the two to disagree -- a separate
    lookup can silently return ``None``, and a report with no provenance looks
    identical to a report with clean provenance.
    """
    data_dir = Path(data_dir)
    raw_dir, proc_dir = data_dir / "raw", data_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    key = _cache_key(symbols, start, end, source)
    processed_path = proc_dir / f"{key}.parquet"
    meta_path = proc_dir / f"{key}.meta.json"

    if use_cache and processed_path.exists():
        logger.info("Loading cached panel: %s", processed_path)
        long = standardise(pd.read_parquet(processed_path))
        prov = _verify_cached_provenance(meta_path, processed_path, source, long)
    else:
        long, actual_source, note = _fetch(
            symbols, start, end, source, allow_synthetic_fallback, seed
        )
        validate_price_panel(long, raise_on_error=True)

        # A fallback panel is stored under the source it ACTUALLY came from, so
        # it can never occupy the slot the real provider is expected to fill. A
        # later run asking for yfinance will find nothing and re-download rather
        # than silently inheriting simulated prices.
        write_key = _cache_key(symbols, start, end, actual_source)
        processed_path = proc_dir / f"{write_key}.parquet"
        meta_path = proc_dir / f"{write_key}.meta.json"

        # Timestamped, never-overwritten snapshot of exactly what was ingested.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        long.to_parquet(raw_dir / f"{write_key}_{stamp}.parquet", index=False)
        long.to_parquet(processed_path, index=False)

        # Checksum the *persisted* form, not the in-memory frame. Parquet does
        # not round-trip every dtype identically (dates come back as
        # milliseconds, for one), so hashing what is in memory at write time
        # produces a digest the reader can never reproduce -- an integrity check
        # that fails every single time, which is worse than none at all because
        # it teaches the reader to ignore it.
        long = standardise(pd.read_parquet(processed_path))

        prov = DataProvenance(
            source=actual_source,
            symbols=tuple(s.upper() for s in symbols),
            start=start,
            end=end,
            downloaded_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            n_rows=len(long),
            first_date=str(long["date"].min().date()),
            last_date=str(long["date"].max().date()),
            note=note,
            git_commit=current_git_commit(),
            data_sha256=panel_checksum(long),
        )
        prov.to_json(meta_path)
        logger.info("Wrote %s (%d rows) and provenance %s", processed_path, len(long), meta_path)

    return (to_wide(long, field=field) if wide else long), prov


def _verify_cached_provenance(
    meta_path: Path, processed_path: Path, requested_source: str, long: pd.DataFrame
) -> DataProvenance:
    """Refuse to hand back cached data whose origin cannot be confirmed.

    The failure this prevents: a download fails, synthetic prices are generated
    as a fallback, and every subsequent run quietly trains on simulated data
    while the report says "yfinance". Cheap to prevent, and impossible to notice
    afterwards from the numbers alone -- synthetic momentum panels look
    perfectly plausible.
    """
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Cached panel {processed_path} has no provenance record at {meta_path}. "
            "Its origin cannot be verified; delete the cached file and re-download, "
            "or pass use_cache=False."
        )

    prov = DataProvenance.from_json(meta_path)
    if prov.source != requested_source:
        raise ValueError(
            f"Cached panel at {processed_path} was produced by source "
            f"{prov.source!r}, but {requested_source!r} was requested. "
            "Refusing to substitute one for the other."
        )
    if prov.source == "synthetic":
        logger.warning(
            "Loaded SYNTHETIC prices from cache -- results describe a simulated "
            "panel and carry no information about real markets."
        )

    digest = panel_checksum(long)
    if prov.data_sha256 and digest != prov.data_sha256:
        logger.warning(
            "Cached panel checksum %s does not match the recorded %s; the file has "
            "changed since it was written.", digest[:12], prov.data_sha256[:12],
        )
    return prov


def _fetch(
    symbols: Sequence[str],
    start: str,
    end: str,
    source: str,
    allow_synthetic_fallback: bool,
    seed: int,
) -> tuple[pd.DataFrame, str, str]:
    """Fetch from the requested source, falling back to synthetic if permitted."""
    if source == "synthetic":
        return (
            synthetic_prices(symbols, start, end, seed=seed),
            "synthetic",
            f"Deterministic synthetic GBM panel, seed={seed}. NOT REAL MARKET DATA.",
        )

    try:
        return download_prices(symbols, start, end), "yfinance", ""
    except Exception as exc:
        if not allow_synthetic_fallback:
            raise
        logger.error("Download failed (%s); falling back to synthetic data", exc)
        return (
            synthetic_prices(symbols, start, end, seed=seed),
            "synthetic",
            f"yfinance download failed ({exc}); fell back to synthetic GBM, seed={seed}. "
            "NOT REAL MARKET DATA.",
        )


def read_provenance(
    symbols: Sequence[str],
    start: str,
    end: str,
    source: str = "yfinance",
    data_dir: str | Path = "data",
) -> DataProvenance | None:
    """Return the provenance record for a cached dataset, if one exists."""
    path = Path(data_dir) / "processed" / f"{_cache_key(symbols, start, end, source)}.meta.json"
    return DataProvenance.from_json(path) if path.exists() else None
