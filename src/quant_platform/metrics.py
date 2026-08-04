"""
Performance and risk metrics.
=============================

Every statistic here is implemented from its definition rather than imported, so
that the conventions are visible and testable. Where a convention is contested
(Sortino's denominator, VaR's sign, turnover's direction) the choice is stated in
the docstring.

Core definitions, for daily returns with ``P = 252``:

    Sharpe   = sqrt(P) * mean(r - rf/P) / std(r - rf/P, ddof=1)
    CAGR     = (V_T / V_0) ^ (P / N) - 1
    DD_t     = V_t / max_{s<=t} V_s - 1

All functions accept a ``pandas.Series`` of **simple** period returns and are
NaN-tolerant: NaNs are dropped, not filled, before estimation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "annualized_return",
    "annualized_volatility",
    "annualized_sharpe",
    "sortino_ratio",
    "drawdown_series",
    "maximum_drawdown",
    "drawdown_details",
    "calmar_ratio",
    "win_rate",
    "annual_turnover",
    "beta",
    "alpha",
    "information_ratio",
    "value_at_risk",
    "expected_shortfall",
    "rolling_sharpe",
    "monthly_return_table",
    "summary_metrics",
    "compare_metrics",
    "sharpe_standard_error",
]

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _clean(returns: pd.Series) -> pd.Series:
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    return returns.replace([np.inf, -np.inf], np.nan).dropna()


def _excess(returns: pd.Series, risk_free_rate: float, periods_per_year: int) -> pd.Series:
    """Convert an annualised risk-free rate to per-period and subtract it."""
    return returns - risk_free_rate / periods_per_year


def _align_pair(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    joined = pd.concat([_clean(a), _clean(b)], axis=1, join="inner").dropna()
    if joined.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    return joined.iloc[:, 0], joined.iloc[:, 1]


# --------------------------------------------------------------------------- #
# Return and risk
# --------------------------------------------------------------------------- #
def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compound annual growth rate: ``(prod(1+r))^(P/N) - 1``.

    Geometric, not arithmetic: this is the rate that actually reproduces the
    terminal wealth, which the arithmetic mean does not.
    """
    r = _clean(returns)
    if len(r) == 0:
        return np.nan
    growth = float((1.0 + r).prod())
    if growth <= 0:
        return -1.0  # capital wiped out
    return growth ** (periods_per_year / len(r)) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised sample standard deviation (``ddof=1``)."""
    r = _clean(returns)
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def annualized_sharpe(
    returns: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """Annualised Sharpe ratio of excess returns.

    ``risk_free_rate`` is annualised and converted to per-period internally.
    Returns NaN rather than infinity when volatility is zero.
    """
    r = _clean(returns)
    if len(r) < 2:
        return np.nan
    excess = _excess(r, risk_free_rate, periods_per_year)
    vol = excess.std(ddof=1)
    if vol < _EPS:
        return np.nan
    return float(np.sqrt(periods_per_year) * excess.mean() / vol)


def sharpe_standard_error(
    sharpe: float,
    n_periods: int,
    periods_per_year: int = 252,
) -> float:
    """Approximate standard error of an annualised Sharpe ratio.

    Under IID returns, ``SE(S) ~ sqrt((1 + S^2/2) / T)`` where ``T`` is the
    sample length **in years**. This is the number that decides whether a
    difference between two strategies means anything: an 18-year daily sample
    gives ``SE ~ 0.25``, so two Sharpe ratios differing by 0.1 are not
    distinguishable, no matter how many decimal places the table shows.

    Real returns are autocorrelated and fat-tailed, which makes this an
    *optimistic* (too small) estimate of the true uncertainty.
    """
    if np.isnan(sharpe) or not n_periods:
        return np.nan
    years = n_periods / periods_per_year
    if years <= 0:
        return np.nan
    return float(np.sqrt((1 + 0.5 * sharpe**2) / years))


def sortino_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """Sharpe with only downside deviation in the denominator.

    Downside deviation uses the **full-sample** denominator -- the sum of squared
    negative excess returns is divided by the total number of observations, not
    by the count of negative ones. This is the standard construction; dividing by
    the negative count would flatter strategies that rarely lose.
    """
    r = _clean(returns)
    if len(r) < 2:
        return np.nan
    excess = _excess(r, risk_free_rate, periods_per_year)
    downside = np.minimum(excess, 0.0)
    dd = np.sqrt((downside**2).sum() / len(excess))
    if dd < _EPS:
        return np.nan
    return float(np.sqrt(periods_per_year) * excess.mean() / dd)


# --------------------------------------------------------------------------- #
# Drawdown
# --------------------------------------------------------------------------- #
def drawdown_series(equity: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak: ``V_t / cummax(V)_t - 1``."""
    equity = equity.dropna()
    if equity.empty:
        return equity
    return equity / equity.cummax() - 1.0


def maximum_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough decline, as a negative fraction."""
    dd = drawdown_series(equity)
    return float(dd.min()) if len(dd) else np.nan


def drawdown_details(equity: pd.Series) -> dict:
    """Depth, peak/trough/recovery dates, and duration of the worst drawdown."""
    dd = drawdown_series(equity)
    if dd.empty:
        return {}

    trough = dd.idxmin()
    peak = equity.loc[:trough].idxmax()
    after = dd.loc[trough:]
    recovered = after[after >= -_EPS]
    recovery = recovered.index[0] if len(recovered) else None

    return {
        "max_drawdown": float(dd.min()),
        "peak_date": peak,
        "trough_date": trough,
        "recovery_date": recovery,
        "drawdown_days": int((trough - peak).days),
        "recovery_days": int((recovery - trough).days) if recovery is not None else None,
        "underwater_days": int((recovery - peak).days) if recovery is not None else None,
        "still_underwater": recovery is None,
    }


def calmar_ratio(
    returns: pd.Series,
    equity: pd.Series | None = None,
    periods_per_year: int = 252,
) -> float:
    """CAGR divided by the absolute maximum drawdown."""
    if equity is None:
        equity = (1.0 + _clean(returns)).cumprod()
    mdd = maximum_drawdown(equity)
    if mdd is None or np.isnan(mdd) or abs(mdd) < _EPS:
        return np.nan
    return float(annualized_return(returns, periods_per_year) / abs(mdd))


# --------------------------------------------------------------------------- #
# Activity
# --------------------------------------------------------------------------- #
def win_rate(returns: pd.Series) -> float:
    """Fraction of periods with a strictly positive return."""
    r = _clean(returns)
    return float((r > 0).mean()) if len(r) else np.nan


def annual_turnover(turnover: pd.Series, periods_per_year: int = 252) -> float:
    """Average turnover per year.

    ``turnover`` is the two-way per-period series from :mod:`quant_platform.costs`
    (see that module for the convention). A value of 4.0 means the portfolio's
    gross weight changes by 400% of NAV per year.
    """
    t = _clean(turnover)
    if len(t) == 0:
        return np.nan
    return float(t.sum() / (len(t) / periods_per_year))


# --------------------------------------------------------------------------- #
# Benchmark-relative
# --------------------------------------------------------------------------- #
def beta(
    returns: pd.Series,
    benchmark: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """OLS beta of strategy excess returns on benchmark excess returns."""
    r, b = _align_pair(returns, benchmark)
    if len(r) < 2:
        return np.nan
    r_ex = _excess(r, risk_free_rate, periods_per_year)
    b_ex = _excess(b, risk_free_rate, periods_per_year)
    var_b = b_ex.var(ddof=1)
    if var_b < _EPS:
        return np.nan
    return float(r_ex.cov(b_ex) / var_b)


def alpha(
    returns: pd.Series,
    benchmark: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualised Jensen's alpha: the intercept of the excess-return regression."""
    r, b = _align_pair(returns, benchmark)
    if len(r) < 2:
        return np.nan
    beta_hat = beta(r, b, risk_free_rate, periods_per_year)
    if np.isnan(beta_hat):
        return np.nan
    r_ex = _excess(r, risk_free_rate, periods_per_year)
    b_ex = _excess(b, risk_free_rate, periods_per_year)
    return float((r_ex.mean() - beta_hat * b_ex.mean()) * periods_per_year)


def information_ratio(
    returns: pd.Series,
    benchmark: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Annualised active return divided by tracking error."""
    r, b = _align_pair(returns, benchmark)
    if len(r) < 2:
        return np.nan
    active = r - b
    te = active.std(ddof=1)
    if te < _EPS:
        return np.nan
    return float(np.sqrt(periods_per_year) * active.mean() / te)


# --------------------------------------------------------------------------- #
# Tail risk
# --------------------------------------------------------------------------- #
def value_at_risk(returns: pd.Series, level: float = 0.05) -> float:
    """Historical VaR: the ``level`` quantile of the period return distribution.

    Returned **as a return**, so a loss is negative: ``-0.021`` means "on the
    worst 5% of days the portfolio lost at least 2.1%". No distributional
    assumption is made -- this is the empirical quantile.
    """
    r = _clean(returns)
    if len(r) == 0:
        return np.nan
    return float(np.quantile(r, level))


def expected_shortfall(returns: pd.Series, level: float = 0.05) -> float:
    """Mean return conditional on being at or below the VaR quantile (CVaR).

    Also returned as a (negative) return. Always at least as bad as
    :func:`value_at_risk`, and unlike VaR it is sensitive to how fat the tail is.
    """
    r = _clean(returns)
    if len(r) == 0:
        return np.nan
    var = np.quantile(r, level)
    tail = r[r <= var]
    return float(tail.mean()) if len(tail) else float(var)


# --------------------------------------------------------------------------- #
# Rolling / tabular views
# --------------------------------------------------------------------------- #
def rolling_sharpe(
    returns: pd.Series,
    window: int = 252,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling annualised Sharpe over a trailing window (default 1 year)."""
    if min_periods is None:
        min_periods = max(2, window // 2)
    excess = _excess(returns, risk_free_rate, periods_per_year)
    mean = excess.rolling(window, min_periods=min_periods).mean()
    std = excess.rolling(window, min_periods=min_periods).std(ddof=1)
    return np.sqrt(periods_per_year) * mean / std.where(std > _EPS)


def monthly_return_table(returns: pd.Series) -> pd.DataFrame:
    """Compound daily returns into a year x month table (fractions, not %)."""
    r = _clean(returns)
    if r.empty:
        return pd.DataFrame()
    monthly = (1.0 + r).resample("ME").prod() - 1.0
    table = pd.DataFrame(
        {"year": monthly.index.year, "month": monthly.index.month, "ret": monthly.values}
    )
    pivot = table.pivot(index="year", columns="month", values="ret")
    pivot.columns = [
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][m - 1]
        for m in pivot.columns
    ]
    annual = (1.0 + r).resample("YE").prod() - 1.0
    pivot["Year"] = pd.Series(annual.values, index=annual.index.year)
    return pivot


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def summary_metrics(
    returns: pd.Series,
    equity: pd.Series | None = None,
    benchmark: pd.Series | None = None,
    turnover: pd.Series | None = None,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    var_level: float = 0.05,
    name: str = "strategy",
) -> pd.Series:
    """Assemble the full metric set into one labelled Series."""
    r = _clean(returns)
    if equity is None:
        equity = (1.0 + r).cumprod()

    out: dict[str, float | str] = {
        "start": str(r.index[0].date()) if len(r) else "",
        "end": str(r.index[-1].date()) if len(r) else "",
        "n_periods": len(r),
        "years": len(r) / periods_per_year if len(r) else np.nan,
        "total_return": float((1.0 + r).prod() - 1.0) if len(r) else np.nan,
        "cagr": annualized_return(r, periods_per_year),
        "ann_volatility": annualized_volatility(r, periods_per_year),
        "sharpe": annualized_sharpe(r, periods_per_year, risk_free_rate),
        "sortino": sortino_ratio(r, periods_per_year, risk_free_rate),
        "max_drawdown": maximum_drawdown(equity),
        "calmar": calmar_ratio(r, equity, periods_per_year),
        "win_rate": win_rate(r),
        "skew": float(r.skew()) if len(r) > 2 else np.nan,
        "kurtosis": float(r.kurtosis()) if len(r) > 3 else np.nan,
        "best_day": float(r.max()) if len(r) else np.nan,
        "worst_day": float(r.min()) if len(r) else np.nan,
        f"var_{int((1 - var_level) * 100)}": value_at_risk(r, var_level),
        f"cvar_{int((1 - var_level) * 100)}": expected_shortfall(r, var_level),
    }

    details = drawdown_details(equity)
    out["underwater_days"] = details.get("underwater_days", np.nan)
    out["still_underwater"] = details.get("still_underwater", np.nan)

    if turnover is not None:
        out["annual_turnover"] = annual_turnover(turnover, periods_per_year)

    if benchmark is not None:
        out["beta"] = beta(r, benchmark, risk_free_rate, periods_per_year)
        out["alpha"] = alpha(r, benchmark, risk_free_rate, periods_per_year)
        out["information_ratio"] = information_ratio(r, benchmark, periods_per_year)

    return pd.Series(out, name=name)


def compare_metrics(results: dict[str, pd.Series]) -> pd.DataFrame:
    """Stack several :func:`summary_metrics` outputs into one comparison table."""
    return pd.DataFrame(results)
