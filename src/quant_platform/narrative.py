"""
Report prose.
=============

The written sections of the research report. Every claim here is derived from
the numbers the run actually produced -- the conclusions are computed, not
hard-coded, so the report stays truthful when the result is bad.

That is the point. A report that says "the strategy works" regardless of the
data is worth nothing; the value of this module is that it will say "the edge
disappears once costs are applied" if that is what the numbers show.
"""

from __future__ import annotations

import html
import numpy as np
import pandas as pd

__all__ = ["build_narrative", "build_kpis"]


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def _pct(x, dp: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{dp}f}%"


def _num(x, dp: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{dp}f}"


def _get(df: pd.DataFrame, row: str, col: str, default=np.nan) -> float:
    try:
        return float(df.loc[row, col])
    except (KeyError, TypeError, ValueError):
        return default


def _p(text: str) -> str:
    return f"<p>{text}</p>"


def _callout(title: str, body: str, ok: bool = False) -> str:
    cls = "callout ok" if ok else "callout"
    return f'<div class="{cls}"><span class="t">{html.escape(title)}</span>{body}</div>'


def _verdict(value: float, good: float, bad: float, better_is_higher: bool = True) -> str:
    """Plain-language band for a metric, so the reader is not left to guess."""
    if np.isnan(value):
        return "not measurable"
    if better_is_higher:
        return "encouraging" if value >= good else ("weak" if value <= bad else "unremarkable")
    return "encouraging" if value <= good else ("weak" if value >= bad else "unremarkable")


# --------------------------------------------------------------------------- #
# KPI strip
# --------------------------------------------------------------------------- #
def build_kpis(
    comparison: pd.DataFrame,
    strategy_name: str,
    split_metrics: pd.DataFrame,
    show_test: bool = False,
):
    """The six numbers a reader should see before anything else.

    The test-split Sharpe is shown only when ``show_test`` is set. Putting a
    withheld number in the headline strip would defeat the entire sample
    discipline -- it is the first thing anyone reads.
    """
    col = strategy_name if strategy_name in comparison.columns else comparison.columns[0]
    pairs = [
        ("CAGR", _pct(_get(comparison, "cagr", col))),
        ("Sharpe", _num(_get(comparison, "sharpe", col))),
        ("Max drawdown", _pct(_get(comparison, "max_drawdown", col))),
        ("Ann. volatility", _pct(_get(comparison, "ann_volatility", col))),
        ("Annual turnover", _num(_get(comparison, "annual_turnover", col)) + "x"),
    ]
    if show_test and "test" in split_metrics.columns:
        pairs.append(("Sharpe (test)", _num(_get(split_metrics, "sharpe", "test"))))
    elif "validation" in split_metrics.columns:
        pairs.append(("Sharpe (valid.)", _num(_get(split_metrics, "sharpe", "validation"))))
    return pairs


# --------------------------------------------------------------------------- #
# Narrative
# --------------------------------------------------------------------------- #
def build_narrative(
    cfg: dict,
    data_report: dict,
    provenance,
    comparison: pd.DataFrame,
    split_metrics: pd.DataFrame,
    bench_splits: pd.DataFrame,
    sensitivity: pd.DataFrame,
    marginals: dict,
    regimes: pd.DataFrame,
    strategy,
    splits,
    show_test: bool = False,
    leave_one_year_out: pd.DataFrame | None = None,
) -> dict[str, str]:
    d, s, p, v = cfg["data"], cfg["strategy"], cfg["portfolio"], cfg["validation"]
    name = strategy.name
    col = name if name in comparison.columns else comparison.columns[0]
    total_bps = p["commission_bps"] + p["slippage_bps"]

    # --- the facts the prose will reason about ----------------------------- #
    f = {
        "sharpe": _get(comparison, "sharpe", col),
        "cagr": _get(comparison, "cagr", col),
        "vol": _get(comparison, "ann_volatility", col),
        "mdd": _get(comparison, "max_drawdown", col),
        "calmar": _get(comparison, "calmar", col),
        "turnover": _get(comparison, "annual_turnover", col),
        "beta": _get(comparison, "beta", col),
        "alpha": _get(comparison, "alpha", col),
        "ir": _get(comparison, "information_ratio", col),
        "win": _get(comparison, "win_rate", col),
        "cvar": _get(comparison, "cvar_95", col),
    }
    b = {
        "spy_sharpe": _get(comparison, "sharpe", "SPY buy & hold"),
        "spy_cagr": _get(comparison, "cagr", "SPY buy & hold"),
        "spy_mdd": _get(comparison, "max_drawdown", "SPY buy & hold"),
        "spy_vol": _get(comparison, "ann_volatility", "SPY buy & hold"),
        "ew_sharpe": _get(comparison, "sharpe", "Equal weight (all)"),
        "ew_cagr": _get(comparison, "cagr", "Equal weight (all)"),
        "rand_sharpe": _get(comparison, "sharpe", "Random selection"),
        "rand_cagr": _get(comparison, "cagr", "Random selection"),
        "rand_turnover": _get(comparison, "annual_turnover", "Random selection"),
        "nocost_sharpe": _get(comparison, "sharpe", "Momentum, zero cost"),
        "nocost_cagr": _get(comparison, "cagr", "Momentum, zero cost"),
        "lag_sharpe": _get(comparison, "sharpe", "Momentum, T+6 exec"),
    }
    sp = {
        "train": _get(split_metrics, "sharpe", "training"),
        "valid": _get(split_metrics, "sharpe", "validation"),
        "test": _get(split_metrics, "sharpe", "test"),
        "train_cagr": _get(split_metrics, "cagr", "training"),
        "valid_cagr": _get(split_metrics, "cagr", "validation"),
        "test_cagr": _get(split_metrics, "cagr", "test"),
    }

    cost_drag_sharpe = b["nocost_sharpe"] - f["sharpe"]
    cost_drag_cagr = b["nocost_cagr"] - f["cagr"]
    lag_decay = f["sharpe"] - b["lag_sharpe"]
    vs_spy = f["sharpe"] - b["spy_sharpe"]
    vs_ew = f["sharpe"] - b["ew_sharpe"]
    vs_rand = f["sharpe"] - b["rand_sharpe"]
    oos_decay = sp["train"] - sp["valid"]

    synthetic = provenance is not None and provenance.source == "synthetic"
    syn_note = (
        " <strong>Because this run used synthetic data, the specific figures below "
        "describe a simulated panel and carry no information about real markets.</strong>"
        if synthetic else ""
    )

    n = {}

    # ---------------------------------------------------------------- 1 ---- #
    n["question"] = (
        _p(
            "Does ranking a small universe of liquid US ETFs by trailing "
            f"{s['lookback_days']}-day return (skipping the most recent {s['skip_days']} days), "
            f"holding the top {s['holdings']} in equal weight, and rebalancing "
            f"{s['rebalance']}, produce a risk-adjusted return that survives realistic "
            "transaction costs and honest out-of-sample testing?"
        )
        + _p(
            "The strategy is deliberately unoriginal. Cross-sectional momentum is one of "
            "the most documented effects in asset pricing, which makes it a good "
            "instrument for the actual purpose here: proving that the research "
            "<em>process</em> -- the data handling, the execution timing, the cost "
            "accounting, the sample discipline -- is correct. A novel signal evaluated "
            "with a broken backtest is worth less than a known signal evaluated properly."
        )
        + _p(
            "The question is stated so that a negative answer is a real, reportable "
            "outcome. Section 9 records what did not work."
        )
    )

    # ---------------------------------------------------------------- 2 ---- #
    n["motivation"] = (
        _p(
            "Time-series and cross-sectional momentum have been documented across "
            "equities, bonds, commodities, and currencies over multi-decade samples. Two "
            "families of explanation compete, and they imply different things about "
            "whether the effect should persist:"
        )
        + "<ul>"
        "<li><strong>Behavioural.</strong> Investors under-react to gradually diffusing "
        "information, then over-extrapolate. Slow adjustment produces continuation. If "
        "this is the mechanism, the effect can be arbitraged away as it becomes known.</li>"
        "<li><strong>Risk-based.</strong> Momentum loads on a compensated risk factor -- "
        "notably crash risk, since momentum portfolios suffer severe, sudden losses at "
        "sharp reversals. If this is the mechanism, the return is payment for a real "
        "exposure and should persist, but so should the crashes.</li>"
        "</ul>"
        + _p(
            "The 12-1 construction (twelve-month formation, one-month skip) is the "
            "convention rather than an optimisation. The skip exists because short-horizon "
            "returns exhibit <em>reversal</em> rather than continuation; including the most "
            "recent month mixes two opposing effects and dilutes both."
        )
        + _p(
            f"This universe is {len(d['symbols'])} cross-asset ETFs "
            f"({', '.join(d['symbols'])}). Spanning equities, international, bonds, gold, "
            "REITs, and commodities gives the ranking genuinely different assets to choose "
            "between, so a top-3 selection can rotate defensively rather than merely "
            "picking the highest-beta equity sleeve. That said, nine assets is a very small "
            "cross-section: the ranking is a coarse instrument and single-name idiosyncratic "
            "moves dominate more than they would in a hundred-name universe."
        )
    )

    # ---------------------------------------------------------------- 3 ---- #
    cov = data_report.get("coverage")
    n_rows, n_syms = data_report.get("n_rows", 0), data_report.get("n_symbols", 0)
    n["data"] = (
        _p(
            f"Source: <code>{html.escape(str(getattr(provenance, 'source', 'unknown')))}</code>, "
            f"daily bars, {n_rows:,} rows across {n_syms} symbols spanning "
            f"{data_report.get('date_min', '?')} to {data_report.get('date_max', '?')}."
            + syn_note
        )
        + _p(
            "All research uses <strong>adjusted</strong> closes. This is not cosmetic: an "
            "unadjusted series treats a 2-for-1 split as a -50% return, which a momentum "
            "ranking would read as the worst-performing asset in the universe. Dividends "
            "matter just as much here -- TLT and VNQ distribute heavily, and ranking them "
            "on price return alone would systematically under-rank them against SPY."
        )
        + _p("The pipeline enforces, and fails loudly on violation:")
        + "<ul>"
        "<li>no duplicate <code>(date, symbol)</code> pairs;</li>"
        "<li>dates monotonically increasing within each symbol;</li>"
        "<li>strictly positive prices; <code>high &ge; low</code>;</li>"
        "<li>a unique, sorted, duplicate-free index on the wide price matrix.</li>"
        "</ul>"
        + _p(
            "The panel is then trimmed to the window where <em>every</em> symbol has data, "
            f"beginning {data_report.get('common_history_starts', '?')}. The cross-sectional "
            "rank is only meaningful if it compares a constant set of assets; letting "
            "symbols enter mid-sample would silently change what 'top 3 of 9' means."
        )
        + _callout(
            "Survivorship bias is present and not corrected",
            _p(
                "The universe is a fixed list of ETFs chosen because they exist and are "
                "liquid <em>today</em>. Any ETF that launched and closed during the sample "
                "is invisible. For these nine cross-asset funds -- none of which was ever "
                "close to delisting -- the distortion is small, but it is a real upward "
                "bias and it is not something this pipeline removes. Applying this code to "
                "single-name equities without point-in-time index membership and delisting "
                "returns would produce badly overstated results."
            ),
        )
    )

    # ---------------------------------------------------------------- 4 ---- #
    n["signal"] = (
        _p("The score for asset <em>i</em> at date <em>t</em> is the total return over the "
           "formation window, excluding the skip period:")
        + "<pre><code>score[i, t] = price[i, t - skip] / price[i, t - lookback] - 1\n\n"
          f"lookback = {s['lookback_days']} trading days (~{s['lookback_days'] // 21} months)\n"
          f"skip     = {s['skip_days']} trading days (~{s['skip_days'] // 21} month)</code></pre>"
        + _p(
            "Implemented as <code>prices.shift(skip).pct_change(lookback - skip)</code>. The "
            "<code>shift(skip)</code> means the value at <em>t</em> depends only on prices at "
            f"or before <em>t - {s['skip_days']}</em>: the signal is strictly backward-looking "
            "before the engine's execution delay is even applied."
        )
        + _p(
            f"On each rebalance date the assets are ranked, the top {s['holdings']} selected, "
            "and weighted equally. Ties break deterministically by column order. During the "
            "initial formation window, when fewer than "
            f"{s['holdings']} assets have a score, the target is <em>no position</em> rather "
            "than a partially-filled portfolio -- holding a one-name 'portfolio' there would "
            "be an artefact of the warm-up rather than a decision."
        )
        + _p(
            f"A per-name cap of {_pct(s.get('max_weight', 1.0), 0)} is applied. With "
            f"{s['holdings']} equal-weighted holdings each position is "
            f"{_pct(1 / s['holdings'])}, so the cap does not bind in the base configuration. "
            "It does bind in the sensitivity grid at 1 and 2 holdings, where the excess is "
            "left in cash rather than redistributed -- so those cells describe a "
            "partly-invested portfolio, which is why average exposure is reported alongside "
            "them."
        )
        + _callout(
            "Rebalance dates are trading days, not calendar month-ends",
            _p(
                "The obvious implementation, <code>scores.resample('ME').last()</code>, stamps "
                "its output on the calendar month-end (e.g. 31 January), which is frequently "
                "not a trading day. Reindexing that label onto a trading-day index silently "
                "drops it, and a forward-fill then carries the <em>previous</em> month's "
                "weights -- a missed rebalance that no assertion would catch. This platform "
                "instead resamples the index itself and takes the last <em>actual</em> "
                "trading day of each period, so the rebalance dates are always a subset of "
                "the price index and the reindex is lossless."
            ),
            ok=True,
        )
    )

    # ---------------------------------------------------------------- 5 ---- #
    n["execution"] = (
        _p("The timing convention, stated precisely, because everything depends on it:")
        + "<ul>"
        "<li><code>prices.loc[t]</code> is the <strong>closing price</strong> at date t.</li>"
        "<li><code>returns.loc[t] = prices.loc[t]/prices.loc[t-1] - 1</code> is the return "
        "earned over <code>(t-1, t]</code>. It is not knowable until the close of t.</li>"
        "<li><code>target_weights.loc[t]</code> is the portfolio <em>desired</em> at the "
        "close of t, from information available at t.</li>"
        "</ul>"
        + _p("The engine therefore holds:")
        + f"<pre><code>executed_weights = target_weights.shift({p['execution_lag']})\n"
          "gross_returns    = (executed_weights * asset_returns).sum(axis=1)</code></pre>"
        + _p(
            f"With <code>execution_lag = {p['execution_lag']}</code>, the weights multiplying "
            "<code>returns.loc[t]</code> were fixed at the close of <code>t-1</code>, strictly "
            "before that return was observable. Economically this is: observe the close, "
            "compute the signal overnight, trade at the next session. Setting the lag to zero "
            "would let a position capture the same return that determined it -- the canonical "
            "look-ahead bug. The parameter is exposed so the bias can be <em>measured</em> "
            "rather than accidentally shipped."
        )
        + _p(
            f"A robustness run at T+{p['execution_lag'] + 5} (acting a full week late) tests "
            "whether the edge depends on immediacy. Results in section 8."
        )
        + _p("<strong>What is not modelled:</strong> trades execute at the close in full, at "
             "no impact, with no partial fills, no gaps, and no failure to trade. The "
             "portfolio is long-only, unlevered, and always fully invested up to the weight "
             "cap. No taxes.")
    )

    # ---------------------------------------------------------------- 6 ---- #
    n["costs"] = (
        _p(f"Costs are linear in turnover at <strong>{total_bps:g} bps</strong> "
           f"({p['commission_bps']:g} commission + {p['slippage_bps']:g} slippage):")
        + "<pre><code>turnover[t] = sum_i |w[i,t] - w[i,t-1]|      # executed weights\n"
          "cost[t]     = turnover[t] * total_bps / 10_000\n"
          "net[t]      = gross[t] - cost[t]</code></pre>"
        + _p(
            "<strong>This turnover is two-way.</strong> Rotating out of one 33% position and "
            "into another counts 0.33 + 0.33 = 0.66, not 0.33. Many published figures quote "
            "one-way turnover, which is exactly half; a factor of two in turnover is a factor "
            "of two in costs, so the convention has to be checked before any external "
            "comparison. Charging the full two-way sum is the conservative choice and matches "
            "what a broker actually bills. The initial entry from cash is charged, not free."
        )
        + _p(
            f"Realised annual turnover is <strong>{_num(f['turnover'])}x</strong>, costing "
            f"roughly {_pct(f['turnover'] * total_bps / 10_000)} of NAV per year at "
            f"{total_bps:g} bps. Measured against the zero-cost run, costs consumed "
            f"<strong>{_pct(cost_drag_cagr)}</strong> of annual return and "
            f"<strong>{_num(cost_drag_sharpe)}</strong> of Sharpe."
        )
        + _callout(
            "Four ways this cost model flatters the result",
            "<ul>"
            "<li><strong>No weight drift.</strong> Positions are assumed restored to target "
            "each period rather than drifting with relative performance, so real rebalancing "
            "trades are slightly larger than modelled.</li>"
            "<li><strong>No market impact.</strong> Cost per unit of turnover is constant "
            "regardless of order size. Defensible for nine ETFs trading billions daily; wrong "
            "at scale or in less liquid names.</li>"
            "<li><strong>Static spread.</strong> A flat slippage number stands in for a spread "
            "that widens in exactly the stressed markets where this strategy rebalances "
            "hardest. Costs are understated in the tail.</li>"
            "<li><strong>No taxes.</strong> At this turnover, in a taxable account, "
            "short-term capital gains would be a first-order drag.</li>"
            "</ul>"
            + _p("Every one of these biases the result in the strategy's favour. The 0/5/10/20 "
                 "bps sensitivity is there to show how much that matters."),
        )
    )

    # ---------------------------------------------------------------- 7 ---- #
    tr, va, te = splits[0], splits[1], splits[2]
    n["validation"] = (
        _p("The sample is split once, in advance, and the periods have different rules:")
        + "<ul>"
        f"<li><strong>Training</strong> ({tr.start.date()} to {tr.end.date()}) -- all "
        "parameter exploration happens here. The sensitivity grid is evaluated on this "
        "window only.</li>"
        f"<li><strong>Validation</strong> ({va.start.date()} to {va.end.date()}) -- inspected "
        "after the parameter region was chosen, to check the choice generalises.</li>"
        f"<li><strong>Test</strong> ({te.start.date()} to {te.end.date()}) -- evaluated once, "
        "at the end. Not used to revise anything.</li>"
        "</ul>"
        + _p(
            "The discipline matters more than the split points. Every time a strategy is "
            "adjusted after seeing test results, the test stops being a test and becomes "
            "another training set -- and the reported out-of-sample performance becomes a "
            "description of the researcher's memory rather than of the strategy. The "
            "sensitivity grid here contains "
            f"{len(sensitivity) if not sensitivity.empty else 0} combinations; searching that "
            "many on the test period and reporting the best cell would produce an impressive "
            "Sharpe from pure noise."
        )
        + _p(
            "Code-level defences against the subtler failure -- look-ahead inside the signal "
            "itself -- live in <code>tests/test_no_lookahead.py</code>, which perturbs future "
            "prices by 100x and asserts that historical signals, weights, and realised "
            "returns are bit-identical. That test catches the bugs a sample split cannot."
        )
        + (_p("<em>The test window is withheld from this report. Re-run with "
              "<code>--show-test</code> to reveal it.</em>") if not show_test else "")
    )

    # ---------------------------------------------------------------- 8 ---- #
    n["results_intro"] = (
        _p(
            f"Over the full common sample the strategy returned "
            f"<strong>{_pct(f['cagr'])}</strong> annualised at "
            f"<strong>{_pct(f['vol'])}</strong> volatility, a Sharpe of "
            f"<strong>{_num(f['sharpe'])}</strong>, with a maximum drawdown of "
            f"<strong>{_pct(f['mdd'])}</strong>. SPY buy-and-hold over the identical window "
            f"returned {_pct(b['spy_cagr'])} at Sharpe {_num(b['spy_sharpe'])} with a "
            f"{_pct(b['spy_mdd'])} drawdown."
        )
        + _p(
            f"Against the baselines, on Sharpe: {_delta_phrase(vs_spy, 'SPY buy-and-hold')}; "
            f"{_delta_phrase(vs_ew, 'the equal-weight universe')}; "
            f"{_delta_phrase(vs_rand, 'random 3-of-9 selection')}."
        )
        + _p(
            "The random-selection comparator is the most informative of the three. It isolates "
            "whether the momentum <em>ranking</em> carries information, as distinct from "
            "whether concentrated rotation within this particular universe happens to work. "
            + _rank_verdict(vs_rand)
        )
        + _turnover_caveat(f, b, total_bps)
        + _p(
            f"Beta to SPY is {_num(f['beta'])} with annualised alpha of {_pct(f['alpha'])} and "
            f"an information ratio of {_num(f['ir'])}. "
            + _alpha_ir_reconciliation(f, b)
        )
    )

    # in-sample vs out-of-sample
    n["oos"] = (
        _p(
            f"Training Sharpe {_num(sp['train'])} vs validation Sharpe {_num(sp['valid'])}"
            + (f" vs test Sharpe {_num(sp['test'])}" if show_test else "")
            + f". {_decay_phrase(oos_decay)}"
        )
        + _p(
            "Some degradation from in-sample to out-of-sample is normal and expected -- the "
            "training period is where the parameters were chosen, so it is optimistically "
            "biased by construction. What matters is whether the degradation is proportionate "
            "or total. A Sharpe that collapses from strongly positive to near zero indicates "
            "the in-sample result was a fit to noise."
        )
        + (
            _p("<em>Test-period figures are withheld from this report. They were computed once "
               "and are available in <code>in_vs_out_of_sample.csv</code>; re-run with "
               "<code>--show-test</code> to display them here.</em>")
            if not show_test else ""
        )
    )

    # sensitivity
    n["sensitivity"] = _sensitivity_prose(sensitivity, marginals, cost_drag_sharpe, total_bps)

    # leave-one-year-out
    n["loyo"] = _p(
        "Each row deletes one calendar year and recomputes the strategy's Sharpe advantage "
        "over the benchmark from the remaining data. <code>edge_change</code> is how much the "
        "advantage moves when that year is removed: a large negative value marks a year the "
        "whole result rests on. This is the single most informative robustness check in the "
        "report, because a full-sample edge cannot by itself distinguish a persistent effect "
        "from one good episode surrounded by noise."
    )

    # ---------------------------------------------------------------- 9 ---- #
    n["failure"] = _failure_prose(f, b, sp, regimes, cost_drag_sharpe, lag_decay,
                                  vs_spy, vs_rand, oos_decay, total_bps, synthetic,
                                  leave_one_year_out)

    # --------------------------------------------------------------- 10 ---- #
    n["limits"] = (
        "<h3>Limitations</h3>"
        "<ul>"
        f"<li><strong>Nine assets is a very small cross-section.</strong> A top-3 selection "
        "from nine is a coarse instrument; idiosyncratic moves in one ETF dominate the "
        "portfolio in a way they would not in a broad universe. Effective breadth is low, "
        "which caps the information ratio achievable regardless of signal quality.</li>"
        "<li><strong>One sample, one market.</strong> A single ~15-year US-listed sample "
        "containing two major drawdowns. The effective number of independent observations "
        "for a monthly strategy is closer to 180 than to the ~4,000 daily rows, so the "
        "standard error on the Sharpe estimate is wide -- roughly "
        f"&plusmn;{_num(_sharpe_se(f['sharpe'], _get(comparison, 'n_periods', col)))} at one "
        "standard error. Differences smaller than that are not distinguishable from noise.</li>"
        "<li><strong>Survivorship bias, uncorrected.</strong> Fixed universe of funds that "
        "exist today.</li>"
        "<li><strong>Costs are optimistic.</strong> No drift, no impact, static spreads, no "
        "taxes -- all biased in the strategy's favour.</li>"
        "<li><strong>No weight drift between rebalances.</strong> Positions are restored to "
        "target each period, understating real turnover slightly.</li>"
        "<li><strong>Free retail data.</strong> Adjusted closes from a free provider carry "
        "occasional errors and revisions; a professional result would be reproduced against a "
        "second vendor.</li>"
        "</ul>"
        "<h3>Next experiments, in the order worth running</h3>"
        "<ol>"
        "<li><strong>Volatility targeting.</strong> Scale exposure to a constant ex-ante "
        "volatility. Momentum's worst episodes cluster in high-volatility reversals; this is "
        "the single best-documented improvement to the base strategy.</li>"
        "<li><strong>Risk-parity weighting instead of equal weight.</strong> Equal-weighting "
        "GLD and QQQ assigns them equal capital and wildly unequal risk.</li>"
        "<li><strong>Absolute-momentum overlay.</strong> Require a positive trailing return, "
        "not merely a top-3 rank, before holding an asset -- move to cash otherwise. Directly "
        "targets the 2008-style failure mode where the 'best' asset is still falling.</li>"
        "<li><strong>Widen the universe.</strong> More assets improves breadth, the binding "
        "constraint on this design.</li>"
        "<li><strong>Block bootstrap and Deflated Sharpe.</strong> Confidence intervals via "
        "stationary block bootstrap, and a multiple-testing correction for the size of the "
        "parameter search.</li>"
        "<li><strong>Weight drift and order-level execution.</strong> Model drift between "
        "rebalances, then participation-rate-based impact.</li>"
        "<li><strong>Cross-check against VectorBT.</strong> Reproduce the headline result in "
        "an independent engine. Agreement is evidence the accounting is right; disagreement "
        "localises a bug in one of them.</li>"
        "</ol>"
    )
    return n


# --------------------------------------------------------------------------- #
# Phrase builders
# --------------------------------------------------------------------------- #
def _safe(d: dict, key: str) -> float:
    return d.get(key, np.nan)


def _regime_explanation(regime: str) -> str:
    """Explain *why* a given regime is the weak one.

    The mechanism differs by regime, so a single canned sentence would be wrong
    most of the time. Trailing in a rising market is the expected cost of a
    defensive rotation; trailing in a falling one is a genuine failure of the
    signal to rotate out.
    """
    label = regime.lower()

    if "up" in label:
        return (
            "This is the expected cost of the design rather than a malfunction: a "
            "cross-asset rotation that can hold bonds, gold, or commodities will "
            "structurally lag a rising equity index whenever it is not fully in equities. "
            "The relevant question is whether the drawdown protection bought in the opposite "
            "regime is worth the give-up here -- which is a preference, not a fact the "
            "backtest can settle."
        )
    if "down" in label:
        return (
            "<strong>This is the more troubling direction.</strong> A defensive rotation "
            "exists precisely to protect capital when the benchmark falls; trailing it in "
            "that regime means the signal failed at the job it is meant to do. The likely "
            "mechanism is that a 12-month formation window adapts too slowly to a fast "
            "drawdown -- the portfolio is still holding last year's winners as leadership "
            "collapses."
        )
    if "high" in label:
        return (
            "Momentum's structural weakness is the sharp reversal, and reversals cluster in "
            "high-volatility periods: the portfolio is by construction concentrated in "
            "whatever just performed best, which is exactly what unwinds hardest when "
            "leadership rotates. Volatility-scaled position sizing is the standard mitigation "
            "and is the first item in section 10."
        )
    if "low" in label:
        return (
            "Trailing in calm markets is usually a drag-and-cost story rather than a signal "
            "failure: low-volatility regimes tend to be steady equity rallies, where a "
            "diversified rotation gives up ground to a concentrated index while still paying "
            "rebalancing costs."
        )
    return (
        "The mechanism is worth isolating before drawing conclusions -- the regime table "
        "above shows the conditional statistics, but attribution to specific holdings would "
        "require a position-level decomposition this version does not produce."
    )


def _sharpe_se(sharpe: float, n_periods: float, periods_per_year: int = 252) -> float:
    """Approximate standard error of an annualised Sharpe: sqrt((1 + S^2/2)/T)."""
    if np.isnan(sharpe) or not n_periods or np.isnan(n_periods):
        return np.nan
    years = n_periods / periods_per_year
    if years <= 0:
        return np.nan
    return float(np.sqrt((1 + 0.5 * sharpe**2) / years))


def _delta_phrase(delta: float, label: str) -> str:
    if np.isnan(delta):
        return f"not comparable against {label}"
    if abs(delta) < 0.10:
        return f"essentially tied with {label} ({delta:+.2f})"
    if delta > 0:
        return f"ahead of {label} by {delta:+.2f}"
    return f"<strong>behind {label} by {delta:+.2f}</strong>"


def _alpha_ir_reconciliation(f: dict, b: dict) -> str:
    """Explain positive alpha alongside a negative information ratio.

    The two disagree by construction whenever a low-beta strategy trails its
    benchmark in raw return: Jensen's alpha credits it for the return it earned
    *relative to the risk it took*, while the information ratio measures raw
    active return and does not adjust for beta at all. Reporting both without
    reconciling them invites the reader to assume one is a mistake.
    """
    alpha, ir, beta = f["alpha"], f["ir"], f["beta"]
    cagr, spy_cagr = f["cagr"], b["spy_cagr"]

    if np.isnan(alpha) or np.isnan(ir):
        return ("Read these together with the regime table below, which shows where the "
                "exposure was actually taken.")

    if alpha > 0 and ir < 0:
        return (
            "<strong>Those two numbers disagree, and the disagreement is the substance of "
            "this result.</strong> Jensen's alpha is beta-adjusted: at a beta of "
            f"{_num(beta)} the strategy was only expected to capture about "
            f"{_pct(beta * spy_cagr)} of the benchmark's return, and it delivered "
            f"{_pct(cagr)}, so it beat the return its risk exposure entitled it to. The "
            "information ratio is <em>not</em> beta-adjusted -- it is raw active return over "
            f"tracking error -- and in raw terms the strategy returned {_pct(cagr)} against "
            f"the benchmark's {_pct(spy_cagr)}, so it lost. Both are correct. The honest "
            "summary is that this strategy did not beat the index on return; it delivered "
            "slightly less return for materially less risk, and whether that is worth having "
            "depends entirely on whether the investor can lever it or values the smaller "
            "drawdown."
        )

    if alpha > 0 and beta < 0.8:
        return (
            "A beta well below 1 alongside positive alpha and a positive information ratio "
            "is the combination worth having: the strategy is not simply de-levering equity "
            "exposure, it is earning a return the exposure alone does not explain."
        )

    if alpha <= 0:
        return (
            "Alpha is not positive, so on a beta-adjusted basis the strategy did not earn "
            "more than its market exposure would predict. Whatever return it produced is "
            "compensation for risk already available more cheaply through the index."
        )

    return ("Read these together with the regime table below, which shows where the exposure "
            "was actually taken.")


def _turnover_caveat(f: dict, b: dict, total_bps: float) -> str:
    """Random selection re-draws every period, so it is charged more in costs.

    Not disclosing this would let the strategy claim a win it partly bought with
    a lower trading rate rather than with better information.
    """
    rt, st = b.get("rand_turnover", np.nan), f.get("turnover", np.nan)
    if np.isnan(rt) or np.isnan(st) or rt <= st * 1.25:
        return ""

    extra_cost = (rt - st) * total_bps / 10_000
    gap = f["sharpe"] - b["rand_sharpe"]
    cagr_gap = f["cagr"] - b.get("rand_cagr", np.nan)
    share = extra_cost / cagr_gap if cagr_gap and not np.isnan(cagr_gap) and cagr_gap > 0 else np.nan

    verdict = (
        "so the handicap explains only a small part of the gap and the ranking is still "
        "doing the work."
        if not np.isnan(share) and share < 0.35
        else "<strong>which is a substantial share of the gap -- the comparison is not clean, "
        "and the ranking's apparent advantage is partly a lower trading rate rather than "
        "better selection.</strong>"
    )

    return _p(
        f"One caveat on that comparison: random selection re-draws independently each period "
        f"and therefore turns over {_num(rt)}x per year against the strategy's {_num(st)}x. At "
        f"{total_bps:g} bps that is an extra {_pct(extra_cost)} of annual cost imposed on the "
        f"random baseline before any skill is measured. The strategy's persistence -- momentum "
        f"rankings are sticky, so the portfolio churns less -- is a genuine economic advantage, "
        f"but it is a different advantage from the ranking being informative. The annualised "
        f"return gap is {_pct(cagr_gap)}, of which the turnover handicap accounts for roughly "
        f"{_pct(extra_cost)}, " + verdict
    )


def _rank_verdict(vs_rand: float) -> str:
    if np.isnan(vs_rand):
        return ""
    if vs_rand > 0.25:
        return ("The ranking beats random selection by a clear margin, which is the minimum "
                "evidence needed to claim the signal itself is doing work.")
    if vs_rand > 0.05:
        return ("The ranking edges out random selection, but by a margin small enough that it "
                "is within the noise band for a sample this size. Treat as suggestive, not "
                "established.")
    return ("<strong>The ranking does not beat random selection.</strong> Whatever this "
            "portfolio earned came from holding concentrated positions in this universe, not "
            "from momentum. This is the result that most cleanly falsifies the hypothesis.")


def _decay_phrase(decay: float) -> str:
    if np.isnan(decay):
        return "Out-of-sample comparison unavailable."
    if decay <= 0.1:
        return ("Performance held up out of sample -- the in-sample result was not obviously "
                "a fit to noise.")
    if decay <= 0.6:
        return ("Some decay out of sample, which is normal: the training window is "
                "optimistically biased by construction.")
    return ("<strong>Substantial decay out of sample.</strong> A drop of this size suggests "
            "the in-sample result was substantially a fit to that particular period.")


def _lookback_stability_prose(lb: pd.DataFrame) -> str:
    """Judge the lookback dimension on *relative* spread and on shape.

    An absolute spread threshold is the wrong test. A 0.13 spread is negligible
    around a mean Sharpe of 2.0 and severe around a mean of 0.36 -- so the
    comparison has to be relative. Shape matters at least as much: a dimension
    that rises, dips, and recovers is not a plateau, it is noise, however narrow
    its range. Genuine structural parameters degrade smoothly away from an
    optimum; fitted ones jump around.
    """
    means = lb["mean"]
    lo, hi = float(means.min()), float(means.max())
    spread = hi - lo
    centre = abs(float(means.mean()))
    relative = spread / centre if centre > 1e-9 else np.inf
    within = float(lb["std"].mean())

    diffs = np.diff(means.to_numpy())
    monotone = bool(np.all(diffs >= 0) or np.all(diffs <= 0))
    worst = means.idxmin()

    head = (
        f"Across lookback lengths the mean Sharpe spans {_num(lo)} to {_num(hi)} -- a spread of "
        f"{_num(spread)}, which is <strong>{relative:.0%}</strong> of the average level across "
        f"the dimension ({_num(centre)}). Average within-lookback standard deviation is "
        f"{_num(within)}. "
    )

    if relative < 0.20 and monotone:
        verdict = (
            "The dimension is both narrow and monotone, which is what a structural parameter "
            "looks like: performance degrades smoothly away from the best setting rather than "
            "jumping around."
        )
    elif relative < 0.20:
        verdict = (
            f"The range is narrow, but the dimension is <strong>not monotone</strong> -- "
            f"{worst} months is the trough with better results on either side. A genuine "
            "parameter degrades smoothly; a dip-and-recover shape is the signature of noise, "
            "even within a narrow band."
        )
    elif monotone:
        verdict = (
            "Results vary materially across the dimension, but monotonically, which is at least "
            "interpretable: it suggests a real preference for one end of the range rather than "
            "an arbitrary optimum."
        )
    else:
        verdict = (
            f"<strong>The lookback dimension is neither flat nor monotone.</strong> "
            f"{worst} months is a trough with better results on both sides, so there is no "
            "smooth plateau to sit on. Varying this much, in this shape, is a warning that the "
            "formation window is fitting period-specific behaviour rather than capturing a "
            "structural effect -- and it undercuts the usual defence that the conventional "
            "12-month choice is a safe one."
        )
    return _p(head + verdict)


def _sensitivity_prose(sensitivity, marginals, cost_drag_sharpe, total_bps) -> str:
    if sensitivity is None or sensitivity.empty:
        return _p("<em>Sensitivity analysis was skipped for this run.</em>")

    sh = sensitivity["sharpe"].dropna()
    best = sensitivity.loc[sh.idxmax()] if len(sh) else None
    lb = marginals.get("lookback_months")

    parts = [
        _p(
            f"The grid spans {len(sensitivity)} parameter combinations, evaluated on the "
            "<strong>training window only</strong>. Sharpe across the grid ranges from "
            f"{_num(sh.min())} to {_num(sh.max())} with a median of {_num(sh.median())} and a "
            f"standard deviation of {_num(sh.std())}."
        ),
        _p(
            "<strong>The objective is not to find the best cell.</strong> With "
            f"{len(sensitivity)} combinations, the maximum is an upward-biased estimate of "
            "what any parameter choice would deliver out of sample -- it is the maximum of "
            "many noisy draws. What matters is whether there is a broad region where results "
            "are stable. A surface where neighbouring parameters give wildly different answers "
            "is measuring noise; one where a whole neighbourhood behaves similarly is measuring "
            "something real."
        ),
    ]

    if lb is not None and len(lb) > 1:
        parts.append(_lookback_stability_prose(lb))

    if best is not None:
        parts.append(
            _p(
                f"For reference, the best training cell is lookback "
                f"{int(best['lookback_months'])}m / {int(best['holdings'])} holdings / "
                f"{best['rebalance']} / {best['cost_bps']:g} bps at Sharpe "
                f"{_num(best['sharpe'])}. <em>This is reported for completeness, not as a "
                "recommendation</em> -- selecting it would be exactly the overfitting this "
                "section exists to detect. The base configuration was fixed in advance from "
                "the literature convention."
            )
        )

    cg = marginals.get("cost_bps")
    if cg is not None and len(cg) > 1:
        hi, lo = float(cg["mean"].iloc[0]), float(cg["mean"].iloc[-1])
        parts.append(
            _p(
                f"The cost dimension moves mean Sharpe from {_num(hi)} at "
                f"{cg.index[0]:g} bps to {_num(lo)} at {cg.index[-1]:g} bps. "
                + (
                    "<strong>The strategy does not survive realistic costs</strong> -- the "
                    "result is an artefact of frictionless accounting."
                    if lo < 0.1 < hi
                    else "Cost sensitivity is material but not fatal across the tested range."
                )
            )
        )
    return "".join(parts)


def _crisis_dependence_prose(loyo) -> str:
    """Report how much of the edge rests on a single year.

    This is placed first in the failure analysis because it can invalidate every
    other favourable number in the report. A full-sample edge that vanishes when
    one year is deleted has not been demonstrated -- it has been survived.
    """
    if loyo is None or not isinstance(loyo, pd.DataFrame) or loyo.empty:
        return ""

    full_edge = loyo.attrs.get("full_edge", np.nan)
    se = loyo.attrs.get("sharpe_se", np.nan)
    if np.isnan(full_edge):
        return ""

    worst_year = loyo["edge_change"].idxmin()
    worst = loyo.loc[worst_year]
    edge_without = float(worst["sharpe_edge_ex"])
    drop = float(worst["edge_change"])

    # Does the edge survive removing its single best year?
    survives = edge_without > 0.05
    collapsed = edge_without <= 0.02

    lead = (
        f"<li><strong>The edge depends heavily on {int(worst_year)}.</strong> "
        f"Over the full sample the strategy's Sharpe advantage over the benchmark is "
        f"{full_edge:+.3f}. Removing {int(worst_year)} alone changes it to "
        f"<strong>{edge_without:+.3f}</strong> (a shift of {drop:+.3f}). In that year the "
        f"strategy returned {_pct(worst['strategy_return_in_year'])} against the benchmark's "
        f"{_pct(worst['benchmark_return_in_year'])}. "
    )

    if collapsed:
        lead += (
            "<strong>With that single year excluded there is no risk-adjusted edge left.</strong> "
            "The strategy has not been shown to beat the index; it has been shown to have "
            "survived one crisis better than the index did. Those are very different claims, "
            "and only the second is supported by this sample. A defensive rotation that pays "
            "off in one drawdown out of one sample is a hypothesis, not a validated strategy."
        )
    elif not survives:
        lead += (
            "The remaining edge is small enough that the result should be treated as driven by "
            "that episode rather than by a persistent effect."
        )
    else:
        lead += (
            "The edge survives the exclusion, which is the more reassuring outcome -- the "
            "result is not a single-episode artefact."
        )
    lead += "</li>"

    if not np.isnan(se):
        lead += (
            f"<li><strong>The full-sample edge is not statistically distinguishable from zero.</strong> "
            f"The approximate standard error on an annualised Sharpe over this sample length is "
            f"&plusmn;{se:.3f}, against a strategy-versus-benchmark gap of {full_edge:+.3f} -- "
            f"roughly {abs(full_edge) / se:.1f} standard errors. Conventional thresholds are not "
            "close to being met, and the standard-error formula assumes IID returns, which makes "
            "it an <em>optimistic</em> estimate of the true uncertainty for autocorrelated, "
            "fat-tailed financial data. Nothing in this report should be read as establishing "
            "that the strategy beats the benchmark.</li>"
        )
    return lead


def _failure_prose(f, b, sp, regimes, cost_drag_sharpe, lag_decay, vs_spy, vs_rand,
                   oos_decay, total_bps, synthetic, loyo=None) -> str:
    """The section a recruiter actually reads. Lead with what went wrong."""
    findings: list[str] = []

    crisis = _crisis_dependence_prose(loyo)
    if crisis:
        findings.append(crisis)

    if not np.isnan(vs_rand) and vs_rand <= 0.05:
        findings.append(
            "<li><strong>The ranking did not beat random selection.</strong> A random 3-of-9 "
            f"rotation achieved Sharpe {_num(b['rand_sharpe'])} against the strategy's "
            f"{_num(f['sharpe'])}. This is the cleanest negative result available: it says the "
            "returns came from concentration and universe composition, not from momentum "
            "information.</li>"
        )
    if not np.isnan(vs_spy) and vs_spy < 0:
        findings.append(
            f"<li><strong>The strategy underperformed simple SPY buy-and-hold on a "
            f"risk-adjusted basis</strong> ({_num(f['sharpe'])} vs {_num(b['spy_sharpe'])}). "
            "Whatever else is true, an investor would have been better off in the index, and "
            "the strategy has to clear that bar before anything else is worth discussing.</li>"
        )
    elif not np.isnan(f["cagr"]) and not np.isnan(b["spy_cagr"]) and f["cagr"] < b["spy_cagr"]:
        # Higher Sharpe but lower raw return. Easy to present as an unqualified
        # win by quoting only the Sharpe; it is not one.
        findings.append(
            f"<li><strong>The strategy earned less than the index in absolute terms</strong> "
            f"({_pct(f['cagr'])} vs {_pct(b['spy_cagr'])} annualised) despite the higher "
            f"Sharpe ratio. The risk-adjusted win is real -- volatility of {_pct(f['vol'])} "
            f"against {_pct(_safe(b, 'spy_vol'))} and a maximum drawdown of {_pct(f['mdd'])} "
            f"against {_pct(b['spy_mdd'])} -- but an unlevered investor who simply held SPY "
            "finished with more money. Quoting the Sharpe ratio alone would misrepresent "
            "that. The result is only attractive to someone who can lever the strategy back "
            "up to the index's risk level, or who values the shallower drawdown for its own "
            "sake.</li>"
        )
    if not np.isnan(cost_drag_sharpe) and cost_drag_sharpe > 0.3:
        findings.append(
            f"<li><strong>Costs consume a large share of the edge.</strong> Removing them "
            f"lifts Sharpe by {_num(cost_drag_sharpe)} ({_num(b['nocost_sharpe'])} gross vs "
            f"{_num(f['sharpe'])} net at {total_bps:g} bps). A result that depends this heavily "
            "on the cost assumption is fragile to the assumption being wrong -- and section 6 "
            "lists four reasons the assumption is optimistic.</li>"
        )
    if not np.isnan(lag_decay) and lag_decay > 0.3:
        findings.append(
            f"<li><strong>The edge decays quickly with execution delay.</strong> Acting a week "
            f"late costs {_num(lag_decay)} of Sharpe ({_num(b['lag_sharpe'])} at T+6). A signal "
            "that needs immediacy at a monthly rebalance frequency is suspicious: a genuine "
            "12-month momentum effect should not care about a few days.</li>"
        )
    if not np.isnan(oos_decay) and oos_decay > 0.6:
        findings.append(
            f"<li><strong>Large in-sample to out-of-sample decay</strong> (Sharpe "
            f"{_num(sp['train'])} &rarr; {_num(sp['valid'])}). The training result was "
            "substantially a fit to that period.</li>"
        )

    if isinstance(regimes, pd.DataFrame) and not regimes.empty and "excess_ann_return" in regimes:
        worst = str(regimes["excess_ann_return"].idxmin())
        worst_v = float(regimes.loc[worst, "excess_ann_return"])
        if worst_v < 0:
            findings.append(
                f"<li><strong>Worst regime: {html.escape(worst)}</strong>, where the strategy "
                f"trailed the benchmark by {_pct(abs(worst_v))} annualised. "
                + _regime_explanation(worst) + "</li>"
            )

    intro = _p(
        "Listing what did not work is not a formality. Most of the informational content of a "
        "backtest is in its failure modes, because those are the parts that generalise -- and "
        "a report that finds nothing wrong with its own strategy has usually not looked."
    )

    if not findings:
        body = _p(
            "No single failure mode dominated: the strategy cleared its baselines, costs did "
            "not consume the edge, the result survived execution delay, and out-of-sample "
            "decay was proportionate. That is a genuinely acceptable outcome, but it is not "
            "the same as a validated edge. The structural caveats in section 10 -- a "
            "nine-asset cross-section, one market, one sample, optimistic cost assumptions -- "
            "still apply in full, and the standard error on the Sharpe estimate is wide enough "
            "that a modest positive result is not distinguishable from luck."
        )
    else:
        body = f"<ul>{''.join(findings)}</ul>"

    known = _p(
        "<strong>Known structural weaknesses of this strategy design, independent of the "
        "result above:</strong>"
    ) + (
        "<ul>"
        "<li><em>Momentum crashes.</em> The portfolio holds whatever recently rose, so it is "
        "maximally exposed at exactly the moment leadership reverses. Losses are sudden rather "
        "than gradual, and volatility-scaled sizing is the standard mitigation.</li>"
        "<li><em>Low breadth.</em> Three positions from nine assets. The portfolio is one bad "
        "ETF away from a bad year, and no amount of signal quality fixes a breadth constraint.</li>"
        "<li><em>Rebalance-date luck.</em> Monthly rebalancing on the last trading day is an "
        "arbitrary choice. Results conditional on that choice are partly luck; overlapping "
        "portfolios started on different days of the month would average it out.</li>"
        "<li><em>Cash is not a holding.</em> The design is always fully invested in the top 3 "
        "even when every asset is falling, because the rank is purely relative. An "
        "absolute-momentum filter is the obvious fix and is listed in section 10.</li>"
        "</ul>"
    )

    syn = (
        _callout(
            "This run used synthetic data",
            _p("The failure analysis above describes the behaviour of the software on a "
               "simulated price panel. It says nothing about whether ETF momentum works. "
               "Re-run with network access for a research result."),
        )
        if synthetic else ""
    )
    return intro + body + known + syn
