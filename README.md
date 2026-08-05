# Cross-Sectional ETF Momentum — Reproducible Research Platform

[![tests](https://github.com/monishkk/quant-research-platform/actions/workflows/tests.yml/badge.svg)](https://github.com/monishkk/quant-research-platform/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**[→ Read the full research report](https://monishkk.github.io/quant-research-platform/)**

A small, tested backtesting platform, built to evaluate one strategy properly rather than many
strategies badly — and rigorous enough to conclude that this one does not survive scrutiny.

The test case is **cross-sectional momentum on nine liquid US ETFs**: rank by trailing 12-month
return (skipping the most recent month), hold the top 3 equally weighted, rebalance monthly,
charge realistic costs, and compare against honest baselines across separated in-sample and
out-of-sample periods.

The strategy is deliberately unoriginal. It is an instrument for proving the research *process*
is correct — a novel signal evaluated with a broken backtest is worth less than a known signal
evaluated properly.

---

## Headline result

Full sample 2008-02-08 to 2025-12-30, real daily adjusted prices, **5 bps per unit of traded
notional** (so a round trip costs 10 bps), T+1 execution:

| | Momentum (top 3) | SPY buy & hold | Equal weight | Random 3-of-9 |
|---|---|---|---|---|
| CAGR | 10.78% | **11.65%** | 7.78% | 6.34% |
| Volatility | **14.84%** | 19.93% | 15.86% | 18.55% |
| Sharpe | **0.76** | 0.65 | 0.56 | 0.43 |
| Max drawdown | **-26.32%** | -51.48% | -44.68% | -50.19% |
| Annual turnover | 4.57x | 0.00x | 0.33x | 15.91x |

![Equity curve: momentum vs SPY, equal weight, and random selection](reports/momentum/equity_curve.png)

**The honest reading — and the actual finding of this project:**

> **Excluding 2008, the Sharpe advantage over SPY falls from +0.111 to −0.029. It reverses.**

No other year moves the edge by more than 0.07.

A stationary block bootstrap (5,000 paired resamples) puts the 95% confidence interval on that
+0.111 advantage at **[−0.258, +0.439]**, with **p = 0.530**. And because the 192-cell sensitivity
grid is itself a search, the edge has to clear the best Sharpe that many trials of pure noise would
be expected to produce — a hurdle of **0.404**. Against it the **Deflated Sharpe Ratio is 0.700**,
short of the conventional 0.95 bar.

Two things cut the other way, and the report says so: the parameters were pre-committed from the
literature rather than picked from the grid, and the deflation assumes trials are independent when
neighbouring cells share most of their trades. But the direction of the result does not change.

Three further caveats the headline table hides:

- The strategy earned **less money** than SPY (10.78% vs 11.65%) — it won on risk, not return.
- Alpha is +5.36% (Newey–West t = 2.06, p = 0.040) but the information ratio is **−0.11**. Both
  are correct: alpha is beta-adjusted, the information ratio is not. The alpha clears 5% by a
  small margin and would not survive a correction for the number of tests here.
- It beat SPY in only **7 of 18 calendar years**.

What it *is*: a defensive cross-asset rotation. It gives up 6.9%/yr in rising markets (76% of the
sample) and gains 12.6%/yr in falling ones (21%). That is a coherent economic story, and it
explains the low beta, the halved drawdown, and the below-index return.

What did **not** fail: costs (0.015 of Sharpe), execution delay (0.09 at T+6), and above all the
ranking itself — against **300 random 3-of-9 rotations** on the same calendar the strategy ranks
above *every one*, and still does when the random paths are handed zero transaction costs
(p < 0.005 either way). The signal beats chance within this universe; it just does not beat a
passive index. The implementation is sound *under the conventions it states* —
weight drift, market impact, taxes and a non-zero risk-free rate are all unmodelled, and each is
listed below. The honest reading of a sound implementation is that the effect is small and
crisis-dependent.

A negative result, reported as one. Full write-up: [`reports/RESEARCH_REPORT.md`](reports/RESEARCH_REPORT.md)
(or the generated `reports/momentum/research_report.html`).

---

## Quick start

```bash
git clone https://github.com/monishkk/quant-research-platform.git && cd quant-research-platform
python -m venv .venv && .venv\Scripts\activate    # Windows
pip install -r requirements.txt && pip install -e .
```

Reproduce every artefact with one command:

```bash
python -m quant_platform.run --config configs/momentum.yaml
```

Run the test suite:

```bash
python -m pytest
```

197 tests, ~8 seconds. They are the reason to trust anything above.

---

## What one command produces

```
reports/momentum/
├── research_report.html        # self-contained 10-section report, images embedded
├── metrics.csv                 # strategy vs all baselines
├── in_vs_out_of_sample.csv     # training / validation / test splits
├── sensitivity.csv             # 192-combination parameter grid
├── regimes.csv                 # performance by market regime
├── calendar_years.csv          # year-by-year vs benchmark
├── leave_one_year_out.csv      # how much of the edge rests on any single year
├── significance.csv            # bootstrap CI, p-value, Deflated Sharpe, noise hurdle
├── random_null_paths.csv       # 300 random 3-of-9 rotations: the empirical null
├── strategy_timeseries.csv     # strategy + benchmark returns, turnover, costs, exposure
├── equity_curve.png            drawdown.png            rolling_sharpe.png
├── rolling_volatility.png      turnover.png            exposure.png
├── monthly_returns.png         sensitivity.png         in_vs_out_sample.png
├── bootstrap_sharpe.png        # where the observed edge sits in its own noise distribution
```

Useful flags:

| Flag | Effect |
|---|---|
| `--show-test` | show the test split in the report (it is always computed and written to CSV) |
| `--no-cache` | force a fresh download instead of reading cached Parquet |
| `--skip-sensitivity` | skip the 192-combination grid |
| `--output-dir DIR` | write elsewhere |
| `-v` | debug logging |

---

## The four things that make this a backtest rather than a chart

### 1. One shift, in one place

```python
executed = target_weights.shift(execution_lag)     # lag >= 1: what we want, when
held     = drift(executed, asset_returns)          # what we actually have
gross    = (held * asset_returns).sum(axis=1)
```

`prices.loc[t]` is the **close** at t, so `returns.loc[t]` is the return over `(t-1, t]` and is
not knowable until t closes. `target_weights.loc[t]` is what you *want* at the close of t. The
shift means the weights multiplying `returns.loc[t]` were fixed at the close of `t-1`, strictly
before that return existed.

`execution_lag=0` is permitted but emits a warning — it exists so the bias can be *measured*,
not shipped.

### 2. Rebalance dates are real trading days

The obvious implementation is wrong:

```python
monthly = scores.resample("ME").last()          # stamped on 31 January
daily   = monthly.reindex(prices.index).ffill()  # 31 Jan is a Sunday -> row dropped
```

The label is silently dropped and the forward-fill carries the *previous* month's weights — a
missed rebalance no assertion would catch. This platform resamples the index itself and takes the
last **actual** trading day of each period, so rebalance dates are always a subset of the price
index and the reindex is lossless.

### 3. Parameters were fixed before any result was inspected

The 192-combination sensitivity grid runs on the **training window only**. With that many cells, a
Sharpe of 1.5 somewhere in the grid is what noise alone produces.

The test split is kept out of the report body unless you pass `--show-test` — but it is computed on
every run and committed in `in_vs_out_of_sample.csv`, so "withheld" would overstate it. The claim
that actually matters is checkable: `git log -p configs/momentum.yaml` shows every strategy
parameter was set in the initial commit and never changed. Later commits fixed engine and reporting
defects; none altered a strategy choice.

### 4. Leave-one-year-out, which is what actually found the answer

A full-sample edge cannot distinguish a persistent effect from one good year attached to a decade
of nothing. Deleting one year at a time can:

```python
validation.leave_one_year_out(strategy_returns, benchmark_returns)
```

Every other diagnostic in this project said the strategy was fine. This one said the edge was
2008. It is the single highest-value check in the repository, and it is the reason the headline
above is a negative result rather than a Sharpe ratio.

---

## Repository layout

```
src/quant_platform/
├── data.py         download, validate, cache to Parquet, record provenance
├── returns.py      simple/log/cumulative returns, rolling volatility
├── signals.py      momentum score -> ranks -> target weights -> rebalance schedule
├── costs.py        turnover definition and the linear cost model
├── portfolio.py    the engine: execution lag, costs, equity curve
├── metrics.py      every performance statistic, implemented from its definition
├── significance.py stationary block bootstrap, Probabilistic and Deflated Sharpe
├── validation.py   sample splits, baselines, sensitivity grid, regime analysis
├── reporting.py    matplotlib figures and the HTML report
├── narrative.py    the report's prose, with conclusions computed from the results
└── run.py          CLI entry point

tests/
├── test_data.py           schema, cleaning, validators, return identities
├── test_portfolio.py      engine accounting, costs, weights, reproducibility
├── test_metrics.py        every metric against an analytically known answer
├── test_no_lookahead.py   perturbation tests for causality
├── test_significance.py   bootstrap construction, deflation, known-answer inference
└── test_validation.py     splits, baselines, sensitivity, end-to-end determinism

notebooks/
├── 01_data_exploration.ipynb    adjusted vs unadjusted, coverage, correlations
├── 02_momentum_research.ipynb   signal construction, weights verified by hand
└── 03_results.ipynb             performance, splits, sensitivity
```

`returns.py` and `narrative.py` are additions to the original module plan: return transforms
deserve their own module rather than being mixed into `metrics.py`, and separating the report's
prose from its rendering keeps `reporting.py` about figures.

Notebooks call into `src/`. They do not contain the platform.

---

## Conventions, stated explicitly

Different backtest implementations produce different results for the same strategy, mostly
through undocumented convention choices. Ours:

| Choice | This platform | Why it matters |
|---|---|---|
| Price series | Adjusted close | Unadjusted reads a 2-for-1 split as -50% |
| Return type | Simple (arithmetic) | Portfolio returns are the weighted arithmetic mean; log returns do not aggregate cross-sectionally |
| Execution | T+1 | Target set at close t is held for the return from t to t+1 — i.e. a closing-auction fill at t. The 21-day signal skip gives three weeks of notice, so this is implementable, not optimistic |
| Turnover | **Two-way**: `sum|Δw|` | Many papers quote one-way, exactly half. A 2x difference in turnover is a 2x difference in costs |
| Initial entry | Charged | Entering from cash is a real trade |
| Cost model | Linear, 5 bps per unit of one-way traded notional (a round trip costs 10) | No impact, no drift, static spreads |
| Weight cap | Cap, residual to **cash** | Renormalising would ignore the limit it was asked to enforce |
| Warm-up | No position until the signal exists | A 1-name portfolio during warm-up is an artefact, not a decision |
| Sortino denominator | Full-sample | Dividing by the count of negative periods flatters strategies that rarely lose |
| VaR / CVaR | Returned as returns (negative = loss) | Avoids sign ambiguity |

---

## Known limitations

These are real and uncorrected:

- **Survivorship bias.** Fixed universe of ETFs that exist today. Small for these nine
  cross-asset funds; severe if you point this code at single-name equities without
  point-in-time index membership and delisting returns.
- **Weight drift is now modelled.** Holdings drift with relative performance between rebalances
  and trades are priced off the actual pre-trade position, so turnover reflects what a portfolio
  would really have to trade. Correcting this raised measured turnover from 4.33x to **4.57x** and
  moved Sharpe from 0.777 to **0.764**. It mattered far more to the equal-weight baseline, whose
  constant target had previously produced a reported turnover of *zero* for a monthly-rebalanced
  mandate.
- **Optimistic costs.** No market impact, static spreads, no taxes. All bias the result in the
  strategy's favour.
- **A zero risk-free rate, which does flatter the strategy.** Its volatility (14.9%) is below
  SPY's (19.9%), so a positive risk-free rate cuts its Sharpe by more. The +0.124 edge becomes
  **+0.090 at 2%** and +0.073 at 3%; realistic bill yields over this sample put the honest figure
  nearer +0.10. The direction reinforces the conclusion, which is no reason to leave it unstated.
- **Low breadth.** Three positions from nine assets caps the achievable information ratio
  regardless of signal quality.
- **One sample.** ~18 years, one market. For a monthly strategy the effective sample is closer
  to 180 independent observations than to 4,500 daily rows, so the standard error on Sharpe is
  roughly ±0.25. Differences smaller than that are not distinguishable from luck.

## Next experiments

1. **Volatility targeting** — scale exposure to constant ex-ante volatility. Best-documented
   improvement to base momentum.
2. **Risk-parity weighting** — equal-weighting GLD and QQQ gives equal capital, wildly unequal risk.
3. **Absolute-momentum overlay** — require a positive trailing return, not just a top-3 rank.
   Targets the failure mode where the "best" asset is still falling.
4. **Wider universe** — breadth is the binding constraint.
5. **Walk-forward validation** — re-select parameters on a rolling past-only window and evaluate
forward, repeatedly. The sensitivity grid shows the lookback dimension is unstable; this tests the
consequence directly. The largest remaining gap.
6. **Cross-check against VectorBT** — agreement is evidence the accounting is right.

---

## Data

Free adjusted ETF prices via `yfinance`. Raw and processed panels are cached to
`data/` as Parquet with a provenance sidecar recording source, symbols, date range,
and retrieval timestamp.

**The synthetic fallback is off by default.** If the download fails the run stops rather than
quietly substituting simulated prices — a synthetic momentum panel produces entirely
plausible-looking results, which is exactly what makes silent substitution dangerous. Set
`allow_synthetic_fallback: true` to opt in; those runs are stored under a `synthetic_*` cache key
so they can never occupy the real provider's slot, and every artefact from them is labelled.

Cached panels are refused unless their provenance sidecar confirms the source they came from, and
the sidecar records the git commit and a SHA-256 of the panel contents — so a silently revised
price history is detectable rather than invisible. (The test suite is unaffected: it builds its
own panels with `source="synthetic"` and never touches the network.)

## Requirements

Python 3.10+; pandas, NumPy, SciPy, Matplotlib, pyarrow, PyYAML, yfinance, pytest.
Verified on Python 3.12 with pandas 3.0 and NumPy 2.5.

## License

MIT
