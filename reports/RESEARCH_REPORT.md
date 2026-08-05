# Cross-sectional momentum on liquid US ETFs

**A reproducible test of whether ranking nine liquid ETFs by trailing 12-1 month return produces an investable edge after costs.**

Sample: 2008-02-08 to 2025-12-30 (4,502 trading days, 17.9 years) · Data: Yahoo Finance adjusted closes
Reproduce: `python -m quant_platform.run --config configs/momentum.yaml`

---

## Summary of findings

The strategy produces a higher Sharpe ratio than SPY (0.76 vs 0.65) with roughly half the maximum drawdown (-26% vs -51%). It also earned **less money** than SPY (10.78% vs 11.65% annualised).

More importantly, **the entire risk-adjusted edge is attributable to a single year.** Excluding 2008, the Sharpe advantage over SPY falls from +0.111 to **-0.029** — it reverses.

Two significance tests agree. A stationary block bootstrap puts the 95% interval on that +0.111 advantage at **[-0.258, +0.439]**, p = 0.530. And because the 192-cell parameter grid is itself a search, the result must clear the best Sharpe such a search would produce from pure noise — **0.404**; against that hurdle the Deflated Sharpe Ratio is **0.700**, short of the conventional 0.95 bar.

The correct conclusion is not "cross-sectional ETF momentum works." It is:

> This strategy behaves as a defensive cross-asset rotation. It sacrificed return in rising markets, protected capital in falling ones, and its measured advantage over a passive index in this sample is indistinguishable from noise once the 2008 crisis is removed.

That is a negative result, and it is reported as one.

---

## 1. Research question

Does ranking a small universe of liquid US ETFs by trailing 252-day return (skipping the most recent 21 days), holding the top 3 in equal weight, and rebalancing monthly, produce a risk-adjusted return that survives realistic transaction costs and honest out-of-sample testing?

The strategy is deliberately unoriginal. Cross-sectional momentum is among the most documented effects in asset pricing, which makes it a good instrument for the actual purpose: proving the research *process* — data handling, execution timing, cost accounting, sample discipline — is correct. A novel signal evaluated with a broken backtest is worth less than a known signal evaluated properly.

The question was framed so a negative answer would be reportable. It was.

## 2. Economic motivation

Momentum has been documented across equities, bonds, commodities, and currencies over multi-decade samples. Two explanations compete, with different implications for persistence:

- **Behavioural.** Investors under-react to gradually diffusing information, then over-extrapolate. If this is the mechanism, the effect can be arbitraged away as it becomes known.
- **Risk-based.** Momentum loads on a compensated risk factor — notably crash risk, since momentum portfolios suffer sudden severe losses at reversals. If this is the mechanism, the return is payment for a real exposure, and so are the crashes.

The 12-1 construction (twelve-month formation, one-month skip) is convention, not optimisation. The skip exists because short-horizon returns exhibit *reversal* rather than continuation; including the most recent month mixes opposing effects.

The universe — SPY, QQQ, IWM, EFA, EEM, TLT, GLD, VNQ, DBC — spans equities, international, bonds, gold, REITs, and commodities. This matters: it lets a top-3 selection rotate genuinely defensively rather than merely picking the highest-beta equity sleeve. That property turns out to drive the entire result.

## 3. Data and cleaning

43,011 daily rows across 9 symbols, 2007-01-03 to 2025-12-30. All nine ETFs have complete coverage from 2007, so no symbol enters mid-sample to distort the cross-sectional rank.

All research uses **adjusted** closes. This is not cosmetic: an unadjusted series treats a 2-for-1 split as a -50% return, which a momentum ranking reads as the worst asset in the universe. Dividends matter equally — TLT and VNQ distribute heavily, and ranking on price return alone would systematically under-rank them against SPY.

The pipeline enforces and fails loudly on: duplicate `(date, symbol)` pairs; non-monotonic dates within a symbol; non-positive prices; `high < low`; and a non-unique or unsorted price index.

**Survivorship bias is present and uncorrected.** The universe is a fixed list of ETFs chosen because they exist and are liquid *today*. For these nine cross-asset funds the distortion is small — none was ever near delisting — but it is a real upward bias. Applying this code to single-name equities without point-in-time index membership and delisting returns would badly overstate results.

## 4. Signal definition

```
score[i, t] = price[i, t - 21] / price[i, t - 252] - 1
```

Implemented as `prices.shift(skip).pct_change(lookback - skip)`. The `shift(skip)` means the value at *t* depends only on prices at or before *t-21* — strictly backward-looking before the execution delay is even applied. This is verified by perturbation tests, not by inspection.

On each rebalance date, assets are ranked, the top 3 selected, weighted equally. During the formation window, when fewer than 3 assets have scores, the target is *no position* rather than a partially-filled portfolio.

A 40% per-name cap is applied. At 3 equal-weighted holdings each position is 33.3%, so the cap does not bind in the base configuration. It does bind at 1 and 2 holdings in the sensitivity grid, where excess is left **in cash** rather than redistributed — those cells describe a partly-invested portfolio (average exposure 0.40 and 0.80 respectively), which is why exposure is reported alongside them.

**A subtle trap avoided.** The obvious implementation, `scores.resample("ME").last()`, stamps output on the calendar month-end, which is frequently not a trading day. Reindexing that label onto a trading-day index silently drops it, and a forward-fill then carries the *previous* month's weights — a missed rebalance no assertion would catch.

In this sample **67 of 227 calendar month-ends (29.5%) are not trading days**, and the naive implementation produces different holdings from the correct one on **866 of 4,779 days — 18.1% of the sample**. That is not a rounding difference; it is a materially different strategy, arrived at silently. The platform instead resamples the index itself and takes the last *actual* trading day of each period, guaranteeing rebalance dates are a subset of the price index.

## 5. Execution assumptions

- `prices.loc[t]` is the **close** at t.
- `returns.loc[t] = prices.loc[t]/prices.loc[t-1] - 1` is the return over `(t-1, t]`, unknowable until t closes.
- `target_weights.loc[t]` is the portfolio *desired* at the close of t.

The engine holds `executed_weights = target_weights.shift(1)`, so the weights multiplying `returns.loc[t]` were fixed at the close of `t-1`.

**Precisely which return is earned, and what that assumes.** A position formed from the signal at `t-1` earns the close-to-close return from `t-1` to `t`. Earning that return requires already holding the position at the close of `t-1` — so the model assumes a **closing-auction fill on the day the target is set**, not a fill during the following session. That is implementable here rather than optimistic, because the signal is not fresh: with a 21-day skip, the score at `t-1` depends only on prices at or before `t-22`, giving three weeks of notice to work the order. But it is a stronger assumption than "trade the next session", and it is the one the arithmetic actually makes. The T+6 baseline exists to bound what looser execution costs: Sharpe falls from 0.76 to 0.67.

`execution_lag=0` is permitted but warns — it exists so the bias can be *measured*, not shipped.

**Not modelled:** trades execute at the close in full, at no impact, with no partial fills or failures to trade. Long-only, unlevered, no taxes.

## 6. Transaction-cost model

Linear in turnover at 5 bps (2 commission + 3 slippage):

```
turnover[t] = sum_i |w[i,t] - w[i,t-1]|     # executed weights, TWO-WAY
cost[t]     = turnover[t] * 5 / 10_000
```

Turnover is **two-way**: rotating out of one 33% position into another counts 0.66, not 0.33. Many published figures quote one-way turnover, exactly half. A factor of two in turnover is a factor of two in costs, so the convention must be checked before any external comparison. Charging the full two-way sum is conservative and matches what a broker bills. The initial entry from cash is charged.

**The 5 bps is per unit of traded notional, not per round trip.** Buying into a position costs 5 bps; selling out of it costs another 5. A complete round trip therefore costs **10 bps**, and a full rotation from one holding to another costs 10 bps (5 to sell, 5 to buy). Earlier drafts of this report described the assumption as "5 bps round-trip", which understated it by half — the code was always right, the label was not. Given that this section exists to argue conventions must be checked before comparison, getting its own label wrong was worth correcting explicitly rather than silently.

Realised annual turnover is **4.57x** (2.28x one-way), costing ~0.23% of NAV per year. Measured against the zero-cost run, costs consumed **0.25%** of annual return and **0.015** of Sharpe.

**Costs are not what kills this strategy** — a useful finding in itself, and the reason the failure analysis focuses elsewhere.

Four ways the model still flatters the result:

1. **Weight drift is modelled** (it was not in earlier versions of this report). Holdings now drift with relative performance between rebalances, and each trade is priced off the actual pre-trade position rather than off the previous target. Correcting this moved the strategy's Sharpe from 0.777 to **0.764** and its measured turnover from 4.33x to **4.57x** — the trades that restore a drifted portfolio are real and are now billed. The effect on the *baselines* was larger: an equal-weight mandate has a target that never changes, so the old accounting reported **zero** annual turnover for a portfolio that must trade every month; it now reports 0.33x.
2. **No market impact.** Constant cost per unit of turnover regardless of size. Fine for nine ETFs trading billions daily; wrong at scale.
3. **Static spread.** A flat slippage number stands in for a spread that widens in exactly the stressed markets where this strategy rebalances hardest.
4. **No taxes.** At 4.3x turnover in a taxable account, short-term capital gains would be a first-order drag.
5. **A zero risk-free rate, which does flatter the strategy.** Every Sharpe here is computed against `rf = 0`, and uninvested cash earns nothing. Because the strategy's volatility (14.9%) is below the benchmark's (19.9%), a positive risk-free rate reduces the strategy's Sharpe *more* than the benchmark's:

| Risk-free rate | Strategy Sharpe | SPY Sharpe | Edge |
|---|---|---|---|
| 0.0% (as reported) | 0.764 | 0.653 | **+0.111** |
| 1.0% | 0.697 | 0.603 | +0.094 |
| 2.0% | 0.629 | 0.552 | +0.077 |
| 3.0% | 0.562 | 0.502 | +0.060 |

Average short-term Treasury yields over 2008–2025 were roughly 1.3%, which puts the honest edge nearer **+0.08 than +0.111**. This does not change the conclusion — it moves it further in the direction the report already argues — but the headline figure is specifically *the excess-Sharpe gap under a zero risk-free rate*, and that assumption is doing about 0.03 of visible work.

6. **The benchmark's entry cost falls outside the reported window.** SPY buy-and-hold is routed through the identical engine and *is* charged its entry, but it enters in January 2007, before the common aligned start of 2008-02-08, so that charge is sliced away and its reported turnover is exactly zero. The strategy's entry is charged inside the window. The asymmetry is worth ~5 bps once across 17.9 years — negligible, and it works against the strategy.

Items 1–4 and 6 are small. Item 5 is the one a reader should keep in mind.

## 7. Validation procedure

| Window | Dates | Rule |
|---|---|---|
| Training | 2008-02-08 → 2018-12-31 | All parameter exploration. Sensitivity grid evaluated here only. |
| Validation | 2019-01-01 → 2022-12-31 | Inspected after the parameter region was chosen. |
| Test | 2023-01-01 → 2025-12-30 | Evaluated once. Not used to revise anything. |

The 192-combination sensitivity grid ran on the **training window only**. With that many cells, a Sharpe of 1.5 somewhere in the grid is what noise alone produces.

**On the word "withheld".** The test split is hidden from the report body and console unless `--show-test` is passed — but it is computed on every run and written to `in_vs_out_of_sample.csv`, which is committed to this repository. Anyone can read it, and by now so have I. The defensible claim is therefore not that the numbers were never seen; it is this:

> **Every strategy parameter was fixed before any result was inspected, and none has been changed since.**

That is checkable rather than asserted: `git log -p configs/momentum.yaml` shows the lookback, skip, holdings, rebalance frequency, weight cap, cost assumptions and split boundaries were all set in the initial commit and never touched. Subsequent commits fixed engine and reporting defects — a sensitivity grid that compared different windows, two report tables that silently rendered as "No data.", a mislabelled cost convention — none of which altered a strategy choice. Corrections to measurement after seeing an outcome are legitimate; corrections to the *strategy* would not have been, and did not happen.

Code-level defences against look-ahead live in `tests/test_no_lookahead.py`, which perturbs future prices by 100x and asserts historical signals, weights, and realised returns are bit-identical. That test catches bugs a sample split cannot.

## 8. Results

### Strategy vs baselines (full sample)

| | **Momentum** | SPY B&H | Equal weight | Random 3-of-9 | Zero cost | T+6 exec |
|---|---|---|---|---|---|---|
| CAGR | 10.78% | **11.65%** | 7.78% | 6.34% | 11.03% | 9.35% |
| Volatility | **14.84%** | 19.93% | 15.50% | 18.45% | 14.84% | 14.90% |
| Sharpe | **0.764** | 0.653 | 0.561 | 0.426 | 0.779 | 0.675 |
| Sortino | 1.077 | 0.923 | 0.790 | 0.600 | 1.099 | 0.950 |
| Max drawdown | **-26.32%** | -51.48% | -44.65% | -50.19% | -26.29% | -26.70% |
| Calmar | 0.409 | 0.226 | 0.174 | 0.126 | 0.420 | 0.350 |
| Worst day | **-5.05%** | -10.94% | -8.35% | -12.92% | -5.05% | -5.06% |
| Kurtosis | **4.64** | 14.86 | 11.01 | 24.30 | 4.63 | 4.46 |
| Annual turnover | 4.57x | 0.00x | 0.33x | 15.91x | 4.57x | 4.48x |
| Beta / Alpha | 0.46 / 5.36% | — | 0.72 / -0.66% | 0.79 / -2.39% | 0.46 / 5.59% | 0.46 / 4.03% |
| Information ratio | -0.11 | — | -0.53 | -0.49 | -0.09 | -0.19 |

### Reading this honestly

The strategy **beat SPY on Sharpe and lost on return.** It made less money while taking less risk.

Alpha is +5.36% but the information ratio is **-0.11**. These disagree by construction: Jensen's alpha is beta-adjusted (at beta 0.46 the strategy was only expected to capture ~5.4% of the benchmark's return and delivered 10.78%), while the information ratio uses raw active return and does not adjust for beta at all. Both are correct. Quoting only the alpha would misrepresent the result.

**Alpha with an error bar.** A point estimate invites over-reading, so alpha is reported with a Newey-West standard error (heteroskedasticity- and autocorrelation-robust, which for daily data is the appropriate correction): **t = 2.06, p = 0.040, 95% CI [+0.25%, +10.47%]**.

That is marginally significant, and it deserves the same skepticism this report applies elsewhere rather than a victory lap. It is one test among several run on one sample; the interval very nearly touches zero; the Sharpe *difference* over the same data is not significant at all (p = 0.530); and the raw active return is negative. The consistent reading is that the strategy earned slightly more than its low beta entitled it to, on evidence that clears a 5% threshold by a small margin and would not survive a correction for the number of tests in this report.

**Against random selection.** The strategy beats a seeded random 3-of-9 rotation by 0.34 of Sharpe — the minimum evidence needed to claim the ranking carries information. One caveat: random selection re-draws independently each month and turns over 15.82x/year against the strategy's 4.33x, an extra 0.57% of annual cost. The CAGR gap is 4.36%, so the turnover handicap explains only ~13% of it. The ranking is doing real work relative to random.

**Costs and execution delay.** Costs remove only 0.014 of Sharpe. Acting a full week late (T+6) costs 0.11 of Sharpe — modest, which is reassuring: a 12-month signal that collapsed on a few days' delay would be a microstructure artefact rather than an investable effect.

### In-sample vs out-of-sample

| | Training | Validation | Test |
|---|---|---|---|
| Strategy Sharpe | 0.570 | 0.814 | 1.554 |
| SPY Sharpe | 0.491 | 0.659 | 1.444 |
| **Edge** | **+0.079** | **+0.155** | **+0.110** |
| Strategy CAGR | 7.73% | 11.97% | 22.56% |
| SPY CAGR | 8.14% | 13.09% | 23.35% |
| Strategy max DD | -26.18% | -20.49% | -13.80% |

The strategy's Sharpe *rises* from training to test — the opposite of the usual overfitting signature. But **so does SPY's**, from 0.491 to 1.444. Almost all of that improvement is the market becoming easier, not the strategy generalising.

The right quantity is the **edge**, and it is stable but small: +0.079, +0.155, +0.110. Reporting the raw Sharpe progression as evidence the strategy generalises would be a serious misreading, and it is the most common way in-sample/out-of-sample tables get abused.

Note also that in every window the strategy's CAGR is *below* SPY's. The edge is entirely a risk story.

### Is the ranking better than chance?

The most informative baseline is not SPY but *random selection*: does the momentum ranking carry information, or would any concentrated 3-of-9 rotation in this universe have done as well? Earlier versions of this report answered that with a single seeded random path, which answers almost nothing — one draw from a wide distribution, where the verdict depends partly on the seed.

The current version runs **300 independent random paths** on the same rebalance calendar, with the same costs and the same execution lag.

| | Random, costed | Random, cost-free | Momentum |
|---|---|---|---|
| Mean Sharpe | 0.450 | 0.496 | — |
| Std deviation | 0.084 | 0.085 | — |
| 95th percentile | 0.595 | 0.645 | — |
| Strategy Sharpe | | | **0.764** |
| Strategy percentile | **100%** | **100%** | |
| Empirical p-value | **< 0.005** | **< 0.005** | |

The strategy outranks every one of the 300 random paths. It still does when the random paths are handed **zero transaction costs** — an advantage the strategy is not given — which matters because random reselection churns about 16x a year against the strategy's 4.6x, so a cost-matched comparison partly rewards momentum for being cheap to hold rather than good at choosing. Removing that confound leaves the selection effect standing.

**This is the one clearly positive finding in the report, and it is narrower than it sounds.** It says the ranking beats chance *within this universe*. It does not say the resulting portfolio beats a passive index — the sections above establish that it does not, and that what edge exists over SPY rests on 2008. Both can be true: a signal can order nine assets better than a coin while still producing a portfolio no better than simply owning the market.

### Is the edge distinguishable from luck?

Two separate questions, answered with two separate tools.

**Is +0.111 different from zero?** The analytic Sharpe standard error assumes iid normal returns; these returns are neither (skew −0.18, excess kurtosis 4.64). A **stationary block bootstrap** — 5,000 resamples, geometric blocks averaging 17 days — resamples the strategy and benchmark on *identical* draws, so their correlation survives and the interval is on the difference itself rather than on two independent levels.

| | |
|---|---|
| Sharpe advantage | **+0.111** |
| 95% confidence interval | **[−0.258, +0.439]** |
| Bootstrap standard error | 0.178 |
| Two-sided p-value | **0.530** |

The interval comfortably contains zero. Excluding 2008 the advantage is −0.029.

**Would +0.111 look impressive even from a worthless strategy?** Searching 192 configurations is itself a source of apparent performance: the maximum of 192 noisy draws is positive even when every draw has zero expectation. Given the dispersion actually observed across the grid, the expected best Sharpe from 192 trials of pure noise is **0.404**.

| | Against zero (PSR) | Against the 0.404 noise hurdle (DSR) |
|---|---|---|
| Training window | 0.969 | **0.700** |
| Full sample | 0.999 | 0.938 |

Measured against zero the strategy is convincing. Measured against what the search itself could have produced, it is not: **0.700 falls short of the conventional 0.95 bar.**

Two qualifications, both favouring the strategy, and both stated because omitting them would overstate the case. The base configuration was fixed in advance from the literature convention, so it never benefited from the search being penalised here. And the Deflated Sharpe correction assumes independent trials, whereas neighbouring grid cells share most of their trades — the effective number of trials is smaller than 192 and the true hurdle is therefore below 0.404. The stricter number is reported anyway, because it is the one that would apply had the parameters been chosen by looking at the grid.

Reproduce from `reports/momentum/significance.csv`.

### Parameter sensitivity (training window only, 192 combinations)

All 192 cells are evaluated on **one common window** (2,708 days, starting 2008-04-01). Cells go live on different dates — a 3-month lookback forms a signal months before a 12-month one, and a quarterly rebalance waits for the next quarter end — so scoring each on its own start date would let the grid compare *periods* as well as parameters. Given that this sample's earliest months are crisis months that dominate the result, that confound is not negligible: it was worth 0.015 of Sharpe on the monthly cells.

(This is why the base cell below reads 0.555 rather than the 0.570 in the split table above: the headline strategy runs from 2008-02-08, the grid from 2008-04-01.)

Sharpe across the grid: mean 0.352, median 0.360, std 0.147, range -0.046 to 0.601.

| Lookback | Mean Sharpe | | Holdings | Mean Sharpe | Avg exposure |
|---|---|---|---|---|---|
| 3m | 0.405 | | 1 | 0.214 | 0.40 |
| 6m | 0.388 | | 2 | 0.310 | 0.80 |
| 9m | 0.276 | | 3 | 0.411 | 1.00 |
| 12m | 0.337 | | 4 | 0.472 | 1.00 |

| Rebalance | Mean Sharpe | | Cost (bps) | Mean Sharpe |
|---|---|---|---|---|
| Weekly | 0.379 | | 0 | 0.403 |
| Monthly | 0.397 | | 5 | 0.374 |
| Quarterly | 0.280 | | 10 | 0.345 |
| | | | 20 | 0.286 |

**Lookback is not flat.** 9 months (0.276) is materially worse than 3 months (0.405), and the chosen 12 months (0.337) is below both 3m and 6m. There is no smooth plateau — the non-monotonicity suggests the lookback dimension is measuring period-specific noise more than a structural parameter.

**The holdings trend is largely an artefact.** The apparent monotone improvement 1→2→3 mostly reflects the 40% cap leaving 1- and 2-holding portfolios 60% and 20% in cash respectively. The only clean comparison is 3 vs 4 holdings (both fully invested): 0.411 vs 0.472, so more diversification helped. Reading "concentration is bad" off the full column would be wrong.

**Quarterly rebalancing is genuinely worse** (0.280 vs monthly 0.397) — and now demonstrably so rather than apparently so, since both are measured on identical dates.

**Costs are benign**, degrading mean Sharpe by 0.11 across the whole 0→20 bps range.

The pre-committed base configuration (12m / 3 holdings / monthly / 5 bps) scores **0.549**, against a grid median of 0.360. The best cell (Sharpe 0.601) is reported for completeness, not as a recommendation — selecting it would be exactly the overfitting this section exists to detect, and it is a zero-cost cell at the highest turnover setting in the grid.

### Performance by market regime

| Regime | % of sample | Strategy ann. return | SPY ann. return | Excess | Strategy Sharpe | SPY Sharpe |
|---|---|---|---|---|---|---|
| Benchmark up | 75.9% | 17.23% | 23.95% | **-6.71%** | 1.268 | 1.672 |
| Benchmark down | 21.3% | -8.19% | -21.33% | **+13.14%** | -0.348 | -0.531 |
| Volatility: low | 32.9% | 4.80% | 10.80% | -6.00% | 0.468 | 1.054 |
| Volatility: mid | 32.9% | 12.85% | 13.69% | -0.84% | 0.908 | 0.961 |
| Volatility: high | 32.9% | 14.51% | 10.14% | **+4.36%** | 0.851 | 0.474 |

This is the clearest table in the report. The strategy is a **defensive rotation**: it gives up 6.7% annually in rising markets (76% of the sample) and gains 13.1% annually in falling ones (21%). Same pattern across volatility terciles — it loses in calm markets and wins in stressed ones.

That is a coherent economic story rather than a statistical accident, and it explains every other number: the low beta, the halved drawdown, the below-index CAGR, and the negative information ratio.

### Calendar years

The strategy beat SPY in only **8 of 18 years**.

Best relative years: 2008 (+27.7pp), 2022 (+9.0pp), 2025 (+7.7pp) — all stress years.
Worst: 2016 (-18.6pp), 2021 (-16.8pp), 2023 (-10.5pp), 2018 (-9.4pp).

Its worst absolute year was 2018 (-13.95% while SPY lost only 4.57%) — a year with a sharp Q4 reversal, precisely the momentum failure mode.

### How much of the edge is one year?

Each row deletes one calendar year and recomputes the Sharpe advantage over SPY from the remaining data.

| Year removed | Edge without it | Change vs full sample (+0.111) |
|---|---|---|
| **2008** | **-0.015** | **-0.139** |
| 2020 | +0.091 | -0.033 |
| 2025 | +0.094 | -0.030 |
| 2009 | +0.105 | -0.019 |
| 2019 | +0.109 | -0.015 |
| … | … | … |
| 2016 | +0.197 | +0.072 |

**Removing 2008 eliminates the edge entirely.** No other year moves it by more than 0.07, and excluding 2008 the advantage does not merely vanish — it turns negative, at **-0.029**.

The analytic Sharpe standard error for this sample length is ±0.27, but that is the error on a *single* Sharpe, not on the difference between two correlated ones — the wrong yardstick for the claim being made. The paired bootstrap above gives the right one: **0.178**, putting the gap at 0.62 standard errors. Either way: p = 0.530.

## 9. Failure analysis

**1. The edge is one crisis.** Stated above and worth restating: excluding 2008, there is no risk-adjusted edge over SPY. The strategy has not been shown to beat the index; it has been shown to have survived one crisis better than the index did. Those are different claims and only the second is supported.

**2. The result is not statistically significant, by either test.** The bootstrap 95% interval on the +0.111 advantage is [−0.258, +0.439] (p = 0.530), and the Deflated Sharpe Ratio against the 192-cell search hurdle is 0.700 — short of 0.95. Nothing here establishes that the strategy beats the benchmark.

**3. It earned less money than the index.** 10.78% vs 11.65% annualised. Only attractive to an investor who can lever it back to the index's risk level, or who values the shallower drawdown for its own sake.

**4. It lost to SPY in 10 of 18 calendar years,** including four years by more than 9 percentage points.

**5. The lookback parameter is unstable.** The 9-month result (0.276) sits well below the 3-month one (0.405) with no monotone structure, which is a warning sign that the formation window is fitting period-specific behaviour.

**6. What did *not* fail:** costs (only 0.015 of Sharpe), execution delay (0.09 at T+6), the ranking versus 300 random rotations (the strategy outranks every one, p < 0.005), and the in-sample/out-of-sample edge. The strategy's problems are not implementation artefacts — the implementation is sound *under the conventions stated in sections 5 and 6*, and the honest reading of a sound implementation is that the effect is small and crisis-dependent.

**Structural weaknesses of the design, independent of this sample:**

- **Momentum crashes.** The portfolio holds whatever recently rose, so it is maximally exposed when leadership reverses. Losses are sudden, not gradual. 2018 shows this directly.
- **Low breadth.** Three positions from nine assets caps the achievable information ratio regardless of signal quality.
- **Rebalance-date luck.** Monthly rebalancing on the last trading day is arbitrary; overlapping portfolios started on different days would average it out.
- **Cash is not a holding.** The rank is purely relative, so the strategy is always fully invested in the top 3 even when every asset is falling. An absolute-momentum filter is the obvious fix.

## 10. Limitations and next experiments

### Limitations

- **Nine assets is a very small cross-section.** Idiosyncratic moves in one ETF dominate the portfolio in a way they would not in a broad universe.
- **One sample, one market.** ~18 years containing two major drawdowns. For a monthly strategy the effective sample is closer to 180 independent observations than to 4,502 daily rows.
- **Survivorship bias, uncorrected.**
- **Costs are optimistic** — no drift, no impact, static spreads, no taxes.
- **Free retail data.** A professional result would be reproduced against a second vendor.
- **Regime classification is descriptive**, computed from trailing benchmark windows. It is attribution, not a tradeable rule.

### Next experiments, in order of expected value

1. **Volatility targeting.** Scale exposure to constant ex-ante volatility. The best-documented improvement to base momentum, and it targets this strategy's exact weakness — losses cluster in high-volatility reversals.
2. **Absolute-momentum overlay.** Require a positive trailing return, not merely a top-3 rank; hold cash otherwise. Directly addresses the 2018-style failure where the "best" asset is still falling.
3. **Risk-parity weighting.** Equal-weighting GLD and QQQ assigns equal capital and wildly unequal risk.
4. **Widen the universe.** Breadth is the binding constraint on this design.
5. **Walk-forward validation.** Re-select parameters on a rolling past-only window and evaluate forward, repeatedly. The sensitivity grid shows the lookback dimension is unstable; walk-forward tests the consequence directly, by asking whether re-optimising on information available at the time would have helped. Given the instability the expected answer is no — worth demonstrating rather than assuming. This is now the largest remaining gap.
6. **Weight drift and order-level execution.**
7. **Cross-check against VectorBT.** Agreement is evidence the accounting is right; disagreement localises a bug.

---

## Reproducibility

```bash
python -m quant_platform.run --config configs/momentum.yaml
```

Regenerates every figure, table, and CSV in `reports/momentum/`. The configuration file is the complete description of the run. 197 tests (`python -m pytest`) cover the engine accounting, every metric against an analytically known answer, look-ahead perturbation tests, and end-to-end determinism.

All numbers in this report are produced by that command. The leave-one-year-out table is `leave_one_year_out.csv`. The 2010-onwards figure is re-derivable from `strategy_timeseries.csv`, which ships both the strategy and the benchmark return series precisely so that sub-period claims can be checked rather than taken on trust.
