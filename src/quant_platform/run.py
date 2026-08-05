"""
Reproducible entry point.
=========================

One command regenerates every artefact in ``reports/``::

    python -m quant_platform.run --config configs/momentum.yaml

Nothing about the run is decided here: every parameter comes from the YAML, so
the config file plus this module is a complete description of the result.
"""

from __future__ import annotations

import argparse
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from quant_platform import __version__, metrics, reporting, validation
from quant_platform.data import load_prices_with_provenance, validate_price_panel
from quant_platform.portfolio import run_backtest
from quant_platform.returns import simple_returns
from quant_platform.signals import build_target_weights

logger = logging.getLogger("quant_platform.run")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict:
    """Read the YAML config and check that required keys are present."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))

    required = {
        "data": ["symbols", "start", "end"],
        "strategy": ["lookback_days", "skip_days", "holdings", "rebalance"],
        "portfolio": ["initial_capital", "commission_bps", "slippage_bps"],
        "validation": ["training_end", "validation_end", "test_end"],
        "reporting": ["output_dir"],
    }
    for section, keys in required.items():
        if section not in cfg:
            raise KeyError(f"Config is missing the '{section}' section")
        missing = [k for k in keys if k not in cfg[section]]
        if missing:
            raise KeyError(f"Config section '{section}' is missing: {missing}")

    cfg["portfolio"].setdefault("execution_lag", 1)
    cfg["portfolio"].setdefault("max_weight", cfg["strategy"].get("max_weight"))
    cfg["strategy"].setdefault("max_weight", cfg["portfolio"].get("max_weight"))
    cfg["reporting"].setdefault("risk_free_rate", 0.0)
    cfg["reporting"].setdefault("periods_per_year", 252)
    cfg["data"].setdefault("benchmark", "SPY")
    cfg["data"].setdefault("source", "yfinance")
    # Default off: a config that omits the key should fail on a dead data source
    # rather than silently substituting simulated prices.
    cfg["data"].setdefault("allow_synthetic_fallback", False)
    cfg.setdefault("seed", 42)
    return cfg


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("peewee").setLevel(logging.WARNING)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m quant_platform.run",
        description="Run the cross-sectional ETF momentum research pipeline end to end.",
    )
    parser.add_argument("--config", default="configs/momentum.yaml", help="path to the YAML config")
    parser.add_argument("--output-dir", default=None, help="override reporting.output_dir")
    parser.add_argument("--no-cache", action="store_true", help="force a fresh download")
    parser.add_argument("--skip-sensitivity", action="store_true", help="skip the parameter grid")
    parser.add_argument("--show-test", action="store_true",
                        help="include the untouched test split in the report")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)
    started = datetime.now(timezone.utc)
    cfg = load_config(args.config)

    np.random.seed(cfg["seed"])  # global seed; every stochastic path also seeds locally

    out_dir = Path(args.output_dir or cfg["reporting"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    reporting.apply_style()

    d, s, p, v, rep = cfg["data"], cfg["strategy"], cfg["portfolio"], cfg["validation"], cfg["reporting"]
    ppy, rf = rep["periods_per_year"], rep["risk_free_rate"]
    run_warnings: list[str] = []

    # ---- 1. Data ---------------------------------------------------------- #
    logger.info("=" * 68)
    logger.info("quant_platform %s  |  config: %s", __version__, args.config)
    logger.info("=" * 68)

    # Provenance comes back from the same call that produced the data, so the
    # two cannot disagree. Looking the sidecar up separately by cache key could
    # silently return None, and a report with missing provenance is
    # indistinguishable from one with clean provenance.
    long, provenance = load_prices_with_provenance(
        symbols=d["symbols"],
        start=d["start"],
        end=d["end"],
        source=d["source"],
        use_cache=not args.no_cache,
        allow_synthetic_fallback=d.get("allow_synthetic_fallback", False),
        seed=cfg["seed"],
        wide=False,
    )
    report = validate_price_panel(long, raise_on_error=True)

    if provenance is not None and provenance.source == "synthetic":
        run_warnings.append(
            "This run used SYNTHETIC data, not real market prices. Every number below "
            "is a test of the software, not a research finding. Re-run with network "
            "access to reproduce against real ETF history."
        )
    for w in report["warnings"][:6]:
        logger.info("data note: %s", w)

    from quant_platform.data import to_wide

    prices = to_wide(long, "adjusted_close")

    # Restrict to the window where every symbol has data, so the cross-sectional
    # ranking always compares the same number of assets.
    common_start = long.groupby("symbol")["date"].min().max()
    prices = prices.loc[prices.index >= common_start]
    prices = prices.dropna(axis=0, how="any")
    logger.info(
        "Panel: %d symbols x %d days (%s -> %s)",
        prices.shape[1], prices.shape[0], prices.index[0].date(), prices.index[-1].date(),
    )

    asset_returns = simple_returns(prices)

    # ---- 2. Signal -> target weights -------------------------------------- #
    target_weights = build_target_weights(
        prices,
        lookback=s["lookback_days"],
        skip=s["skip_days"],
        holdings=s["holdings"],
        rebalance=s["rebalance"],
        max_weight=s.get("max_weight"),
    )
    # Explicit raises, not asserts: these are the constraints the strategy is
    # claimed to satisfy, and `python -O` strips assert statements entirely.
    # A guarantee that disappears under a common interpreter flag is not one.
    if not (target_weights.abs().sum(axis=1) <= 1.000001).all():
        worst = target_weights.abs().sum(axis=1).max()
        raise ValueError(f"weights exceed 100% of capital (max gross exposure {worst:.6f})")
    if not (target_weights >= -1e-12).all().all():
        raise ValueError("long-only strategy produced a negative weight")

    # ---- 3. Backtest ------------------------------------------------------ #
    strategy = run_backtest(
        returns=asset_returns,
        target_weights=target_weights,
        initial_capital=p["initial_capital"],
        commission_bps=p["commission_bps"],
        slippage_bps=p["slippage_bps"],
        execution_lag=p["execution_lag"],
        periods_per_year=ppy,
        name=f"Momentum (top {s['holdings']})",
    )
    logger.info("Strategy: %r", strategy)

    # ---- 4. Baselines ----------------------------------------------------- #
    baselines = validation.build_baselines(
        prices,
        asset_returns,
        benchmark_symbol=d["benchmark"],
        lookback=s["lookback_days"],
        skip=s["skip_days"],
        holdings=s["holdings"],
        rebalance=s["rebalance"],
        max_weight=s.get("max_weight"),
        initial_capital=p["initial_capital"],
        commission_bps=p["commission_bps"],
        slippage_bps=p["slippage_bps"],
        execution_lag=p["execution_lag"],
        seed=cfg["seed"],
    )
    all_results = validation.align_results({strategy.name: strategy, **baselines})
    strategy = all_results[strategy.name]
    benchmark_returns = all_results["SPY buy & hold"].net_returns

    # ---- 5. Metrics ------------------------------------------------------- #
    comparison = pd.DataFrame(
        {
            name: metrics.summary_metrics(
                r.net_returns, equity=r.equity,
                benchmark=benchmark_returns if name != "SPY buy & hold" else None,
                turnover=r.turnover, periods_per_year=ppy, risk_free_rate=rf, name=name,
            )
            for name, r in all_results.items()
        }
    )

    splits = validation.make_splits(
        strategy.index, v["training_end"], v["validation_end"], v["test_end"]
    )
    logger.info("Sample splits:")
    for sp in splits:
        logger.info("  %s", sp)

    split_metrics = validation.evaluate_splits(
        strategy, splits, benchmark=benchmark_returns, periods_per_year=ppy, risk_free_rate=rf
    )
    bench_splits = validation.evaluate_splits(
        all_results["SPY buy & hold"], splits, periods_per_year=ppy, risk_free_rate=rf
    )

    regimes = validation.regime_analysis(strategy.net_returns, benchmark_returns, ppy)
    years = validation.calendar_year_table(strategy.net_returns, benchmark_returns)
    loyo = validation.leave_one_year_out(strategy.net_returns, benchmark_returns, ppy, rf)

    # ---- 6. Sensitivity (training window only) ---------------------------- #
    sensitivity = pd.DataFrame()
    marginals: dict[str, pd.DataFrame] = {}
    if not args.skip_sensitivity:
        sens_cfg = cfg.get("sensitivity", {})
        train = splits[0]
        logger.info("Sensitivity grid on the TRAINING window only (%s -> %s)",
                    train.start.date(), train.end.date())
        sensitivity = validation.run_sensitivity(
            prices,
            asset_returns,
            lookback_months=sens_cfg.get("lookback_months", [3, 6, 9, 12]),
            holdings_grid=sens_cfg.get("holdings", [1, 2, 3, 4]),
            rebalance_grid=sens_cfg.get("rebalance", ["weekly", "monthly", "quarterly"]),
            cost_grid=sens_cfg.get("cost_bps", [0, 5, 10, 20]),
            skip=s["skip_days"],
            max_weight=s.get("max_weight"),
            initial_capital=p["initial_capital"],
            execution_lag=p["execution_lag"],
            eval_start=train.start,
            eval_end=train.end,
            periods_per_year=ppy,
            risk_free_rate=rf,
        )
        marginals = validation.sensitivity_marginals(sensitivity, "sharpe")
        logger.info("Sensitivity grid: %d combinations evaluated", len(sensitivity))

    # ---- 6b. Statistical significance -------------------------------------- #
    # Runs after the grid because deflation needs to know how large the search
    # was: the hurdle a result must clear depends on how many were tried.
    sig_cfg = cfg.get("significance", {})
    sig_results, sig_table = _run_significance(
        strategy.net_returns, benchmark_returns, sensitivity,
        train_returns=strategy.slice(splits[0].start, splits[0].end).net_returns,
        n_boot=sig_cfg.get("n_boot", 5000),
        block_length=sig_cfg.get("block_length"),
        confidence=sig_cfg.get("confidence", 0.95),
        periods_per_year=ppy,
        risk_free_rate=rf,
        seed=cfg.get("seed", 42),
    )

    # ---- 7. Figures ------------------------------------------------------- #
    logger.info("Rendering figures -> %s", out_dir)
    equity_curves = {name: r.equity for name, r in all_results.items()}
    headline = {strategy.name: strategy.net_returns, "SPY buy & hold": benchmark_returns}

    figures = {
        "equity": reporting.plot_equity_curves(
            equity_curves, out_dir / "equity_curve.png", splits=splits),
        "drawdown": reporting.plot_drawdown(
            {k: equity_curves[k] for k in list(equity_curves)[:3]},
            out_dir / "drawdown.png", splits=splits),
        "rolling_sharpe": reporting.plot_rolling_sharpe(
            headline, out_dir / "rolling_sharpe.png", 252, ppy, splits=splits),
        "rolling_vol": reporting.plot_rolling_volatility(
            headline, out_dir / "rolling_volatility.png", 63, ppy, splits=splits),
        "turnover": reporting.plot_turnover(
            strategy.turnover, out_dir / "turnover.png", strategy.costs, splits=splits),
        "exposure": reporting.plot_exposure(
            strategy.exposure, strategy.executed_weights, out_dir / "exposure.png", splits=splits),
        "monthly": reporting.plot_monthly_heatmap(
            strategy.net_returns, out_dir / "monthly_returns.png",
            f"Monthly returns (%) -- {strategy.name}"),
        "splits": reporting.plot_split_bars(split_metrics, out_dir / "in_vs_out_sample.png"),
    }
    if not sensitivity.empty:
        figures["sensitivity"] = reporting.plot_sensitivity_heatmap(
            sensitivity, out_dir / "sensitivity.png", "sharpe", "lookback_months", "holdings")
    if sig_results.get("bootstrap") is not None:
        bs = sig_results["bootstrap"]
        figures["bootstrap"] = reporting.plot_bootstrap_distribution(
            bs["draws"], bs["difference"], bs["ci_low"], bs["ci_high"],
            out_dir / "bootstrap_sharpe.png", p_value=bs["p_value"])

    # ---- 8. CSV artefacts ------------------------------------------------- #
    comparison.to_csv(out_dir / "metrics.csv")
    split_metrics.to_csv(out_dir / "in_vs_out_of_sample.csv")
    bench_splits.to_csv(out_dir / "benchmark_splits.csv")
    if not regimes.empty:
        regimes.to_csv(out_dir / "regimes.csv")
    if not years.empty:
        years.to_csv(out_dir / "calendar_years.csv")
    if not loyo.empty:
        loyo.to_csv(out_dir / "leave_one_year_out.csv")
    if not sensitivity.empty:
        sensitivity.to_csv(out_dir / "sensitivity.csv", index=False)
    if not sig_table.empty:
        sig_table.to_csv(out_dir / "significance.csv", index=False)
    # The benchmark series ships alongside the strategy so that any sub-period
    # claim in the report (e.g. "the edge over 2010-2025 is negative") can be
    # re-derived from this one file, rather than having to be taken on trust.
    pd.DataFrame(
        {"equity": strategy.equity, "net_return": strategy.net_returns,
         "gross_return": strategy.gross_returns, "turnover": strategy.turnover,
         "cost": strategy.costs, "exposure": strategy.exposure,
         "benchmark_net_return": benchmark_returns,
         "benchmark_equity": all_results["SPY buy & hold"].equity}
    ).to_csv(out_dir / "strategy_timeseries.csv")

    # ---- 9. Report -------------------------------------------------------- #
    from quant_platform.narrative import build_kpis, build_narrative

    narrative = build_narrative(
        cfg, report, provenance, comparison, split_metrics, bench_splits,
        sensitivity, marginals, regimes, strategy, splits, show_test=args.show_test,
        leave_one_year_out=loyo, significance=sig_results,
    )
    kpis = build_kpis(comparison, strategy.name, split_metrics, show_test=args.show_test)

    tables = {
        "coverage": report["coverage"].assign(
            first=lambda x: x["first"].dt.date.astype(str),
            last=lambda x: x["last"].dt.date.astype(str),
        ),
        "baselines": comparison,
        "splits": split_metrics if args.show_test else split_metrics.drop(columns=["test"], errors="ignore"),
        "regimes": regimes.round(4),
        "years": years.round(4),
        "loyo": loyo.round(4),
        "significance": sig_table,
        **{f"sens_{k}": _label_marginal(df, k) for k, df in marginals.items()},
    }

    report_path = reporting.build_html_report(
        title="Cross-sectional momentum on liquid US ETFs",
        subtitle=(
            "A reproducible test of whether ranking nine liquid ETFs by trailing "
            "12-1 month return produces an investable edge after costs."
        ),
        config=cfg,
        provenance=provenance,
        figures=figures,
        tables=tables,
        narrative=narrative,
        kpis=kpis,
        warnings_=run_warnings,
        output_path=out_dir / "research_report.html",
    )

    # ---- 10. Console summary --------------------------------------------- #
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    _print_summary(
        comparison, split_metrics, strategy.name, out_dir, report_path,
        elapsed, run_warnings, show_test=args.show_test,
    )
    return 0


def _run_significance(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    sensitivity: pd.DataFrame,
    train_returns: pd.Series,
    n_boot: int,
    block_length: float | None,
    confidence: float,
    periods_per_year: int,
    risk_free_rate: float,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    """Bootstrap the Sharpe advantage and deflate it for the size of the search.

    Returns the raw results (including the bootstrap draws, for plotting) and a
    tidy one-row-per-quantity table for the CSV.
    """
    from quant_platform import significance as sig

    results: dict = {"bootstrap": None, "deflation_full": None, "deflation_train": None}
    rows: list[dict] = []

    def add(metric: str, value, note: str = "") -> None:
        rows.append({"metric": metric, "value": value, "note": note})

    try:
        bs = sig.bootstrap_sharpe_difference(
            strategy_returns, benchmark_returns, n_boot=n_boot,
            block_length=block_length, confidence=confidence,
            periods_per_year=periods_per_year, risk_free_rate=risk_free_rate, seed=seed,
        )
        results["bootstrap"] = bs
        pct = int(round(confidence * 100))
        add("strategy_sharpe", round(bs["strategy_sharpe"], 6))
        add("benchmark_sharpe", round(bs["benchmark_sharpe"], 6))
        add("sharpe_difference", round(bs["difference"], 6), "strategy minus benchmark")
        add(f"ci_low_{pct}", round(bs["ci_low"], 6), "stationary block bootstrap")
        add(f"ci_high_{pct}", round(bs["ci_high"], 6), "stationary block bootstrap")
        add("bootstrap_se", round(bs["bootstrap_se"], 6), "SE of the difference, paired resampling")
        add("p_value", round(bs["p_value"], 6), "two-sided, null of no difference")
        add("block_length_days", bs["block_length"], "mean geometric block length")
        add("n_resamples", bs["n_boot"])
        # The default block length is a rate (n^(1/3)), not a data-driven
        # estimator, so the honest thing is to show the conclusion does not
        # depend on it rather than to assert that it does not.
        for L in (1, 5, 10, 25, 40, 60, 90):
            alt = sig.bootstrap_sharpe_difference(
                strategy_returns, benchmark_returns, n_boot=max(1000, n_boot // 5),
                block_length=L, confidence=confidence,
                periods_per_year=periods_per_year, risk_free_rate=risk_free_rate, seed=seed,
            )
            rows.append({
                "metric": f"p_value_block_{L}", "value": round(alt["p_value"], 6),
                "note": f"block-length robustness; CI [{alt['ci_low']:+.3f}, {alt['ci_high']:+.3f}]",
            })
    except (ValueError, KeyError) as exc:
        logger.warning("Bootstrap skipped: %s", exc)

    if not sensitivity.empty and "sharpe" in sensitivity:
        trials = sensitivity["sharpe"].dropna()
        try:
            full = sig.deflated_sharpe_ratio(
                strategy_returns, trial_sharpes=trials,
                periods_per_year=periods_per_year, risk_free_rate=risk_free_rate)
            train = sig.deflated_sharpe_ratio(
                train_returns, trial_sharpes=trials,
                periods_per_year=periods_per_year, risk_free_rate=risk_free_rate)
            results["deflation_full"] = full
            results["deflation_train"] = train

            add("n_trials", full["n_trials"], "sensitivity grid cells")
            add("noise_hurdle_sharpe", round(full["noise_hurdle_sharpe"], 6),
                "expected best Sharpe from this many trials of pure noise")
            add("best_grid_sharpe", round(float(trials.max()), 6), "training window")
            add("psr_vs_zero_full", round(full["psr_vs_zero"], 6))
            add("deflated_sharpe_full", round(full["deflated_sharpe"], 6))
            add("psr_vs_zero_training", round(train["psr_vs_zero"], 6))
            add("deflated_sharpe_training", round(train["deflated_sharpe"], 6),
                "same window as the grid; the like-for-like comparison")
        except ValueError as exc:
            logger.warning("Deflation skipped: %s", exc)

    return results, pd.DataFrame(rows)


def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a metrics table to float for display.

    ``summary_metrics`` deliberately mixes types -- start/end dates are strings
    and ``still_underwater`` is a bool, both of which belong in the CSV. That
    makes the assembled frame ``object`` dtype, and ``DataFrame.round()`` is a
    silent no-op on object columns: the console would print full float precision
    while appearing to be rounded.
    """
    return df.apply(pd.to_numeric, errors="coerce")


def _label_marginal(df: pd.DataFrame, dim: str) -> pd.DataFrame:
    out = df.copy()
    out.index = [f"{dim.replace('_', ' ')} = {i}" for i in out.index]
    return out


def _print_summary(comparison, split_metrics, strategy_name, out_dir, report_path,
                   elapsed, warns, show_test: bool = False):
    line = "=" * 74
    print("\n" + line)
    print("RESULTS".center(74))
    print(line)

    rows = ["cagr", "ann_volatility", "sharpe", "sortino", "max_drawdown", "calmar", "annual_turnover"]
    show = _numeric(comparison.loc[[r for r in rows if r in comparison.index]])
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(show.round(3).to_string())

    if not split_metrics.empty:
        print("\n" + "-" * 74)
        print(f"IN-SAMPLE vs OUT-OF-SAMPLE  ({strategy_name})")
        print("-" * 74)
        keep = [r for r in ["cagr", "ann_volatility", "sharpe", "max_drawdown", "calmar"]
                if r in split_metrics.index]
        # The test split is withheld by default. Printing it here would defeat the
        # sample discipline the report enforces: a number you have seen cannot be
        # un-seen, and every glance makes the next strategy revision less honest.
        display = split_metrics if show_test else split_metrics.drop(columns=["test"], errors="ignore")
        print(_numeric(display.loc[keep]).round(3).to_string())
        if not show_test and "test" in split_metrics.columns:
            print("\n  [test split withheld -- computed and saved to in_vs_out_of_sample.csv;"
                  "\n   pass --show-test to display it once the strategy is final]")

    if warns:
        print("\n" + "!" * 74)
        for w in warns:
            print("! " + w)
        print("!" * 74)

    print(f"\nArtefacts written to: {out_dir.resolve()}")
    print(f"Report:               {report_path.resolve()}")
    print(f"Completed in {elapsed:.1f}s  "
          f"(python {platform.python_version()}, pandas {pd.__version__}, numpy {np.__version__})")
    print(line + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
