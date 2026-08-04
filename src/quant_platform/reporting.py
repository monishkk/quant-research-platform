"""
Charts and the HTML research report.
====================================

All figures are written as PNG *and* embedded as base64 in a single
self-contained ``research_report.html``, so the report can be emailed or hosted
without carrying an asset folder.

Chart conventions
-----------------
* Equity curves are plotted on a **log** y-axis. On a linear axis a 15-year
  compounding series makes every early drawdown invisible; log scale makes equal
  *percentage* moves equal *visual* moves, which is what matters when judging
  risk.
* Every strategy in a comparison starts at the same capital on the same date.
* Out-of-sample boundaries are drawn on the time-series charts, because "where
  does the in-sample period end" is the first question worth asking of any
  equity curve.
"""

from __future__ import annotations

import base64
import html
import logging
from io import BytesIO
from pathlib import Path

import matplotlib


def _ensure_non_gui_backend() -> None:
    """Select a file-writing backend for headless runs, without breaking Jupyter.

    An unconditional ``matplotlib.use("Agg")`` here would override the inline
    backend that IPython installs, and every plot in every notebook would
    silently render nothing -- no error, just missing figures. So only switch
    when the active backend is not already an interactive/inline one.
    """
    current = matplotlib.get_backend().lower()
    interactive = ("inline", "ipympl", "widget", "nbagg", "notebook")
    if any(token in current for token in interactive):
        return
    matplotlib.use("Agg")


_ensure_non_gui_backend()

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from quant_platform import metrics

logger = logging.getLogger(__name__)

__all__ = [
    "apply_style",
    "plot_equity_curves",
    "plot_drawdown",
    "plot_rolling_sharpe",
    "plot_rolling_volatility",
    "plot_turnover",
    "plot_exposure",
    "plot_monthly_heatmap",
    "plot_sensitivity_heatmap",
    "plot_split_bars",
    "build_html_report",
]

# A restrained palette: one strong colour for the strategy, greys/blues for
# comparators, so the eye goes to the right line without a legend lookup.
PALETTE = {
    "strategy": "#0B5FFF",
    "benchmark": "#D1495B",
    "neutral": "#6C757D",
    "accent": "#00897B",
    "muted": "#ADB5BD",
    "warn": "#E8A33D",
    "grid": "#E4E7EB",
    "text": "#1F2933",
}
SERIES_COLORS = [
    PALETTE["strategy"],
    PALETTE["benchmark"],
    PALETTE["accent"],
    PALETTE["warn"],
    PALETTE["neutral"],
    "#8E44AD",
    "#2E7D32",
]


def apply_style() -> None:
    """Set global matplotlib defaults once, so every figure matches."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 140,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": PALETTE["grid"],
            "axes.labelcolor": PALETTE["text"],
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.color": PALETTE["text"],
            "ytick.color": PALETTE["text"],
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.size": 10,
            "lines.linewidth": 1.6,
        }
    )


def _save(fig: plt.Figure, path: Path | None) -> str:
    """Write the figure to disk (if a path is given) and return a base64 PNG."""
    buf = BytesIO()
    fig.savefig(buf, format="png")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _mark_splits(ax, splits, y_text: float = 0.02) -> None:
    """Draw vertical boundaries between training / validation / test."""
    if not splits:
        return
    for sp in splits:
        if sp.name in ("validation", "test"):
            ax.axvline(sp.start, color=PALETTE["neutral"], linestyle="--", linewidth=1.0, alpha=0.8)
            ax.text(
                sp.start,
                y_text,
                f" {sp.name}",
                transform=ax.get_xaxis_transform(),
                fontsize=8,
                color=PALETTE["neutral"],
                rotation=90,
                va="bottom",
                ha="left",
            )


def _fmt_date_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


# --------------------------------------------------------------------------- #
# Time-series charts
# --------------------------------------------------------------------------- #
def plot_equity_curves(
    curves: dict[str, pd.Series],
    path: Path | None = None,
    title: str = "Equity curves (net of costs)",
    splits=None,
    log_scale: bool = True,
) -> str:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for i, (label, series) in enumerate(curves.items()):
        s = series.dropna()
        if s.empty:
            continue
        ax.plot(
            s.index,
            s.values,
            label=label,
            color=SERIES_COLORS[i % len(SERIES_COLORS)],
            linewidth=2.0 if i == 0 else 1.3,
            alpha=1.0 if i == 0 else 0.85,
        )
    if log_scale:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_title(title + ("  [log scale]" if log_scale else ""))
    ax.set_ylabel("Portfolio value")
    ax.legend(loc="upper left", ncol=2)
    _mark_splits(ax, splits)
    _fmt_date_axis(ax)
    return _save(fig, path)


def plot_drawdown(
    curves: dict[str, pd.Series],
    path: Path | None = None,
    title: str = "Drawdown from running peak",
    splits=None,
) -> str:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, (label, equity) in enumerate(curves.items()):
        dd = metrics.drawdown_series(equity.dropna())
        if dd.empty:
            continue
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        ax.plot(dd.index, dd.values * 100, label=label, color=color, linewidth=1.5 if i == 0 else 1.1)
        if i == 0:
            ax.fill_between(dd.index, dd.values * 100, 0, color=color, alpha=0.18)
    ax.set_title(title)
    ax.set_ylabel("Drawdown (%)")
    ax.axhline(0, color=PALETTE["text"], linewidth=0.8)
    ax.legend(loc="lower left", ncol=2)
    _mark_splits(ax, splits)
    _fmt_date_axis(ax)
    return _save(fig, path)


def plot_rolling_sharpe(
    returns: dict[str, pd.Series],
    path: Path | None = None,
    window: int = 252,
    periods_per_year: int = 252,
    splits=None,
) -> str:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, (label, r) in enumerate(returns.items()):
        rs = metrics.rolling_sharpe(r.dropna(), window, periods_per_year)
        ax.plot(rs.index, rs.values, label=label, color=SERIES_COLORS[i % len(SERIES_COLORS)],
                linewidth=1.6 if i == 0 else 1.1)
    ax.axhline(0, color=PALETTE["text"], linewidth=0.8)
    ax.axhline(1, color=PALETTE["muted"], linewidth=0.8, linestyle=":")
    ax.set_title(f"Rolling {window}-day annualised Sharpe")
    ax.set_ylabel("Sharpe")
    ax.legend(loc="upper left", ncol=2)
    _mark_splits(ax, splits)
    _fmt_date_axis(ax)
    return _save(fig, path)


def plot_rolling_volatility(
    returns: dict[str, pd.Series],
    path: Path | None = None,
    window: int = 63,
    periods_per_year: int = 252,
    splits=None,
) -> str:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, (label, r) in enumerate(returns.items()):
        rv = r.dropna().rolling(window, min_periods=window // 2).std(ddof=1) * np.sqrt(
            periods_per_year
        )
        ax.plot(rv.index, rv.values * 100, label=label,
                color=SERIES_COLORS[i % len(SERIES_COLORS)], linewidth=1.5 if i == 0 else 1.1)
    ax.set_title(f"Rolling {window}-day annualised volatility")
    ax.set_ylabel("Volatility (%)")
    ax.legend(loc="upper left", ncol=2)
    _mark_splits(ax, splits)
    _fmt_date_axis(ax)
    return _save(fig, path)


def plot_turnover(
    turnover: pd.Series,
    path: Path | None = None,
    costs: pd.Series | None = None,
    splits=None,
) -> str:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 5.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    t = turnover.dropna()
    ax1.bar(t.index, t.values * 100, width=3.0, color=PALETTE["strategy"], alpha=0.75)
    ax1.set_title("Turnover per rebalance (two-way, % of NAV)")
    ax1.set_ylabel("Turnover (%)")
    _mark_splits(ax1, splits)

    if costs is not None:
        cum = costs.dropna().cumsum() * 100
        ax2.plot(cum.index, cum.values, color=PALETTE["benchmark"])
        ax2.fill_between(cum.index, cum.values, 0, color=PALETTE["benchmark"], alpha=0.15)
        ax2.set_ylabel("Cumulative\ncost (%)")
        ax2.set_title("Cumulative transaction-cost drag", fontsize=10)
    _fmt_date_axis(ax2)
    fig.align_ylabels()
    return _save(fig, path)


def plot_exposure(
    exposure: pd.Series,
    weights: pd.DataFrame | None = None,
    path: Path | None = None,
    splits=None,
) -> str:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6.0), sharex=True, gridspec_kw={"height_ratios": [1, 2]}
    )
    e = exposure.dropna()
    ax1.plot(e.index, e.values * 100, color=PALETTE["strategy"])
    ax1.fill_between(e.index, e.values * 100, 0, color=PALETTE["strategy"], alpha=0.15)
    ax1.set_ylabel("Invested (%)")
    ax1.set_title("Gross exposure (remainder held in cash)")
    ax1.set_ylim(0, max(105, float(e.max()) * 100 * 1.1))
    _mark_splits(ax1, splits)

    if weights is not None and not weights.empty:
        w = weights.loc[:, weights.abs().sum() > 0]
        ax2.stackplot(
            w.index,
            *[w[c].values * 100 for c in w.columns],
            labels=list(w.columns),
            colors=plt.cm.tab20(np.linspace(0, 1, len(w.columns))),
            alpha=0.9,
        )
        ax2.set_ylabel("Weight (%)")
        ax2.set_title("Holdings through time", fontsize=10)
        ax2.legend(loc="upper left", ncol=5, fontsize=8)
    _fmt_date_axis(ax2)
    fig.align_ylabels()
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# Tabular charts
# --------------------------------------------------------------------------- #
def plot_monthly_heatmap(
    returns: pd.Series,
    path: Path | None = None,
    title: str = "Monthly returns (%)",
) -> str:
    table = metrics.monthly_return_table(returns)
    if table.empty:
        return ""
    months = table.drop(columns=["Year"], errors="ignore")

    fig, ax = plt.subplots(figsize=(11, max(3.2, 0.34 * len(months) + 1.6)))
    values = months.values.astype(float) * 100
    finite = values[np.isfinite(values)]
    lim = max(abs(np.nanpercentile(finite, 2)), abs(np.nanpercentile(finite, 98)), 1.0)

    im = ax.imshow(
        values,
        cmap="RdYlGn",
        norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim),
        aspect="auto",
    )
    ax.set_xticks(range(len(months.columns)), months.columns)
    ax.set_yticks(range(len(months.index)), months.index)
    ax.set_title(title)
    ax.grid(False)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                        color="#1F2933")
    fig.colorbar(im, ax=ax, shrink=0.7, label="%")
    return _save(fig, path)


def plot_sensitivity_heatmap(
    sensitivity: pd.DataFrame,
    path: Path | None = None,
    metric: str = "sharpe",
    index: str = "lookback_months",
    columns: str = "holdings",
    title: str | None = None,
) -> str:
    """Average ``metric`` over the remaining grid dimensions and plot the surface.

    Averaging (rather than showing the max) is deliberate: the maximum over a
    grid is an upward-biased estimate of what a parameter choice would deliver
    out of sample. The mean shows whether a *region* works.
    """
    if sensitivity.empty or metric not in sensitivity.columns:
        return ""
    pivot = sensitivity.pivot_table(index=index, columns=columns, values=metric, aggfunc="mean")

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    values = pivot.values.astype(float)
    lim = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
    im = ax.imshow(values, cmap="RdYlGn", norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim),
                   aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xlabel(columns.replace("_", " "))
    ax.set_ylabel(index.replace("_", " "))
    ax.set_title(title or f"Mean {metric} by {index} x {columns} (training window)")
    ax.grid(False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label=metric)
    return _save(fig, path)


def plot_bootstrap_distribution(
    draws,
    observed: float,
    ci_low: float,
    ci_high: float,
    path: Path | None = None,
    title: str = "Bootstrap distribution of the Sharpe advantage over the benchmark",
    p_value: float | None = None,
) -> str:
    """Where the observed edge sits inside its own sampling distribution.

    The single most honest chart in the report: if the zero line sits comfortably
    inside the mass, the edge is not distinguishable from luck, and that is
    visible at a glance in a way a p-value in a table is not.
    """
    import numpy as np

    fig, ax = plt.subplots(figsize=(11, 4.6))
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]

    # Shared bin edges for both layers. Letting each call choose its own bins
    # over its own range misaligns the overlay, so the highlighted interval
    # appears ragged instead of nesting exactly inside the outer distribution.
    _, edges = np.histogram(draws, bins=60)
    ax.hist(draws, bins=edges, color=SERIES_COLORS[0], alpha=0.45,
            edgecolor="none", label="Bootstrap resamples")

    inside = (draws >= ci_low) & (draws <= ci_high)
    ax.hist(draws[inside], bins=edges, color=SERIES_COLORS[0], alpha=0.90,
            edgecolor="none", label="95% confidence interval")

    ax.axvline(0.0, color=PALETTE.get("bad", "#c0392b"), linewidth=2.0,
               label="No advantage (zero)")
    ax.axvline(observed, color=PALETTE.get("good", "#1e8449"), linewidth=2.0,
               linestyle="--", label=f"Observed ({observed:+.3f})")

    ax.set_xlabel("Sharpe ratio difference (strategy - benchmark)")
    ax.set_ylabel("Resamples")
    subtitle = f"   [95% CI {ci_low:+.3f} to {ci_high:+.3f}"
    subtitle += f", p = {p_value:.3f}]" if p_value is not None else "]"
    ax.set_title(title + subtitle)
    ax.legend(loc="upper right", fontsize=8)
    return _save(fig, path)


def plot_split_bars(
    split_metrics: pd.DataFrame,
    path: Path | None = None,
    rows=("sharpe", "cagr", "max_drawdown"),
) -> str:
    """In-sample vs out-of-sample bars -- the headline honesty chart."""
    cols = [c for c in ("training", "validation", "test") if c in split_metrics.columns]
    if not cols:
        return ""

    fig, axes = plt.subplots(1, len(rows), figsize=(3.7 * len(rows), 3.6))
    axes = np.atleast_1d(axes)
    colors = [PALETTE["strategy"], PALETTE["accent"], PALETTE["warn"]]

    for ax, row in zip(axes, rows):
        if row not in split_metrics.index:
            continue
        vals = [float(split_metrics.loc[row, c]) for c in cols]
        scale = 100 if row in ("cagr", "max_drawdown") else 1
        bars = ax.bar(cols, [v * scale for v in vals], color=colors[: len(cols)], alpha=0.9)
        ax.set_title(row.replace("_", " "), fontsize=11, fontweight="bold")
        ax.axhline(0, color=PALETTE["text"], linewidth=0.8)
        ax.grid(axis="x", visible=False)
        for bar, v in zip(bars, vals):
            ax.annotate(
                f"{v * scale:.2f}" + ("%" if scale == 100 else ""),
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom" if bar.get_height() >= 0 else "top",
                fontsize=9,
            )
    fig.suptitle("In-sample vs out-of-sample", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #
_CSS = """
:root { --ink:#1F2933; --muted:#6C757D; --line:#E4E7EB; --accent:#0B5FFF;
        --warnbg:#FFF6E5; --warnln:#E8A33D; --okbg:#EAF7EE; --okln:#2E7D32; }
* { box-sizing: border-box; }
body { margin:0; padding:0; background:#F7F8FA; color:var(--ink);
       font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
       line-height:1.62; }
.wrap { max-width:1080px; margin:0 auto; padding:48px 28px 96px; background:#fff;
        box-shadow:0 1px 3px rgba(0,0,0,.06); }
h1 { font-size:2.0rem; margin:0 0 .2em; letter-spacing:-.02em; }
h2 { font-size:1.32rem; margin:2.4em 0 .6em; padding-bottom:.3em;
     border-bottom:2px solid var(--line); letter-spacing:-.01em; }
h3 { font-size:1.06rem; margin:1.8em 0 .5em; color:#334155; }
p, li { font-size:.95rem; }
.sub { color:var(--muted); font-size:.95rem; margin:0 0 1.6em; }
.meta { display:flex; flex-wrap:wrap; gap:10px; margin:1.2em 0 2em; }
.chip { background:#F1F3F5; border:1px solid var(--line); border-radius:999px;
        padding:5px 13px; font-size:.8rem; color:#495057; }
.chip b { color:var(--ink); font-weight:600; }
table { border-collapse:collapse; width:100%; margin:1em 0 1.6em; font-size:.85rem; }
th, td { padding:7px 11px; text-align:right; border-bottom:1px solid var(--line); }
th { background:#F8F9FA; font-weight:600; font-size:.78rem; text-transform:uppercase;
     letter-spacing:.04em; color:#495057; }
th:first-child, td:first-child { text-align:left; font-weight:500; }
tbody tr:hover { background:#FAFBFC; }
.fig { margin:1.6em 0 2.2em; }
.fig img { width:100%; height:auto; border:1px solid var(--line); border-radius:6px; }
.cap { font-size:.83rem; color:var(--muted); margin-top:.6em; }
.callout { border-left:4px solid var(--warnln); background:var(--warnbg);
           padding:14px 18px; margin:1.4em 0; border-radius:0 6px 6px 0; font-size:.9rem; }
.callout.ok { border-left-color:var(--okln); background:var(--okbg); }
.callout .t { font-weight:700; display:block; margin-bottom:.3em; }
.grid2 { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:24px; }
.kpi { border:1px solid var(--line); border-radius:8px; padding:14px 16px; }
.kpi .l { font-size:.72rem; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }
.kpi .v { font-size:1.5rem; font-weight:650; margin-top:.15em; letter-spacing:-.02em; }
.kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; margin:1.4em 0 2em; }
code { background:#F1F3F5; padding:1px 6px; border-radius:4px; font-size:.86em;
       font-family:"SF Mono",Menlo,Consolas,monospace; }
pre { background:#1F2933; color:#E4E7EB; padding:14px 18px; border-radius:8px;
      overflow-x:auto; font-size:.82rem; line-height:1.55; }
pre code { background:none; color:inherit; padding:0; }
.pos { color:#2E7D32; font-weight:600; } .neg { color:#C62828; font-weight:600; }
footer { margin-top:3.5em; padding-top:1.4em; border-top:1px solid var(--line);
         color:var(--muted); font-size:.82rem; }
.toc { background:#F8F9FA; border:1px solid var(--line); border-radius:8px; padding:16px 22px; }
.toc ol { margin:.4em 0; padding-left:1.4em; } .toc a { color:var(--accent); text-decoration:none; }
.toc a:hover { text-decoration:underline; }
.scroll { overflow-x:auto; }
@media (max-width:640px){ .wrap{padding:28px 16px 64px;} h1{font-size:1.5rem;} }
"""


def _img(b64: str, caption: str = "") -> str:
    if not b64:
        return ""
    cap = f'<div class="cap">{html.escape(caption)}</div>' if caption else ""
    return f'<div class="fig"><img src="data:image/png;base64,{b64}" alt="{html.escape(caption)}">{cap}</div>'


def _table(df: pd.DataFrame, float_fmt: str = "{:.3f}", pct_rows=()) -> str:
    """Render a DataFrame as HTML, formatting selected rows as percentages."""
    if df is None or df.empty:
        return "<p><em>No data.</em></p>"

    out = df.copy()
    for r in out.index:
        if r in pct_rows:
            out.loc[r] = out.loc[r].map(
                lambda v: f"{v * 100:.2f}%" if isinstance(v, (int, float, np.floating)) and pd.notna(v) else v
            )

    def fmt(v):
        if isinstance(v, str):
            return html.escape(v)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "&ndash;"
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        if isinstance(v, (float, np.floating)):
            return float_fmt.format(v)
        return html.escape(str(v))

    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in out.columns)
    body = "".join(
        "<tr><td>" + html.escape(str(idx)).replace("_", " ") + "</td>"
        + "".join(f"<td>{fmt(v)}</td>" for v in row) + "</tr>"
        for idx, row in zip(out.index, out.values)
    )
    return f'<div class="scroll"><table><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _kpis(pairs: list[tuple[str, str]]) -> str:
    cells = "".join(
        f'<div class="kpi"><div class="l">{html.escape(l)}</div><div class="v">{html.escape(v)}</div></div>'
        for l, v in pairs
    )
    return f'<div class="kpis">{cells}</div>'


def build_html_report(
    *,
    title: str,
    subtitle: str,
    config: dict,
    provenance,
    figures: dict[str, str],
    tables: dict[str, pd.DataFrame],
    narrative: dict[str, str],
    kpis: list[tuple[str, str]],
    warnings_: list[str],
    output_path: Path,
) -> Path:
    """Assemble the self-contained HTML research report."""
    sections: list[str] = []

    banner = ""
    if warnings_:
        items = "".join(f"<li>{html.escape(w)}</li>" for w in warnings_)
        banner = (
            f'<div class="callout"><span class="t">Read this before the numbers</span>'
            f"<ul>{items}</ul></div>"
        )

    meta_chips = "".join(
        f'<span class="chip">{html.escape(k)}: <b>{html.escape(str(v))}</b></span>'
        for k, v in [
            ("Universe", f"{len(config['data']['symbols'])} ETFs"),
            ("Signal", f"{config['strategy']['lookback_days']}d / skip {config['strategy']['skip_days']}d"),
            ("Holdings", config["strategy"]["holdings"]),
            ("Rebalance", config["strategy"]["rebalance"]),
            ("Costs", f"{config['portfolio']['commission_bps'] + config['portfolio']['slippage_bps']:g} bps"),
            ("Execution", f"T+{config['portfolio']['execution_lag']}"),
            ("Data source", getattr(provenance, "source", "unknown")),
        ]
    )

    toc = """
    <div class="toc"><b>Contents</b><ol>
      <li><a href="#q">Research question</a></li>
      <li><a href="#motivation">Economic motivation</a></li>
      <li><a href="#data">Data and cleaning</a></li>
      <li><a href="#signal">Signal definition</a></li>
      <li><a href="#execution">Execution assumptions</a></li>
      <li><a href="#costs">Transaction-cost model</a></li>
      <li><a href="#validation">Validation procedure</a></li>
      <li><a href="#results">Results</a></li>
      <li><a href="#failure">Failure analysis</a></li>
      <li><a href="#limits">Limitations and next experiments</a></li>
    </ol></div>
    """

    def section(anchor: str, heading: str, *blocks: str) -> str:
        return f'<h2 id="{anchor}">{html.escape(heading)}</h2>' + "".join(b for b in blocks if b)

    pct = ("cagr", "ann_volatility", "max_drawdown", "total_return", "win_rate",
           "alpha", "best_day", "worst_day", "var_95", "cvar_95")

    sections.append(section("q", "1. Research question", narrative.get("question", "")))
    sections.append(section("motivation", "2. Economic motivation", narrative.get("motivation", "")))
    sections.append(
        section(
            "data", "3. Data and cleaning",
            narrative.get("data", ""),
            _table(tables.get("coverage", pd.DataFrame())),
        )
    )
    sections.append(section("signal", "4. Signal definition", narrative.get("signal", "")))
    sections.append(section("execution", "5. Execution assumptions", narrative.get("execution", "")))
    sections.append(
        section(
            "costs", "6. Transaction-cost model",
            narrative.get("costs", ""),
            _img(figures.get("turnover", ""), "Turnover per rebalance and the cumulative cost drag."),
        )
    )
    sections.append(section("validation", "7. Validation procedure", narrative.get("validation", "")))

    sections.append(
        section(
            "results", "8. Results",
            _kpis(kpis),
            narrative.get("results_intro", ""),
            "<h3>Strategy vs baselines (full sample)</h3>",
            _table(tables.get("baselines", pd.DataFrame()), pct_rows=pct),
            _img(figures.get("equity", ""), "Equity curves, net of costs, log scale. Dashed lines mark the out-of-sample boundaries."),
            _img(figures.get("drawdown", ""), "Drawdown from running peak."),
            "<h3>In-sample vs out-of-sample</h3>",
            narrative.get("oos", ""),
            _table(tables.get("splits", pd.DataFrame()), pct_rows=pct),
            _img(figures.get("splits", ""), "Headline metrics by sample split."),
            "<h3>Stability through time</h3>",
            _img(figures.get("rolling_sharpe", ""), "Rolling 1-year Sharpe. A strategy whose edge is real should not be a single spike."),
            _img(figures.get("rolling_vol", ""), "Rolling 3-month annualised volatility."),
            _img(figures.get("monthly", ""), "Monthly returns (%)."),
            "<h3>Exposure and holdings</h3>",
            _img(figures.get("exposure", ""), "Gross exposure and the composition of the portfolio through time."),
            "<h3>Parameter sensitivity</h3>",
            narrative.get("sensitivity", ""),
            _img(figures.get("sensitivity", ""), "Mean Sharpe across the parameter grid, training window only."),
            # Render whatever marginal tables were supplied, in the order they
            # were built, rather than naming them here. The dimension names come
            # from the sensitivity config ("lookback_months", "cost_bps", ...),
            # so a hardcoded list silently drops any table whose key does not
            # match -- which is exactly what happened to lookback and cost.
            *(_table(tables[k]) for k in tables if k.startswith("sens_")),
            "<h3>Performance by market regime</h3>",
            _table(tables.get("regimes", pd.DataFrame())),
            "<h3>Calendar years</h3>",
            _table(tables.get("years", pd.DataFrame())),
            "<h3>How much of the edge is one year?</h3>",
            narrative.get("loyo", ""),
            _table(tables.get("loyo", pd.DataFrame())),
            "<h3>Is the edge distinguishable from luck?</h3>",
            narrative.get("significance", ""),
            _img(figures.get("bootstrap", ""),
                 "Bootstrap distribution of the Sharpe advantage. If zero sits inside the "
                 "shaded interval, the edge is within sampling noise."),
            _table(tables.get("significance", pd.DataFrame())),
        )
    )

    sections.append(section("failure", "9. Failure analysis", narrative.get("failure", "")))
    sections.append(section("limits", "10. Limitations and next experiments", narrative.get("limits", "")))

    prov_note = ""
    if provenance is not None:
        prov_note = (
            f"Data source <code>{html.escape(str(provenance.source))}</code>, "
            f"retrieved {html.escape(str(provenance.downloaded_at_utc))}, "
            f"{provenance.n_rows:,} rows spanning {provenance.first_date} to {provenance.last_date}."
        )

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{_CSS}</style></head>
<body><div class="wrap">
<h1>{html.escape(title)}</h1>
<p class="sub">{html.escape(subtitle)}</p>
<div class="meta">{meta_chips}</div>
{banner}
{toc}
{''.join(sections)}
<footer>
<p>{prov_note}</p>
<p>Generated by <code>quant_platform</code> from <code>configs/momentum.yaml</code>.
Reproduce with <code>python -m quant_platform.run --config configs/momentum.yaml</code>.</p>
</footer>
</div></body></html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(doc, encoding="utf-8")
    logger.info("Wrote report: %s", output_path)
    return output_path
