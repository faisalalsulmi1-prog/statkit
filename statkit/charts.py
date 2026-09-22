"""Charts — one figure family per test, rendered to an in-memory PNG (PLAN §7).

L3 (load-bearing): every chart is built in memory and saved to an ``io.BytesIO``
buffer — NEVER to disk. matplotlib is driven through ``Figure`` +
``FigureCanvasAgg`` directly (NO ``pyplot`` global state: safer under Streamlit
re-runs, no GUI backend, no font/temp side effects). The mechanical guard is
``tests/test_no_ai_no_io.py`` (a ``savefig("path")`` string arg fails the build;
``savefig(buf)`` is fine).

Public shape:
  * PRIMITIVES (``box_strip`` / ``scatter_fit`` / ``hist_qq`` / ``paired_slope`` /
    ``stacked_pct_bar`` / ``means_lines`` / ``corr_heatmap`` / ``resid_panels`` /
    ``bar``) each take the data they draw and return an ``io.BytesIO`` PNG.
  * ``chart(result) -> ((caption, png_bytes), ...)`` is the dispatcher wired into
    every ``TestSpec.chart``. It reads the raw data a chart needs from the
    ``Result`` (the runner stashes the canonical frame on ``result.arrays`` — see
    ``advise.advised``), maps the test to its §7 figure(s), and returns caption +
    PNG bytes. A blocked Result, ``describe`` and ``logistic`` (no §7 chart) all
    return ``()``.

§7 mapping (family -> figure): F-2G indep -> box_strip; paired -> paired_slope +
hist_qq(diff); one-sample -> hist_qq. F-KG -> box_strip (+ means_lines for the
two-way interaction). F-RANK2 same as F-2G. F-ASSOC -> scatter_fit (fit line for
Pearson/OLS; none for rank). F-ASSOC-M -> corr_heatmap. F-CAT -> stacked_pct_bar
(a grouped bar for goodness-of-fit counts / a one-proportion bar). F-REG ->
scatter_fit (simple) + resid_panels. F-CHK -> hist_qq (normality) / box_strip
(homogeneity, which is about spread).

Only stdlib (io) + numpy + pandas + scipy + matplotlib (L3 allowlist). No I/O.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy import stats

from . import levels

# Okabe-Ito colour-blind-safe palette (6 hues + a neutral grey), one constant.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
_GREY = "#666666"
_FIGSIZE = (6.0, 4.0)
_DPI = 150


# --------------------------------------------------------------------------
# figure scaffolding (no pyplot)
# --------------------------------------------------------------------------
def _fig(cols: int = 1):
    fig = Figure(figsize=(_FIGSIZE[0] * cols if cols > 1 else _FIGSIZE[0],
                          _FIGSIZE[1]), dpi=_DPI)
    FigureCanvasAgg(fig)                       # attach an Agg canvas for savefig
    axes = fig.subplots(1, cols, squeeze=False)[0]
    return fig, list(axes)


def _png(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")   # BytesIO -> in memory
    buf.seek(0)
    return buf


def _color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


# --------------------------------------------------------------------------
# primitives (each returns an io.BytesIO PNG)
# --------------------------------------------------------------------------
def box_strip(groups: dict, ylabel: str = "", title: str = "") -> io.BytesIO:
    """Box plot + jittered points per group (§7: box+jitter, never bar-of-means)."""
    fig, (ax,) = _fig()
    labels = list(groups)
    data = [np.asarray(groups[k], float) for k in labels]
    data = [d[np.isfinite(d)] for d in data]
    ax.boxplot(data, tick_labels=[str(l) for l in labels], showfliers=False,
               medianprops=dict(color=_GREY))
    rng = np.random.default_rng(0)
    for i, d in enumerate(data):
        if len(d):
            jitter = rng.uniform(-0.12, 0.12, len(d))
            ax.scatter(np.full(len(d), i + 1) + jitter, d, s=18, alpha=0.6,
                       color=_color(i), edgecolors="none")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    return _png(fig)


def scatter_fit(x, y, xlabel: str = "", ylabel: str = "",
                fit: bool = True) -> io.BytesIO:
    """Scatter of x vs y with an optional least-squares fit line (§7)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    fig, (ax,) = _fig()
    ax.scatter(x, y, s=22, alpha=0.7, color=_color(0), edgecolors="none")
    if fit and len(x) >= 2 and np.ptp(x) > 0:
        b, a = np.polyfit(x, y, 1)                 # slope, intercept
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, a + b * xs, color=_color(4), lw=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    return _png(fig)


def hist_qq(values, label: str = "", vline=None,
            vline_label: str = "") -> io.BytesIO:
    """Two panels: a histogram (optional reference line) and a normal Q-Q plot.

    ``vline`` draws a dashed reference line on the histogram; ``vline_label``
    (optional) labels it in a small legend so the reader knows what it marks."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    fig, (ax1, ax2) = _fig(cols=2)
    ax1.hist(v, bins=min(20, max(5, len(v) // 2 or 1)), color=_color(0), alpha=0.8)
    if vline is not None:
        ax1.axvline(float(vline), color=_color(4), lw=2, ls="--",
                    label=vline_label or None)
        if vline_label:
            ax1.legend(fontsize=8)
    ax1.set_xlabel(label or "value")
    ax1.set_ylabel("count")
    ax1.set_title("Histogram")
    if len(v) >= 3 and np.ptp(v) > 0:
        (osm, osr), (slope, intercept, _) = stats.probplot(v, dist="norm")
        ax2.scatter(osm, osr, s=18, alpha=0.7, color=_color(0), edgecolors="none")
        ax2.plot(osm, slope * osm + intercept, color=_color(4), lw=2)
    ax2.set_xlabel("theoretical quantiles")
    ax2.set_ylabel("sample quantiles")
    ax2.set_title("Q-Q plot")
    fig.tight_layout()
    return _png(fig)


def paired_slope(before, after, label_before: str = "before",
                 label_after: str = "after", ylabel: str = "") -> io.BytesIO:
    """One line per unit connecting its before/after value (slope/dumbbell)."""
    a = np.asarray(before, float)
    b = np.asarray(after, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    fig, (ax,) = _fig()
    for ai, bi in zip(a, b):
        ax.plot([0, 1], [ai, bi], color=_GREY, alpha=0.5, lw=1,
                marker="o", markersize=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([str(label_before), str(label_after)])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    return _png(fig)


def stacked_pct_bar(xtab: pd.DataFrame, xlabel: str = "",
                    legend_title: str = "") -> io.BytesIO:
    """One 100%-stacked bar per row of a contingency table, split by column."""
    counts = xtab.to_numpy(float)
    row_tot = counts.sum(axis=1, keepdims=True)
    pct = np.divide(counts, row_tot, out=np.zeros_like(counts),
                    where=row_tot > 0) * 100.0
    rows = [str(i) for i in xtab.index]
    cols = [str(c) for c in xtab.columns]
    fig, (ax,) = _fig()
    bottom = np.zeros(len(rows))
    xpos = np.arange(len(rows))
    for j, c in enumerate(cols):
        ax.bar(xpos, pct[:, j], bottom=bottom, label=c, color=_color(j))
        bottom += pct[:, j]
    ax.set_xticks(xpos)
    ax.set_xticklabels(rows)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("percent")
    ax.set_ylim(0, 100)
    ax.legend(title=legend_title, fontsize=8)
    return _png(fig)


def bar(labels, values, ylabel: str = "", ref=None, ref_label: str = "",
        errs=None) -> io.BytesIO:
    """A simple bar chart (goodness-of-fit counts / a single proportion).

    ``values`` may be a 1-D sequence (one series) or a dict {series: seq} for a
    grouped bar (e.g. observed vs expected). Optional ``ref`` draws a horizontal
    reference line; ``errs`` draws symmetric error bars on a single series."""
    labels = [str(l) for l in labels]
    xpos = np.arange(len(labels))
    fig, (ax,) = _fig()
    if isinstance(values, dict):
        series = list(values)
        w = 0.8 / len(series)
        for k, name in enumerate(series):
            ax.bar(xpos + (k - (len(series) - 1) / 2) * w,
                   np.asarray(values[name], float), width=w, label=name,
                   color=_color(k))
        ax.legend(fontsize=8)
    else:
        yerr = np.asarray(errs, float) if errs is not None else None
        ax.bar(xpos, np.asarray(values, float), width=0.6, color=_color(0),
               yerr=yerr, capsize=4)
    if ref is not None:
        ax.axhline(float(ref), color=_color(4), lw=2, ls="--",
                   label=ref_label or None)
        if ref_label:
            ax.legend(fontsize=8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    return _png(fig)


def means_lines(cell_means: pd.DataFrame, factor_a: str = "",
                factor_b: str = "", ylabel: str = "") -> io.BytesIO:
    """Two-way interaction plot: mean outcome vs factor A, one line per factor B.

    ``cell_means`` is a DataFrame indexed by factor-A level, columns = factor-B
    levels, values = cell means."""
    fig, (ax,) = _fig()
    a_levels = levels.display_index(cell_means.index)   # '1'/'2', not '1.0' (S-B(c))
    xpos = np.arange(len(a_levels))
    for j, b in enumerate(cell_means.columns):
        ax.plot(xpos, cell_means[b].to_numpy(float), marker="o",
                color=_color(j), label=levels.display(b))
    ax.set_xticks(xpos)
    ax.set_xticklabels(a_levels)
    ax.set_xlabel(factor_a)
    ax.set_ylabel(ylabel)
    ax.legend(title=factor_b, fontsize=8)
    ax.grid(alpha=0.3)
    return _png(fig)


def corr_heatmap(rmatrix: pd.DataFrame) -> io.BytesIO:
    """A correlation matrix as a heatmap with r values annotated (§7)."""
    m = rmatrix.to_numpy(float)
    labels = [str(c) for c in rmatrix.columns]
    fig, (ax,) = _fig()
    im = ax.imshow(m, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if np.isfinite(m[i, j]):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                        color="black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _png(fig)


def resid_panels(fitted, resid) -> io.BytesIO:
    """Regression diagnostics: residuals vs fitted, and a normal Q-Q of resid."""
    f = np.asarray(fitted, float)
    e = np.asarray(resid, float)
    ok = np.isfinite(f) & np.isfinite(e)
    f, e = f[ok], e[ok]
    fig, (ax1, ax2) = _fig(cols=2)
    ax1.scatter(f, e, s=18, alpha=0.7, color=_color(0), edgecolors="none")
    ax1.axhline(0, color=_color(4), lw=1.5, ls="--")
    ax1.set_xlabel("fitted values")
    ax1.set_ylabel("residuals")
    ax1.set_title("Residuals vs fitted")
    if len(e) >= 3 and np.ptp(e) > 0:
        (osm, osr), (slope, intercept, _) = stats.probplot(e, dist="norm")
        ax2.scatter(osm, osr, s=18, alpha=0.7, color=_color(0), edgecolors="none")
        ax2.plot(osm, slope * osm + intercept, color=_color(4), lw=2)
    ax2.set_xlabel("theoretical quantiles")
    ax2.set_ylabel("residual quantiles")
    ax2.set_title("Normal Q-Q")
    fig.tight_layout()
    return _png(fig)


# --------------------------------------------------------------------------
# data extraction from a Result (the runner stashed the canonical frame)
# --------------------------------------------------------------------------
def _frame(r):
    return r.arrays.get("_frame")


def _cols(r):
    return r.arrays.get("_cols") or {}


def _group_split(r):
    """{level: outcome array} from the canonical long outcome+group frame."""
    df = _frame(r)
    if df is None or "outcome" not in df.columns or "group" not in df.columns:
        return {}
    order = list(dict.fromkeys(df["group"].tolist()))
    return {levels.display(g): df.loc[df["group"] == g, "outcome"].to_numpy(float)
            for g in order}


def _measures(r):
    """{label: array} for the repeated-measure wide columns."""
    df = _frame(r)
    if df is None:
        return {}
    mcols = [c for c in df.columns if c.startswith("measures__")]
    if not mcols:
        return {}
    headers = _cols(r).get("measures")
    labels = (list(headers) if headers and len(headers) == len(mcols)
              else [f"condition {i + 1}" for i in range(len(mcols))])
    return {levels.display(l): df[c].to_numpy(float) for l, c in zip(labels, mcols)}


def _xy(r):
    df = _frame(r)
    if df is not None and "x" in df.columns and "y" in df.columns:
        return df["x"].to_numpy(float), df["y"].to_numpy(float)
    return r.arrays.get("x"), r.arrays.get("y")


def _var_values(r):
    """[(label, array)] for the normality 'variables' role."""
    df = _frame(r)
    headers = _cols(r).get("variables", ())
    out = []
    if df is None:
        return out
    for i, hdr in enumerate(headers):
        col = f"variables__{i}" if f"variables__{i}" in df.columns else hdr
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
            out.append((str(hdr), v))
    return out


def _observed_xtab(r):
    """The observed row x col contingency table (chi2_ind / fisher).

    Prefer the already-display-labeled ``extra['observed']`` (cat.py relabels it
    to '0'/'1' for a numeric-coded 2x2, matching the observed-counts TABLE); fall
    back to a raw rebuild from the frame only when it is absent."""
    observed = r.extra.get("observed")
    if observed is not None:
        return observed
    df = _frame(r)
    if df is None or "row" not in df.columns or "col" not in df.columns:
        return None
    return pd.crosstab(df["row"], df["col"])


def _lbl(r, role, default):
    return r.labels.get(role, default)


# --------------------------------------------------------------------------
# per-test figure builders -> ((caption, io.BytesIO), ...)
# --------------------------------------------------------------------------
def _c_box_outcome(r, title=""):
    groups = _group_split(r)
    if len(groups) < 2:
        return ()
    yl = _lbl(r, "outcome", "outcome")
    return ((f"{yl} by group", box_strip(groups, ylabel=yl, title=title)),)


def _c_hist_one(r):
    df = _frame(r)
    if df is None or "outcome" not in df.columns:
        return ()
    yl = _lbl(r, "outcome", "value")
    mu0 = r.extra.get("mu0")                    # None for tests without a null mean
    vline_label = f"μ₀ = {mu0:g}" if mu0 is not None else ""
    return ((f"Distribution of {yl}",
             hist_qq(df["outcome"].to_numpy(float), label=yl,
                     vline=mu0, vline_label=vline_label)),)


def _c_paired(r):
    df = _frame(r)
    if df is None or "before" not in df.columns or "after" not in df.columns:
        return ()
    lb = _lbl(r, "before", "before")
    la = _lbl(r, "after", "after")
    before = df["before"].to_numpy(float)
    after = df["after"].to_numpy(float)
    return (
        (f"{lb} vs {la} (paired)",
         paired_slope(before, after, lb, la, ylabel="value")),
        ("Distribution of the differences",
         hist_qq(before - after, label=f"{lb} − {la}")),
    )


def _c_measures_box(r):
    m = _measures(r)
    if len(m) < 2:
        return ()
    return (("Values across the repeated measurements",
             box_strip(m, ylabel="value")),)


def _c_two_way(r):
    df = _frame(r)
    if df is None or not {"outcome", "factor_a", "factor_b"} <= set(df.columns):
        return ()
    cm = (df.groupby(["factor_a", "factor_b"])["outcome"].mean()
          .unstack("factor_b"))
    fa = _lbl(r, "factor_a", "factor A")
    fb = _lbl(r, "factor_b", "factor B")
    yl = _lbl(r, "outcome", "outcome")
    return ((f"Interaction of {fa} and {fb}",
             means_lines(cm, factor_a=fa, factor_b=fb, ylabel=f"mean {yl}")),)


def _c_scatter(r, fit):
    x, y = _xy(r)
    if x is None or y is None:
        return ()
    xl = _lbl(r, "x", "x")
    yl = _lbl(r, "y", "y")
    return ((f"{xl} vs {yl}", scatter_fit(x, y, xl, yl, fit=fit)),)


def _c_corr_heatmap(r):
    if r.table is None:
        return ()
    return (("Correlation matrix", corr_heatmap(r.table)),)


def _c_stacked_from_observed(r):
    xt = _observed_xtab(r)
    if xt is None or xt.empty:
        return ()
    xl = _lbl(r, "row", "")
    return ((f"Composition by {xl}" if xl else "Composition by group",
             stacked_pct_bar(xt, xlabel=xl)),)


def _c_gof_bar(r):
    if r.table is None:
        return ()
    t = r.table
    return (("Observed vs expected counts",
             bar(t["category"].tolist(),
                 {"observed": t["observed"].to_numpy(float),
                  "expected": t["expected"].to_numpy(float)},
                 ylabel="count")),)


def _c_mcnemar(r):
    if r.table is None:
        return ()
    lb = _lbl(r, "before", "before")
    return ((f"Change from {lb}", stacked_pct_bar(r.table, xlabel=lb,
                                                  legend_title=_lbl(r, "after", "after"))),)


def _c_cochran(r):
    if r.table is None:
        return ()
    t = r.table
    xt = pd.DataFrame({"success": t["proportion"].to_numpy(float),
                       "other": 1.0 - t["proportion"].to_numpy(float)},
                      index=[str(m) for m in t["measure"]])
    success = r.labels.get("success", "success")
    return ((f"Proportion of '{success}' across conditions",
             stacked_pct_bar(xt, xlabel="condition")),)


def _c_prop1(r):
    if r.estimate is None:
        return ()
    p_hat = float(r.estimate[1])
    success = r.labels.get("success", "success")
    errs = None
    if r.estimate_ci is not None:
        lo, hi = r.estimate_ci
        errs = [max(0.0, p_hat - lo), max(0.0, hi - p_hat)]
        errs = [[errs[0]], [errs[1]]]
    p0 = r.extra.get("p0")                      # None for tests without a null prop.
    ref_label = f"p₀ = {p0:g}" if p0 is not None else ""
    return ((f"Observed proportion of '{success}'",
             bar([f"'{success}'"], [p_hat], ylabel="proportion", errs=errs,
                 ref=p0, ref_label=ref_label)),)


def _c_prop2(r):
    df = _frame(r)
    if df is None or "outcome" not in df.columns or "group" not in df.columns:
        return ()
    xt = pd.crosstab(df["group"], df["outcome"])
    xt.index = [levels.display(v) for v in xt.index]       # '0'/'1', not '0.0' (F4)
    xt.columns = [levels.display(v) for v in xt.columns]
    return ((f"Proportion of '{r.labels.get('success', 'success')}' by group",
             stacked_pct_bar(xt, xlabel=_lbl(r, "group", ""))),)


def _c_ols_simple(r):
    x, y = _xy(r)
    figs = []
    if x is not None and y is not None:
        xl = _lbl(r, "x", "x")
        yl = _lbl(r, "outcome", "y")
        figs.append((f"{yl} vs {xl} with fit line",
                     scatter_fit(x, y, xl, yl, fit=True)))
        slope = r.extra.get("slope")
        intercept = r.extra.get("intercept")
        if slope is not None and intercept is not None:
            xa = np.asarray(x, float)
            fitted = intercept + slope * xa
            figs.append(("Regression diagnostics",
                         resid_panels(fitted, np.asarray(y, float) - fitted)))
    return tuple(figs)


def _c_resid(r):
    f = r.arrays.get("fitted")
    e = r.arrays.get("resid")
    if f is None or e is None:
        return ()
    return (("Regression diagnostics", resid_panels(f, e)),)


def _c_normality(r):
    figs = []
    for label, values in _var_values(r):
        figs.append((f"Distribution of {label}",
                     hist_qq(values, label=label)))
    return tuple(figs) if figs else ()


_CHART = {
    # F-2G
    "t_1s": _c_hist_one,
    "t_ind": _c_box_outcome,
    "t_paired": _c_paired,
    # F-KG
    "anova_1w": _c_box_outcome,
    "anova_2w": _c_two_way,
    "rm_anova": _c_measures_box,
    "kruskal": _c_box_outcome,
    "friedman": _c_measures_box,
    # F-RANK2
    "mwu": _c_box_outcome,
    "wilcoxon": lambda r: (_c_paired(r) if _frame(r) is not None
                           and "after" in _frame(r).columns else _c_hist_one(r)),
    # F-ASSOC
    "pearson": lambda r: _c_scatter(r, fit=True),
    "spearman": lambda r: _c_scatter(r, fit=False),
    "kendall": lambda r: _c_scatter(r, fit=False),
    "corr_matrix": _c_corr_heatmap,
    # F-CAT
    "chi2_ind": _c_stacked_from_observed,
    "chi2_gof": _c_gof_bar,
    "fisher": _c_stacked_from_observed,
    "mcnemar": _c_mcnemar,
    "cochran_q": _c_cochran,
    "prop_1": _c_prop1,
    "prop_2": _c_prop2,
    # F-REG
    "ols_simple": _c_ols_simple,
    "ols_multi": _c_resid,
    "logistic": lambda r: (),          # §7 assigns no chart to logistic
    # F-CHK
    "normality": _c_normality,
    "homogeneity": _c_box_outcome,
    # describe: Table 1 only, no chart
    "describe": lambda r: (),
}


def chart(r) -> tuple:
    """Wired into every ``TestSpec.chart``: ``Result -> ((caption, png_bytes), ...)``.

    A blocked Result, or a test with no §7 chart, yields ``()``. PNG bytes are the
    buffer's ``getvalue()`` — the report embeds them via ``BytesIO -> add_picture``.
    """
    if getattr(r, "status", None) == "blocked":
        return ()
    builder = _CHART.get(r.test_id)
    if builder is None:                    # pragma: no cover - registry is closed
        return ()
    figs = builder(r)
    return tuple((cap, buf.getvalue()) for cap, buf in figs)


# {test_id: chart fn} — registry._wire_charts() swaps each spec.chart.
CHARTS = {tid: chart for tid in _CHART}
