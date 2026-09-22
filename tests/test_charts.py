"""RED-first tests for statkit/charts.py (Chunk 13, PLAN §7).

DoD slice: EVERY charted family renders a non-empty PNG. A chart is a pure
function of a ``Result`` (the wired ``spec.chart`` signature); the runner stashes
the canonical frame onto ``Result.arrays`` so a chart can be drawn from a Result
alone. We therefore build real ``Result`` objects by running the WIRED runners
(``REGISTRY[id].run`` -> advise.advised, which stashes the frame) on small
synthetic data, then assert:

  * ``charts.chart(result)`` returns ``((caption, png_bytes), ...)`` with PNG
    magic bytes and a non-trivial size, one entry per figure §7 assigns;
  * ``describe`` / ``logistic`` (no §7 chart) and any blocked Result -> ``()``;
  * the L3 contract: charts render to memory (BytesIO), never to disk (the
    mechanical guard is tests/test_no_ai_no_io.py; here we just prove the
    primitives return an in-memory buffer).

NO disk, NO AI (L3). matplotlib is driven via Figure + Agg (no pyplot global
state), asserted indirectly by the gate + by these renders succeeding headless.
"""
from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd
import pytest

from statkit import bind
from statkit.infer import Dataset
from statkit.model import ColumnProfile, Kind
from statkit.registry import REGISTRY

N, O, CAT, B, ID = (Kind.NUMERIC, Kind.ORDINAL, Kind.CATEGORICAL, Kind.BINARY, Kind.ID)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------
# compact dataset / bound builders (same shape as tests/test_means.py)
# --------------------------------------------------------------------------
def _isna(v):
    return v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v))


def _profile(name, kinds, values):
    non_null = [v for v in values if not _isna(v)]
    if kinds[0] in (Kind.NUMERIC, Kind.ID, Kind.DATE, Kind.EMPTY):
        levels = ()
    else:
        seen, levels = set(), []
        for v in non_null:
            s = str(v)
            if s not in seen:
                seen.add(s)
                levels.append(s)
        levels = tuple(levels)
    return ColumnProfile(
        name=name, kinds=tuple(kinds), n_total=len(values),
        n_missing=len(values) - len(non_null), n_levels=len(levels), levels=levels)


def _ds(cols, excel_rows=None):
    df_cols, profiles, n = {}, [], None
    for name, (values, kinds) in cols.items():
        n = len(values) if n is None else n
        non_null = [v for v in values if not _isna(v)]
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                      for v in non_null)
        if kinds[0] in (N, O, B, ID) and numeric:
            df_cols[name] = pd.array(
                [None if _isna(v) else float(v) for v in values], dtype="float64")
        else:
            df_cols[name] = pd.array(
                [pd.NA if _isna(v) else str(v) for v in values], dtype="string")
        profiles.append(_profile(name, kinds, values))
    df = pd.DataFrame(df_cols)
    excel_rows = tuple(excel_rows or range(2, 2 + (n or 0)))
    return Dataset(df=df, profiles=tuple(profiles), excel_rows=excel_rows)


def _run(test_id, cols, columns, layout=None, params=None):
    """Build a bound and run it through the WIRED runner (frame-stashing)."""
    b = bind.bind(REGISTRY[test_id], _ds(cols), columns, layout=layout, params=params)
    return REGISTRY[test_id].run(b)


# --------------------------------------------------------------------------
# per-test synthetic cases -> (cols, columns, params, expected_fig_count)
# --------------------------------------------------------------------------
_R = np.random.default_rng(2026)


def _cases():
    A6 = [1, 2, 3, 4, 5, 6]
    B6 = [3, 4, 5, 6, 7, 8]
    x = list(np.round(_R.normal(10, 3, 24), 2))
    y = list(np.round(np.array(x) * 1.5 + _R.normal(0, 2, 24), 2))
    z = list(np.round(_R.normal(5, 2, 24), 2))
    grp3 = (["A"] * 8 + ["B"] * 8 + ["C"] * 8)
    out3 = list(np.round(_R.normal(0, 1, 8), 2)) + \
        list(np.round(_R.normal(1.5, 1, 8), 2)) + \
        list(np.round(_R.normal(3, 1, 8), 2))
    pre = list(np.round(_R.normal(10, 2, 10), 2))
    post = list(np.round(np.array(pre) + _R.normal(2, 1, 10), 2))
    m0 = list(np.round(_R.normal(10, 2, 8), 2))
    m1 = list(np.round(np.array(m0) + 1 + _R.normal(0, 1, 8), 2))
    m2 = list(np.round(np.array(m0) + 2 + _R.normal(0, 1, 8), 2))
    # 2x2 chi2 with expected counts >= 5 (so it does NOT route to Fisher)
    ci_row = ["A"] * 20 + ["B"] * 20
    ci_col = (["X"] * 15 + ["Y"] * 5) + (["X"] * 5 + ["Y"] * 15)
    # 2x2 factorial for two-way ANOVA (>= 3 per cell)
    fa, fb, y2 = [], [], []
    for a in ("lo", "hi"):
        for bb in ("p", "q"):
            for v in np.round(_R.normal(10 + (a == "hi") * 3 + (bb == "q") * 2, 1.5, 5), 2):
                fa.append(a); fb.append(bb); y2.append(float(v))
    return {
        # F-2G
        "t_1s": ({"Y": (A6 + B6, (N,))}, {"outcome": ("Y",)}, {"mu0": 5.0}, 1),
        "t_ind": ({"Y": (A6 + B6, (N,)), "G": (["A"] * 6 + ["B"] * 6, (CAT,))},
                  {"outcome": ("Y",), "group": ("G",)}, None, 1),
        "t_paired": ({"Pre": (pre, (N,)), "Post": (post, (N,))},
                     {"before": ("Pre",), "after": ("Post",)}, None, 2),
        # F-KG
        "anova_1w": ({"Y": (out3, (N,)), "G": (grp3, (CAT,))},
                     {"outcome": ("Y",), "group": ("G",)}, None, 1),
        "anova_2w": ({"Y": (y2, (N,)), "A": (fa, (CAT,)), "Bf": (fb, (CAT,))},
                     {"outcome": ("Y",), "factor_a": ("A",), "factor_b": ("Bf",)},
                     None, 1),
        "rm_anova": ({"C1": (m0, (N,)), "C2": (m1, (N,)), "C3": (m2, (N,))},
                     {"measures": ("C1", "C2", "C3")}, None, 1),
        "kruskal": ({"Y": (out3, (N,)), "G": (grp3, (CAT,))},
                    {"outcome": ("Y",), "group": ("G",)}, None, 1),
        "friedman": ({"C1": (m0, (N,)), "C2": (m1, (N,)), "C3": (m2, (N,))},
                     {"measures": ("C1", "C2", "C3")}, None, 1),
        # F-RANK2
        "mwu": ({"Y": (A6 + B6, (N,)), "G": (["A"] * 6 + ["B"] * 6, (CAT,))},
                {"outcome": ("Y",), "group": ("G",)}, None, 1),
        "wilcoxon": ({"Pre": (pre, (N,)), "Post": (post, (N,))},
                     {"before": ("Pre",), "after": ("Post",)}, None, 2),
        # F-ASSOC
        "pearson": ({"X": (x, (N,)), "Y": (y, (N,))}, {"x": ("X",), "y": ("Y",)},
                    None, 1),
        "spearman": ({"X": (x, (N,)), "Y": (y, (N,))}, {"x": ("X",), "y": ("Y",)},
                     None, 1),
        "kendall": ({"X": (x, (N,)), "Y": (y, (N,))}, {"x": ("X",), "y": ("Y",)},
                    None, 1),
        # F-ASSOC-M
        "corr_matrix": ({"X": (x, (N,)), "Y": (y, (N,)), "Z": (z, (N,))},
                        {"variables": ("X", "Y", "Z")}, None, 1),
        # F-CAT
        "chi2_ind": ({"R": (ci_row, (CAT,)), "Cc": (ci_col, (CAT,))},
                     {"row": ("R",), "col": ("Cc",)}, None, 1),
        "chi2_gof": ({"Cat": (["a"] * 10 + ["b"] * 8 + ["c"] * 12, (CAT,))},
                     {"category": ("Cat",)}, None, 1),
        "fisher": ({"R": (["A"] * 6 + ["B"] * 6, (CAT,)),
                    "Cc": (["X", "X", "X", "Y", "Y", "Y"] * 2, (CAT,))},
                   {"row": ("R",), "col": ("Cc",)}, None, 1),
        "mcnemar": ({"Pre": (["Yes", "No"] * 6 + ["Yes"] * 4 + ["No"] * 4, (B,)),
                     "Post": (["Yes"] * 6 + ["No"] * 6 + ["No"] * 4 + ["Yes"] * 4, (B,))},
                    {"before": ("Pre",), "after": ("Post",)}, None, 1),
        "cochran_q": ({"M1": ([1, 0, 1, 0, 1, 1, 0, 1, 0, 1], (B,)),
                       "M2": ([0, 0, 1, 1, 1, 0, 0, 1, 1, 1], (B,)),
                       "M3": ([1, 1, 1, 0, 0, 1, 1, 1, 0, 0], (B,))},
                      {"measures": ("M1", "M2", "M3")}, None, 1),
        "prop_1": ({"Out": (["Yes"] * 13 + ["No"] * 7, (B,))},
                   {"outcome": ("Out",)}, {"success": "Yes", "p0": 0.5}, 1),
        "prop_2": ({"Out": (["Yes"] * 7 + ["No"] * 5 + ["Yes"] * 4 + ["No"] * 8, (B,)),
                    "G": (["A"] * 12 + ["B"] * 12, (B,))},
                   {"outcome": ("Out",), "group": ("G",)}, {"success": "Yes"}, 1),
        # F-REG
        "ols_simple": ({"Y": (y, (N,)), "X": (x, (N,))},
                       {"outcome": ("Y",), "x": ("X",)}, None, 2),
        "ols_multi": ({"Y": (y, (N,)), "X1": (x, (N,)), "X2": (z, (N,))},
                      {"outcome": ("Y",), "predictors": ("X1", "X2")}, None, 1),
        # F-CHK
        "normality": ({"X": (x, (N,))}, {"variables": ("X",)}, None, 1),
        "homogeneity": ({"Y": (out3, (N,)), "G": (grp3, (CAT,))},
                        {"outcome": ("Y",), "group": ("G",)}, None, 1),
    }


CASES = _cases()
CHARTED = sorted(CASES)
NO_CHART = ["describe", "logistic"]


def _assert_png(png_bytes):
    assert isinstance(png_bytes, (bytes, bytearray)), type(png_bytes)
    assert png_bytes[:8] == PNG_MAGIC, "not a PNG"
    assert len(png_bytes) > 1000, f"PNG suspiciously small: {len(png_bytes)} bytes"


# ==========================================================================
# every charted family renders the §7-assigned figure(s)
# ==========================================================================
@pytest.mark.parametrize("test_id", CHARTED)
def test_charted_family_renders_png(test_id):
    from statkit import charts
    cols, columns, params, n_figs = CASES[test_id]
    r = _run(test_id, cols, columns, params=params)
    assert r.status == "ok", (test_id, [f.text for f in r.findings])
    figs = REGISTRY[test_id].chart(r)          # the wired spec.chart(result)
    assert isinstance(figs, tuple), test_id
    assert len(figs) == n_figs, (test_id, len(figs))
    for caption, png in figs:
        assert isinstance(caption, str) and caption, test_id
        _assert_png(png)
    # charts.chart is the same public entry the spec.chart is wired to
    assert charts.chart(r) is not None


def test_every_family_has_a_charted_member():
    """DoD: 'every family renders'. Each family with a §7 chart has >=1 member
    that produces a figure in CASES."""
    fams = {REGISTRY[tid].family for tid in CHARTED}
    # families that §7 assigns a chart to:
    expected = {"F-2G", "F-KG", "F-RANK2", "F-ASSOC", "F-ASSOC-M", "F-CAT",
                "F-REG", "F-CHK"}
    assert expected <= fams


# ==========================================================================
# no-chart contract: describe / logistic / blocked -> ()
# ==========================================================================
def test_describe_has_no_chart():
    r = _run("describe", {"Y": ([1, 2, 3, 4, 5], (N,))}, {"variables": ("Y",)})
    assert REGISTRY["describe"].chart(r) == ()


def test_logistic_has_no_chart():
    # a clean, separable-free logistic fit
    out = (["No"] * 12 + ["Yes"] * 12)
    xp = list(np.round(_R.normal(0, 1, 24), 2))
    r = _run("logistic", {"O": (out, (B,)), "P": (xp, (N,))},
             {"outcome": ("O",), "predictors": ("P",)},
             params={"success": "Yes"})
    assert r.status == "ok", [f.text for f in r.findings]
    assert REGISTRY["logistic"].chart(r) == ()


def test_blocked_result_has_no_chart():
    # constant column -> S8 block -> no chart
    r = _run("t_ind", {"Y": ([5, 5, 5, 5, 5, 5], (N,)),
                       "G": (["A"] * 3 + ["B"] * 3, (CAT,))},
             {"outcome": ("Y",), "group": ("G",)})
    assert r.status == "blocked"
    from statkit import charts
    assert charts.chart(r) == ()


# ==========================================================================
# primitives return an in-memory BytesIO PNG (L3: no disk)
# ==========================================================================
def test_primitives_return_bytesio_png():
    from statkit import charts
    a = np.array([1.0, 2, 3, 4, 5])
    b = np.array([2.0, 3, 4, 5, 6])
    buf = charts.box_strip({"A": a, "B": b}, ylabel="Y")
    assert isinstance(buf, io.BytesIO)
    assert buf.getvalue()[:8] == PNG_MAGIC
    for buf in (
        charts.scatter_fit(a, b, "x", "y", fit=True),
        charts.hist_qq(np.concatenate([a, b, a]), label="v"),
        charts.paired_slope(a, b, "pre", "post", ylabel="v"),
        charts.corr_heatmap(pd.DataFrame([[1.0, 0.5], [0.5, 1.0]],
                                         index=["x", "y"], columns=["x", "y"])),
        charts.resid_panels(np.array([1.0, 2, 3, 4, 5]),
                            np.array([0.1, -0.2, 0.05, -0.1, 0.15])),
    ):
        assert isinstance(buf, io.BytesIO)
        assert buf.getvalue()[:8] == PNG_MAGIC


# ==========================================================================
# S12 (batch 2B): the null-reference line must be DRIVEN BY THE RESULT
# ==========================================================================
# A PNG is opaque, so we prove the line is wired from the Result — not merely
# that a PNG renders — by rendering the SAME chart with and without the null
# scalar in ``r.extra`` and asserting the bytes DIFFER (matplotlib Agg output
# is deterministic for an identical figure — verified empirically).
#
# Expected-RED list (must fail BEFORE the charts.py fix, for the right reason):
#   * t_1s / one-sample wilcoxon: png(with mu0) != png(without mu0) is RED
#     because ``_c_hist_one`` currently ignores ``r.extra["mu0"]`` — both
#     renders draw NO vline, so they are byte-identical (== , not !=).
#   * prop_1: png(with p0) != png(without p0) is RED because ``_c_prop1``
#     currently never passes ``ref``/``ref_label`` — both renders are identical.
#   * hist_qq(..., vline_label=...) is RED with TypeError — the kwarg does not
#     exist yet.
# The "no mu0/p0 -> still a valid PNG" half of each test is a regression guard
# for the None fallback (green before and after); it rides in the same test.

def _without(r, key):
    """A copy of Result ``r`` with ``key`` dropped from ``.extra`` (frame kept)."""
    import dataclasses
    return dataclasses.replace(
        r, extra={k: v for k, v in r.extra.items() if k != key})


def test_t1s_hist_shows_mu0_line_driven_by_result():
    cols, columns, params, _ = CASES["t_1s"]
    r = _run("t_1s", cols, columns, params=params)
    assert r.status == "ok"
    assert r.extra.get("mu0") == 5.0
    (_, png_with), = REGISTRY["t_1s"].chart(r)
    (_, png_without), = REGISTRY["t_1s"].chart(_without(r, "mu0"))
    _assert_png(png_with)
    _assert_png(png_without)                  # fallback: no mu0 -> still valid PNG
    assert png_with != png_without, "mu0 line not driven by the Result"


def test_wilcoxon_one_sample_hist_shows_mu0_line_driven_by_result():
    # one-sample wilcoxon: a lone outcome column + mu0 (bind.py:160), so the
    # dispatcher lambda routes to _c_hist_one (no 'after' column).
    r = _run("wilcoxon", {"Y": ([2.0, 4, 3, 6, 7, 8, 9, 4, 10, 6], (N,))},
             {"outcome": ("Y",)}, params={"mu0": 5.0})
    assert r.status == "ok", [f.text for f in r.findings]
    assert r.extra.get("mu0") == 5.0
    figs = REGISTRY["wilcoxon"].chart(r)
    assert len(figs) == 1, "one-sample wilcoxon -> a single hist_qq figure"
    png_with = figs[0][1]
    png_without = REGISTRY["wilcoxon"].chart(_without(r, "mu0"))[0][1]
    _assert_png(png_with)
    _assert_png(png_without)                  # fallback
    assert png_with != png_without, "mu0 line not driven by the Result"


def test_prop1_bar_shows_p0_line_driven_by_result():
    cols, columns, params, _ = CASES["prop_1"]
    r = _run("prop_1", cols, columns, params=params)
    assert r.status == "ok"
    assert r.extra.get("p0") == 0.5
    (_, png_with), = REGISTRY["prop_1"].chart(r)
    (_, png_without), = REGISTRY["prop_1"].chart(_without(r, "p0"))
    _assert_png(png_with)
    _assert_png(png_without)                  # fallback: no p0 -> still valid PNG
    assert png_with != png_without, "p0 line not driven by the Result"


def test_hist_qq_vline_label_renders_legend():
    from statkit import charts
    v = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2, 3, 4, 5, 6])
    labelled = charts.hist_qq(v, label="x", vline=5, vline_label="μ₀ = 5")
    plain = charts.hist_qq(v, label="x", vline=5)
    assert isinstance(labelled, io.BytesIO)
    assert labelled.getvalue()[:8] == PNG_MAGIC
    # the legend entry is extra content -> the labelled render must differ
    assert labelled.getvalue() != plain.getvalue()


# ==========================================================================
# F6 — the Cochran chart caption NAMES the success level it plots
# ==========================================================================
def test_cochran_chart_caption_names_success_level():
    from statkit import charts
    cols, columns, params, _ = CASES["cochran_q"]      # 0/1 numeric coding
    r = _run("cochran_q", cols, columns, params=params)
    assert r.status == "ok"
    (caption, _), = charts.chart(r)
    assert caption == "Proportion of '1' across conditions"


# ==========================================================================
# F4 — numeric-coded group / measure tick labels render "1"/"2", not "1.0"
# ==========================================================================
def test_group_split_displays_numeric_group_levels():
    from statkit import charts
    from statkit.model import Result
    df = pd.DataFrame({"outcome": [1.0, 2.0, 3.0, 4.0],
                       "group": pd.array([1.0, 1.0, 2.0, 2.0], dtype="float64")})
    r = Result(test_id="t_ind", test_name="x", status="ok",
               arrays={"_frame": df})
    assert list(charts._group_split(r).keys()) == ["1", "2"]


def test_measures_displays_numeric_headers():
    from statkit import charts
    from statkit.model import Result
    df = pd.DataFrame({"measures__0": [1.0, 0.0], "measures__1": [0.0, 1.0]})
    r = Result(test_id="rm_anova", test_name="x", status="ok",
               arrays={"_frame": df, "_cols": {"measures": (1.0, 2.0)}})
    assert list(charts._measures(r).keys()) == ["1", "2"]


# ==========================================================================
# S-B(c) — the two-way interaction plot (means_lines) shows numeric-coded
# factor levels as "1"/"2", not "1.0"/"2.0" (x-ticks = factor A, legend =
# factor B). Text levels ("Male"/"Female") pass through unchanged.
# --------------------------------------------------------------------------
# A PNG is opaque, so (as with prop_2 above) we pin the LABELS by comparison:
# a numeric-coded plot must render byte-identically to the equivalent '1'/'2'
# STRING-coded plot (proving the ticks + legend read '1'/'2'), and must DIFFER
# from a '1.0'/'2.0' string-coded plot. Agg output is deterministic for an
# identical figure; between the frames only the label coding differs.
# ==========================================================================
_ML_VALS = [[10.0, 12.0], [11.0, 15.0]]


def test_means_lines_numeric_levels_render_as_int_labels():
    from statkit import charts
    num = pd.DataFrame(_ML_VALS, index=pd.Index([1.0, 2.0]),
                       columns=pd.Index([1.0, 2.0]))
    clean = pd.DataFrame(_ML_VALS, index=pd.Index(["1", "2"]),
                         columns=pd.Index(["1", "2"]))
    dotzero = pd.DataFrame(_ML_VALS, index=pd.Index(["1.0", "2.0"]),
                           columns=pd.Index(["1.0", "2.0"]))
    png_num = charts.means_lines(num).getvalue()
    _assert_png(png_num)
    assert png_num == charts.means_lines(clean).getvalue(), \
        "numeric levels not routed through display (still '1.0'/'2.0')"
    assert png_num != charts.means_lines(dotzero).getvalue(), \
        "numeric levels rendered as '1.0'/'2.0'"


def test_means_lines_text_levels_pass_through():
    """Text factor levels render verbatim (display passes strings through) -> a
    valid PNG; a regression guard that the fix does not mangle worded levels."""
    from statkit import charts
    text = pd.DataFrame(_ML_VALS, index=pd.Index(["Male", "Female"]),
                        columns=pd.Index(["Drug", "Placebo"]))
    png_text = charts.means_lines(text).getvalue()
    _assert_png(png_text)
    # differs from a numeric-coded render (worded vs "1"/"2" labels)
    num = pd.DataFrame(_ML_VALS, index=pd.Index([1.0, 2.0]),
                       columns=pd.Index([1.0, 2.0]))
    assert png_text != charts.means_lines(num).getvalue()


# --------------------------------------------------------------------------
# F4 residual: the chi2/fisher STACKED bar + the prop_2 chart must show the
# same display labels as the observed-counts TABLE ("0"/"1", not "0.0"/"1.0").
# --------------------------------------------------------------------------
def test_observed_xtab_prefers_display_labeled_observed():
    """The stacked-bar crosstab reuses the already-display-labeled
    ``extra['observed']`` (cat.py builds it as '0'/'1'), NOT a raw rebuild that
    reads '0.0'/'1.0' off the numeric-coded frame."""
    from statkit import charts
    from statkit.model import Result
    raw = pd.DataFrame({"row": [0.0, 0.0, 1.0, 1.0],
                        "col": [0.0, 1.0, 0.0, 1.0]})            # numeric-coded
    observed = pd.DataFrame([[1, 1], [1, 1]],
                            index=["0", "1"], columns=["0", "1"])
    r = Result(test_id="chi2_ind", test_name="x", status="ok",
               arrays={"_frame": raw}, extra={"observed": observed})
    xt = charts._observed_xtab(r)
    assert list(xt.index) == ["0", "1"]
    assert list(xt.columns) == ["0", "1"]


def test_observed_xtab_falls_back_to_frame_when_no_observed():
    """No ``extra['observed']`` (e.g. a Result built without it) -> rebuild from
    the frame so the chart still renders."""
    from statkit import charts
    from statkit.model import Result
    raw = pd.DataFrame({"row": ["A", "A", "B", "B"],
                        "col": ["X", "Y", "X", "Y"]})
    r = Result(test_id="chi2_ind", test_name="x", status="ok",
               arrays={"_frame": raw})
    xt = charts._observed_xtab(r)
    assert list(xt.index) == ["A", "B"]
    assert list(xt.columns) == ["X", "Y"]


def test_prop2_chart_displays_numeric_levels():
    """A numeric-coded prop_2 chart renders the SAME labels as the equivalent
    string-coded one -> its tick/legend labels read '0'/'1', not '0.0'/'1.0'.
    (Agg output is deterministic for an identical figure; the only difference
    between the two frames is the coding of otherwise-identical labels.)"""
    from statkit import charts
    from statkit.model import Result
    num = pd.DataFrame({"group": [0.0, 0.0, 1.0, 1.0],
                        "outcome": [0.0, 1.0, 0.0, 1.0]})
    strg = pd.DataFrame({"group": ["0", "0", "1", "1"],
                         "outcome": ["0", "1", "0", "1"]})
    rn = Result(test_id="prop_2", test_name="x", status="ok",
                arrays={"_frame": num}, labels={"success": "1"})
    rs = Result(test_id="prop_2", test_name="x", status="ok",
                arrays={"_frame": strg}, labels={"success": "1"})
    (_, png_num), = charts.chart(rn)           # public entry -> PNG bytes
    (_, png_str), = charts.chart(rs)
    _assert_png(png_num)
    assert png_num == png_str, "prop_2 numeric levels not routed through display"
