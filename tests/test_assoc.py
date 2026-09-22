"""Golden tests for statkit.assoc runners (Chunk 10; PLAN §3 relate family).

Non-circular goldens: each test pairs a HAND-CHECKED / PUBLISHED constant (cited
in the test) with a DIRECT cross-call to the scipy routine the PLAN names
(pearson r/p == scipy.stats.pearsonr, spearman == spearmanr, kendall τ-b ==
kendalltau). Runners take a canonical ``Bound`` (built by the real ``bind``) and
return a ``Result``; a blocked structural check yields ``status='blocked'``.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kendalltau, norm, pearsonr, spearmanr

from statkit import assoc, bind
from statkit.infer import Dataset
from statkit.model import ColumnProfile, Kind
from statkit.registry import REGISTRY

N, O, C, B = (Kind.NUMERIC, Kind.ORDINAL, Kind.CATEGORICAL, Kind.BINARY)


# --------------------------------------------------------------------------
# minimal Dataset builder (mirrors tests/test_bind.py; dtype from default kind)
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
        n_missing=len(values) - len(non_null), n_levels=len(levels), levels=levels,
    )


def _ds(cols, excel_rows=None):
    df_cols, profiles, n = {}, [], None
    for name, (values, kinds) in cols.items():
        n = len(values) if n is None else n
        non_null = [v for v in values if not _isna(v)]
        numeric = all(isinstance(v, (int, float)) for v in non_null)
        if kinds[0] in (Kind.NUMERIC, Kind.ORDINAL, Kind.BINARY, Kind.ID) and numeric:
            df_cols[name] = pd.array(
                [None if _isna(v) else float(v) for v in values], dtype="float64")
        else:
            df_cols[name] = pd.array(
                [pd.NA if _isna(v) else str(v) for v in values], dtype="string")
        profiles.append(_profile(name, kinds, values))
    df = pd.DataFrame(df_cols)
    excel_rows = tuple(excel_rows or range(2, 2 + n))
    return Dataset(df=df, profiles=tuple(profiles), excel_rows=excel_rows)


def _bind(test_id, cols, columns, layout="long", params=None):
    return bind.bind(REGISTRY[test_id], _ds(cols), columns, layout=layout,
                     params=params)


# ==========================================================================
# PEARSON
# ==========================================================================
# Hand example: x=[1..5], y=[2,4,5,4,5]. Sxy=6, Sxx=10, Syy=6 ->
# r = 6/sqrt(60) = 0.7745967 ; r^2 = 0.6 (worked from the product-moment
# definition, independent of the module's code).
PX = [1.0, 2.0, 3.0, 4.0, 5.0]
PY = [2.0, 4.0, 5.0, 4.0, 5.0]


def test_pearson_hand_value_and_scipy_crosscall():
    b = _bind("pearson", {"H": (PX, (N,)), "W": (PY, (N,))},
              {"x": ("H",), "y": ("W",)})
    r = assoc.pearson(b)
    assert r.status == "ok"
    assert r.statistic[0] == "r"
    assert r.statistic[1] == pytest.approx(6 / math.sqrt(60), abs=1e-6)   # hand
    ref = pearsonr(PX, PY)                                                 # cross-call
    assert r.statistic[1] == pytest.approx(ref.statistic, rel=1e-12)
    assert r.p == pytest.approx(ref.pvalue, rel=1e-12)
    assert r.test_name == "Pearson correlation"


def test_pearson_r_squared_and_label():
    b = _bind("pearson", {"H": (PX, (N,)), "W": (PY, (N,))},
              {"x": ("H",), "y": ("W",)})
    r = assoc.pearson(b)
    assert r.extra["r2"] == pytest.approx(0.6, abs=1e-9)
    assert r.effect == ("r", pytest.approx(6 / math.sqrt(60), abs=1e-6))
    assert r.effect_label == "large"   # .77 > .5


def test_pearson_ci_is_scipy_fisher_z():
    b = _bind("pearson", {"H": (PX, (N,)), "W": (PY, (N,))},
              {"x": ("H",), "y": ("W",)})
    r = assoc.pearson(b)
    ci = pearsonr(PX, PY).confidence_interval()
    assert r.effect_ci[0] == pytest.approx(ci.low, rel=1e-9)
    assert r.effect_ci[1] == pytest.approx(ci.high, rel=1e-9)


def test_pearson_constant_column_is_blocked_not_nan():
    # A constant column -> r is NaN (silent garbage); S8 must block it.
    b = _bind("pearson", {"H": ([5.0] * 6, (N,)), "W": ([1, 2, 3, 4, 5, 6], (N,))},
              {"x": ("H",), "y": ("W",)})
    r = assoc.pearson(b)
    assert r.status == "blocked"
    assert r.p is None
    assert any(f.severity == "block" for f in r.findings)


# ==========================================================================
# SPEARMAN — Wikipedia worked example (rho = -29/165 = -0.1757576)
# ==========================================================================
IQ = [106, 86, 100, 101, 99, 103, 97, 113, 112, 110]
TV = [7, 0, 27, 50, 28, 29, 20, 12, 6, 17]


def test_spearman_wikipedia_value_and_crosscall():
    b = _bind("spearman", {"IQ": (IQ, (N,)), "TV": (TV, (N,))},
              {"x": ("IQ",), "y": ("TV",)})
    r = assoc.spearman(b)
    assert r.status == "ok"
    assert r.statistic[0] == "rho"
    assert r.statistic[1] == pytest.approx(-29 / 165, abs=1e-6)     # published
    ref = spearmanr(IQ, TV)                                          # cross-call
    assert r.statistic[1] == pytest.approx(ref.statistic, rel=1e-12)
    assert r.p == pytest.approx(ref.pvalue, rel=1e-12)


def test_spearman_ci_is_bonett_wright_fisher_z():
    b = _bind("spearman", {"IQ": (IQ, (N,)), "TV": (TV, (N,))},
              {"x": ("IQ",), "y": ("TV",)})
    r = assoc.spearman(b)
    rho, n = spearmanr(IQ, TV).statistic, 10
    z = math.atanh(rho)
    se = math.sqrt((1 + rho ** 2 / 2) / (n - 3))       # Bonett & Wright (2000)
    crit = norm.ppf(0.975)
    lo, hi = math.tanh(z - crit * se), math.tanh(z + crit * se)
    assert r.effect_ci[0] == pytest.approx(lo, rel=1e-9)
    assert r.effect_ci[1] == pytest.approx(hi, rel=1e-9)


# ==========================================================================
# KENDALL τ-b  (scipy kendalltau default variant='b')
# ==========================================================================
def test_kendall_taub_with_ties_hand_and_crosscall():
    # x=[1,1,2,3], y=[1,2,2,3]: C=4, D=0, tie-in-x=1, tie-in-y=1 ->
    # tau-b = (C-D)/sqrt((C+D+Tx)(C+D+Ty)) = 4/sqrt(5*5) = 0.8 (hand).
    kx, ky = [1, 1, 2, 3], [1, 2, 2, 3]
    b = _bind("kendall", {"X": (kx, (O,)), "Y": (ky, (O,))},
              {"x": ("X",), "y": ("Y",)})
    r = assoc.kendall(b)
    assert r.status == "ok"
    assert r.statistic[0].startswith("tau")
    assert r.statistic[1] == pytest.approx(0.8, abs=1e-9)            # hand τ-b
    ref = kendalltau(kx, ky)                                          # cross-call (variant b)
    assert r.statistic[1] == pytest.approx(ref.statistic, rel=1e-12)
    assert r.p == pytest.approx(ref.pvalue, rel=1e-12)


def test_kendall_ci_is_fieller_fisher_z():
    kx = [1, 2, 3, 4, 5, 6, 7, 8]
    ky = [1, 3, 2, 4, 6, 5, 8, 7]
    b = _bind("kendall", {"X": (kx, (N,)), "Y": (ky, (N,))},
              {"x": ("X",), "y": ("Y",)})
    r = assoc.kendall(b)
    tau, n = kendalltau(kx, ky).statistic, 8
    z = math.atanh(tau)
    se = math.sqrt(0.437 / (n - 4))                    # Fieller-Hartley-Pearson
    crit = norm.ppf(0.975)
    lo, hi = math.tanh(z - crit * se), math.tanh(z + crit * se)
    assert r.effect_ci[0] == pytest.approx(lo, rel=1e-9)
    assert r.effect_ci[1] == pytest.approx(hi, rel=1e-9)


# ==========================================================================
# CORR_MATRIX — pairwise via scipy per pair + Holm-adjusted off-diagonal p's.
# (SNAG: bind() does listwise deletion, so complete-case == pairwise here.)
# ==========================================================================
V1 = [1.0, 2, 3, 4, 5, 6]
V2 = [2.0, 1, 4, 3, 6, 5]
V3 = [1.0, 3, 2, 5, 4, 7]


def test_corr_matrix_pearson_matches_scipy_and_holm():
    b = _bind("corr_matrix", {"v1": (V1, (N,)), "v2": (V2, (N,)), "v3": (V3, (N,))},
              {"variables": ("v1", "v2", "v3")}, params={"method": "pearson"})
    r = assoc.corr_matrix(b)
    assert r.status == "ok"
    R = r.table                                     # r matrix, labelled by header
    assert list(R.columns) == ["v1", "v2", "v3"]
    assert R.loc["v1", "v2"] == pytest.approx(pearsonr(V1, V2).statistic, rel=1e-9)
    assert R.loc["v1", "v3"] == pytest.approx(0.890769, abs=1e-6)
    assert R.loc["v2", "v3"] == pytest.approx(0.494872, abs=1e-6)
    # Holm-adjusted off-diagonal p-values (order v1v2, v1v3, v2v3)
    holm = r.extra["p_holm"]
    assert holm.loc["v1", "v2"] == pytest.approx(0.083125, abs=1e-5)
    assert holm.loc["v1", "v3"] == pytest.approx(0.051736, abs=1e-5)
    assert holm.loc["v2", "v3"] == pytest.approx(0.318289, abs=1e-5)


def test_corr_matrix_diagonal_is_one_and_n_recorded():
    b = _bind("corr_matrix", {"v1": (V1, (N,)), "v2": (V2, (N,)), "v3": (V3, (N,))},
              {"variables": ("v1", "v2", "v3")})
    r = assoc.corr_matrix(b)
    assert r.table.loc["v1", "v1"] == pytest.approx(1.0)
    assert r.extra["n"].loc["v1", "v2"] == 6        # complete-case n


def test_corr_matrix_method_spearman_differs():
    b = _bind("corr_matrix", {"v1": (V1, (N,)), "v2": (V2, (N,)), "v3": (V3, (N,))},
              {"variables": ("v1", "v2", "v3")}, params={"method": "spearman"})
    r = assoc.corr_matrix(b)
    assert r.table.loc["v1", "v3"] == pytest.approx(spearmanr(V1, V3).statistic, rel=1e-9)
    assert r.table.loc["v1", "v3"] == pytest.approx(0.885714, abs=1e-6)   # != pearson .890769


# ==========================================================================
# S5 — corr_matrix n-matrix is genuinely pairwise-complete (batch 1A)
# ==========================================================================
def test_corr_matrix_n_matrix_is_pairwise():
    a = [None, 2, 3, 4, 5, 6, 7, 8, 9, 10]        # NaN in row 0
    bb = [2, 1, 4, None, 6, 5, 8, 7, 10, 9]       # NaN in row 3
    cc = [1, 3, 2, 5, 4, 7, None, 9, 8, 11]       # NaN in row 6
    b = _bind("corr_matrix", {"a": (a, (N,)), "b": (bb, (N,)), "c": (cc, (N,))},
              {"variables": ("a", "b", "c")}, layout="long")
    r = assoc.corr_matrix(b)
    Ns = r.extra["n"]
    # each pair drops exactly its two distinct NaN rows -> 8; diagonal keeps 9.
    assert Ns.loc["a", "b"] == 8
    assert Ns.loc["a", "c"] == 8
    assert Ns.loc["b", "c"] == 8
    assert Ns.loc["a", "a"] == 9
    assert r.n["used"] == 8                         # min pairwise-complete n


# ==========================================================================
# F5 — Spearman/Kendall at a PERFECT correlation must not crash on atanh(1)
# (x=1..12, y=2x+1 is perfectly monotone -> coef == 1.0; a Fisher-z CI is
# undefined there, so the CI is None and the result still computes.)
# ==========================================================================
_MONO_X = list(range(1, 13))
_MONO_Y = [2 * i + 1 for i in _MONO_X]


def test_spearman_perfect_monotone_ci_is_none_not_crash():
    b = _bind("spearman", {"X": (_MONO_X, (N,)), "Y": (_MONO_Y, (N,))},
              {"x": ("X",), "y": ("Y",)})
    r = assoc.spearman(b)                                   # must not raise
    assert r.status == "ok"
    assert r.statistic[1] == pytest.approx(1.0, abs=1e-12)  # rho == 1
    assert r.effect_ci is None                              # atanh(1): no finite CI


def test_kendall_perfect_monotone_ci_is_none_not_crash():
    b = _bind("kendall", {"X": (_MONO_X, (N,)), "Y": (_MONO_Y, (N,))},
              {"x": ("X",), "y": ("Y",)})
    r = assoc.kendall(b)                                    # must not raise
    assert r.status == "ok"
    assert r.statistic[1] == pytest.approx(1.0, abs=1e-12)  # tau-b == 1
    assert r.effect_ci is None


# ==========================================================================
# F10 — the p-matrix diagonal is a self-correlation, not a hypothesis: it must
# render as "—" (NaN), never "p < .001" (0.0). Off-diagonal p/Holm untouched.
# ==========================================================================
def test_corr_matrix_p_diagonal_is_nan_not_zero():
    b = _bind("corr_matrix", {"v1": (V1, (N,)), "v2": (V2, (N,)), "v3": (V3, (N,))},
              {"variables": ("v1", "v2", "v3")}, params={"method": "pearson"})
    r = assoc.corr_matrix(b)
    praw, holm = r.extra["p_raw"], r.extra["p_holm"]
    for v in ("v1", "v2", "v3"):
        assert math.isnan(praw.loc[v, v])     # was 0.0 -> "p < .001"
        assert math.isnan(holm.loc[v, v])
    assert r.table.loc["v1", "v1"] == pytest.approx(1.0)   # coef diagonal stays 1
    # off-diagonal Holm values are the published goldens (byte-identical, F10 guard)
    assert holm.loc["v1", "v2"] == pytest.approx(0.083125, abs=1e-5)
    assert holm.loc["v1", "v3"] == pytest.approx(0.051736, abs=1e-5)
    assert holm.loc["v2", "v3"] == pytest.approx(0.318289, abs=1e-5)


# ==========================================================================
# F11 (corr) — a matrix pair with < 3 overlapping rows must not crash scipy:
# blank that cell (NaN) + record a note naming the pair and its overlap n.
# ==========================================================================
def test_corr_matrix_pair_with_too_few_overlaps_is_nan_with_note():
    aa = [1, 3, 2, 5, None, None, None]      # non-NaN rows 0-3
    bb = [None, None, None, 7, 4, 6, 5]      # non-NaN rows 3-6  (overlap w/ aa = row 3)
    cc = [2, 1, 4, 3, 6, 5, 8]               # all 7 rows
    b = _bind("corr_matrix",
              {"aa": (aa, (N,)), "bb": (bb, (N,)), "cc": (cc, (N,))},
              {"variables": ("aa", "bb", "cc")}, layout="long")
    r = assoc.corr_matrix(b)                              # must not raise
    assert r.status == "ok"
    assert math.isnan(r.table.loc["aa", "bb"])            # degenerate pair blanked
    assert math.isnan(r.extra["p_raw"].loc["aa", "bb"])
    assert math.isnan(r.extra["p_holm"].loc["aa", "bb"])
    assert not math.isnan(r.table.loc["aa", "cc"])        # good pairs still computed
    assert not math.isnan(r.table.loc["bb", "cc"])
    assert r.extra["n"].loc["aa", "bb"] == 1              # the true overlap recorded
    notes = [f for f in r.findings if f.code == "PAIR_N"]
    assert len(notes) == 1
    assert "aa" in notes[0].text and "bb" in notes[0].text
    assert "1" in notes[0].text                           # names the overlap n


# ==========================================================================
# module wiring
# ==========================================================================
def test_runners_mapping_covers_the_four_ids():
    assert set(assoc.RUNNERS) == {"pearson", "spearman", "kendall", "corr_matrix"}
    for tid, fn in assoc.RUNNERS.items():
        assert callable(fn)
