"""RED-first goldens for statkit.ranks (Chunk 9 runners: mwu, wilcoxon, kruskal,
friedman).

Non-circular design (task + PLAN §9.1 runners): every runner is checked against
(a) a hand-checked / published constant AND (b) a direct scipy cross-call, so the
literal is the independent oracle and the cross-call guards the wiring (args:
alternative='two-sided', zero_method='wilcox', method='auto', ...).

Runner signature (registry/model): ``run(bound) -> Result`` where the Bound is the
canonical frame from bind() (columns named by role). Each runner: check -> (block
Result if blocked) -> statistic -> effect -> Result; NaN statistic/p or an
all-zero Wilcoxon -> status='blocked' (silent-garbage gate, PLAN §5.2).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import scipy.stats as ss

from statkit import bind, ranks
from statkit.infer import Dataset
from statkit.model import ColumnProfile, Kind
from statkit.registry import REGISTRY

N, O, C, B, ID = (Kind.NUMERIC, Kind.ORDINAL, Kind.CATEGORICAL, Kind.BINARY, Kind.ID)


# --------------------------------------------------------------------------
# Dataset / bind helpers (same idiom as tests/test_bind.py)
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
        numeric_vals = all(isinstance(v, (int, float)) for v in non_null)
        if kinds[0] in (Kind.NUMERIC, Kind.ORDINAL, Kind.BINARY, Kind.ID) and numeric_vals:
            df_cols[name] = pd.array(
                [None if _isna(v) else float(v) for v in values], dtype="float64")
        else:
            df_cols[name] = pd.array(
                [pd.NA if _isna(v) else str(v) for v in values], dtype="string")
        profiles.append(_profile(name, kinds, values))
    df = pd.DataFrame(df_cols)
    excel_rows = tuple(excel_rows or range(2, 2 + n))
    return Dataset(df=df, profiles=tuple(profiles), excel_rows=excel_rows)


def _two_group(a, b, la="A", lb="B"):
    vals = list(a) + list(b)
    grp = [la] * len(a) + [lb] * len(b)
    return _ds({"Score": (vals, (N,)), "Grp": (grp, (C,))})


# ==========================================================================
# mwu — Mann-Whitney U
# ==========================================================================
def test_mwu_tiny_hand_case_U0_and_rank_biserial_one():
    # a=[1,2,3] all below b=[4,5,6]: U (for a) = 0 -> exact two-sided p = 0.1;
    # rank-biserial r = 1 - 2*0/(3*3) = 1.0. Fully hand-checkable.
    ds = _two_group([1, 2, 3], [4, 5, 6])
    b = bind.bind(REGISTRY["mwu"], ds, {"outcome": ("Score",), "group": ("Grp",)})
    r = ranks.mwu(b)
    assert r.status == "ok"
    assert r.statistic[0] == "U" and r.statistic[1] == 0.0
    assert r.p == pytest.approx(0.1, abs=1e-9)
    assert abs(r.effect[1]) == pytest.approx(1.0)
    assert r.effect_label == "large"
    assert r.higher == "B"                                   # B values are higher


def test_mwu_matches_scipy_and_published_constant():
    # Published (SciPy docs example): U=17, two-sided p=0.1111 for these samples.
    males = [19, 22, 16, 29, 24]
    females = [20, 11, 17, 12]
    ds = _two_group(males, females, la="Males", lb="Females")
    b = bind.bind(REGISTRY["mwu"], ds, {"outcome": ("Score",), "group": ("Grp",)})
    r = ranks.mwu(b)
    assert r.p == pytest.approx(0.1111, abs=5e-4)            # published constant
    # cross-call: same p as scipy called directly on the same split
    ref = ss.mannwhitneyu(np.array(males, float), np.array(females, float),
                          alternative="two-sided", method="auto")
    assert r.p == pytest.approx(ref.pvalue)


def test_mwu_prints_the_method_actually_used_exact():
    # both n<=8 and no ties -> exact; the note must be truthful (== scipy exact p)
    ds = _two_group([1, 2, 3], [4, 5, 6])
    b = bind.bind(REGISTRY["mwu"], ds, {"outcome": ("Score",), "group": ("Grp",)})
    r = ranks.mwu(b)
    assert any("exact" in v for v in r.variant_notes)
    ref = ss.mannwhitneyu([1., 2., 3.], [4., 5., 6.],
                          alternative="two-sided", method="exact")
    assert r.p == pytest.approx(ref.pvalue)


# ==========================================================================
# wilcoxon — signed-rank (paired + one-sample vs mu0)
# ==========================================================================
def test_wilcoxon_paired_matches_scipy_and_hand_rank_biserial():
    x = [20, 18, 24, 14, 5, 18, 14, 12, 19, 17]
    y = [16, 10, 20, 10, 3, 17, 12, 10, 15, 14]      # every diff > 0 (no zeros/ties issues)
    ds = _ds({"Pre": (x, (N,)), "Post": (y, (N,))})
    b = bind.bind(REGISTRY["wilcoxon"], ds, {"before": ("Pre",), "after": ("Post",)})
    r = ranks.wilcoxon(b)
    assert r.status == "ok"
    ref = ss.wilcoxon(np.array(x, float), np.array(y, float),
                      zero_method="wilcox", alternative="two-sided")
    assert r.statistic[0] == "W" and r.statistic[1] == pytest.approx(ref.statistic)
    assert r.p == pytest.approx(ref.pvalue)
    assert r.p == pytest.approx(0.00195, abs=1e-4)   # published/hand value
    # all diffs positive -> W- = 0 -> matched-pairs rank-biserial = +1.0
    assert r.effect[1] == pytest.approx(1.0)
    assert r.estimate[1] == pytest.approx(np.median(np.array(x) - np.array(y)))


def test_wilcoxon_one_sample_branch_on_absence_of_after():
    # a lone outcome column + mu0 -> wilcoxon(x - mu0)
    x = [5.0, 6.5, 4.0, 7.0, 8.0, 3.5, 6.0]
    ds = _ds({"Change": (x, (N, O))})
    b = bind.bind(REGISTRY["wilcoxon"], ds, {"outcome": ("Change",)},
                  layout="long", params={"mu0": 5.0})
    r = ranks.wilcoxon(b)
    assert r.status == "ok"
    ref = ss.wilcoxon(np.array(x, float) - 5.0, zero_method="wilcox",
                      alternative="two-sided")
    assert r.statistic[1] == pytest.approx(ref.statistic)
    assert r.p == pytest.approx(ref.pvalue)
    assert "one-sample" in r.test_name.lower()


def test_wilcoxon_all_zero_differences_is_blocked():
    ds = _ds({"Pre": ([5, 6, 7, 8], (N,)), "Post": ([5, 6, 7, 8], (N,))})
    b = bind.bind(REGISTRY["wilcoxon"], ds, {"before": ("Pre",), "after": ("Post",)})
    r = ranks.wilcoxon(b)
    assert r.status == "blocked"


# ==========================================================================
# kruskal — Kruskal-Wallis H (+ Dunn post-hoc)
# ==========================================================================
def _three_group(a, b, c):
    vals = list(a) + list(b) + list(c)
    grp = ["A"] * len(a) + ["B"] * len(b) + ["C"] * len(c)
    return _ds({"Score": (vals, (N,)), "Grp": (grp, (C,))})


def test_kruskal_matches_scipy_with_epsilon_squared_and_dunn():
    a, b, c = [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]
    ds = _three_group(a, b, c)
    bd = bind.bind(REGISTRY["kruskal"], ds, {"outcome": ("Score",), "group": ("Grp",)})
    r = ranks.kruskal(bd)
    assert r.status == "ok"
    ref = ss.kruskal(np.array(a, float), np.array(b, float), np.array(c, float))
    assert r.statistic[0] == "H" and r.statistic[1] == pytest.approx(ref.statistic)
    assert r.p == pytest.approx(ref.pvalue)
    assert r.statistic[1] == pytest.approx(12.5, abs=1e-9)          # hand value
    # epsilon^2 = (H - k + 1)/(n - k) = (12.5 - 3 + 1)/(15 - 3) = 0.875
    assert r.effect[1] == pytest.approx(0.875, abs=1e-9)
    # significant -> Dunn post-hoc auto-runs, one row per pair, Holm column present
    assert r.posthoc is not None and len(r.posthoc) == 3
    assert "p_holm" in r.posthoc.columns


def test_kruskal_not_significant_has_no_posthoc():
    # SciPy docs example: H=0.771, p=0.68 -> no post-hoc
    x1, x2, x3 = [2.9, 3.0, 2.5, 2.6, 3.2], [3.8, 2.7, 4.0, 2.4], [2.8, 3.4, 3.7, 2.2, 2.0]
    ds = _three_group(x1, x2, x3)
    bd = bind.bind(REGISTRY["kruskal"], ds, {"outcome": ("Score",), "group": ("Grp",)})
    r = ranks.kruskal(bd)
    assert r.p == pytest.approx(0.68, abs=1e-2)
    assert r.posthoc is None


def test_kruskal_constant_outcome_is_blocked():
    ds = _three_group([5, 5], [5, 5], [5, 5])
    bd = bind.bind(REGISTRY["kruskal"], ds, {"outcome": ("Score",), "group": ("Grp",)})
    r = ranks.kruskal(bd)
    assert r.status == "blocked"


# ==========================================================================
# friedman — Friedman test (+ pairwise Wilcoxon-Holm post-hoc)
# ==========================================================================
def test_friedman_tiny_hand_case_chi2_6_and_kendalls_w_one():
    # perfect concordance: 3 subjects rank 3 treatments identically ->
    # chi2_r = 6.0, p = 0.0498, Kendall's W = chi2/(n(k-1)) = 6/(3*2) = 1.0
    ds = _ds({"T1": ([1, 2, 3], (N,)), "T2": ([2, 3, 4], (N,)), "T3": ([3, 4, 5], (N,))})
    bd = bind.bind(REGISTRY["friedman"], ds, {"measures": ("T1", "T2", "T3")}, layout="wide")
    r = ranks.friedman(bd)
    assert r.status == "ok"
    ref = ss.friedmanchisquare([1., 2., 3.], [2., 3., 4.], [3., 4., 5.])
    assert r.statistic[1] == pytest.approx(ref.statistic)
    assert r.statistic[1] == pytest.approx(6.0)
    assert r.p == pytest.approx(0.049787, abs=1e-5)
    assert r.effect[1] == pytest.approx(1.0)                        # Kendall's W


def test_friedman_significant_runs_pairwise_wilcoxon_holm():
    from synth import long_k3
    from statkit import grid, clean, infer as inf
    data, name = long_k3(subjects=8)
    ds = inf.infer(clean.clean(grid.load(data, name)[0]))
    bd = bind.bind(REGISTRY["friedman"], ds,
                   {"subject": ("Subject",), "condition": ("Condition",),
                    "outcome": ("Score",)}, layout="long")
    r = ranks.friedman(bd)
    assert r.status == "ok"
    cols = [bd.data[c].to_numpy(float) for c in bd.data.columns]
    ref = ss.friedmanchisquare(*cols)
    assert r.statistic[1] == pytest.approx(ref.statistic)
    assert r.p == pytest.approx(ref.pvalue)
    assert r.posthoc is not None and len(r.posthoc) == 3           # 3 pairs of 3 conditions
    assert "p_holm" in r.posthoc.columns


# ==========================================================================
# B1 / NICE — MWU direction from mean ranks + rank-biserial sign (batch 1A)
# ==========================================================================
def test_mwu_direction_follows_mean_ranks_not_means():
    # a has a huge outlier (1000) so its raw MEAN exceeds b's, but by rank b is
    # higher: a occupies ranks 1-10 + 22 (mean rank 7), b ranks 11-21 (mean 16).
    # `higher` must follow the MEAN RANKS, which is what MWU actually tests (B1).
    a = list(range(1, 11)) + [1000]
    b = list(range(11, 22))
    ds = _two_group(a, b, la="a", lb="b")
    bd = bind.bind(REGISTRY["mwu"], ds, {"outcome": ("Score",), "group": ("Grp",)})
    r = REGISTRY["mwu"].run(bd)
    assert r.higher == "b"                                    # was "a" (raw means)
    assert r.estimate[0] == "Hodges-Lehmann median difference (a − b)"


def test_mwu_rank_biserial_sign_matches_higher():
    # sign convention (Kerby 2014): rank-biserial is positive when groups[0]
    # tends higher, negative otherwise. Here b (=groups[1]) is higher -> negative.
    a = list(range(1, 11)) + [1000]
    b = list(range(11, 22))
    ds = _two_group(a, b, la="a", lb="b")
    bd = bind.bind(REGISTRY["mwu"], ds, {"outcome": ("Score",), "group": ("Grp",)})
    r = REGISTRY["mwu"].run(bd)
    # concrete red: b (=groups[1]) is rank-higher, so rb must be NEGATIVE
    # (magnitude unchanged; today rb = +0.82 -- vacuously agreed with a wrong `higher`).
    assert r.effect[1] < 0
    assert abs(r.effect[1]) == pytest.approx(1 - 2 * 11 / (11 * 11))   # |rb| = 0.818
    assert np.sign(r.effect[1]) == (1 if r.higher == r.groups[0] else -1)


# ==========================================================================
# S12 — wilcoxon paired reports condition names + rank-based direction (1A)
# ==========================================================================
def test_wilcoxon_paired_reports_names_and_direction():
    x = [20, 18, 24, 14, 5, 18, 14, 12, 19, 17]
    y = [16, 10, 20, 10, 3, 17, 12, 10, 15, 14]
    ds = _ds({"Pre": (x, (N,)), "Post": (y, (N,))})
    b = bind.bind(REGISTRY["wilcoxon"], ds, {"before": ("Pre",), "after": ("Post",)})
    r = ranks.wilcoxon(b)
    assert r.labels == {"before": "Pre", "after": "Post"}     # was {}
    assert r.estimate[0] == "median of differences (Pre − Post)"
    assert r.higher in r.groups


# ==========================================================================
# B6 — friedman long layout posthoc uses condition labels, not measures__i (1A)
# ==========================================================================
def test_friedman_long_layout_posthoc_uses_condition_labels():
    from synth import long_k3
    from statkit import grid, clean, infer as inf
    data, name = long_k3(subjects=8)
    ds = inf.infer(clean.clean(grid.load(data, name)[0]))
    bd = bind.bind(REGISTRY["friedman"], ds,
                   {"subject": ("Subject",), "condition": ("Condition",),
                    "outcome": ("Score",)}, layout="long")
    r = ranks.friedman(bd)
    assert r.posthoc is not None
    names = {"Baseline", "Week4", "Week8"}
    assert set(r.posthoc["group1"]) | set(r.posthoc["group2"]) <= names


# ==========================================================================
# F1 / F4 — numeric-coded group labels (1/2, 0/1, 1/2/3) must run, not crash,
# and read as display strings ('1'/'2'), not floats (1.0/2.0). The group column
# is stored as float64 (kind B/O), so _group_arrays keys the dict by float and
# `_n(bound, **{la: n1})` used to raise `TypeError: keywords must be strings`.
# ==========================================================================
def _two_group_numeric(a, b, la=1.0, lb=2.0):
    vals = list(a) + list(b)
    grp = [la] * len(a) + [lb] * len(b)
    return _ds({"Score": (vals, (N,)), "Grp": (grp, (B,))})


def _three_group_numeric(a, b, c, codes=(1.0, 2.0, 3.0)):
    vals = list(a) + list(b) + list(c)
    grp = ([codes[0]] * len(a) + [codes[1]] * len(b) + [codes[2]] * len(c))
    return _ds({"Score": (vals, (N,)), "Grp": (grp, (O,))})


def test_mwu_numeric_coded_group_runs_and_labels_are_display_strings():
    # F1: the commonest student encoding (Group = 1/2). Used to crash in _n.
    ds = _two_group_numeric([1, 2, 3], [4, 5, 6], la=1.0, lb=2.0)
    b = bind.bind(REGISTRY["mwu"], ds, {"outcome": ("Score",), "group": ("Grp",)})
    r = ranks.mwu(b)
    assert r.status == "ok"
    assert r.statistic[0] == "U" and r.statistic[1] == 0.0     # same U as the A/B case
    assert r.p == pytest.approx(0.1, abs=1e-9)
    assert r.groups == ("1", "2")                              # display strings, not 1.0/2.0
    assert r.higher == "2"                                     # group-2 values are higher
    assert r.estimate[0] == "Hodges-Lehmann median difference (1 − 2)"
    assert set(r.n) >= {"1", "2"}                              # per-group counts keyed by display


def test_kruskal_numeric_coded_group_runs_and_labels_are_display_strings():
    # F1 (kruskal shape): 1/2/3-coded groups must run and read '1'/'2'/'3'.
    a, b, c = [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]
    ds = _three_group_numeric(a, b, c)
    bd = bind.bind(REGISTRY["kruskal"], ds, {"outcome": ("Score",), "group": ("Grp",)})
    r = ranks.kruskal(bd)
    assert r.status == "ok"
    assert r.statistic[1] == pytest.approx(12.5, abs=1e-9)     # hand value, unchanged
    assert r.groups == ("1", "2", "3")                         # display strings
    assert set(r.n) >= {"1", "2", "3"}
    # significant -> Dunn post-hoc labels also read as display strings, not floats
    assert r.posthoc is not None
    assert set(r.posthoc["group1"]) | set(r.posthoc["group2"]) <= {"1", "2", "3"}


# ==========================================================================
# F7 — the Kruskal-Wallis effect is η²_H (H-based eta-squared), NOT epsilon^2.
# (H - k + 1)/(n - k) is what Tomczak & Tomczak (2014) call eta-squared (H);
# their epsilon^2 is the different H/(n - 1). Only the LABEL was wrong.
# ==========================================================================
def test_kruskal_effect_label_is_eta_squared_h_not_epsilon():
    a, b, c = [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]
    ds = _three_group(a, b, c)
    bd = bind.bind(REGISTRY["kruskal"], ds, {"outcome": ("Score",), "group": ("Grp",)})
    r = ranks.kruskal(bd)
    assert r.effect[0] == "eta-squared (H)"                    # was "epsilon-squared"
    assert r.effect_source == "Tomczak & Tomczak (2014)"       # they define this exact formula
    assert r.effect[1] == pytest.approx(0.875, abs=1e-9)       # NUMBER unchanged


# ==========================================================================
# module contract
# ==========================================================================
def test_runners_mapping_exposes_all_four():
    assert set(ranks.RUNNERS) == {"mwu", "wilcoxon", "kruskal", "friedman"}
    for fn in ranks.RUNNERS.values():
        assert callable(fn)
