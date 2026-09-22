"""RED-first goldens + garbage gates for statkit/means.py (Chunk 8, PLAN §10).

Seven runners: describe, t_1s, t_ind (Welch default), t_paired, anova_1w,
anova_2w, rm_anova. Each is a ``Callable[[Bound], Result]`` (the registry runner
signature); the session wires them in by swapping ``_todo`` -> the function.

NON-CIRCULAR GOLDENS (PLAN §9.1 runners list): every statistic is checked
against an INDEPENDENT oracle -- a hand-computed constant or a widely published
dataset value -- AND against a direct scipy/statsmodels cross-call on the same
data. Sources cited per test. The hand values were verified in a real venv
(pandas 3.0.6 / scipy 1.18.1 / statsmodels 0.15.0) before being pasted here.

GARBAGE GATES: a constant column, n<min, all-missing, or wrong-arity input must
yield ``status == "blocked"`` carrying the S-check's reason (via check.py) --
never a NaN result or a crash (PLAN §5.2).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from statkit import bind, means
from statkit.infer import Dataset
from statkit.model import ColumnProfile, Kind
from statkit.registry import REGISTRY

# NOTE: aliased CAT (not C) so it never shadows patsy's C() inside a formula fit.
N, O, CAT, B, ID = (Kind.NUMERIC, Kind.ORDINAL, Kind.CATEGORICAL, Kind.BINARY, Kind.ID)

APPROX = dict(rel=1e-4, abs=1e-4)


# --------------------------------------------------------------------------
# dataset / bound builders (mirrors tests/test_bind.py)
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
    """cols = {name: (values, kinds)}; dtype from the default kind."""
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


def _bound(test_id, cols, columns, layout=None, params=None):
    return bind.bind(REGISTRY[test_id], _ds(cols), columns, layout=layout,
                     params=params)


def _blocks(result):
    return {f.code for f in result.findings if f.severity == "block"}


# ==========================================================================
# 1. describe  (Table-1 descriptives)
# ==========================================================================
def test_describe_numeric_matches_numpy_pandas():
    # Independent oracle: numpy/pandas on the same array.
    y = [2, 4, 6, 8, 10]
    b = _bound("describe", {"Y": (y, (N,))}, {"variables": ("Y",)})
    r = means.describe(b)
    assert r.status == "ok"
    row = r.descriptives.iloc[0]
    assert row["n"] == 5
    assert row["mean"] == pytest.approx(np.mean(y))                 # 6.0
    assert row["sd"] == pytest.approx(np.std(y, ddof=1))            # 3.16228
    assert row["median"] == pytest.approx(6.0)
    assert row["iqr"] == pytest.approx(4.0)                         # Q3-Q1 = 8-4
    assert row["min"] == 2 and row["max"] == 10


def test_describe_by_group_splits():
    b = _bound("describe",
               {"Y": ([1, 2, 3, 4], (N,)), "G": (["A", "A", "B", "B"], (CAT,))},
               {"variables": ("Y",), "group": ("G",)})
    r = means.describe(b)
    d = r.descriptives.set_index("group")["mean"]
    assert d["A"] == pytest.approx(1.5) and d["B"] == pytest.approx(3.5)


def test_describe_constant_is_ok_not_blocked():
    # describe must summarise a constant column (S8 would wrongly block it).
    b = _bound("describe", {"Y": ([5, 5, 5, 5], (N,))}, {"variables": ("Y",)})
    r = means.describe(b)
    assert r.status == "ok"
    assert r.descriptives.iloc[0]["sd"] == pytest.approx(0.0)


def test_describe_empty_blocks():
    b = _bound("describe", {"Y": ([None, None, None], (N,))},
               {"variables": ("Y",)})
    assert means.describe(b).status == "blocked"


# ==========================================================================
# 2. t_1s  (one-sample t-test)
# ==========================================================================
def test_t_1s_golden():
    # Hand oracle: x=[2,4,6,8,10], mu0=5 -> mean 6, sd sqrt(10), SE sqrt(2),
    # t = 1/sqrt(2) = 0.70710678, df = 4. (Student 1908.)
    x = [2, 4, 6, 8, 10]
    b = _bound("t_1s", {"Y": (x, (N,))}, {"outcome": ("Y",)}, params={"mu0": 5.0})
    r = means.t_1s(b)
    assert r.status == "ok"
    assert r.statistic[0] == "t"
    assert r.statistic[1] == pytest.approx(1 / math.sqrt(2), **APPROX)   # hand
    assert r.df == (4.0,)
    # direct scipy cross-call on the same data
    sc = stats.ttest_1samp(x, 5.0)
    assert r.statistic[1] == pytest.approx(float(sc.statistic))
    assert r.p == pytest.approx(float(sc.pvalue))
    # estimate = mean - mu0 = 1; CI is scipy's mean-CI shifted by mu0
    assert r.estimate[1] == pytest.approx(1.0)
    ci = sc.confidence_interval()
    assert r.estimate_ci == pytest.approx((ci.low - 5.0, ci.high - 5.0))
    # Cohen's d one-sample = (6-5)/sqrt(10) = 0.3162278  (independent formula)
    assert r.effect[0] == "Cohen's d"
    assert r.effect[1] == pytest.approx(1 / math.sqrt(10), **APPROX)


def test_t_1s_constant_blocks_S8():
    b = _bound("t_1s", {"Y": ([5, 5, 5, 5], (N,))}, {"outcome": ("Y",)},
               params={"mu0": 0.0})
    r = means.t_1s(b)
    assert r.status == "blocked" and "S8" in _blocks(r)


def test_t_1s_too_few_blocks_S5():
    b = _bound("t_1s", {"Y": ([1, 2], (N,))}, {"outcome": ("Y",)},
               params={"mu0": 0.0})
    r = means.t_1s(b)
    assert r.status == "blocked" and "S5" in _blocks(r)


# ==========================================================================
# 3. t_ind  (Welch's independent-samples t-test -- Welch is the DEFAULT)
# ==========================================================================
def test_t_ind_welch_golden():
    # Hand oracle: A=[1..5] (mean 3, var 2.5), B=[3..7] (mean 5, var 2.5).
    # Welch SE = sqrt(2.5/5 + 2.5/5) = 1 -> t = (3-5)/1 = -2.0, Welch df = 8.
    A = [1, 2, 3, 4, 5]
    B = [3, 4, 5, 6, 7]
    b = _bound("t_ind",
               {"Y": (A + B, (N,)), "G": (["A"] * 5 + ["B"] * 5, (CAT,))},
               {"outcome": ("Y",), "group": ("G",)})
    r = means.t_ind(b)
    assert r.status == "ok"
    assert r.test_name == "Welch's independent-samples t-test"
    assert r.statistic[1] == pytest.approx(-2.0, **APPROX)      # hand
    assert r.df == (8.0,)                                       # hand Welch df
    # direct scipy cross-call, Welch (equal_var=False), the D1 default
    sc = stats.ttest_ind(A, B, equal_var=False, alternative="two-sided")
    assert r.statistic[1] == pytest.approx(float(sc.statistic))
    assert r.p == pytest.approx(float(sc.pvalue))
    # estimate = mean(A) - mean(B) = -2; direction: B is higher
    assert r.estimate[1] == pytest.approx(-2.0)
    assert r.higher == "B"
    assert r.groups == ("A", "B")
    # min n = 5 < 20 -> Hedges' g = d*J, d=-2/sqrt(2.5), J=1-3/31 -> -1.142499
    assert r.effect[0] == "Hedges' g"
    d = -2 / math.sqrt(2.5)
    g = d * (1 - 3 / (4 * 8 - 1))
    assert r.effect[1] == pytest.approx(g, **APPROX)


def test_t_ind_default_is_welch_not_student():
    # D1 / STATE: Welch is the default. The registry param `equal_var` defaults
    # False; the runner must pass scipy equal_var=False (NOT negated).
    # unequal spread so Welch's df is fractional and differs from Student's 12.
    A = [1, 2, 3, 4, 5, 6, 7]
    B = [10, 10, 11, 9, 10, 50, 10]
    b = _bound("t_ind",
               {"Y": (A + B, (N,)), "G": (["A"] * 7 + ["B"] * 7, (CAT,))},
               {"outcome": ("Y",), "group": ("G",)})
    r = means.t_ind(b)
    welch = stats.ttest_ind(A, B, equal_var=False)
    student = stats.ttest_ind(A, B, equal_var=True)
    assert r.statistic[1] == pytest.approx(float(welch.statistic))  # Welch default
    assert r.df[0] == pytest.approx(float(welch.df))               # fractional df
    assert r.df[0] != pytest.approx(float(student.df))             # not Student's 12


def test_t_ind_student_opt_in():
    A, B = [1, 2, 3, 4, 5], [3, 4, 5, 6, 8]
    b = _bound("t_ind",
               {"Y": (A + B, (N,)), "G": (["A"] * 5 + ["B"] * 5, (CAT,))},
               {"outcome": ("Y",), "group": ("G",)}, params={"equal_var": True})
    r = means.t_ind(b)
    assert r.test_name == "Student's independent-samples t-test"
    sc = stats.ttest_ind(A, B, equal_var=True)
    assert r.statistic[1] == pytest.approx(float(sc.statistic))


def test_t_ind_three_groups_blocks_S2():
    b = _bound("t_ind",
               {"Y": (list(range(9)), (N,)), "G": (["A", "B", "C"] * 3, (CAT,))},
               {"outcome": ("Y",), "group": ("G",)})
    r = means.t_ind(b)
    assert r.status == "blocked" and "S2" in _blocks(r)


def test_t_ind_all_missing_blocks():
    b = _bound("t_ind",
               {"Y": ([None, None, None, None], (N,)),
                "G": (["A", "B", "A", "B"], (CAT,))},
               {"outcome": ("Y",), "group": ("G",)})
    r = means.t_ind(b)
    assert r.status == "blocked" and "S5" in _blocks(r)


# ==========================================================================
# 4. t_paired  (paired-samples t-test)
# ==========================================================================
def test_t_paired_golden_student_sleep():
    # Published oracle: R `sleep` dataset (Student 1908; R docs):
    #   t = -4.0621, df = 9, p = 0.002833 for paired extra hours (drug1 vs drug2).
    g1 = [0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0]
    g2 = [1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4]
    b = _bound("t_paired",
               {"Pre": (g1, (N,)), "Post": (g2, (N,))},
               {"before": ("Pre",), "after": ("Post",)})
    r = means.t_paired(b)
    assert r.status == "ok"
    assert r.statistic[1] == pytest.approx(-4.0621, abs=1e-3)   # published
    assert r.df == (9.0,)
    assert r.p == pytest.approx(0.002833, abs=1e-5)             # published
    # direct scipy cross-call
    sc = stats.ttest_rel(g1, g2)
    assert r.statistic[1] == pytest.approx(float(sc.statistic))
    assert r.p == pytest.approx(float(sc.pvalue))
    # mean of differences (before - after) = -1.58
    assert r.estimate[1] == pytest.approx(-1.58, abs=1e-6)


def test_t_paired_too_few_pairs_blocks_S6():
    b = _bound("t_paired", {"Pre": ([1.0, 2.0], (N,)), "Post": ([2.0, 3.0], (N,))},
               {"before": ("Pre",), "after": ("Post",)})
    r = means.t_paired(b)
    assert r.status == "blocked" and "S6" in _blocks(r)


# ==========================================================================
# 5. anova_1w  (one-way ANOVA)
# ==========================================================================
# R `PlantGrowth` -- a standard published dataset.
_PG_CTRL = [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14]
_PG_TRT1 = [4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69]
_PG_TRT2 = [6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26]


def _pg_bound():
    vals = _PG_CTRL + _PG_TRT1 + _PG_TRT2
    grp = ["ctrl"] * 10 + ["trt1"] * 10 + ["trt2"] * 10
    return _bound("anova_1w", {"W": (vals, (N,)), "G": (grp, (CAT,))},
                  {"outcome": ("W",), "group": ("G",)})


def test_anova_1w_golden_plantgrowth():
    # Published: summary(aov(weight ~ group, PlantGrowth)) -> F(2,27)=4.846,
    # p=0.01591. eta^2 = SS_between/SS_total = 0.2641.
    r = means.anova_1w(_pg_bound())
    assert r.status == "ok"
    assert r.statistic[0] == "F"
    assert r.statistic[1] == pytest.approx(4.8461, abs=1e-3)    # published
    assert r.df == (2.0, 27.0)
    assert r.p == pytest.approx(0.01591, abs=1e-4)              # published
    # direct scipy cross-call
    F, p = stats.f_oneway(_PG_CTRL, _PG_TRT1, _PG_TRT2)
    assert r.statistic[1] == pytest.approx(float(F))
    assert r.p == pytest.approx(float(p))
    assert r.effect[0] == "η²"
    assert r.effect[1] == pytest.approx(0.26415, abs=1e-4)


def test_anova_1w_posthoc_tukey_when_significant():
    r = means.anova_1w(_pg_bound())
    assert r.posthoc_name == "Tukey HSD"
    assert r.posthoc is not None and len(r.posthoc) == 3       # 3 pairs


def test_anova_1w_welch_opt_in():
    b = _bound("anova_1w",
               {"W": (_PG_CTRL + _PG_TRT1 + _PG_TRT2, (N,)),
                "G": (["ctrl"] * 10 + ["trt1"] * 10 + ["trt2"] * 10, (CAT,))},
               {"outcome": ("W",), "group": ("G",)}, params={"welch": True})
    r = means.anova_1w(b)
    assert r.status == "ok" and r.test_name == "Welch's ANOVA"
    from statsmodels.stats.oneway import anova_oneway
    res = anova_oneway([_PG_CTRL, _PG_TRT1, _PG_TRT2], use_var="unequal",
                       welch_correction=True)
    assert r.statistic[1] == pytest.approx(float(res.statistic))
    assert r.posthoc_name == "Games-Howell"


def test_anova_1w_two_groups_blocks_S2():
    b = _bound("anova_1w",
               {"W": (list(range(10)), (N,)), "G": (["A", "B"] * 5, (CAT,))},
               {"outcome": ("W",), "group": ("G",)})
    r = means.anova_1w(b)
    assert r.status == "blocked" and "S2" in _blocks(r)


# ==========================================================================
# 6. anova_2w  (two-way ANOVA, Type II, Sum contrasts)
# ==========================================================================
def _twoway_bound():
    # Balanced 2x2x3 design; hand SS decomposition (means-based) is the
    # independent oracle, confirmed against statsmodels anova_lm(typ=2):
    #   SS_A=12 (F=12), SS_B=27 (F=27), SS_AB=3 (F=3), SS_resid=8 (df=8).
    cells = {("a1", "b1"): [1, 2, 3], ("a1", "b2"): [3, 4, 5],
             ("a2", "b1"): [2, 3, 4], ("a2", "b2"): [6, 7, 8]}
    y, fa, fb = [], [], []
    for (a, bb), vv in cells.items():
        for v in vv:
            y.append(v); fa.append(a); fb.append(bb)
    return _bound("anova_2w",
                  {"Y": (y, (N,)), "A": (fa, (CAT,)), "B": (fb, (CAT,))},
                  {"outcome": ("Y",), "factor_a": ("A",), "factor_b": ("B",)})


def test_anova_2w_golden_balanced():
    r = means.anova_2w(_twoway_bound())
    assert r.status == "ok"
    t = r.table
    # hand SS decomposition (independent oracle)
    assert t.loc["factor_a", "sum_sq"] == pytest.approx(12.0)
    assert t.loc["factor_b", "sum_sq"] == pytest.approx(27.0)
    assert t.loc["interaction", "sum_sq"] == pytest.approx(3.0)
    assert t.loc["Residual", "sum_sq"] == pytest.approx(8.0)
    assert t.loc["factor_a", "F"] == pytest.approx(12.0)
    assert t.loc["factor_b", "F"] == pytest.approx(27.0)
    assert t.loc["interaction", "F"] == pytest.approx(3.0)
    assert t.loc["Residual", "df"] == pytest.approx(8.0)
    # primary reported term = the interaction
    assert r.statistic == ("F", pytest.approx(3.0))
    assert r.df == (1.0, 8.0)
    assert r.p == pytest.approx(0.121503, abs=1e-5)
    # partial eta^2 for the interaction = 3 / (3 + 8)
    assert r.effect[0] == "partial η²"
    assert r.effect[1] == pytest.approx(3 / 11, abs=1e-6)


def test_anova_2w_crosscall_statsmodels():
    # pipeline-integrity cross-call: a fresh statsmodels fit on the same data.
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    b = _twoway_bound()
    df = pd.DataFrame({"outcome": b.data["outcome"].to_numpy(float),
                       "factor_a": b.data["factor_a"].astype(str).to_numpy(),
                       "factor_b": b.data["factor_b"].astype(str).to_numpy()})
    m = smf.ols("outcome ~ C(factor_a, Sum)*C(factor_b, Sum)", data=df).fit()
    ref = anova_lm(m, typ=2)
    r = means.anova_2w(b)
    assert r.table.loc["factor_a", "F"] == pytest.approx(
        float(ref.loc["C(factor_a, Sum)", "F"]))
    assert r.table.loc["interaction", "F"] == pytest.approx(
        float(ref.loc["C(factor_a, Sum):C(factor_b, Sum)", "F"]))


def test_anova_2w_empty_cell_blocks_S19():
    b = _bound("anova_2w",
               {"Y": ([1, 2, 3, 4, 5, 6], (N,)),
                "A": (["X", "X", "X", "Y", "Y", "Y"], (CAT,)),
                "B": (["P", "Q", "P", "P", "P", "P"], (CAT,))},
               {"outcome": ("Y",), "factor_a": ("A",), "factor_b": ("B",)})
    r = means.anova_2w(b)
    assert r.status == "blocked" and "S19" in _blocks(r)


# ==========================================================================
# 7. rm_anova  (repeated-measures ANOVA)
# ==========================================================================
def _rm_bound():
    # 4 subjects x 3 conditions. Hand SS decomposition (independent oracle):
    #   SS_cond=74/3, SS_subj=18, SS_err=2, df=(2,6) -> F=37.0.
    m0 = [10, 11, 9, 12]
    m1 = [12, 14, 11, 13]
    m2 = [13, 16, 12, 15]
    return _bound("rm_anova",
                  {"C1": (m0, (N,)), "C2": (m1, (N,)), "C3": (m2, (N,))},
                  {"measures": ("C1", "C2", "C3")})


def test_rm_anova_golden():
    r = means.rm_anova(_rm_bound())
    assert r.status == "ok"
    assert r.statistic[0] == "F"
    assert r.statistic[1] == pytest.approx(37.0, abs=1e-3)      # hand oracle
    assert r.df == (2.0, 6.0)
    # direct statsmodels cross-call (AnovaRM) on the same data
    from statsmodels.stats.anova import AnovaRM
    rows = []
    for s, (a, bb, c) in enumerate(zip([10, 11, 9, 12], [12, 14, 11, 13],
                                       [13, 16, 12, 15])):
        rows += [(s, "c0", float(a)), (s, "c1", float(bb)), (s, "c2", float(c))]
    ldf = pd.DataFrame(rows, columns=["subject", "condition", "outcome"])
    at = AnovaRM(ldf, "outcome", "subject", within=["condition"]).fit().anova_table
    assert r.statistic[1] == pytest.approx(float(at["F Value"].iloc[0]))
    assert r.p == pytest.approx(float(at["Pr > F"].iloc[0]))
    # partial eta^2 = F*df1/(F*df1+df2) = 74/80
    assert r.effect[1] == pytest.approx(74 / 80, abs=1e-6)


def test_rm_anova_too_few_subjects_blocks_S6():
    b = _bound("rm_anova",
               {"C1": ([1.0, 2.0], (N,)), "C2": ([2.0, 3.0], (N,)),
                "C3": ([3.0, 4.0], (N,))},
               {"measures": ("C1", "C2", "C3")})
    r = means.rm_anova(b)
    assert r.status == "blocked" and "S6" in _blocks(r)


# ==========================================================================
# B6 / S12 — long-layout condition labels + mu0 plumbing (batch 1A)
# ==========================================================================
def test_t_paired_long_layout_names_conditions():
    # long sheet, "Post" first-seen -> before=Post, after=Pre (the pivot order).
    ds = _ds({
        "Pt":   (["p1", "p1", "p2", "p2", "p3", "p3"], (ID, CAT)),
        "Time": (["Post", "Pre", "Post", "Pre", "Post", "Pre"], (CAT,)),
        "BP":   ([110, 120, 118, 130, 115, 140], (N,)),
    })
    b = bind.bind(REGISTRY["t_paired"], ds,
                  {"subject": ("Pt",), "condition": ("Time",), "outcome": ("BP",)},
                  layout="long")
    r = means.t_paired(b)
    assert r.labels["before"] == "Post"
    assert r.estimate[0] == "mean difference (Post − Pre)"


def test_rm_anova_long_layout_groups_are_condition_labels():
    from synth import long_k3
    from statkit import grid, clean, infer as inf
    data, name = long_k3(subjects=6)
    ds = inf.infer(clean.clean(grid.load(data, name)[0]))
    b = bind.bind(REGISTRY["rm_anova"], ds,
                  {"subject": ("Subject",), "condition": ("Condition",),
                   "outcome": ("Score",)}, layout="long")
    r = means.rm_anova(b)
    assert r.groups == ("Baseline", "Week4", "Week8")


def test_t_1s_carries_mu0():
    b = _bound("t_1s", {"Y": ([2, 4, 6, 8, 10], (N,))}, {"outcome": ("Y",)},
               params={"mu0": 5.0})
    r = means.t_1s(b)
    assert r.extra["mu0"] == 5.0


# ==========================================================================
# F4 — numeric-coded group columns render as "1"/"2", never "1.0"/"2.0"
# (Group=1/2, Sex=0/1 stored as floats). The array SPLIT stays keyed on the
# raw value; only the LABELS a student reads change. Existing string-level
# goldens are untouched because levels.display(str) == str.
# ==========================================================================
def test_t_ind_numeric_coded_group_labels_are_display_strings():
    # Same numbers as the Welch golden, but the group is coded 1/2 (stored as
    # floats via the BINARY kind). Before the fix: groups ("1.0","2.0"),
    # higher "2.0", estimate "mean difference (1.0 − 2.0)".
    # NB: don't name a local 'B' here -- it shadows the Kind.BINARY alias.
    g1 = [1, 2, 3, 4, 5]      # group 1, mean 3
    g2 = [3, 4, 5, 6, 7]      # group 2, mean 5 -> higher
    b = _bound("t_ind",
               {"Y": (g1 + g2, (N,)), "G": ([1] * 5 + [2] * 5, (B,))},
               {"outcome": ("Y",), "group": ("G",)})
    r = means.t_ind(b)
    assert r.status == "ok"
    assert r.statistic[1] == pytest.approx(-2.0, **APPROX)   # numbers unchanged
    assert r.groups == ("1", "2")                            # NOT ("1.0","2.0")
    assert r.higher == "2"                                   # NOT "2.0"
    assert r.estimate[0] == "mean difference (1 − 2)"        # NOT (1.0 − 2.0)
    assert "1" in r.n and "2" in r.n                          # per-group n keys


def test_anova_1w_numeric_coded_group_labels_and_posthoc():
    # PlantGrowth numbers, groups coded 1/2/3 (stored as floats). Post-hoc
    # (significant -> Tukey) labels must read "1"/"2"/"3", never "1.0" etc.
    vals = _PG_CTRL + _PG_TRT1 + _PG_TRT2
    grp = [1] * 10 + [2] * 10 + [3] * 10
    b = _bound("anova_1w", {"W": (vals, (N,)), "G": (grp, (O,))},
               {"outcome": ("W",), "group": ("G",)})
    r = means.anova_1w(b)
    assert r.status == "ok"
    assert r.statistic[1] == pytest.approx(4.8461, abs=1e-3)  # numbers unchanged
    assert r.groups == ("1", "2", "3")                        # NOT ("1.0",...)
    labels = set(r.posthoc["group1"]) | set(r.posthoc["group2"])
    assert labels == {"1", "2", "3"}                          # no "1.0"/"2.0"/"3.0"


def test_describe_numeric_coded_group_reads_display():
    b = _bound("describe",
               {"Y": ([1, 2, 3, 4], (N,)), "G": ([1, 1, 2, 2], (B,))},
               {"variables": ("Y",), "group": ("G",)})
    r = means.describe(b)
    assert r.status == "ok"
    assert set(r.descriptives["group"]) == {"1", "2"}         # NOT {"1.0","2.0"}


def test_describe_categorical_by_numeric_coded_group_reads_display():
    b = _bound("describe",
               {"Cat": (["x", "y", "x", "y"], (CAT,)), "G": ([1, 1, 2, 2], (B,))},
               {"variables": ("Cat",), "group": ("G",)})
    r = means.describe(b)
    assert r.status == "ok"
    assert set(r.extra["categoricals"]["Cat"]) == {"1", "2"}  # group keys, not "1.0"


# ==========================================================================
# runner registry-signature contract
# ==========================================================================
def test_runners_mapping_covers_chunk8():
    assert set(means.RUNNERS) == {
        "describe", "t_1s", "t_ind", "t_paired", "anova_1w", "anova_2w",
        "rm_anova"}
    for fn in means.RUNNERS.values():
        assert callable(fn)
