"""RED-first goldens for statkit.cat (Chunk 9 runners: chi2_ind, chi2_gof, fisher,
mcnemar, cochran_q).

DoD specifics proven here (task / PLAN §10): the no-Yates chi-square value differs
from the Yates-corrected one (correction=False); Fisher fires (block -> suggest
fisher) when an expected count is < 5 in a 2x2; Fisher labels 2x2 vs R x C
correctly; McNemar switches exact <-> chi2 at b+c=25 and keeps a valid p when
b*c=0 (snag 15). Each runner: hand/published constant + a direct scipy/statsmodels
cross-call.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import scipy.stats as ss
from scipy.stats import contingency
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar as sm_mcnemar

from statkit import bind, cat
from statkit.infer import Dataset
from statkit.model import ColumnProfile, Kind
from statkit.registry import REGISTRY

N, O, C, B = Kind.NUMERIC, Kind.ORDINAL, Kind.CATEGORICAL, Kind.BINARY


# --------------------------------------------------------------------------
# helpers (same idiom as tests/test_bind.py)
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


def _grid_bind(spec_id, row_labels, count_cols):
    """A count-grid Dataset -> Bound via the table binder (chi2_ind / fisher)."""
    cols = {"Region": (row_labels, (C,))}
    for name, vals in count_cols.items():
        cols[name] = (vals, (N,))
    ds = _ds(cols)
    return bind.bind(REGISTRY[spec_id], ds, {"counts": tuple(count_cols)}, layout="table")


def _long_bind(spec_id, row_name, col_name, cells):
    """A long two-column Dataset -> Bound (chi2_ind / fisher long layout).

    ``cells`` = {(row_value, col_value): count} -> one row per counted unit, so the
    crosstab reproduces those counts exactly and the source headers survive on the
    Bound (a table/count-grid bind carries only 'counts', no row/col headers).
    """
    rows, cols = [], []
    for (rv, cv), k in cells.items():
        rows.extend([rv] * k)
        cols.extend([cv] * k)
    ds = _ds({row_name: (rows, (C,)), col_name: (cols, (C,))})
    return bind.bind(REGISTRY[spec_id], ds,
                     {"row": (row_name,), "col": (col_name,)}, layout="long")


def _pairs_ds(pairs):
    """before/after wide Dataset from a list of (before, after) pairs."""
    before = [p[0] for p in pairs]
    after = [p[1] for p in pairs]
    return _ds({"Before": (before, (B,)), "After": (after, (B,))})


# ==========================================================================
# chi2_ind — no Yates + Fisher routing + Cramer's V
# ==========================================================================
def test_chi2_ind_uses_correction_false_and_differs_from_yates():
    # [[12,8],[5,15]] all expected >= 5. correction=False p=.02516 (sig);
    # Yates p=.05497 (not sig) -> a genuine sign-flip across alpha.
    b = _grid_bind("chi2_ind", ["R1", "R2"], {"C1": [12, 5], "C2": [8, 15]})
    r = cat.chi2_ind(b)
    assert r.status == "ok"
    ref = ss.chi2_contingency(np.array([[12, 8], [5, 15]]), correction=False)
    yates = ss.chi2_contingency(np.array([[12, 8], [5, 15]]), correction=True)
    assert r.statistic[1] == pytest.approx(ref.statistic)
    assert r.p == pytest.approx(ref.pvalue)
    assert r.p == pytest.approx(0.02516, abs=1e-4)          # published constant
    assert r.p < 0.05 < yates.pvalue                        # no-Yates != Yates (sign-flip)


def test_chi2_ind_routes_to_fisher_when_expected_below_5():
    # [[8,2],[1,5]] -> expected min 2.625 < 5 in a 2x2 -> block, suggest fisher
    b = _grid_bind("chi2_ind", ["R1", "R2"], {"C1": [8, 1], "C2": [2, 5]})
    r = cat.chi2_ind(b)
    assert r.status == "blocked"
    assert any(f.suggest_test == "fisher" for f in r.findings)


def test_chi2_ind_rxc_reports_cramers_v_and_residuals():
    b = _grid_bind("chi2_ind", ["North", "South", "East"],
                   {"A": [30, 20, 25], "B": [25, 30, 20], "Cc": [20, 25, 30]})
    r = cat.chi2_ind(b)
    assert r.status == "ok"
    assert r.effect[0] == "Cramér's V"
    assert r.table is not None                               # adjusted residuals (R x C)
    assert r.expected is not None


def test_chi2_2x2_result_names_variables_and_orients_odds_ratio():
    # sex x smoke, counts I control: F/no=15 F/yes=5  M/no=8 M/yes=12.
    # crosstab sorts -> index ["F","M"], columns ["no","yes"]; every expected >=5
    # (no Fisher routing). Odds of "no" are HIGHER for F (15/5=3 > 8/12=0.67).
    b = _long_bind("chi2_ind", "sex", "smoke",
                   {("F", "no"): 15, ("F", "yes"): 5,
                    ("M", "no"): 8, ("M", "yes"): 12})
    r = cat.chi2_ind(b)
    assert r.status == "ok"
    assert r.labels == {"row": "sex", "col": "smoke"}        # was {} (S4)
    assert r.extra["observed"].loc["F", "no"] == 15          # observed reaches the report
    assert r.extra["or"] == {"col_level": "no", "row_level": "F", "row_ref": "M"}
    # hand sample OR = (a*d)/(b*c) = (15*12)/(5*8) = 4.5 = odds(no|F)/odds(no|M) > 1,
    # i.e. exactly the orientation the 'or' dict names (col_level 'no', row 'F' vs 'M').
    assert (15 * 12) / (5 * 8) == pytest.approx(4.5)
    # estimate is the conditional MLE OR on [[F/no,F/yes],[M/no,M/yes]] -> same direction
    mle = contingency.odds_ratio(np.array([[15, 5], [8, 12]])).statistic
    assert r.estimate[1] == pytest.approx(mle)
    assert r.estimate[1] > 1


def test_chi2_rxc_carries_observed_and_residuals():
    b = _grid_bind("chi2_ind", ["North", "South", "East"],
                   {"A": [30, 20, 25], "B": [25, 30, 20], "Cc": [20, 25, 30]})
    r = cat.chi2_ind(b)
    assert r.status == "ok"
    assert r.extra["observed"].shape == (3, 3)              # observed 3x3 crosstab
    assert r.extra["adjusted_residuals"].shape == (3, 3)    # was KeyError (S4)
    # a count-grid (table layout) has no source headers -> generic variable names
    assert r.labels == {"row": "row variable", "col": "column variable"}


# ==========================================================================
# chi2_gof — goodness of fit + Cohen's w
# ==========================================================================
def test_chi2_gof_equal_expected_matches_scipy_and_cohens_w():
    cats = ["A"] * 10 + ["B"] * 20 + ["Cc"] * 30 + ["D"] * 40
    ds = _ds({"Grade": (cats, (C,))})
    b = bind.bind(REGISTRY["chi2_gof"], ds, {"category": ("Grade",)}, layout="long")
    r = cat.chi2_gof(b)
    assert r.status == "ok"
    ref = ss.chisquare([10, 20, 30, 40])
    assert r.statistic[1] == pytest.approx(ref.statistic)
    assert r.statistic[1] == pytest.approx(20.0)            # hand value
    assert r.p == pytest.approx(ref.pvalue)
    # Cohen's w = sqrt(chi2 / N) = sqrt(20/100) = 0.4472
    assert r.effect[1] == pytest.approx(math.sqrt(0.2), abs=1e-6)


def test_chi2_gof_numeric_coded_categories_display_without_trailing_zero():
    # S-B: a die-face column coded 1..6 (numeric) must read "1".."6" in groups=
    # and the category table, not "1.0".."6.0". Counts/stat are unchanged.
    faces = [1, 2, 3, 4, 5, 6] * 6                       # 36 rolls, 6 each
    ds = _ds({"Die": (faces, (O,))})
    b = bind.bind(REGISTRY["chi2_gof"], ds, {"category": ("Die",)}, layout="long")
    r = cat.chi2_gof(b)
    assert r.status == "ok"
    assert set(r.groups) == {"1", "2", "3", "4", "5", "6"}
    assert not any("." in g for g in r.groups)          # no "1.0"
    assert r.table["category"].tolist() == ["1", "2", "3", "4", "5", "6"]


def test_chi2_gof_custom_proportions_mendel_9331():
    # Mendel's peas 315:101:108:32 against 9:3:3:1 -> chi2 = 0.470 (textbook)
    cats = (["Round"] * 315 + ["Yellow"] * 101 + ["Green"] * 108 + ["Wrinkled"] * 32)
    ds = _ds({"Phenotype": (cats, (C,))})
    b = bind.bind(REGISTRY["chi2_gof"], ds, {"category": ("Phenotype",)},
                  layout="long", params={"expected": (9 / 16, 3 / 16, 3 / 16, 1 / 16)})
    r = cat.chi2_gof(b)
    assert r.statistic[1] == pytest.approx(0.470, abs=1e-2)


# ==========================================================================
# fisher — exact (2x2 labelled 'Fisher's exact test'; R x C 'Freeman-Halton')
# ==========================================================================
def test_fisher_2x2_matches_scipy_and_reports_conditional_or():
    b = _grid_bind("fisher", ["R1", "R2"], {"C1": [3, 1], "C2": [1, 3]})
    r = cat.fisher(b)
    assert r.status == "ok"
    assert "fisher's exact" in r.test_name.lower()
    ref = ss.fisher_exact(np.array([[3, 1], [1, 3]]), alternative="two-sided")
    assert r.p == pytest.approx(ref.pvalue)
    assert r.p == pytest.approx(0.4857, abs=1e-3)           # published constant
    cond = contingency.odds_ratio(np.array([[3, 1], [1, 3]]))
    assert r.estimate[1] == pytest.approx(cond.statistic)   # conditional MLE OR (6.408)


def test_fisher_rxc_is_freeman_halton_with_table_probability():
    # scipy's R x C fisher_exact is Monte Carlo (non-deterministic); the runner SEEDS
    # it so the p is reproducible (PLAN D18) and close to the exact Freeman-Halton p.
    b = _grid_bind("fisher", ["R1", "R2"], {"A": [10, 2], "B": [5, 8], "Cc": [3, 7]})
    r = cat.fisher(b)
    assert r.status == "ok"
    assert "freeman" in r.test_name.lower()                 # Fisher-Freeman-Halton
    assert r.statistic[0] == "table probability"
    # reproducible: same input -> same p on a second call
    b2 = _grid_bind("fisher", ["R1", "R2"], {"A": [10, 2], "B": [5, 8], "Cc": [3, 7]})
    assert cat.fisher(b2).p == r.p
    # close to the exact Freeman-Halton p (hand-enumerated oracle, 0.023917)
    assert r.p == pytest.approx(0.023917, abs=3e-3)


def test_fisher_rxc_name_says_monte_carlo():
    # S13: the R x C name must say it IS a (seeded) Monte-Carlo p, so "exact" and
    # "estimate" don't sit in one paragraph of the report.
    b = _grid_bind("fisher", ["R1", "R2"], {"A": [10, 2], "B": [5, 8], "Cc": [3, 7]})
    r = cat.fisher(b)
    assert r.status == "ok"
    assert r.test_name == "Fisher-Freeman-Halton test (Monte Carlo p)"
    assert r.statistic[0] == "table probability"             # unchanged for R x C


def test_fisher_2x2_reports_one_odds_ratio():
    # S13: the sample OR (statistic) is a duplicate of the conditional MLE OR
    # (estimate); report ONE odds ratio -> statistic is dropped.
    b = _grid_bind("fisher", ["R1", "R2"], {"C1": [3, 1], "C2": [1, 3]})
    r = cat.fisher(b)
    assert r.status == "ok"
    assert r.statistic is None                                # sample OR dropped (S13)
    assert r.estimate[0] == "odds ratio"                     # the one OR: conditional MLE
    assert isinstance(r.estimate[1], float)
    # S4: observed crosstab + 2x2 OR orientation reach the report
    assert r.extra["observed"].loc["R1", "C1"] == 3
    assert r.extra["or"] == {"col_level": "C1", "row_level": "R1", "row_ref": "R2"}


# ==========================================================================
# mcnemar — exact/chi2 switch (snag 15: b*c=0 keeps a valid p, no block)
# ==========================================================================
def test_mcnemar_exact_branch_when_bc_small():
    pairs = ([("Yes", "Yes")] * 6 + [("Yes", "No")] * 4
             + [("No", "Yes")] * 3 + [("No", "No")] * 5)     # b+c = 7 < 25 -> exact
    b = bind.bind(REGISTRY["mcnemar"], _pairs_ds(pairs),
                  {"before": ("Before",), "after": ("After",)})
    r = cat.mcnemar(b)
    assert r.status == "ok"
    assert "exact" in r.test_name.lower()
    ref = sm_mcnemar(np.array([[5, 3], [4, 6]]), exact=True, correction=True)
    assert r.p == pytest.approx(ref.pvalue)
    assert r.p == pytest.approx(1.0)


def test_mcnemar_chi2_branch_when_bc_large():
    pairs = ([("Yes", "Yes")] * 10 + [("No", "No")] * 10
             + [("Yes", "No")] * 15 + [("No", "Yes")] * 25)  # b+c = 40 >= 25 -> chi2
    b = bind.bind(REGISTRY["mcnemar"], _pairs_ds(pairs),
                  {"before": ("Before",), "after": ("After",)})
    r = cat.mcnemar(b)
    assert r.status == "ok"
    assert "χ²" in r.test_name or "chi" in r.test_name.lower()
    # hand continuity chi2 = (|b-c|-1)^2/(b+c) = (|25-15|-1)^2/40 = 2.025
    assert r.statistic[1] == pytest.approx(2.025, abs=1e-6)
    assert r.p == pytest.approx(0.15473, abs=1e-4)


def test_mcnemar_bc_zero_keeps_p_but_or_not_estimable_snag15():
    pairs = ([("Yes", "Yes")] * 6 + [("Yes", "No")] * 3
             + [("No", "No")] * 5)                            # c = 0, b = 3 -> b*c = 0
    b = bind.bind(REGISTRY["mcnemar"], _pairs_ds(pairs),
                  {"before": ("Before",), "after": ("After",)})
    r = cat.mcnemar(b)
    assert r.status == "ok"                                   # NOT blocked (snag 15)
    assert r.estimate is None                                 # OR not estimable
    assert any("not estimable" in v.lower() for v in r.variant_notes)
    assert r.p == pytest.approx(0.25, abs=1e-6)               # valid exact p kept


def test_mcnemar_no_discordant_pairs_is_blocked():
    pairs = [("Yes", "Yes")] * 6 + [("No", "No")] * 5        # b = c = 0 -> block
    b = bind.bind(REGISTRY["mcnemar"], _pairs_ds(pairs),
                  {"before": ("Before",), "after": ("After",)})
    r = cat.mcnemar(b)
    assert r.status == "blocked"


# ==========================================================================
# cochran_q — Q + per-column proportions + pairwise McNemar-Holm
# ==========================================================================
def test_cochran_q_matches_statsmodels_with_proportions_and_posthoc():
    X = [[1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 1, 0], [1, 1, 0],
         [1, 0, 0], [1, 1, 1], [1, 1, 0], [0, 0, 0], [1, 1, 1]]
    ds = _ds({"T1": ([r[0] for r in X], (B,)),
              "T2": ([r[1] for r in X], (B,)),
              "T3": ([r[2] for r in X], (B,))})
    b = bind.bind(REGISTRY["cochran_q"], ds,
                  {"measures": ("T1", "T2", "T3")}, layout="wide")
    r = cat.cochran_q(b)
    assert r.status == "ok"
    ref = cochrans_q(np.array(X))
    assert r.statistic[0] == "Q" and r.statistic[1] == pytest.approx(ref.statistic)
    assert r.statistic[1] == pytest.approx(7.0)
    assert r.p == pytest.approx(ref.pvalue)
    assert r.p == pytest.approx(0.0302, abs=1e-4)
    # per-column proportions 0.8 / 0.7 / 0.3
    props = r.table["proportion"].tolist()
    assert props == pytest.approx([0.8, 0.7, 0.3])
    # significant -> pairwise McNemar-Holm, one row per pair
    assert r.posthoc is not None and len(r.posthoc) == 3
    assert "p_holm" in r.posthoc.columns


def test_cochran_q_labels_the_success_level_numeric_coded():
    # F6: the Result must NAME the success level it reports proportions of, and
    # a numeric 0/1 coding renders "1", not "1.0". Proportions are UNCHANGED.
    X = [[1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 1, 0], [1, 1, 0],
         [1, 0, 0], [1, 1, 1], [1, 1, 0], [0, 0, 0], [1, 1, 1]]
    ds = _ds({"T1": ([r[0] for r in X], (B,)),
              "T2": ([r[1] for r in X], (B,)),
              "T3": ([r[2] for r in X], (B,))})
    b = bind.bind(REGISTRY["cochran_q"], ds,
                  {"measures": ("T1", "T2", "T3")}, layout="wide")
    r = cat.cochran_q(b)
    assert r.labels.get("success") == "1"                    # not "1.0"
    assert r.table["proportion"].tolist() == pytest.approx([0.8, 0.7, 0.3])


# ==========================================================================
# F4 — numeric-coded row/col levels render "0"/"1", not "0.0"/"1.0"
# ==========================================================================
def _numeric_2x2_bound(spec_id):
    """A 0/1 numeric-coded 2×2 bound to row/col (long layout) -> float crosstab."""
    cells = {(0, 0): 8, (0, 1): 2, (1, 0): 3, (1, 1): 9}
    rows, cols = [], []
    for (rv, cv), k in cells.items():
        rows.extend([rv] * k)
        cols.extend([cv] * k)
    ds = _ds({"Rr": (rows, (B,)), "Cc": (cols, (B,))})
    return bind.bind(REGISTRY[spec_id], ds,
                     {"row": ("Rr",), "col": ("Cc",)}, layout="long")


def test_fisher_numeric_levels_display_without_trailing_zero():
    r = cat.fisher(_numeric_2x2_bound("fisher"))
    assert r.status == "ok"
    assert list(r.extra["observed"].index) == ["0", "1"]
    assert list(r.extra["observed"].columns) == ["0", "1"]
    assert r.extra["or"] == {"col_level": "0", "row_level": "0", "row_ref": "1"}


def test_chi2_ind_numeric_levels_display_without_trailing_zero():
    # expected cells all >= 5 so it does not route to Fisher
    cells = {(0, 0): 18, (0, 1): 7, (1, 0): 6, (1, 1): 19}
    rows, cols = [], []
    for (rv, cv), k in cells.items():
        rows.extend([rv] * k)
        cols.extend([cv] * k)
    ds = _ds({"Rr": (rows, (B,)), "Cc": (cols, (B,))})
    b = bind.bind(REGISTRY["chi2_ind"], ds,
                  {"row": ("Rr",), "col": ("Cc",)}, layout="long")
    r = cat.chi2_ind(b)
    assert r.status == "ok"
    assert list(r.extra["observed"].index) == ["0", "1"]
    assert list(r.expected.index) == ["0", "1"]
    assert r.extra["or"]["col_level"] == "0"


def test_mcnemar_numeric_levels_display_without_trailing_zero():
    pairs = ([(1, 1)] * 6 + [(1, 0)] * 4 + [(0, 1)] * 3 + [(0, 0)] * 5)
    b = bind.bind(REGISTRY["mcnemar"], _pairs_ds(pairs),
                  {"before": ("Before",), "after": ("After",)})
    r = cat.mcnemar(b)
    assert r.status == "ok"
    assert list(r.table.index) == ["0", "1"]
    assert list(r.table.columns) == ["0", "1"]


# ==========================================================================
# module contract
# ==========================================================================
def test_runners_mapping_exposes_all_five():
    assert set(cat.RUNNERS) == {"chi2_ind", "chi2_gof", "fisher", "mcnemar", "cochran_q"}
    for fn in cat.RUNNERS.values():
        assert callable(fn)


# ==========================================================================
# S-L — an R×C Fisher table whose Monte-Carlo p/statistic come back as a
# length-1 array (a degenerate table) must not crash on float(res.pvalue).
# ==========================================================================
def test_fisher_rxc_degenerate_table_returns_scalar_p():
    b = _grid_bind("fisher", ["A", "B"], {"x": [1, 2], "y": [1, 0], "z": [1, 0]})
    r = cat.fisher(b)                                        # pre-fix: TypeError (0-d cast)
    assert r.status == "ok"
    assert 0.0 <= r.p <= 1.0
    assert isinstance(r.p, float)
    assert r.statistic[0] == "table probability"
    assert math.isfinite(r.statistic[1])
