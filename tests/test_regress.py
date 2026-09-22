"""Golden tests for statkit.regress runners (Chunk 10; PLAN §3 predict family).

The four Chunk-10 DoD specifics are each pinned by a golden:
  * separation block  — a perfectly separable logistic fit returns status
    'blocked' with NO odds ratios (PLAN §5.2 / logistic row).
  * reference level    — a categorical predictor's dropped level is reported and
    the correct level is the baseline (Treatment coding).
  * VIF                — matches the hand identity VIF = 1/(1 - r^2) for two
    numeric predictors (statsmodels is the orthogonal oracle).
  * typ=2              — the ANOVA-of-regression uses Type II sums of squares
    (a categorical term's SS differs sharply from the Type I value).

Regression uses statsmodels OLS/Logit, never scipy.linregress (PLAN D5); the
ols_simple golden is cross-checked against scipy.stats.linregress (an
independent implementation) AND the closed-form slope Sxy/Sxx.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import linregress

from statkit import bind, regress
from statkit.infer import Dataset
from statkit.model import ColumnProfile, Kind

N, O, C, B = (Kind.NUMERIC, Kind.ORDINAL, Kind.CATEGORICAL, Kind.BINARY)


# --------------------------------------------------------------------------
# minimal Dataset builder (mirrors tests/test_bind.py)
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


def _ds(cols):
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
    return Dataset(df=df, profiles=tuple(profiles), excel_rows=tuple(range(2, 2 + n)))


def _bind(test_id, cols, columns, params=None):
    from statkit.registry import REGISTRY
    return bind.bind(REGISTRY[test_id], _ds(cols), columns, layout="long",
                     params=params)


# ==========================================================================
# OLS SIMPLE — n=12; vs scipy.stats.linregress (orthogonal) + hand slope
# ==========================================================================
SX = list(range(1, 13))
SY = [3.0, 5, 4, 8, 7, 9, 10, 13, 12, 14, 16, 15]


def test_ols_simple_slope_intercept_r2_vs_linregress():
    b = _bind("ols_simple", {"x": (SX, (N,)), "y": (SY, (N,))},
              {"outcome": ("y",), "x": ("x",)})
    r = regress.ols_simple(b)
    assert r.status == "ok"
    lr = linregress(SX, SY)                                   # independent impl
    assert r.estimate == ("slope", pytest.approx(lr.slope, rel=1e-9))
    xa, ya = np.array(SX, float), np.array(SY, float)
    hand = ((xa - xa.mean()) * (ya - ya.mean())).sum() / ((xa - xa.mean()) ** 2).sum()
    assert r.estimate[1] == pytest.approx(hand, rel=1e-9)     # closed-form slope
    assert r.extra["intercept"] == pytest.approx(lr.intercept, rel=1e-9)
    assert r.extra["r2"] == pytest.approx(lr.rvalue ** 2, rel=1e-9)
    assert r.extra["r2"] == pytest.approx(0.950304, abs=1e-6)


def test_ols_simple_slope_ci():
    b = _bind("ols_simple", {"x": (SX, (N,)), "y": (SY, (N,))},
              {"outcome": ("y",), "x": ("x",)})
    r = regress.ols_simple(b)
    assert r.estimate_ci[0] == pytest.approx(0.997259, abs=1e-5)
    assert r.estimate_ci[1] == pytest.approx(1.380363, abs=1e-5)


# ==========================================================================
# OLS MULTI — VIF hand identity (two numeric predictors)
# ==========================================================================
MX1 = list(range(1, 13))
MX2 = [2.0, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]
MY = [10.0, 12, 9, 14, 13, 11, 20, 22, 19, 15, 17, 16]


def test_ols_multi_vif_matches_hand_identity_and_flags():
    b = _bind("ols_multi", {"x1": (MX1, (N,)), "x2": (MX2, (N,)), "y": (MY, (N,))},
              {"outcome": ("y",), "predictors": ("x1", "x2")})
    r = regress.ols_multi(b)
    assert r.status == "ok"
    rr = np.corrcoef(MX1, MX2)[0, 1]
    hand_vif = 1.0 / (1.0 - rr ** 2)
    assert r.extra["vif"]["x1"] == pytest.approx(hand_vif, rel=1e-6)
    assert r.extra["vif"]["x2"] == pytest.approx(hand_vif, rel=1e-6)
    assert r.extra["vif"]["x1"] == pytest.approx(12.172024, abs=1e-4)
    # VIF > 5 must raise a naming flag (PLAN §5.1 S6/advise D6)
    assert any("x1" in f.text and f.severity == "flag" for f in r.findings)


# ==========================================================================
# OLS MULTI — reference level + typ=2 ANOVA (categorical predictor)
# ==========================================================================
CY = [10.0, 12, 9, 14, 13, 11, 20, 22, 19, 8, 7, 9]
CXN = [1.0, 2, 1, 3, 3, 2, 5, 6, 5, 1, 1, 2]
CG = ["A", "A", "A", "B", "B", "B", "C", "C", "C", "A", "A", "B"]


def test_ols_multi_reference_level_named_and_dropped():
    b = _bind("ols_multi", {"g": (CG, (C,)), "xn": (CXN, (N,)), "y": (CY, (N,))},
              {"outcome": ("y",), "predictors": ("xn", "g")})
    r = regress.ols_multi(b)
    assert r.status == "ok"
    assert r.extra["reference"]["g"] == "A"                 # default = first sorted
    # coefficient table: B and C present, reference A absent
    terms = " ".join(r.table.index)
    assert "B" in terms and "C" in terms
    assert not any(t.strip() == "g = 'A'" for t in r.table.index)


def test_ols_multi_coefficients_match_statsmodels():
    b = _bind("ols_multi", {"g": (CG, (C,)), "xn": (CXN, (N,)), "y": (CY, (N,))},
              {"outcome": ("y",), "predictors": ("xn", "g")})
    r = regress.ols_multi(b)
    coef = r.table["coef"]
    assert coef["xn"] == pytest.approx(3.229730, abs=1e-5)
    assert coef["g = 'B'"] == pytest.approx(-1.648649, abs=1e-5)
    assert coef["g = 'C'"] == pytest.approx(-2.216216, abs=1e-5)
    assert r.extra["r2"] == pytest.approx(0.968990, abs=1e-6)


# --------------------------------------------------------------------------
# S-G — Treatment-coded categorical coefficient rows NAME the level ("g = 'B'"),
# never leaking patsy's "g[T.B]" suffix, in the table AND the rendered sentence.
# --------------------------------------------------------------------------
def test_ols_multi_coefficient_labels_name_the_level():
    from statkit import sentences
    b = _bind("ols_multi", {"g": (CG, (C,)), "xn": (CXN, (N,)), "y": (CY, (N,))},
              {"outcome": ("y",), "predictors": ("xn", "g")})
    r = regress.ols_multi(b)
    assert r.status == "ok"
    labels = list(r.table.index)
    assert not any("[T." in lbl for lbl in labels), labels
    assert "g = 'B'" in labels and "g = 'C'" in labels
    sent = sentences.render(r)
    assert "[T." not in sent
    assert "g = 'B'" in sent


def test_logistic_coefficient_labels_name_the_level():
    # a Treatment-coded categorical predictor in the logistic path shares the
    # relabeller, so its odds-ratio rows must read "g = 'B'", not "g[T.B]".
    gg = ["A", "B", "C"] * 8                                # 24 rows, balanced
    ys = ([0, 1] * 12)
    b = _bind("logistic", {"g": (gg, (C,)), "y": (ys, (B,))},
              {"outcome": ("y",), "predictors": ("g",)}, params={"success": "1"})
    r = regress.logistic(b)
    assert r.status == "ok", [f.text for f in r.findings]
    labels = list(r.table.index)
    assert not any("[T." in lbl for lbl in labels), labels
    assert "g = 'B'" in labels and "g = 'C'" in labels


def test_ols_multi_uses_type_two_sums_of_squares():
    b = _bind("ols_multi", {"g": (CG, (C,)), "xn": (CXN, (N,)), "y": (CY, (N,))},
              {"outcome": ("y",), "predictors": ("xn", "g")})
    r = regress.ols_multi(b)
    anova = r.extra["anova_typ2"]
    g_row = [ix for ix in anova.index if "g" in ix][0]
    # Type II SS for the categorical term = 3.0135; Type I would be 239.45.
    assert anova.loc[g_row, "sum_sq"] == pytest.approx(3.013514, abs=1e-4)
    assert anova.loc[g_row, "sum_sq"] != pytest.approx(239.45, abs=1.0)


# ==========================================================================
# LOGISTIC — 2x2 OR identity, success mapping, separation block
# ==========================================================================
# group 1: 8 events / 2 non ; group 0: 3 events / 7 non ; OR = (8*7)/(2*3) = 9.3333
LG = [1] * 10 + [0] * 10
LY = [1] * 8 + [0] * 2 + [1] * 3 + [0] * 7


def test_logistic_odds_ratio_equals_2x2_table_or():
    b = _bind("logistic", {"grp": (LG, (N,)), "y": (LY, (B,))},
              {"outcome": ("y",), "predictors": ("grp",)})
    r = regress.logistic(b)
    assert r.status == "ok"
    assert r.extra["or"]["grp"] == pytest.approx(56 / 6, rel=1e-6)   # hand 2x2 OR
    assert r.extra["success"] == "1"     # display string (NICE 1.0 -> 1), used by the sentence


def test_logistic_success_level_flips_odds_ratio():
    b0 = _bind("logistic", {"grp": (LG, (N,)), "y": (LY, (B,))},
               {"outcome": ("y",), "predictors": ("grp",)}, params={"success": 0.0})
    r0 = regress.logistic(b0)
    assert r0.extra["success"] == "0"    # display string (NICE 0.0 -> 0)
    assert r0.extra["or"]["grp"] == pytest.approx(6 / 56, rel=1e-6)  # reciprocal


def test_logistic_perfect_separation_is_blocked_with_no_odds_ratios():
    ys = [0] * 6 + [1] * 6
    xs = list(range(1, 13))
    b = _bind("logistic", {"x": (xs, (N,)), "y": (ys, (B,))},
              {"outcome": ("y",), "predictors": ("x",)})
    r = regress.logistic(b)
    assert r.status == "blocked"
    assert "or" not in r.extra or not r.extra.get("or")
    assert any(f.severity == "block" and "separation" in f.text.lower()
               for f in r.findings)


# --------------------------------------------------------------------------
# B3 — a numeric 0/1 outcome chosen by its DISPLAY string "1"
# --------------------------------------------------------------------------
def test_logistic_numeric_outcome_accepts_display_string_success():
    # 60-row float 0/1 outcome; the student picks the success level as "1" (a
    # string from a dropdown). Before B3, "1" != 1.0 made _y all zero and the
    # failed fit was reported as "separation".
    ys = [0, 1] * 30                                   # no relationship -> fits
    xs = [float(i) for i in range(60)]
    b = _bind("logistic", {"x": (xs, (N,)), "y": (ys, (B,))},
              {"outcome": ("y",), "predictors": ("x",)}, params={"success": "1"})
    r = regress.logistic(b)
    assert r.status == "ok"
    assert r.extra["success"] == "1"                  # NICE display string, not 1.0


def test_logistic_unfittable_design_is_not_called_separation():
    # two identical numeric predictors -> a perfectly collinear (singular) design.
    # statsmodels raises LinAlgError, NOT a separation warning; the block must say
    # so, not mislabel it "separation" (the S15 bug that made B3 misleading).
    xs = list(range(1, 15))
    ys = [0, 1] * 7
    b = _bind("logistic",
              {"x1": (xs, (N,)), "x2": (list(xs), (N,)), "y": (ys, (B,))},
              {"outcome": ("y",), "predictors": ("x1", "x2")})
    r = regress.logistic(b)
    assert r.status == "blocked"
    block = next(f for f in r.findings if f.severity == "block")
    assert "could not be fitted" in block.text
    assert "separation" not in block.text.lower()
    assert block.code == "FIT"


# ==========================================================================
# module wiring
# ==========================================================================
def test_runners_mapping_covers_the_three_ids():
    assert set(regress.RUNNERS) == {"ols_simple", "ols_multi", "logistic"}
    for fn in regress.RUNNERS.values():
        assert callable(fn)
