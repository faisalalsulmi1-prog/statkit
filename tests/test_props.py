"""RED-first goldens for statkit.props (Chunk 9 runners: prop_1, prop_2).

DoD specific (task / PLAN D6): proportion CIs are WILSON (prop_1) / Newcombe
(prop_2), never the normal-approximation (Wald). Each runner is checked against a
published Wilson worked value + a closed-form hand oracle, a statsmodels
cross-call, and an explicit assertion that Wilson != Wald.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import scipy.stats as ss
from statsmodels.stats.proportion import (confint_proportions_2indep,
                                          proportion_confint, proportions_ztest)

from statkit import bind, props
from statkit.infer import Dataset
from statkit.model import ColumnProfile, Kind

N, O, C, B = Kind.NUMERIC, Kind.ORDINAL, Kind.CATEGORICAL, Kind.BINARY
Z = 1.959963984540054


# --------------------------------------------------------------------------
# helpers (same idiom as tests/test_bind.py)
# --------------------------------------------------------------------------
def _isna(v):
    return v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v))


def _profile(name, kinds, values):
    non_null = [v for v in values if not _isna(v)]
    seen, levels = set(), []
    for v in non_null:
        s = str(v)
        if s not in seen:
            seen.add(s)
            levels.append(s)
    return ColumnProfile(
        name=name, kinds=tuple(kinds), n_total=len(values),
        n_missing=len(values) - len(non_null), n_levels=len(levels), levels=tuple(levels))


def _ds(cols):
    df_cols, profiles, n = {}, [], None
    for name, (values, kinds) in cols.items():
        n = len(values) if n is None else n
        non_null = [v for v in values if not _isna(v)]
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                      for v in non_null)
        if kinds[0] in (N, O, B, Kind.ID) and numeric:
            df_cols[name] = pd.array(
                [None if _isna(v) else float(v) for v in values], dtype="float64")
        else:
            df_cols[name] = pd.array(
                [pd.NA if _isna(v) else str(v) for v in values], dtype="string")
        profiles.append(_profile(name, kinds, values))
    return Dataset(df=pd.DataFrame(df_cols), profiles=tuple(profiles),
                   excel_rows=tuple(range(2, 2 + n)))


def _hand_wilson(k, n, z=Z):
    p = k / n
    cen = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = (z / (1 + z * z / n)) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return cen - half, cen + half


# ==========================================================================
# prop_1 — one-sample proportion (exact binomial) + Wilson CI
# ==========================================================================
def test_prop_1_wilson_ci_matches_published_and_hand_and_differs_from_wald():
    # k=10, n=100: Wilson 95% CI = [0.0552, 0.1744] (published; Brown, Cai &
    # DasGupta 2001). Wald = [0.0412, 0.1588] -> Wilson must NOT equal Wald.
    outcome = ["Yes"] * 10 + ["No"] * 90
    b = _bind_prop("prop_1", outcome, {"success": "Yes", "p0": 0.5})
    r = props.prop_1(b)
    assert r.status == "ok"
    assert r.estimate[1] == pytest.approx(0.10)                    # p-hat
    lo, hi = r.estimate_ci
    assert (lo, hi) == pytest.approx((0.05523, 0.17437), abs=1e-4)  # published Wilson
    assert (lo, hi) == pytest.approx(_hand_wilson(10, 100))         # closed-form oracle
    wald = proportion_confint(10, 100, method="normal")
    assert abs(lo - wald[0]) > 1e-3 and abs(hi - wald[1]) > 1e-3    # Wilson != Wald
    # exact binomial p (k=10, n=100, p0=.5) is ~0 -> p < .001
    assert r.p == pytest.approx(ss.binomtest(10, 100, 0.5).pvalue, abs=1e-9)
    assert r.p < 0.001


def test_prop_1_uses_binomtest_against_custom_p0_and_cohens_h():
    outcome = ["Success"] * 8 + ["Fail"] * 12                      # k=8, n=20
    b = _bind_prop("prop_1", outcome, {"success": "Success", "p0": 0.3})
    r = props.prop_1(b)
    assert r.p == pytest.approx(ss.binomtest(8, 20, 0.3).pvalue, abs=1e-9)
    # Cohen's h = 2asin(sqrt(.4)) - 2asin(sqrt(.3))
    h = 2 * math.asin(math.sqrt(0.4)) - 2 * math.asin(math.sqrt(0.3))
    assert r.effect[1] == pytest.approx(h)


def test_prop_1_default_success_level_picks_positive_token():
    # no explicit success -> "yes" recognised as the event level
    outcome = ["yes"] * 6 + ["no"] * 14
    b = _bind_prop("prop_1", outcome, {"p0": 0.5})
    r = props.prop_1(b)
    assert r.estimate[1] == pytest.approx(6 / 20)


# --------------------------------------------------------------------------
# B3 — a numeric 0/1 outcome chosen by its DISPLAY string ("0"/"1")
# --------------------------------------------------------------------------
def test_prop_1_success_display_string_matches_numeric_level():
    # outcome stored as floats 1.0/0.0 (13 ones, 7 zeros); the student picks the
    # success level by its display string "0". Before B3 the "0" != 1.0 mismatch
    # fell through to the positive token and reported 0.65 (the proportion of 1).
    outcome = [1.0] * 13 + [0.0] * 7
    b = _bind_prop("prop_1", outcome, {"success": "0", "p0": 0.5}, kinds=(B,))
    r = props.prop_1(b)
    assert r.status == "ok"
    assert r.estimate[1] == pytest.approx(0.35)     # 7 zeros / 20, NOT 0.65
    assert r.labels["success"] == "0"               # NICE: not "0.0"


def test_prop_1_unknown_success_level_blocks():
    outcome = ["Yes"] * 6 + ["No"] * 14
    b = _bind_prop("prop_1", outcome, {"success": "maybe", "p0": 0.5})
    r = props.prop_1(b)
    assert r.status == "blocked"
    assert any(f.code == "LEVEL" and f.severity == "block" for f in r.findings)


def test_prop_1_carries_p0():
    outcome = ["Yes"] * 8 + ["No"] * 12
    b = _bind_prop("prop_1", outcome, {"success": "Yes", "p0": 0.3})
    r = props.prop_1(b)
    assert r.extra["p0"] == 0.3


# ==========================================================================
# prop_2 — two-sample proportion z-test + Newcombe CI
# ==========================================================================
def test_prop_2_matches_statsmodels_z_and_newcombe_ci():
    # group A: 30/100 ; group B: 20/120
    outcome = ["Yes"] * 30 + ["No"] * 70 + ["Yes"] * 20 + ["No"] * 100
    group = ["A"] * 100 + ["B"] * 120
    b = _bind_prop2(outcome, group, {"success": "Yes"})
    r = props.prop_2(b)
    assert r.status == "ok"
    z, p = proportions_ztest([30, 20], [100, 120])
    assert r.statistic[0] == "z" and r.statistic[1] == pytest.approx(z)
    assert r.p == pytest.approx(p)
    assert r.p == pytest.approx(0.01878, abs=1e-4)                 # published constant
    assert r.estimate[1] == pytest.approx(30 / 100 - 20 / 120)    # p1 - p2
    ci = confint_proportions_2indep(30, 100, 20, 120, method="newcomb")
    assert r.estimate_ci == pytest.approx((ci[0], ci[1]))         # Newcombe, not Wald
    assert r.higher == "A"


# --------------------------------------------------------------------------
# F4 — a numeric-coded GROUP column (Sex = 0/1 stored as floats). The split
# stays keyed on the raw value; the group LABELS a student reads must render
# "0"/"1", never "0.0"/"1.0" (sentence, n keys, groups, higher).
# --------------------------------------------------------------------------
def test_prop_2_numeric_coded_group_labels_are_display_strings():
    # group 0: 30/100 Yes ; group 1: 20/120 Yes (same numbers as the golden).
    outcome = ["Yes"] * 30 + ["No"] * 70 + ["Yes"] * 20 + ["No"] * 100
    group = [0] * 100 + [1] * 120
    b = _bind_prop2(outcome, group, {"success": "Yes"})
    r = props.prop_2(b)
    assert r.status == "ok"
    assert r.estimate[1] == pytest.approx(30 / 100 - 20 / 120)  # numbers unchanged
    assert r.groups == ("0", "1")                               # NOT ("0.0","1.0")
    assert r.higher == "0"                                      # NOT "0.0"
    assert r.estimate[0] == "difference in proportion 'Yes' (0 − 1)"  # not (0.0 − 1.0)
    assert "0" in r.n and "1" in r.n                            # per-group n keys


# ==========================================================================
# bind helpers (kept below the tests they serve)
# ==========================================================================
def _bind_prop(spec_id, outcome, params, kinds=(B, C)):
    from statkit.registry import REGISTRY
    ds = _ds({"Outcome": (outcome, kinds)})
    return bind.bind(REGISTRY[spec_id], ds, {"outcome": ("Outcome",)},
                     layout="long", params=params)


def _bind_prop2(outcome, group, params):
    from statkit.registry import REGISTRY
    ds = _ds({"Outcome": (outcome, (B, C)), "Group": (group, (B, C))})
    return bind.bind(REGISTRY["prop_2"], ds,
                     {"outcome": ("Outcome",), "group": ("Group",)},
                     layout="long", params=params)


# ==========================================================================
# module contract
# ==========================================================================
def test_runners_mapping_exposes_both():
    assert set(props.RUNNERS) == {"prop_1", "prop_2"}
    for fn in props.RUNNERS.values():
        assert callable(fn)
