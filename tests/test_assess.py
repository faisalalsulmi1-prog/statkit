"""RED-first goldens + garbage gates for statkit/assess.py (assumption-tool menu).

Two standalone menu runners, same signature as every other runner
(``Callable[[Bound], Result]``): the L4 assumption tools ``normality`` and
``homogeneity``.

NON-CIRCULAR GOLDENS: every statistic is cross-checked against an INDEPENDENT
oracle -- ``scipy.stats.shapiro`` / ``statsmodels ... lilliefors`` /
``scipy.stats.levene(center="median")`` called directly on the same data, which
bypasses all of assess.py's binding/selection/extraction work. An orthogonal
check pins Brown-Forsythe to MEDIAN centring (it must NOT equal mean-centred
Levene). GARBAGE GATES: too-small n / a singleton group must yield
``status == "blocked"`` with a stated reason -- never a NaN result or a crash.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors

from statkit import assess, bind
from statkit.infer import Dataset
from statkit.model import ColumnProfile, Kind
from statkit.registry import REGISTRY

N, O, CAT, B, ID = (Kind.NUMERIC, Kind.ORDINAL, Kind.CATEGORICAL, Kind.BINARY, Kind.ID)
APPROX = dict(rel=1e-6, abs=1e-9)


# --------------------------------------------------------------------------
# dataset / bound builders (mirrors tests/test_means.py)
# --------------------------------------------------------------------------
def _isna(v):
    return v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v))


def _profile(name, kinds, values):
    non_null = [v for v in values if not _isna(v)]
    if kinds[0] in (N, ID, Kind.DATE, Kind.EMPTY):
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


def _ds(cols):
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
    return Dataset(df=pd.DataFrame(df_cols), profiles=tuple(profiles),
                   excel_rows=tuple(range(2, 2 + (n or 0))))


def _bound(test_id, cols, columns, layout=None, params=None):
    return bind.bind(REGISTRY[test_id], _ds(cols), columns, layout=layout,
                     params=params)


def _blocks(result):
    return {f.code for f in result.findings if f.severity == "block"}


def _shapiro_check(result):
    return next(c for c in result.checks if c.name.startswith("Shapiro"))


X = [2.1, 3.4, 1.9, 5.6, 4.2, 3.3, 2.8, 4.9, 3.1, 2.2]   # n=10, no ties


# ==========================================================================
# normality
# ==========================================================================
def test_normality_shapiro_matches_scipy_and_carries_decision():
    b = _bound("normality", {"Score": (X, (N,))}, {"variables": ("Score",)})
    r = assess.normality(b)
    assert r.status == "ok"
    w, p = stats.shapiro(X)                                  # independent oracle
    assert r.statistic == ("W", pytest.approx(w, **APPROX))
    assert r.p == pytest.approx(p, **APPROX)
    chk = _shapiro_check(r)
    assert chk.statistic == pytest.approx(w, **APPROX)
    assert chk.p == pytest.approx(p, **APPROX)
    assert chk.passed == bool(p >= 0.05)                     # decision on the Check
    assert not math.isnan(r.statistic[1])


def test_normality_includes_lilliefors_matching_statsmodels():
    b = _bound("normality", {"Score": (X, (N,))}, {"variables": ("Score",)})
    r = assess.normality(b)
    lil = next(c for c in r.checks if c.name.startswith("Lilliefors"))
    stat, p = lilliefors(np.asarray(X, dtype=float), dist="norm", pvalmethod="table")
    assert lil.statistic == pytest.approx(stat, **APPROX)
    assert lil.p == pytest.approx(p, **APPROX)


def test_normality_splits_by_group():
    ya, yb = [1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 8.0, 13.0, 7.0, 11.0]
    b = _bound("normality",
               {"Score": (ya + yb, (N,)),
                "G": (["A"] * 5 + ["B"] * 5, (CAT,))},
               {"variables": ("Score",), "group": ("G",)})
    r = assess.normality(b)
    assert r.status == "ok"
    shap = [c for c in r.checks if c.name.startswith("Shapiro")]
    assert len(shap) == 2                                    # one per group level
    wa, _ = stats.shapiro(ya)
    wb, _ = stats.shapiro(yb)
    stats_seen = sorted(c.statistic for c in shap)
    assert stats_seen == pytest.approx(sorted([wa, wb]), **APPROX)


def test_normality_too_small_n_blocks_without_nan():
    b = _bound("normality", {"Score": ([1.0, 2.0], (N,))}, {"variables": ("Score",)})
    r = assess.normality(b)
    assert r.status == "blocked"
    assert "S5" in _blocks(r)                                # min_n=3
    assert r.statistic is None                               # never a NaN statistic


# ==========================================================================
# homogeneity  (Brown-Forsythe = Levene center="median")
# ==========================================================================
GA = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
GB = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
GC = [5.0, 5.5, 6.0, 4.5, 5.2, 4.8]


def test_homogeneity_matches_brown_forsythe_and_carries_decision():
    y = GA + GB + GC
    g = ["A"] * 6 + ["B"] * 6 + ["C"] * 6
    b = _bound("homogeneity", {"Y": (y, (N,)), "G": (g, (CAT,))},
               {"outcome": ("Y",), "group": ("G",)})
    r = assess.homogeneity(b)
    assert r.status == "ok"
    stat, p = stats.levene(GA, GB, GC, center="median")      # independent oracle
    assert r.statistic == ("W", pytest.approx(stat, **APPROX))
    assert r.p == pytest.approx(p, **APPROX)
    bf = next(c for c in r.checks if c.name.startswith("Brown-Forsythe"))
    assert bf.passed == bool(p >= 0.05)
    assert not math.isnan(r.statistic[1])


def test_homogeneity_is_median_centred_not_mean_centred():
    # Orthogonal check: Brown-Forsythe (median) must NOT equal classic Levene (mean).
    y = GA + GB + GC
    g = ["A"] * 6 + ["B"] * 6 + ["C"] * 6
    b = _bound("homogeneity", {"Y": (y, (N,)), "G": (g, (CAT,))},
               {"outcome": ("Y",), "group": ("G",)})
    r = assess.homogeneity(b)
    mean_stat, _ = stats.levene(GA, GB, GC, center="mean")
    assert abs(r.statistic[1] - mean_stat) > 1e-6


def test_homogeneity_singleton_group_blocks_without_nan():
    # group B has a single observation -> variance undefined; must block cleanly.
    y = [1.0, 2.0, 3.0, 4.0, 5.0, 9.0]
    g = ["A"] * 5 + ["B"]
    b = _bound("homogeneity", {"Y": (y, (N,)), "G": (g, (CAT,))},
               {"outcome": ("Y",), "group": ("G",)})
    r = assess.homogeneity(b)
    assert r.status == "blocked"
    assert r.statistic is None                               # no NaN leaked


def test_homogeneity_too_small_n_blocks():
    b = _bound("homogeneity", {"Y": ([1.0, 2.0], (N,)), "G": (["A", "B"], (CAT,))},
               {"outcome": ("Y",), "group": ("G",)})
    r = assess.homogeneity(b)
    assert r.status == "blocked"
    assert "S5" in _blocks(r)


# ==========================================================================
# S-B — a numeric-coded group renders "1"/"2" in labels, not "1.0"/"2.0"
# ==========================================================================
def test_normality_numeric_coded_group_labels_without_trailing_zero():
    ya, yb = [1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 8.0, 13.0, 7.0, 11.0]
    b = _bound("normality",
               {"Score": (ya + yb, (N,)),
                "G": ([1] * 5 + [2] * 5, (B,))},        # numeric-coded group
               {"variables": ("Score",), "group": ("G",)})
    r = assess.normality(b)
    assert r.status == "ok"
    names = [c.name for c in r.checks]
    assert any(n.endswith("(1)") for n in names)        # not "(1.0)"
    assert not any("1.0" in n or "2.0" in n for n in names)


def test_homogeneity_numeric_coded_group_labels_without_trailing_zero():
    y = GA + GB
    b = _bound("homogeneity",
               {"Y": (y, (N,)), "G": ([1] * 6 + [2] * 6, (B,))},
               {"outcome": ("Y",), "group": ("G",)})
    r = assess.homogeneity(b)
    assert r.status == "ok"
    assert r.groups == ("1", "2")                        # not ("1.0", "2.0")


def test_homogeneity_singleton_numeric_group_names_it_without_trailing_zero():
    # the block message (line 123) must render the offending numeric level as "2".
    y = [1.0, 2.0, 3.0, 4.0, 5.0, 9.0]
    b = _bound("homogeneity",
               {"Y": (y, (N,)), "G": ([1, 1, 1, 1, 1, 2], (B,))},
               {"outcome": ("Y",), "group": ("G",)})
    r = assess.homogeneity(b)
    assert r.status == "blocked"
    reason = " ".join(f.text for f in r.findings)
    assert "2.0" not in reason and "2" in reason


# ==========================================================================
# module contract
# ==========================================================================
def test_runners_mapping_exposes_both():
    assert set(assess.RUNNERS) == {"normality", "homogeneity"}
    for fn in assess.RUNNERS.values():
        assert callable(fn)
