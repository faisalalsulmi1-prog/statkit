"""RED-first gate for the registry wiring (PLAN §10, Chunk-1 follow-on).

The registry ships 27 ``TestSpec``s whose ``run`` started life as the ``_todo``
stub. This gate asserts every one has been pointed at its REAL runner, that the
27 wired functions are EXACTLY the union of the seven ``RUNNERS`` maps (and cover
all 27 ids with nothing left over), and that dispatch actually works end-to-end
through the registry for one test per family -> a populated, non-blocked Result.
"""
from __future__ import annotations

import math

import pandas as pd

from statkit import (assess, assoc, bind, cat, means, props, ranks, regress,
                     registry)
from statkit.infer import Dataset
from statkit.model import ColumnProfile, Kind
from statkit.registry import REGISTRY

N, O, CAT, B, ID = (Kind.NUMERIC, Kind.ORDINAL, Kind.CATEGORICAL, Kind.BINARY, Kind.ID)

RUNNER_MODULES = (means, ranks, cat, props, assoc, regress, assess)


# --------------------------------------------------------------------------
# 1 + 2. every run is a real runner; the 27 == union of the 7 RUNNERS maps
# --------------------------------------------------------------------------
def test_no_run_is_the_todo_stub():
    for spec in REGISTRY.values():
        assert callable(spec.run), spec.id
        assert spec.run is not registry._todo, spec.id
        assert getattr(spec.run, "__name__", "") != "_todo", spec.id


def test_wired_functions_are_exactly_the_union_of_runner_maps():
    union: dict = {}
    total = 0
    for m in RUNNER_MODULES:
        total += len(m.RUNNERS)
        union.update(m.RUNNERS)
    assert total == 27, f"RUNNERS maps overlap or miscount: {total} entries"
    assert set(union) == set(REGISTRY), "runner ids != registry ids"
    for sid, spec in REGISTRY.items():
        # Chunk 11 wraps each runner with advise.advised(); the raw runner is
        # preserved on __wrapped__, so identity is asserted through the wrapper.
        raw = getattr(spec.run, "__wrapped__", spec.run)
        assert raw is union[sid], sid                       # points at the real fn


def test_t_ind_library_string_has_no_stray_not():
    # D1 display fix: scipy's equal_var flag equals the param directly.
    lib = REGISTRY["t_ind"].library
    assert "equal_var=not equal_var" not in lib
    assert "equal_var=equal_var" in lib


# --------------------------------------------------------------------------
# 3. end-to-end dispatch through the registry, one per family
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


def _dispatch(test_id, cols, columns, layout=None, params=None):
    """Bind, then run THROUGH the registry entry (proving spec.run dispatch)."""
    spec = REGISTRY[test_id]
    b = bind.bind(spec, _ds(cols), columns, layout=layout, params=params)
    return spec.run(b)


def _populated(r):
    return (r.status == "ok"
            and (r.p is not None or r.statistic is not None
                 or r.descriptives is not None or bool(r.checks)))


def test_end_to_end_dispatch_one_per_family():
    ya = [5.0, 6.0, 7.0, 8.0, 5.5, 6.5]
    yb = [9.0, 10.0, 11.0, 12.0, 9.5, 10.5]
    yc = [1.0, 2.0, 3.0, 4.0, 1.5, 2.5]

    cases = [
        # F-2G — Welch t (means)
        ("t_ind",
         {"Y": (ya + yb, (N,)), "G": (["A"] * 6 + ["B"] * 6, (CAT,))},
         {"outcome": ("Y",), "group": ("G",)}, None, None),
        # F-KG rank — Kruskal-Wallis (ranks)
        ("kruskal",
         {"Y": (ya + yb + yc, (N,)),
          "G": (["A"] * 6 + ["B"] * 6 + ["C"] * 6, (CAT,))},
         {"outcome": ("Y",), "group": ("G",)}, None, None),
        # F-CAT — chi-square independence (cat)
        ("chi2_ind",
         {"R": (["X"] * 15 + ["Y"] * 15, (CAT,)),
          "C": ((["P"] * 10 + ["Q"] * 5) + (["P"] * 4 + ["Q"] * 11), (CAT,))},
         {"row": ("R",), "col": ("C",)}, None, None),
        # F-CAT — one-sample proportion (props)
        ("prop_1",
         {"Out": (["Yes"] * 8 + ["No"] * 12, (B, CAT))},
         {"outcome": ("Out",)}, None, {"success": "Yes", "p0": 0.5}),
        # F-ASSOC — Pearson (assoc)
        ("pearson",
         {"X": ([1.0, 2, 3, 4, 5, 6, 7, 8], (N,)),
          "Y": ([2.0, 4.1, 5.9, 8.2, 9.8, 12.1, 14.0, 16.2], (N,))},
         {"x": ("X",), "y": ("Y",)}, None, None),
        # F-REG — simple OLS (regress); S11 needs n >= 11
        ("ols_simple",
         {"Y": ([2.0, 4, 5, 8, 10, 11, 14, 15, 18, 19, 22, 24], (N,)),
          "X": ([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], (N,))},
         {"outcome": ("Y",), "x": ("X",)}, None, None),
        # F-CHK — normality assumption tool (assess)
        ("normality",
         {"Score": ([2.1, 3.4, 1.9, 5.6, 4.2, 3.3, 2.8, 4.9, 3.1, 2.2], (N,))},
         {"variables": ("Score",)}, None, None),
    ]
    for test_id, cols, columns, layout, params in cases:
        r = _dispatch(test_id, cols, columns, layout=layout, params=params)
        assert _populated(r), (test_id, r.status,
                               [f.text for f in r.findings if f.severity == "block"])
