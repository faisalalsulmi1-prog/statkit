"""Standalone assumption-tool runners (PLAN §3, menu L4): normality, homogeneity.

Two ``Callable[[Bound], Result]`` runners matching the registry runner signature
(``TestSpec.run``), for the two assumption tools that appear as their OWN menu
entries (not the auto ``advise()`` a test runs on itself):

  * ``normality``   -- Shapiro-Wilk + Lilliefors on each bound numeric variable
                       (split by group when one is bound), honouring the n<3 /
                       n>5000 Shapiro skip window (assumptions.py).
  * ``homogeneity`` -- Brown-Forsythe = ``levene(center="median")`` across the
                       groups.

Same flow as every other runner: ``check(bound)`` gates structural blocks first;
the statistic itself is computed via ``assumptions.py``; a group too small to have
a variance, or a NaN result, gates to ``status="blocked"`` -- never a NaN Result
or a crash. Only stdlib + numpy + scipy + statsmodels + pandas (L3 allowlist).
No I/O.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from . import assumptions, levels
from .check import gate
from .model import Bound, Finding, Result


def _n(bound: Bound) -> dict:
    return {"total": bound.n_total, "used": bound.n_used,
            "dropped": bound.n_total - bound.n_used}


def _gate(bound: Bound):
    return gate(bound)


def _blocked(bound: Bound, findings, reason: str = "") -> Result:
    fs = tuple(findings)
    if reason and not any(f.severity == "block" for f in fs):
        fs = fs + (Finding("block", reason, code="RUN"),)
    return Result(test_id=bound.test.id, test_name=bound.test.name,
                  status="blocked", n=_n(bound), findings=fs)


def _levels_in_order(series) -> list:
    return list(dict.fromkeys(series.dropna().tolist()))


def _label(hdr: str, level, multi_var: bool) -> str:
    parts = []
    if multi_var:
        parts.append(str(hdr))
    if level is not None:
        parts.append(levels.display(level))
    return " · ".join(parts)


def _tag(chk, label: str):
    """Disambiguate a Check when several targets are assessed (else keep base name)."""
    return replace(chk, name=f"{chk.name} ({label})") if label else chk


# --------------------------------------------------------------------------
# normality — Shapiro-Wilk + Lilliefors per numeric variable (× group)
# --------------------------------------------------------------------------
def normality(bound: Bound) -> Result:
    findings = _gate(bound)
    blocks = tuple(f for f in findings if f.severity == "block")
    if blocks:
        return _blocked(bound, blocks)

    data = bound.data
    var_headers = bound.columns.get("variables", ())
    multi_var = len(var_headers) > 1
    has_group = "group" in data.columns
    group_levels = _levels_in_order(data["group"]) if has_group else [None]

    checks, primary = [], None
    for i, hdr in enumerate(var_headers):
        col = f"variables__{i}" if f"variables__{i}" in data.columns else hdr
        s = data[col]
        for lv in group_levels:
            sub = s if lv is None else s[data["group"] == lv]
            x = sub.dropna().to_numpy(dtype=float)
            label = _label(hdr, lv, multi_var)
            sw = assumptions.shapiro_normality(x)
            lil = assumptions.lilliefors_normality(x)
            checks.extend((_tag(sw, label), _tag(lil, label)))
            if primary is None and sw.statistic is not None:
                primary = sw

    r = Result(test_id="normality", test_name=bound.test.name, status="ok",
               n=_n(bound), checks=tuple(checks),
               method=("scipy.stats.shapiro + statsmodels lilliefors "
                       "(skipped for n<3 or n>5000)"),
               findings=tuple(f for f in findings if f.severity != "block"))
    if primary is not None:
        r.statistic = ("W", primary.statistic)
        r.p = primary.p
        r.variant_notes = (("No evidence against normality (p ≥ .05)"
                            if primary.passed else "Departs from normality (p < .05)"),)
    elif checks and checks[0].note:
        r.variant_notes = (checks[0].note,)          # e.g. n>5000: Shapiro not run
    return r


# --------------------------------------------------------------------------
# homogeneity — Brown-Forsythe (Levene, center="median") across groups
# --------------------------------------------------------------------------
def homogeneity(bound: Bound) -> Result:
    findings = _gate(bound)
    blocks = tuple(f for f in findings if f.severity == "block")
    if blocks:
        return _blocked(bound, blocks)

    df = bound.data
    level_vals = _levels_in_order(df["group"])
    groups = [df.loc[df["group"] == lv, "outcome"].dropna().to_numpy(dtype=float)
              for lv in level_vals]
    small = [levels.display(lv) for lv, g in zip(level_vals, groups) if len(g) < 2]
    if len(groups) < 2 or small:
        reason = ("Group(s) with fewer than 2 values: " + ", ".join(small)
                  + ". Equality of variances needs at least 2 per group."
                  if small else "Equality of variances needs at least 2 groups.")
        return _blocked(bound, findings, reason)

    bf = assumptions.brown_forsythe(*groups)
    if bf.statistic is None or np.isnan(bf.statistic) or np.isnan(bf.p):
        return _blocked(bound, findings, "The Brown-Forsythe statistic could not "
                        "be computed.")

    r = Result(test_id="homogeneity", test_name=bound.test.name, status="ok",
               statistic=("W", bf.statistic), p=bf.p, checks=(bf,),
               n=_n(bound), groups=tuple(levels.display(lv) for lv in level_vals),
               method="scipy.stats.levene(center='median')  [Brown-Forsythe]",
               variant_notes=(("Equal variances not rejected (p ≥ .05)"
                               if bf.passed else "Variances differ (p < .05)"),),
               findings=tuple(f for f in findings if f.severity != "block"))
    return r


RUNNERS = {"normality": normality, "homogeneity": homogeneity}
