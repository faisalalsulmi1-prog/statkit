"""Proportion test runners (PLAN §3, Chunk 9): prop_1, prop_2.

Correct-by-default (PLAN D6): proportion confidence intervals are WILSON
(one-sample) / Newcombe (two-sample difference), NEVER the normal-approximation
(Wald). prop_1 uses the exact binomial test; prop_2 uses the two-sample z-test.

Each runner is ``run(bound) -> Result``. check() gates structural blocks first; a
NaN statistic/p gates to blocked. Only stdlib + numpy + scipy + statsmodels +
pandas (L3 allowlist). No I/O.
"""
from __future__ import annotations

import numpy as np
import scipy.stats as ss
from statsmodels.stats.proportion import (confint_proportions_2indep,
                                          proportion_confint, proportions_ztest)

from . import effects, levels
from .check import gate
from .model import Bound, Finding, Result


def _n(bound: Bound, **extra) -> dict:
    n = {"total": bound.n_total, "used": bound.n_used,
         "dropped": bound.n_total - bound.n_used}
    n.update(extra)
    return n


def _findings(bound: Bound):
    findings = gate(bound)
    blocks = tuple(f for f in findings if f.severity == "block")
    return findings, blocks


def _blocked(bound: Bound, findings, reason: str = "") -> Result:
    fs = tuple(findings)
    if reason and not any(f.severity == "block" for f in fs):
        fs = fs + (Finding("block", reason, code="RUN"),)
    return Result(test_id=bound.test.id, test_name=bound.test.name,
                  status="blocked", n=_n(bound), findings=fs)


def _nan(*vals) -> bool:
    return any(v is None or (isinstance(v, float) and np.isnan(v)) for v in vals)


# --------------------------------------------------------------------------
# prop_1 — one-sample proportion (exact binomial) + Wilson CI
# --------------------------------------------------------------------------
def prop_1(bound: Bound) -> Result:
    findings, blocks = _findings(bound)
    if blocks:
        return _blocked(bound, blocks)

    col = bound.data["outcome"].dropna()
    success, level_block = levels.pick_success(col, bound.params.get("success"))
    if level_block is not None:
        return _blocked(bound, findings + (level_block,))
    disp = levels.display(success)
    p0 = float(bound.params.get("p0", 0.5))
    k = int((col == success).sum())
    n = int(len(col))
    p_hat = k / n

    p_val = float(ss.binomtest(k, n, p0).pvalue)
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    if _nan(p_val):
        return _blocked(bound, findings, "The p-value could not be computed.")
    h = effects.cohens_h(p_hat, p0)

    return Result(
        test_id="prop_1", test_name="One-sample proportion test (exact binomial)",
        status="ok", p=p_val,
        estimate=(f"proportion '{disp}'", p_hat),
        estimate_ci=(float(lo), float(hi)),
        effect=("Cohen's h", h), effect_label=effects.h_label(h),
        effect_source="Cohen (1988)",
        n=_n(bound, successes=k), labels={"success": disp},
        variant_notes=(f"tested against p₀ = {p0:g}; 95% Wilson interval",),
        method=("scipy.stats.binomtest(k, n, p0) + statsmodels proportion_confint("
                "method='wilson')"),
        findings=findings, extra={"p0": p0, "success": disp})


# --------------------------------------------------------------------------
# prop_2 — two-sample proportion z-test + Newcombe CI
# --------------------------------------------------------------------------
def prop_2(bound: Bound) -> Result:
    findings, blocks = _findings(bound)
    if blocks:
        return _blocked(bound, blocks)

    df = bound.data
    success, level_block = levels.pick_success(
        df["outcome"].dropna(), bound.params.get("success"))
    if level_block is not None:
        return _blocked(bound, findings + (level_block,))
    disp = levels.display(success)
    groups = list(dict.fromkeys(df["group"].dropna().tolist()))
    g1, g2 = groups[0], groups[1]              # raw values -> keep for the split
    d1, d2 = levels.display(g1), levels.display(g2)   # labels a student reads

    def _kn(g):
        sub = df.loc[df["group"] == g, "outcome"].dropna()
        return int((sub == success).sum()), int(len(sub))

    k1, n1 = _kn(g1)
    k2, n2 = _kn(g2)
    p1, p2 = k1 / n1, k2 / n2

    z, p_val = proportions_ztest([k1, k2], [n1, n2])
    z, p_val = float(z), float(p_val)
    if _nan(z, p_val):
        return _blocked(bound, findings, "The z statistic could not be computed.")
    ci = confint_proportions_2indep(k1, n1, k2, n2, method="newcomb")
    h = effects.cohens_h(p1, p2)

    return Result(
        test_id="prop_2", test_name="Two-sample proportion z-test", status="ok",
        statistic=("z", z), p=p_val,
        estimate=(f"difference in proportion '{disp}' ({d1} − {d2})", p1 - p2),
        estimate_ci=(float(ci[0]), float(ci[1])),
        effect=("Cohen's h", h), effect_label=effects.h_label(h),
        effect_source="Cohen (1988)",
        n=_n(bound, **{d1: n1, d2: n2}),
        groups=(d1, d2), higher=d1 if p1 > p2 else d2,
        labels={"success": disp},
        variant_notes=("95% Newcombe difference interval",),
        method=("statsmodels proportions_ztest + confint_proportions_2indep("
                "method='newcomb')"),
        findings=findings, extra={"success": disp})


RUNNERS = {"prop_1": prop_1, "prop_2": prop_2}
