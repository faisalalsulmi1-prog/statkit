"""Association runners — the ``relate`` family (PLAN §3, Chunk 10).

``pearson`` / ``spearman`` / ``kendall`` / ``corr_matrix``. Each is a pure
``run(Bound) -> Result``: it honours the structural checks first (a blocked bind
returns ``status='blocked'`` with the reason, and gates on a NaN statistic/p so a
constant column can never surface as a silent garbage result — PLAN §5.2), then
computes the coefficient via the exact scipy routine the PLAN names, plus a
deterministic closed-form CI (no bootstrap — PLAN D18).

CIs (PLAN D18):
  * Pearson  — ``pearsonr(...).confidence_interval()`` (Fisher z).
  * Spearman — Bonett & Wright (2000): SE = sqrt((1 + r^2/2)/(n-3)).
  * Kendall  — Fieller-Hartley-Pearson: SE = sqrt(0.437/(n-4)).

SNAG (corr_matrix): the PLAN wants *pairwise-complete* correlations, but
``bind()`` performs listwise deletion for the long layout and the runner only
receives the already-cleaned frame, so in this pipeline complete-case ==
pairwise. corr_matrix therefore computes on complete-case data (one shared n);
the per-pair machinery + n-matrix are kept so a future ``bind`` opt-out (skip
listwise for this test) makes it truly pairwise with no runner change. See the
report.

Only numpy + scipy + statsmodels + pandas (L3 allowlist); no I/O, no state.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

from . import effects
from .check import check
from .model import Bound, Finding, Result

_CRIT = norm.ppf(0.975)   # two-sided 95%


# --------------------------------------------------------------------------
# shared structural gate + blocked result
# --------------------------------------------------------------------------
def _gather(bound: Bound) -> tuple[tuple[Finding, ...], tuple[Finding, ...]]:
    """All findings (bound's own + a fresh structural check, deduped) and the
    subset that BLOCKS (PLAN §5.1: a block disables Run)."""
    findings = tuple(bound.findings) + tuple(
        f for f in check(bound) if f not in bound.findings)
    blocks = tuple(f for f in findings if f.severity == "block")
    return findings, blocks


def _n(bound: Bound) -> dict:
    return {"total": bound.n_total, "used": bound.n_used,
            "dropped": bound.n_total - bound.n_used}


def _blocked(bound: Bound, findings: tuple[Finding, ...]) -> Result:
    reason = next((f.text for f in findings if f.severity == "block"),
                  "This test cannot run on the chosen columns.")
    return Result(test_id=bound.test.id, test_name=bound.test.name,
                  status="blocked", n=_n(bound), findings=findings,
                  variant_notes=(reason,))


def _labels(bound: Bound, *roles: str) -> dict:
    return {role: bound.columns[role][0] for role in roles if role in bound.columns}


# --------------------------------------------------------------------------
# the three pairwise correlations
# --------------------------------------------------------------------------
def _corr_runner(bound: Bound, stat_name: str, fn, ci_fn, method_call: str) -> Result:
    findings, blocks = _gather(bound)
    if blocks:
        return _blocked(bound, findings)
    x = bound.data["x"].to_numpy(dtype=float)
    y = bound.data["y"].to_numpy(dtype=float)
    res = fn(x, y)
    coef, p = float(res.statistic), float(res.pvalue)
    if not (math.isfinite(coef) and math.isfinite(p)):
        f = Finding("block", "The correlation is undefined (a column has no "
                    "variation).", code="NAN")
        return _blocked(bound, findings + (f,))
    ci = ci_fn(coef, bound.n_used, res)
    return Result(
        test_id=bound.test.id, test_name=bound.test.name, status="ok",
        statistic=(stat_name, coef), p=p,
        effect=(stat_name, coef), effect_ci=ci,
        effect_label=effects.r_label(coef), effect_source="Cohen (1988)",
        n=_n(bound), labels=_labels(bound, "x", "y"),
        method=method_call, findings=findings,
        extra={"r2": coef ** 2}, arrays={"x": x, "y": y},
    )


def _fisher_ci(coef: float, se: float) -> tuple[float, float] | None:
    if abs(coef) >= 1 - 1e-12:       # atanh diverges at |coef|==1: no finite CI
        return None
    z = math.atanh(coef)
    return math.tanh(z - _CRIT * se), math.tanh(z + _CRIT * se)


def pearson(bound: Bound) -> Result:
    return _corr_runner(
        bound, "r", pearsonr,
        lambda coef, n, res: tuple(res.confidence_interval()),
        "scipy.stats.pearsonr(x, y); 95% CI via Fisher z")


def _spearman_ci(coef, n, res):
    if n <= 3:                       # SE undefined (Bonett-Wright divides by n-3)
        return None
    return _fisher_ci(coef, math.sqrt((1 + coef ** 2 / 2) / (n - 3)))


def _kendall_ci(coef, n, res):
    if n <= 4:                       # SE undefined (Fieller divides by n-4)
        return None
    return _fisher_ci(coef, math.sqrt(0.437 / (n - 4)))


def spearman(bound: Bound) -> Result:
    return _corr_runner(
        bound, "rho", spearmanr, _spearman_ci,
        "scipy.stats.spearmanr(x, y); 95% CI Bonett-Wright Fisher z")


def kendall(bound: Bound) -> Result:
    return _corr_runner(
        bound, "tau-b", kendalltau, _kendall_ci,
        "scipy.stats.kendalltau(x, y, variant='b'); 95% CI Fieller-Hartley-Pearson")


# --------------------------------------------------------------------------
# correlation matrix
# --------------------------------------------------------------------------
def corr_matrix(bound: Bound) -> Result:
    findings, blocks = _gather(bound)
    if blocks:
        return _blocked(bound, findings)
    cols = [c for c in bound.data.columns if c.startswith("variables__")]
    headers = list(bound.columns["variables"])                    # verbatim labels
    method = bound.params.get("method", "pearson")
    fn = spearmanr if method == "spearman" else pearsonr

    k = len(cols)
    R = np.eye(k)
    P = np.zeros((k, k))
    Ns = np.zeros((k, k), dtype=int)
    pair_p: list[float] = []
    pair_ij: list[tuple[int, int]] = []
    pair_notes: list[Finding] = []
    for i in range(k):
        Ns[i, i] = int(bound.data[cols[i]].notna().sum())
        for j in range(i + 1, k):
            a = bound.data[cols[i]].to_numpy(dtype=float)
            b = bound.data[cols[j]].to_numpy(dtype=float)
            mask = ~(np.isnan(a) | np.isnan(b))
            n_ij = int(mask.sum())
            Ns[i, j] = Ns[j, i] = n_ij
            if n_ij < 3:            # scipy needs >=2 and a 2-pt r is degenerate:
                R[i, j] = R[j, i] = np.nan           # blank the cell, don't crash
                P[i, j] = P[j, i] = np.nan
                pair_notes.append(Finding(
                    "flag", f"{headers[i]} × {headers[j]}: only {n_ij} "
                    "overlapping value(s) — correlation left blank "
                    "(needs at least 3).", code="PAIR_N"))
                continue
            res = fn(a[mask], b[mask])
            r, p = float(res.statistic), float(res.pvalue)
            R[i, j] = R[j, i] = r
            P[i, j] = P[j, i] = p
            pair_p.append(p)
            pair_ij.append((i, j))

    holm = P.copy()
    if pair_p:
        adj = multipletests(pair_p, method="holm")[1]
        for (i, j), pa in zip(pair_ij, adj):
            holm[i, j] = holm[j, i] = pa
    # A variable-against-itself is not a hypothesis: the p diagonal is NaN
    # (renders "—"), never 0.0 ("p < .001"). Off-diagonal p/Holm untouched.
    np.fill_diagonal(P, np.nan)
    np.fill_diagonal(holm, np.nan)

    def _df(mat):
        return pd.DataFrame(mat, index=headers, columns=headers)

    # bind() no longer listwise-deletes for corr_matrix, so the frame carries NaN
    # and each pair is masked independently: report the pairwise-complete n's (the
    # off-diagonal counts), not one shared complete-case n (S5).
    off = Ns[np.triu_indices(k, 1)]
    used = int(off.min())
    n = {"total": bound.n_total, "used": used, "used_max": int(off.max()),
         "dropped": bound.n_total - used}
    return Result(
        test_id=bound.test.id, test_name=bound.test.name, status="ok",
        n=n, findings=findings + tuple(pair_notes), table=_df(R),
        method=f"pairwise-complete {method} correlation; off-diagonal p Holm-adjusted",
        extra={"p_raw": _df(P), "p_holm": _df(holm), "n": _df(Ns), "method": method},
    )


RUNNERS = {"pearson": pearson, "spearman": spearman, "kendall": kendall,
           "corr_matrix": corr_matrix}
