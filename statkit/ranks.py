"""Rank-based test runners (PLAN §3, Chunk 9): mwu, wilcoxon, kruskal, friedman.

Each runner is ``run(bound) -> Result`` (the registry signature). The Bound is the
canonical frame from bind() -- columns named by role. Flow (task / PLAN §5.2):

    check(bound) -> a blocked Result if any structural block fires
      -> statistic (scipy, correct-by-default args)
      -> effect size (statkit.effects, ours)
      -> post-hoc iff the omnibus is significant (auto, always adjusted)
      -> Result

Silent-garbage gate: a NaN statistic/p, or an all-zero Wilcoxon, yields
``status='blocked'`` with a finding naming the cause -- scipy returns garbage
(NaN, or a meaningless p=1.0) rather than raising on these (PLAN §5.2).

Only stdlib + numpy + scipy + statsmodels + pandas (L3 allowlist). No I/O.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests

from . import effects, levels
from .check import gate
from .model import Bound, Finding, Result
from .posthoc import dunn


# --------------------------------------------------------------------------
# shared runner scaffolding
# --------------------------------------------------------------------------
def _n(bound: Bound, **extra) -> dict:
    n = {"total": bound.n_total, "used": bound.n_used,
         "dropped": bound.n_total - bound.n_used}
    n.update(extra)
    return n


def _findings(bound: Bound):
    """Structural findings + the block subset (merged with any pre-attached)."""
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


def _group_arrays(bound: Bound):
    """Split the canonical long 'outcome' by the 'group' levels (first-seen order)."""
    df = bound.data
    order = list(dict.fromkeys(df["group"].tolist()))
    return {g: df.loc[df["group"] == g, "outcome"].to_numpy(float) for g in order}


# --------------------------------------------------------------------------
# mwu — Mann-Whitney U
# --------------------------------------------------------------------------
def mwu(bound: Bound) -> Result:
    findings, blocks = _findings(bound)
    if blocks:
        return _blocked(bound, blocks)

    groups = _group_arrays(bound)
    # F1/F4: the split keys off the RAW level (above), but every label the
    # student reads (groups=, higher, the estimate label, the _n keys) is a
    # display string -- a numeric code 1/2 reads '1'/'2', not 1.0/2.0, and
    # float keys no longer crash `_n(bound, **{la: n1})`.
    groups = {levels.display(g): a for g, a in groups.items()}
    (la, a), (lb, b) = list(groups.items())
    n1, n2 = len(a), len(b)

    combined = np.concatenate([a, b])
    no_ties = len(np.unique(combined)) == len(combined)
    exact = no_ties and n1 <= 8 and n2 <= 8
    method = "exact" if exact else "asymptotic"
    res = ss.mannwhitneyu(a, b, alternative="two-sided", method="auto")
    U, p = float(res.statistic), float(res.pvalue)
    if _nan(U, p):
        return _blocked(bound, findings, "The U statistic could not be computed.")

    # sign convention: positive when groups[0] tends higher (Kerby 2014);
    # effects.rank_biserial_u keeps the textbook 1 − 2U/(n₁n₂) form, so negate it.
    rb = -effects.rank_biserial_u(U, n1, n2)
    shift = float(np.median(np.subtract.outer(a, b)))   # Hodges-Lehmann (no CI)
    # direction from MEAN RANKS (what MWU actually tests), not raw means (B1):
    # a lone outlier can flip the raw mean without changing the rank ordering.
    rk = rankdata(combined)
    higher = la if rk[:n1].mean() > rk[n1:].mean() else lb

    return Result(
        test_id="mwu", test_name="Mann-Whitney U test", status="ok",
        statistic=("U", U), p=p,
        estimate=(f"Hodges-Lehmann median difference ({la} − {lb})", shift),
        effect=("rank-biserial correlation", rb),
        effect_label=effects.r_label(rb), effect_source="Cohen (1988)",
        n=_n(bound, **{la: n1, lb: n2}),
        groups=(la, lb), higher=higher,
        variant_notes=(f"{method} method"
                       + ("" if no_ties else "; ties present, tie-corrected"),),
        method=("scipy.stats.mannwhitneyu(a, b, alternative='two-sided', "
                f"method='auto') -> {method}"),
        findings=findings,
        arrays={"groups": {la: a, lb: b}},
    )


# --------------------------------------------------------------------------
# wilcoxon — signed-rank (paired + one-sample vs mu0)
# --------------------------------------------------------------------------
def wilcoxon(bound: Bound) -> Result:
    findings, blocks = _findings(bound)
    if blocks:
        return _blocked(bound, blocks)

    one_sample = "after" not in bound.data.columns
    groups: tuple = ()
    labels: dict = {}
    extra: dict = {}
    b_hdr = a_hdr = None
    if one_sample:                                       # branch on ABSENCE of after
        mu0 = float(bound.params.get("mu0", 0.0))
        x = bound.data["outcome"].to_numpy(float)
        diff = x - mu0
        name = "Wilcoxon signed-rank test (one-sample)"
        est_label = f"median difference from μ₀ = {mu0:g}"
        extra["mu0"] = mu0
    else:
        b_hdr = bound.columns.get("before", ("before",))[0]
        a_hdr = bound.columns.get("after", ("after",))[0]
        before = bound.data["before"].to_numpy(float)
        after = bound.data["after"].to_numpy(float)
        diff = before - after
        name = "Wilcoxon signed-rank test"
        est_label = f"median of differences ({b_hdr} − {a_hdr})"
        groups = (b_hdr, a_hdr)
        labels = {"before": b_hdr, "after": a_hdr}

    nonzero = diff[diff != 0]
    n_zero = int(len(diff) - len(nonzero))
    if len(nonzero) == 0:
        return _blocked(bound, findings,
                        "Every difference is zero; the test cannot run.")

    res = ss.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
    W, p = float(res.statistic), float(res.pvalue)
    if _nan(W, p):
        return _blocked(bound, findings, "The W statistic could not be computed.")

    ar = rankdata(np.abs(nonzero))                      # W+ / W- for rank-biserial
    w_plus = float(ar[nonzero > 0].sum())
    w_minus = float(ar[nonzero < 0].sum())
    rb = effects.rank_biserial_wilcoxon(w_plus, w_minus)
    # rank-based direction: the condition with the larger signed-rank mass is higher.
    higher = (b_hdr if w_plus > w_minus else a_hdr) if not one_sample else None

    notes = [f"{n_zero} zero difference(s) excluded (zero_method='wilcox')"] if n_zero else []

    return Result(
        test_id="wilcoxon", test_name=name, status="ok",
        statistic=("W", W), p=p,
        estimate=(est_label, float(np.median(diff))),
        effect=("matched-pairs rank-biserial correlation", rb),
        effect_label=effects.r_label(rb), effect_source="Cohen (1988)",
        n=_n(bound, pairs=len(diff), nonzero_pairs=len(nonzero)),
        groups=groups, higher=higher, labels=labels,
        variant_notes=tuple(notes),
        method="scipy.stats.wilcoxon(d, zero_method='wilcox', alternative='two-sided')",
        findings=findings, extra=extra,
    )


# --------------------------------------------------------------------------
# kruskal — Kruskal-Wallis H (+ Dunn post-hoc)
# --------------------------------------------------------------------------
def kruskal(bound: Bound) -> Result:
    findings, blocks = _findings(bound)
    if blocks:
        return _blocked(bound, blocks)

    groups = _group_arrays(bound)
    # F1/F4: split keys off the RAW level, but relabel to display strings so a
    # numeric code 1/2/3 reads '1'/'2'/'3' everywhere the student looks (groups=,
    # _n keys, Dunn's group1/group2) -- and float keys no longer crash `_n`.
    groups = {levels.display(g): a for g, a in groups.items()}
    arrs = list(groups.values())
    k = len(arrs)
    total_n = sum(len(a) for a in arrs)
    H, p = (lambda r: (float(r.statistic), float(r.pvalue)))(ss.kruskal(*arrs))
    if _nan(H, p):
        return _blocked(bound, findings, "The H statistic could not be computed "
                        "(all values may be identical).")

    # F7: (H - k + 1)/(n - k) is Tomczak & Tomczak's eta^2_H, not their epsilon^2
    # (= H/(n - 1)); the function keeps the name epsilon_squared but the label
    # reported to the student is eta-squared (H).
    eps2 = effects.epsilon_squared(H, k, total_n)

    posthoc, posthoc_name = None, None
    if p < 0.05:                                        # auto post-hoc iff omnibus sig (T-16)
        posthoc = dunn(groups)
        posthoc_name = "Dunn's test (Holm-adjusted)"

    return Result(
        test_id="kruskal", test_name="Kruskal-Wallis H test", status="ok",
        statistic=("H", H), df=(float(k - 1),), p=p,
        effect=("eta-squared (H)", eps2),
        effect_label=effects.eta_label(eps2),
        effect_source="Tomczak & Tomczak (2014)",
        n=_n(bound, **{g: len(a) for g, a in groups.items()}),
        groups=tuple(groups), posthoc=posthoc, posthoc_name=posthoc_name,
        method="scipy.stats.kruskal(*groups)",
        findings=findings,
        arrays={"groups": groups},
    )


# --------------------------------------------------------------------------
# friedman — Friedman test (+ pairwise Wilcoxon-Holm post-hoc)
# --------------------------------------------------------------------------
def _measure_labels(bound: Bound):
    """Friendly labels for the repeated-measure columns (original headers if any)."""
    cols = list(bound.data.columns)
    headers = bound.columns.get("measures")
    if headers and len(headers) == len(cols):
        return list(headers)
    return cols


def _pairwise_wilcoxon_holm(cols: dict) -> pd.DataFrame:
    names = list(cols)
    rows = []
    for i, j in combinations(range(len(names)), 2):
        res = ss.wilcoxon(cols[names[i]], cols[names[j]],
                          zero_method="wilcox", alternative="two-sided")
        rows.append({"group1": names[i], "group2": names[j],
                     "W": float(res.statistic), "p": float(res.pvalue)})
    df = pd.DataFrame(rows, columns=["group1", "group2", "W", "p"])
    df["p_holm"] = multipletests(df["p"].to_numpy(), method="holm")[1]
    return df


def friedman(bound: Bound) -> Result:
    findings, blocks = _findings(bound)
    if blocks:
        return _blocked(bound, blocks)

    cols = [bound.data[c].to_numpy(float) for c in bound.data.columns]
    n_subj, k = len(bound.data), len(cols)
    res = ss.friedmanchisquare(*cols)
    chi2, p = float(res.statistic), float(res.pvalue)
    if _nan(chi2, p):
        return _blocked(bound, findings, "The Friedman statistic could not be computed.")

    W = effects.kendalls_w(chi2, n_subj, k)

    posthoc, posthoc_name = None, None
    if p < 0.05:
        labels = _measure_labels(bound)
        posthoc = _pairwise_wilcoxon_holm(dict(zip(labels, cols)))
        posthoc_name = "pairwise Wilcoxon signed-rank (Holm-adjusted)"

    return Result(
        test_id="friedman", test_name="Friedman test", status="ok",
        statistic=("χ²", chi2), df=(float(k - 1),), p=p,
        effect=("Kendall's W", W),
        effect_label=effects.w_kendall_label(W),
        effect_source="Tomczak & Tomczak (2014)",
        n=_n(bound, subjects=n_subj, measures=k),
        posthoc=posthoc, posthoc_name=posthoc_name,
        method="scipy.stats.friedmanchisquare(*measures)",
        findings=findings,
    )


RUNNERS = {"mwu": mwu, "wilcoxon": wilcoxon, "kruskal": kruskal, "friedman": friedman}
