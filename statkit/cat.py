"""Categorical / count test runners (PLAN §3, Chunk 9): chi2_ind, chi2_gof,
fisher, mcnemar, cochran_q.

Correct-by-default (PLAN D4, §5.2, snag 15):
  * chi2_ind uses ``correction=False`` (no Yates) and HARD-ROUTES to Fisher when an
    expected count is too small (any <1, or a 2x2 with any <5): a block Result that
    suggests ``fisher``. Effect: phi + conditional OR (2x2) / Cramer's V + adjusted
    residuals (R x C).
  * fisher labels 2x2 as "Fisher's exact test" (one conditional-MLE odds ratio in
    estimate; no duplicate sample OR) and R x C as the "Fisher-Freeman-Halton test
    (Monte Carlo p)" (statistic = the table probability, p only); R x C guarded to
    r*c<=25, N<=2000. Both carry labels + the observed crosstab (S4).
  * mcnemar switches exact (b+c<25) <-> chi2-with-continuity (b+c>=25); when b*c=0 the
    OR is "not estimable" but the p and change % are KEPT (snag 15); b+c=0 blocks.

Each runner is ``run(bound) -> Result``. check() gates structural blocks first; a
NaN statistic/p gates to blocked (silent-garbage). Only stdlib + numpy + scipy +
statsmodels + pandas (L3 allowlist). No I/O.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.stats import contingency
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar as _sm_mcnemar
from statsmodels.stats.multitest import multipletests

from . import effects, levels
from .check import gate
from .model import Bound, Finding, Result

# R x C Fisher-Freeman-Halton: scipy uses a Monte Carlo test (no exact enumeration).
# A fixed seed + many resamples make it REPRODUCIBLE for a student report (PLAN D18).
_FH_SEED = 20260921
_FH_RESAMPLES = 99999


# --------------------------------------------------------------------------
# shared runner scaffolding
# --------------------------------------------------------------------------
def _n(bound: Bound, **extra) -> dict:
    n = {"total": bound.n_total, "used": bound.n_used,
         "dropped": bound.n_total - bound.n_used}
    n.update(extra)
    return n


def _findings(bound: Bound):
    findings = gate(bound)
    blocks = tuple(f for f in findings if f.severity == "block")
    return findings, blocks


def _blocked(bound: Bound, findings, reason: str = "", suggest: str | None = None) -> Result:
    fs = tuple(findings)
    if reason and not any(f.severity == "block" for f in fs):
        fs = fs + (Finding("block", reason, suggest_test=suggest, code="RUN"),)
    return Result(test_id=bound.test.id, test_name=bound.test.name,
                  status="blocked", n=_n(bound), findings=fs)


def _nan(*vals) -> bool:
    return any(v is None or (isinstance(v, float) and np.isnan(v)) for v in vals)


def _display_xtab(tab: pd.DataFrame) -> pd.DataFrame:
    """A crosstab relabeled with student-facing level strings, so a numeric-coded
    0/1 column reads '0'/'1' (not '0.0'/'1.0'). Counts are unchanged: computation
    keys off the raw values (``tab.to_numpy``), only the displayed labels change."""
    tab = tab.copy()
    tab.index = [levels.display(v) for v in tab.index]
    tab.columns = [levels.display(v) for v in tab.columns]
    return tab


# --------------------------------------------------------------------------
# chi2_ind — chi-square test of independence (no Yates + Fisher routing)
# --------------------------------------------------------------------------
def chi2_ind(bound: Bound) -> Result:
    findings, blocks = _findings(bound)
    if blocks:
        return _blocked(bound, blocks)

    tab = pd.crosstab(bound.data["row"], bound.data["col"])
    obs = tab.to_numpy(float)
    if obs.shape[0] < 2 or obs.shape[1] < 2:
        return _blocked(bound, findings, "A chi-square test needs at least two "
                        "categories in each variable; after dropping missing "
                        "values one of them has only one.")
    tab = _display_xtab(tab)                     # student-facing level labels (F4)
    chi2, p, dof, expected = ss.chi2_contingency(obs, correction=False)
    chi2, p, dof = float(chi2), float(p), int(dof)
    r_, c_ = obs.shape

    # Fisher routing (PLAN D4/D3): a block that suggests fisher.
    if (expected < 1).any() or (r_ == 2 and c_ == 2 and (expected < 5).any()):
        f = Finding("block", "Some expected counts are too small for a reliable "
                    "chi-square test; use Fisher's exact test instead.",
                    suggest_test="fisher", code="D3")
        return _blocked(bound, findings + (f,))
    if _nan(chi2, p):
        return _blocked(bound, findings, "The chi-square statistic could not be computed.")

    exp_df = pd.DataFrame(expected, index=tab.index, columns=tab.columns)
    notes = ["no Yates' continuity correction (correction=False)"]

    if r_ == 2 and c_ == 2:
        phi = effects.phi(obs)
        orr = contingency.odds_ratio(obs.astype(int))
        ci = orr.confidence_interval(0.95)
        result = dict(
            effect=("phi", phi), effect_label=effects.phi_w_label(phi),
            estimate=("odds ratio", float(orr.statistic)),
            estimate_ci=(float(ci.low), float(ci.high)),
            expected=exp_df,
            extra={"observed": tab, "or": _or_orientation(tab)})
    else:
        v = effects.cramers_v(obs)
        residuals = _adjusted_residuals(tab)
        result = dict(
            effect=("Cramér's V", v),
            effect_label=effects.cramers_v_label(v, r_, c_),
            expected=exp_df, table=residuals,             # keep table= for compatibility
            extra={"observed": tab, "adjusted_residuals": residuals})

    return Result(
        test_id="chi2_ind", test_name="Chi-square test of independence", status="ok",
        statistic=("χ²", chi2), df=(float(dof),), p=p,
        effect_source="Cohen (1988)",
        n=_n(bound), labels=_table_labels(bound), variant_notes=tuple(notes),
        method="scipy.stats.chi2_contingency(table, correction=False)",
        findings=findings, **result)


def _adjusted_residuals(tab: pd.DataFrame) -> pd.DataFrame:
    """Adjusted standardized residuals; |z|>1.96 is the flagged cell (PLAN §3)."""
    obs = tab.to_numpy(float)
    n = obs.sum()
    row_p = obs.sum(axis=1, keepdims=True) / n
    col_p = obs.sum(axis=0, keepdims=True) / n
    exp = row_p * col_p * n
    z = (obs - exp) / np.sqrt(exp * (1 - row_p) * (1 - col_p))
    return pd.DataFrame(z, index=tab.index, columns=tab.columns)


def _table_labels(bound: Bound) -> dict:
    """Name the row/col variables for the sentence, chart and report (S4). A long
    /two-column bind carries the real headers; a count-grid (table layout) has only
    the 'counts' role, so fall back to generic names."""
    cols = bound.columns
    if cols.get("row") and cols.get("col"):
        return {"row": cols["row"][0], "col": cols["col"][0]}
    return {"row": "row variable", "col": "column variable"}


def _or_orientation(tab: pd.DataFrame) -> dict:
    """Which cells the 2x2 odds ratio compares (S4). contingency.odds_ratio(
    [[a,b],[c,d]]) computes OR = (a/b)/(c/d) = odds(col0 | row0) / odds(col0 | row1),
    so the levels are the first column and the two rows of the crosstab."""
    return {"col_level": levels.display(tab.columns[0]),
            "row_level": levels.display(tab.index[0]),
            "row_ref": levels.display(tab.index[1])}


# --------------------------------------------------------------------------
# chi2_gof — goodness of fit
# --------------------------------------------------------------------------
def chi2_gof(bound: Bound) -> Result:
    findings, blocks = _findings(bound)
    if blocks:
        return _blocked(bound, blocks)

    cats = list(dict.fromkeys(bound.data["category"].tolist()))
    observed = bound.data["category"].value_counts().reindex(cats).to_numpy(float)
    total = float(observed.sum())

    props = bound.params.get("expected") or ()
    f_exp = (np.asarray(props, float) * total) if props else None
    chi2, p = ss.chisquare(observed, f_exp=f_exp)
    chi2, p = float(chi2), float(p)
    if _nan(chi2, p):
        return _blocked(bound, findings, "The chi-square statistic could not be computed.")

    w = effects.cohens_w(chi2, int(total))
    return Result(
        test_id="chi2_gof", test_name="Chi-square goodness-of-fit test", status="ok",
        statistic=("χ²", chi2), df=(float(len(observed) - 1),), p=p,
        effect=("Cohen's w", w), effect_label=effects.phi_w_label(w),
        effect_source="Cohen (1988)",
        n=_n(bound), groups=tuple(levels.display(c) for c in cats),
        table=pd.DataFrame({"category": [levels.display(c) for c in cats],
                            "observed": observed,
                            "expected": (f_exp if f_exp is not None
                                         else np.full(len(observed), total / len(observed)))}),
        method="scipy.stats.chisquare(observed, f_exp=N*p)",
        findings=findings)


# --------------------------------------------------------------------------
# fisher — exact (2x2) / Fisher-Freeman-Halton (R x C)
# --------------------------------------------------------------------------
def fisher(bound: Bound) -> Result:
    findings, blocks = _findings(bound)
    if blocks:
        return _blocked(bound, blocks)

    tab = pd.crosstab(bound.data["row"], bound.data["col"])
    obs = tab.to_numpy()
    r_, c_ = obs.shape
    if r_ < 2 or c_ < 2:
        return _blocked(bound, findings, "Fisher's exact test needs at least two "
                        "categories in each variable; after dropping missing "
                        "values one of them has only one.")
    tab = _display_xtab(tab)                      # student-facing level labels (F4)

    if r_ == 2 and c_ == 2:
        res = ss.fisher_exact(obs, alternative="two-sided")
        p = float(res.pvalue)
        orr = contingency.odds_ratio(obs.astype(int))
        ci = orr.confidence_interval(0.95)
        phi = effects.phi(obs.astype(float))
        return Result(
            test_id="fisher", test_name="Fisher's exact test", status="ok",
            statistic=None, p=p,          # S13: one OR only (the conditional MLE below)
            estimate=("odds ratio", float(orr.statistic)),
            estimate_ci=(float(ci.low), float(ci.high)),
            effect=("phi", phi), effect_label=effects.phi_w_label(phi),
            effect_source="Cohen (1988)",
            n=_n(bound), labels=_table_labels(bound), expected=None,
            extra={"observed": tab, "or": _or_orientation(tab)},
            method="scipy.stats.fisher_exact(table)  # 2x2, conditional MLE OR + exact CI",
            findings=findings)

    # R x C: Fisher-Freeman-Halton. Guard the combinatorial cost (PLAN §3).
    if r_ * c_ > 25 or obs.sum() > 2000:
        return _blocked(bound, findings, "The table is too large for the exact "
                        "Fisher-Freeman-Halton test (needs r*c<=25 and N<=2000).")
    # SNAG: scipy's R x C fisher_exact is Monte Carlo (non-deterministic) by default.
    # A student report must be reproducible (PLAN D18), so we SEED it -> same input
    # gives the same p every run; named honestly as a Monte Carlo estimate.
    mc = ss.MonteCarloMethod(n_resamples=_FH_RESAMPLES,
                             rng=np.random.default_rng(_FH_SEED))
    res = ss.fisher_exact(obs, method=mc)
    # S-L: on a degenerate table scipy returns p/statistic as a length-1 array,
    # not a 0-d scalar; float() on it raises. Flatten and take the lone element.
    pv = np.asarray(res.pvalue).reshape(-1)
    st = np.asarray(res.statistic).reshape(-1)
    if pv.size != 1 or st.size != 1:
        return _blocked(bound, findings, "The exact test p-value could not be computed.")
    p = float(pv[0])
    if _nan(p):
        return _blocked(bound, findings, "The exact test p-value could not be computed.")
    residuals = _adjusted_residuals(tab)
    return Result(
        test_id="fisher", test_name="Fisher-Freeman-Halton test (Monte Carlo p)",
        status="ok",
        statistic=("table probability", float(st[0])), p=p,
        n=_n(bound), labels=_table_labels(bound), table=residuals,
        extra={"observed": tab, "adjusted_residuals": residuals},
        variant_notes=(f"Monte Carlo estimate ({_FH_RESAMPLES:,} resamples, seeded "
                       "for reproducibility)",),
        method=(f"scipy.stats.fisher_exact(table, method=MonteCarloMethod("
                f"n_resamples={_FH_RESAMPLES}, seeded))  # R x C Freeman-Halton"),
        findings=findings)


# --------------------------------------------------------------------------
# mcnemar — exact/chi2 switch, snag-15 OR handling
# --------------------------------------------------------------------------
def mcnemar(bound: Bound) -> Result:
    findings, blocks = _findings(bound)
    if blocks:
        return _blocked(bound, blocks)

    before, after = bound.data["before"], bound.data["after"]
    level_vals = sorted(set(before.dropna()) | set(after.dropna()), key=str)
    tab = pd.crosstab(before, after).reindex(
        index=level_vals, columns=level_vals, fill_value=0)
    mat = tab.to_numpy(float)
    tab = _display_xtab(tab)                      # student-facing level labels (F4)
    b_cell, c_cell = float(mat[0, 1]), float(mat[1, 0])
    disc = b_cell + c_cell
    if disc == 0:
        return _blocked(bound, findings, "No one changed between the two measurements; "
                        "McNemar's test needs at least one discordant pair.")

    exact = disc < 25
    res = _sm_mcnemar(mat, exact=exact, correction=True)
    stat, p = float(res.statistic), float(res.pvalue)
    if _nan(p):
        return _blocked(bound, findings, "The p-value could not be computed.")

    if exact:
        name = "McNemar's test (exact)"
        statistic = ("smaller discordant count", stat)
    else:
        name = "McNemar's test (χ² with continuity correction)"
        statistic = ("χ²", stat)

    notes = [f"{int(disc)} of {bound.n_used} changed ("
             f"{_pct(disc / bound.n_used)})"]
    if b_cell * c_cell == 0:                                # snag 15: OR not estimable
        notes.append("odds ratio not estimable (one direction of change has zero cases)")
        estimate, estimate_ci = None, None
    else:
        orr = b_cell / c_cell
        se = np.sqrt(1.0 / b_cell + 1.0 / c_cell)
        estimate = ("odds ratio", float(orr))
        estimate_ci = (float(np.exp(np.log(orr) - 1.96 * se)),
                       float(np.exp(np.log(orr) + 1.96 * se)))

    return Result(
        test_id="mcnemar", test_name=name, status="ok",
        statistic=statistic, p=p, estimate=estimate, estimate_ci=estimate_ci,
        n=_n(bound, discordant=int(disc)), variant_notes=tuple(notes),
        table=tab, method=f"statsmodels.stats.contingency_tables.mcnemar("
                          f"table, exact={exact}, correction=True)",
        findings=findings)


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


# --------------------------------------------------------------------------
# cochran_q — Cochran's Q (+ pairwise McNemar-Holm)
# --------------------------------------------------------------------------
def _to01(bound: Bound):
    """The measure columns as a 0/1 matrix, their labels, and the success level's
    display string (success -> 1). The success is picked via ``levels.pick_success``
    over the pooled measure levels (the shared positive-token-then-string-sort
    rule), so the 0/1 encoding — and thus the proportions/Q/p — are unchanged."""
    cols = list(bound.data.columns)
    pooled = pd.concat([bound.data[c] for c in cols], ignore_index=True)
    success, _ = levels.pick_success(pooled)
    X = np.column_stack([(bound.data[c] == success).to_numpy(int) for c in cols])
    headers = bound.columns.get("measures")
    lbls = list(headers) if headers and len(headers) == len(cols) else cols
    return X, lbls, levels.display(success)


def _pairwise_mcnemar_holm(X, labels) -> pd.DataFrame:
    rows = []
    for i, j in combinations(range(X.shape[1]), 2):
        t = pd.crosstab(X[:, i], X[:, j]).reindex(index=[0, 1], columns=[0, 1],
                                                  fill_value=0).to_numpy(float)
        disc = t[0, 1] + t[1, 0]
        res = _sm_mcnemar(t, exact=disc < 25, correction=True)
        rows.append({"group1": labels[i], "group2": labels[j], "p": float(res.pvalue)})
    df = pd.DataFrame(rows, columns=["group1", "group2", "p"])
    df["p_holm"] = multipletests(df["p"].to_numpy(), method="holm")[1]
    return df


def cochran_q(bound: Bound) -> Result:
    findings, blocks = _findings(bound)
    if blocks:
        return _blocked(bound, blocks)

    X, labels, success = _to01(bound)
    res = cochrans_q(X)
    Q, p = float(res.statistic), float(res.pvalue)
    if _nan(Q, p):
        return _blocked(bound, findings, "The Q statistic could not be computed.")

    props = X.mean(axis=0)
    table = pd.DataFrame({"measure": labels, "proportion": props})

    posthoc, posthoc_name = None, None
    if p < 0.05:
        posthoc = _pairwise_mcnemar_holm(X, labels)
        posthoc_name = "pairwise McNemar (Holm-adjusted)"

    return Result(
        test_id="cochran_q", test_name="Cochran's Q test", status="ok",
        statistic=("Q", Q), df=(float(X.shape[1] - 1),), p=p,
        n=_n(bound, subjects=X.shape[0], measures=X.shape[1]),
        labels={"success": success},
        table=table, posthoc=posthoc, posthoc_name=posthoc_name,
        method="statsmodels.stats.contingency_tables.cochrans_q(X)",
        findings=findings)


RUNNERS = {"chi2_ind": chi2_ind, "chi2_gof": chi2_gof, "fisher": fisher,
           "mcnemar": mcnemar, "cochran_q": cochran_q}
