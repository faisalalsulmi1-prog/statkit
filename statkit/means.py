"""Mean-comparison runners: describe + the parametric t / F family (PLAN §3, §10).

Seven ``Callable[[Bound], Result]`` runners matching the registry runner
signature (``TestSpec.run``); the session wires each in by swapping the ``_todo``
stub for the function named in ``RUNNERS`` below:

    describe · t_1s · t_ind · t_paired · anova_1w · anova_2w · rm_anova

Correct-by-default (PLAN §1 / STATE locks):
  * **Welch is the default** for ``t_ind`` (``equal_var=False``); Student's only
    on explicit opt-in via the ``equal_var`` param.
  * Every runner GATES via ``check.py`` first: if any structural check BLOCKS
    (constant column, n<min, wrong arity, empty ANOVA cell, ...) the runner
    returns a ``status="blocked"`` Result carrying the reason -- never a NaN
    result or a crash. A NaN statistic/p after computation also blocks (§5.2).
  * Effect sizes come from ``effects.py`` (ours), assumption checks from
    ``assumptions.py``, auto post-hoc from ``posthoc.py`` (Tukey when variances
    are assumed equal, Games-Howell when Welch) and fire only when the omnibus
    test is significant (D16).

Only stdlib + numpy + scipy + statsmodels + pandas (L3 allowlist). No I/O.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import AnovaRM, anova_lm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway

from . import assumptions, effects as eff, levels, posthoc
from .check import gate
from .model import Finding, Result

_ALPHA = 0.05
_Z = 1.959963984540054  # two-sided 95% normal quantile


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def _n_dict(bound) -> dict:
    return {"total": bound.n_total, "used": bound.n_used,
            "dropped": bound.n_total - bound.n_used}


def _outcome_label(bound) -> str:
    """The outcome column's header, or a generic label when the bind was WIDE:
    a wide->long normalisation carries the group columns under ``bound.columns``
    ('groups'), so there is no single 'outcome' header to quote. Sentences and the
    report already fall back with ``labels.get('outcome', 'the outcome')``."""
    cols = bound.columns.get("outcome")
    return cols[0] if cols else "the outcome"


def _gate(bound) -> tuple:
    """Structural findings for this bound (merged with any pre-attached)."""
    return gate(bound)


def _blocked(bound, findings) -> Result:
    return Result(test_id=bound.test.id, test_name=bound.test.name,
                  status="blocked", p=None, n=_n_dict(bound),
                  findings=tuple(findings))


def _nan_block(bound, findings, cause: str) -> Result:
    extra = findings + (Finding("block", f"The test could not be computed: "
                                f"{cause}.", code="NAN"),)
    return _blocked(bound, extra)


def _finite(*xs) -> bool:
    return all(x is not None and np.isfinite(x) for x in xs)


def _levels_in_order(series) -> list:
    """Distinct values of a canonical group/condition column in first-seen
    order (bind has already dropped NaN rows for these frames)."""
    return list(dict.fromkeys(series.tolist()))


def _split_groups(bound) -> dict:
    """{level_label: outcome array} for a long outcome+group frame."""
    data = bound.data
    out = {}
    for lv in _levels_in_order(data["group"]):
        out[levels.display(lv)] = data.loc[
            data["group"] == lv, "outcome"].to_numpy(float)
    return out


def _smd_ci(g: float, n1: int, n2: int) -> tuple:
    """Large-sample CI for a standardized mean difference:
    Var(g) = (n1+n2)/(n1*n2) + g^2 / (2*(n1+n2-2))
    (Hedges & Olkin (1985) eq. 8; Borenstein et al. (2009) §4)."""
    var = (n1 + n2) / (n1 * n2) + g * g / (2.0 * (n1 + n2 - 2))
    se = math.sqrt(var)
    return (g - _Z * se, g + _Z * se)


# ==========================================================================
# describe -- Table 1 (PLAN §3). Pure summary: it does NOT apply the inferential
# S-checks (S8 would wrongly block a constant column that is perfectly
# describable); it only blocks when there is nothing to summarise.
# ==========================================================================
def _num_stats(header, group, s) -> dict:
    s = s.dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    row = {"variable": header, "n": int(s.size), "mean": float(s.mean()),
           "sd": float(s.std(ddof=1)) if s.size > 1 else 0.0,
           "median": float(s.median()), "iqr": float(q3 - q1),
           "min": float(s.min()), "max": float(s.max())}
    if group is not None:
        row = {"variable": header, "group": levels.display(group), **{k: v for k, v in row.items() if k != "variable"}}
    return row


def describe(bound) -> Result:
    n = _n_dict(bound)
    data = bound.data
    var_headers = bound.columns.get("variables", ())
    has_group = "group" in data.columns
    group_levels = _levels_in_order(data["group"]) if has_group else [None]

    # describe keeps every row (no cross-variable listwise, snag 8.3), so "nothing
    # to summarise" is decided on the DATA: block only when no variable column has
    # a single non-missing value.
    def _col_of(i, hdr):
        c = f"variables__{i}"
        return c if c in data.columns else (hdr if hdr in data.columns else c)

    if not any(data[_col_of(i, h)].notna().any() for i, h in enumerate(var_headers)):
        return _blocked(bound, [Finding(
            "block", "No usable rows to summarise (all values were missing).",
            code="S5")])

    num_rows, cats = [], {}
    for i, hdr in enumerate(var_headers):
        col = _col_of(i, hdr)
        s = data[col]
        if pd.api.types.is_numeric_dtype(s):
            for lv in group_levels:
                sub = s if lv is None else s[data["group"] == lv]
                num_rows.append(_num_stats(hdr, lv, sub))
        else:
            counts = {}
            for lv in group_levels:
                sub = s if lv is None else s[data["group"] == lv]
                vc = sub.dropna().value_counts()
                # lv is None only in the no-group case (a throwaway sentinel key
                # retrieved as "None" below); real group levels get their display
                # label so a numeric-coded group reads "1", not "1.0".
                key = "None" if lv is None else levels.display(lv)
                counts[key] = {levels.display(k): int(v) for k, v in vc.items()}
            cats[hdr] = counts if has_group else counts["None"]

    desc = pd.DataFrame(num_rows) if num_rows else None
    r = Result(test_id="describe", test_name="Descriptive statistics (Table 1)",
               status="ok", descriptives=desc, n=n,
               method="pandas describe / value_counts",
               findings=tuple(f for f in _gate(bound) if f.severity != "block"))
    if cats:
        r.extra["categoricals"] = cats
    return r


# ==========================================================================
# t_1s -- one-sample t-test
# ==========================================================================
def t_1s(bound) -> Result:
    findings = _gate(bound)
    if any(f.severity == "block" for f in findings):
        return _blocked(bound, findings)

    x = bound.data["outcome"].to_numpy(float)
    mu0 = float(bound.params.get("mu0", 0.0))
    res = stats.ttest_1samp(x, mu0)
    if not _finite(res.statistic, res.pvalue):
        return _nan_block(bound, findings, "non-finite t/p")
    ci = res.confidence_interval()                    # CI for the mean
    d = eff.cohens_d_onesample(x, mu0)

    return Result(
        test_id="t_1s", test_name="One-sample t-test", status="ok",
        statistic=("t", float(res.statistic)), df=(float(res.df),),
        p=float(res.pvalue),
        estimate=("mean − μ₀", float(x.mean() - mu0)),
        estimate_ci=(float(ci.low - mu0), float(ci.high - mu0)),
        effect=("Cohen's d", float(d)), effect_label=eff.d_label(d),
        effect_source="Cohen (1988)",
        n=_n_dict(bound),
        labels={"outcome": _outcome_label(bound)},
        checks=(assumptions.shapiro_normality(x),),
        method=f"scipy.stats.ttest_1samp(x, {mu0:g})",
        findings=tuple(findings),
        extra={"mu0": mu0},
    )


# ==========================================================================
# t_ind -- independent-samples t-test (Welch by default, D1)
# ==========================================================================
def t_ind(bound) -> Result:
    findings = _gate(bound)
    if any(f.severity == "block" for f in findings):
        return _blocked(bound, findings)

    groups = _split_groups(bound)
    levs = list(groups)
    a, b = groups[levs[0]], groups[levs[1]]
    # PARAM `equal_var` == "assume equal variances (Student's)". Default False
    # -> Welch. scipy's equal_var flag equals the param DIRECTLY (see SNAG note
    # in this module's report: the registry `library` string's `not equal_var`
    # is a typo that would make Student's the default, contradicting D1).
    student = bool(bound.params.get("equal_var", False))
    res = stats.ttest_ind(a, b, equal_var=student, alternative="two-sided")
    if not _finite(res.statistic, res.pvalue):
        return _nan_block(bound, findings, "non-finite t/p")
    ci = res.confidence_interval()
    diff = a.mean() - b.mean()

    small = min(len(a), len(b)) < 20
    if small:
        e, ename = eff.hedges_g(a, b), "Hedges' g"
    else:
        e, ename = eff.cohens_d(a, b), "Cohen's d"
    name = ("Student's independent-samples t-test" if student
            else "Welch's independent-samples t-test")
    higher = levs[0] if a.mean() > b.mean() else levs[1]

    return Result(
        test_id="t_ind", test_name=name, status="ok",
        statistic=("t", float(res.statistic)), df=(float(res.df),),
        p=float(res.pvalue),
        estimate=(f"mean difference ({levs[0]} − {levs[1]})", float(diff)),
        estimate_ci=(float(ci.low), float(ci.high)),
        effect=(ename, float(e)), effect_ci=_smd_ci(e, len(a), len(b)),
        effect_label=eff.d_label(e), effect_source="Cohen (1988)",
        n={**_n_dict(bound), levs[0]: len(a), levs[1]: len(b)},
        groups=tuple(levs), higher=higher,
        labels={"outcome": _outcome_label(bound)},
        checks=(assumptions.brown_forsythe(a, b),
                assumptions.shapiro_normality(a),
                assumptions.shapiro_normality(b)),
        method=(f"scipy.stats.ttest_ind(a, b, equal_var={student}, "
                "alternative='two-sided')"),
        findings=tuple(findings),
    )


# ==========================================================================
# t_paired -- paired-samples t-test
# ==========================================================================
def t_paired(bound) -> Result:
    findings = _gate(bound)
    if any(f.severity == "block" for f in findings):
        return _blocked(bound, findings)

    before = bound.data["before"].to_numpy(float)
    after = bound.data["after"].to_numpy(float)
    diff = before - after
    res = stats.ttest_rel(before, after)
    if not _finite(res.statistic, res.pvalue):
        return _nan_block(bound, findings, "non-finite t/p")
    ci = res.confidence_interval()
    dz = eff.cohens_dz(diff)
    b_hdr = bound.columns.get("before", ("before",))[0]
    a_hdr = bound.columns.get("after", ("after",))[0]

    return Result(
        test_id="t_paired", test_name="Paired-samples t-test", status="ok",
        statistic=("t", float(res.statistic)), df=(float(res.df),),
        p=float(res.pvalue),
        estimate=(f"mean difference ({b_hdr} − {a_hdr})", float(diff.mean())),
        estimate_ci=(float(ci.low), float(ci.high)),
        effect=("Cohen's dz", float(dz)), effect_label=eff.d_label(dz),
        effect_source="Cohen (1988)",
        n={**_n_dict(bound), "pairs": bound.n_used},
        labels={"before": b_hdr, "after": a_hdr},
        higher=b_hdr if before.mean() > after.mean() else a_hdr,
        checks=(assumptions.shapiro_normality(diff),),
        method="scipy.stats.ttest_rel(before, after)",
        findings=tuple(findings),
    )


# ==========================================================================
# anova_1w -- one-way ANOVA (classic F + Tukey; Welch F + Games-Howell on opt-in)
# ==========================================================================
def anova_1w(bound) -> Result:
    findings = _gate(bound)
    if any(f.severity == "block" for f in findings):
        return _blocked(bound, findings)

    groups = _split_groups(bound)
    names = list(groups)
    arrs = [groups[n] for n in names]
    k = len(arrs)
    ntot = sum(len(g) for g in arrs)
    welch = bool(bound.params.get("welch", False))

    if welch:
        res = anova_oneway(arrs, use_var="unequal", welch_correction=True)
        F, p = float(res.statistic), float(res.pvalue)
        df = (float(res.df[0]), float(res.df[1]))
        name = "Welch's ANOVA"
        method = ("statsmodels.stats.oneway.anova_oneway(use_var='unequal', "
                  "welch_correction=True)")
    else:
        F, p = stats.f_oneway(*arrs)
        F, p = float(F), float(p)
        df = (float(k - 1), float(ntot - k))
        name = "One-way ANOVA"
        method = "scipy.stats.f_oneway(*groups)"

    if not _finite(F, p):
        return _nan_block(bound, findings, "non-finite F/p")

    eta = eff.eta_squared(*arrs)
    omega = eff.omega_squared(*arrs)

    posthoc_tbl, posthoc_name = None, None
    if p < _ALPHA:
        if welch:
            posthoc_tbl, posthoc_name = posthoc.games_howell(groups), "Games-Howell"
        else:
            posthoc_tbl, posthoc_name = posthoc.tukey_hsd(groups), "Tukey HSD"

    r = Result(
        test_id="anova_1w", test_name=name, status="ok",
        statistic=("F", F), df=df, p=p,
        effect=("η²", float(eta)), effect_label=eff.eta_label(eta),
        effect_source="Cohen (1988)",
        n={**_n_dict(bound), **{nm: len(groups[nm]) for nm in names}},
        groups=tuple(names),
        labels={"outcome": _outcome_label(bound)},
        checks=(assumptions.brown_forsythe(*arrs),),
        posthoc=posthoc_tbl, posthoc_name=posthoc_name,
        method=method, findings=tuple(findings),
    )
    r.extra["omega_sq"] = float(omega)
    return r


# ==========================================================================
# anova_2w -- two-way ANOVA (Type II, Sum contrasts; Type III when interaction
# is significant -- D5). Balanced -> typ 1/2/3 agree.
# ==========================================================================
_TERMS = {
    "C(factor_a, Sum)": "factor_a",
    "C(factor_b, Sum)": "factor_b",
    "C(factor_a, Sum):C(factor_b, Sum)": "interaction",
}


def anova_2w(bound) -> Result:
    findings = _gate(bound)
    if any(f.severity == "block" for f in findings):
        return _blocked(bound, findings)

    df = pd.DataFrame({
        "outcome": bound.data["outcome"].to_numpy(float),
        "factor_a": bound.data["factor_a"].astype(str).to_numpy(),
        "factor_b": bound.data["factor_b"].astype(str).to_numpy(),
    })
    model = smf.ols("outcome ~ C(factor_a, Sum)*C(factor_b, Sum)", data=df).fit()
    raw = anova_lm(model, typ=2)
    ss_resid = float(raw.loc["Residual", "sum_sq"])
    df_resid = float(raw.loc["Residual", "df"])

    hdr = {"factor_a": bound.columns["factor_a"][0],
           "factor_b": bound.columns["factor_b"][0]}
    hdr["interaction"] = f"{hdr['factor_a']} × {hdr['factor_b']}"

    rows, index = [], []
    for src, key in _TERMS.items():
        ss = float(raw.loc[src, "sum_sq"])
        d = float(raw.loc[src, "df"])
        F = float(raw.loc[src, "F"])
        pv = float(raw.loc[src, "PR(>F)"])
        rows.append({"label": hdr[key], "sum_sq": ss, "df": d, "F": F, "p": pv,
                     "partial_eta_sq": ss / (ss + ss_resid)})
        index.append(key)
    rows.append({"label": "Residual", "sum_sq": ss_resid, "df": df_resid,
                 "F": float("nan"), "p": float("nan"),
                 "partial_eta_sq": float("nan")})
    index.append("Residual")
    table = pd.DataFrame(rows, index=index)

    inter = table.loc["interaction"]
    F_int, p_int, pe_int = float(inter["F"]), float(inter["p"]), float(inter["partial_eta_sq"])
    if not _finite(F_int, p_int):
        return _nan_block(bound, findings, "non-finite F/p")

    variant = ()
    if p_int < _ALPHA:                                 # D5: show Type III too
        t3 = anova_lm(model, typ=3)
        table_t3 = t3.rename(index={s: k for s, k in _TERMS.items()})
        variant = ("The interaction is significant; Type III sums of squares "
                   "(Sum contrasts) are also reported.",)

    r = Result(
        test_id="anova_2w", test_name="Two-way ANOVA (Type II)", status="ok",
        statistic=("F", F_int), df=(float(inter["df"]), df_resid), p=p_int,
        effect=("partial η²", pe_int), effect_label=eff.eta_label(pe_int),
        effect_source="Cohen (1988)",
        n=_n_dict(bound), table=table,
        labels={"outcome": _outcome_label(bound), **hdr},
        variant_notes=variant,
        method=("statsmodels ols('outcome ~ C(factor_a, Sum)*C(factor_b, Sum)') "
                "+ anova_lm(typ=2)"),
        findings=tuple(findings),
    )
    if variant:
        r.extra["anova_type3"] = table_t3
    return r


# ==========================================================================
# rm_anova -- one within-factor repeated-measures ANOVA (no sphericity
# correction -- stated). Canonical WIDE frame: measures__0..k-1, subject = row.
# ==========================================================================
def rm_anova(bound) -> Result:
    findings = _gate(bound)
    if any(f.severity == "block" for f in findings):
        return _blocked(bound, findings)

    mcols = [c for c in bound.data.columns if c.startswith("measures__")]
    labels = bound.columns.get("measures")
    if not labels or len(labels) != len(mcols):
        labels = [f"condition {i + 1}" for i in range(len(mcols))]

    # reshape wide -> long for statsmodels AnovaRM
    long_rows = []
    for subj in range(len(bound.data)):
        for k, c in enumerate(mcols):
            long_rows.append((subj, k, float(bound.data[c].iloc[subj])))
    ldf = pd.DataFrame(long_rows, columns=["subject", "condition", "outcome"])
    at = AnovaRM(ldf, "outcome", "subject", within=["condition"]).fit().anova_table
    F = float(at["F Value"].iloc[0])
    df1 = float(at["Num DF"].iloc[0])
    df2 = float(at["Den DF"].iloc[0])
    p = float(at["Pr > F"].iloc[0])
    if not _finite(F, p):
        return _nan_block(bound, findings, "non-finite F/p")
    pe = eff.partial_eta_squared(F, df1, df2)

    posthoc_tbl, posthoc_name = None, None
    if p < _ALPHA:                                     # pairwise paired t + Holm
        rows = []
        for i in range(len(mcols)):
            for j in range(i + 1, len(mcols)):
                res = stats.ttest_rel(bound.data[mcols[i]].to_numpy(float),
                                      bound.data[mcols[j]].to_numpy(float))
                rows.append({"group1": levels.display(labels[i]),
                             "group2": levels.display(labels[j]),
                             "statistic": float(res.statistic),
                             "p": float(res.pvalue)})
        ph = pd.DataFrame(rows)
        ph["p_holm"] = multipletests(ph["p"].to_numpy(), method="holm")[1]
        posthoc_tbl, posthoc_name = ph, "pairwise paired t (Holm)"

    return Result(
        test_id="rm_anova", test_name="Repeated-measures ANOVA", status="ok",
        statistic=("F", F), df=(df1, df2), p=p,
        effect=("partial η²", float(pe)), effect_label=eff.eta_label(pe),
        effect_source="Cohen (1988)",
        n={**_n_dict(bound), "subjects": bound.n_used},
        groups=tuple(levels.display(x) for x in labels),
        posthoc=posthoc_tbl, posthoc_name=posthoc_name,
        variant_notes=("Sphericity was not tested (no Greenhouse–Geisser / "
                       "Huynh–Feldt correction in the API).",),
        method=("statsmodels AnovaRM(data, 'outcome', 'subject', "
                "within=['condition']).fit()"),
        findings=tuple(findings),
    )


# --------------------------------------------------------------------------
# {test_id: runner} -- the session swaps each into registry.py's spec.run.
# --------------------------------------------------------------------------
RUNNERS = {
    "describe": describe,
    "t_1s": t_1s,
    "t_ind": t_ind,
    "t_paired": t_paired,
    "anova_1w": anova_1w,
    "anova_2w": anova_2w,
    "rm_anova": rm_anova,
}
