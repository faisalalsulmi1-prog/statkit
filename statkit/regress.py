"""Regression runners — the ``predict`` family (PLAN §3 / §5, Chunk 10).

``ols_simple`` / ``ols_multi`` / ``logistic``, each a pure ``run(Bound) ->
Result``. Correct-by-default (PLAN D5):

  * **statsmodels OLS / Logit, never scipy.linregress.**
  * ols_multi builds a **Treatment-coded formula** so a categorical predictor is
    ONE multi-df term: this gives a *named reference level* in the coefficient
    table AND a Type II ``anova_lm(typ=2)`` ANOVA-of-regression (Type I would
    depend on term order). SNAG vs the PLAN's ``get_dummies`` wording — see report.
  * **VIF** is reported per design column (statsmodels
    ``variance_inflation_factor``); VIF > 5 raises a naming flag (PLAN §5 D6).
  * logistic **blocks on perfect separation** (any separation/convergence
    warning, a non-converged fit, or |b| > 15) and shows NO odds ratios
    (PLAN §5.2 / logistic row) — a stated block, never a silent bad fit.

Structural checks run first: a blocked bind returns ``status='blocked'`` with the
reason (S11 already blocks n < predictors + 10, etc.).

Only numpy + scipy + statsmodels + pandas + patsy (L3 allowlist); no I/O, no state.
"""
from __future__ import annotations

import math
import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from . import effects, levels
from .check import check
from .model import Bound, Finding, Result


# --------------------------------------------------------------------------
# shared structural gate + result helpers (kept local; runner modules stay
# independent — Chunk 10 may create only assoc.py / regress.py)
# --------------------------------------------------------------------------
def _gather(bound: Bound) -> tuple[tuple[Finding, ...], tuple[Finding, ...]]:
    findings = tuple(bound.findings) + tuple(
        f for f in check(bound) if f not in bound.findings)
    blocks = tuple(f for f in findings if f.severity == "block")
    return findings, blocks


def _n(bound: Bound) -> dict:
    return {"total": bound.n_total, "used": bound.n_used,
            "dropped": bound.n_total - bound.n_used}


def _blocked(bound: Bound, findings: tuple[Finding, ...], extra=None) -> Result:
    reason = next((f.text for f in findings if f.severity == "block"),
                  "This test cannot run on the chosen columns.")
    return Result(test_id=bound.test.id, test_name=bound.test.name,
                  status="blocked", n=_n(bound), findings=findings,
                  variant_notes=(reason,), extra=extra or {})


def _cohens_f2(r2: float) -> float:
    return effects.cohens_f2(r2) if r2 < 1.0 else math.inf


# --------------------------------------------------------------------------
# predictor -> formula terms + reference levels + canonical->header relabel
# --------------------------------------------------------------------------
def _pick_reference(ref_param, header, series):
    """The baseline level for a categorical predictor: the user's choice
    (resolved from its display string against the column's levels), else the
    first level in sorted order."""
    chosen = None
    if isinstance(ref_param, dict) and header in ref_param:
        chosen = ref_param[header]
    elif isinstance(ref_param, str):
        chosen = ref_param
    if chosen is not None:
        resolved = levels.resolve(chosen, series)
        if resolved is not None:
            return resolved
    return sorted(series.dropna().astype(str).unique())[0]  # default: first sorted


def _terms_and_refs(bound, data, pred_cols, headers):
    ref_param = bound.params.get("reference")
    terms, refs = [], {}
    for i, c in enumerate(pred_cols):
        h = headers[i] if i < len(headers) else c
        if pd.api.types.is_numeric_dtype(data[c]):
            terms.append(c)
        else:
            data[c] = data[c].astype(str)
            ref = _pick_reference(ref_param, h, data[c])
            refs[h] = ref
            terms.append(f"C({c}, Treatment(reference={ref!r}))")
    return terms, refs


def _relabeller(pred_cols, headers):
    # replace longest canonical names first (predictors__10 before predictors__1)
    pairs = sorted(zip(pred_cols, headers), key=lambda p: -len(p[0]))

    def pretty(name: str) -> str:
        for c, h in pairs:
            name = name.replace(c, h)
        name = re.sub(r"C\(([^,]+),\s*Treatment\(reference=[^)]*\)\)", r"\1", name)
        # name the level: patsy's "City[T.NY]" -> readable "City = 'NY'" (S-G)
        return re.sub(r"([^\[\]]+)\[T\.([^\]]+)\]", r"\1 = '\2'", name)

    return pretty


def _coef_table(m, pretty) -> pd.DataFrame:
    ci = m.conf_int()
    rows = {}
    for name in m.model.exog_names:
        rows[pretty(name)] = {
            "coef": float(m.params[name]),
            "ci_low": float(ci.loc[name, 0]),
            "ci_high": float(ci.loc[name, 1]),
            "p": float(m.pvalues[name]),
        }
    return pd.DataFrame(rows).T[["coef", "ci_low", "ci_high", "p"]]


def _vif(m, pretty) -> tuple[dict, tuple[Finding, ...]]:
    exog, names = m.model.exog, m.model.exog_names
    vif, flags = {}, []
    for j, name in enumerate(names):
        if name == "Intercept" or name == "const":
            continue
        v = float(variance_inflation_factor(exog, j))
        label = pretty(name)
        vif[label] = v
        if v > 5:
            flags.append(Finding("flag", f"'{label}' has a high variance inflation "
                                 f"factor (VIF = {v:.1f} > 5): it is collinear with "
                                 "the other predictors.", code="D6"))
    return vif, tuple(flags)


# --------------------------------------------------------------------------
# OLS — simple
# --------------------------------------------------------------------------
def ols_simple(bound: Bound) -> Result:
    findings, blocks = _gather(bound)
    if blocks:
        return _blocked(bound, findings)
    y = bound.data["outcome"].to_numpy(dtype=float)
    x = bound.data["x"].to_numpy(dtype=float)
    m = sm.OLS(y, sm.add_constant(x)).fit()
    intercept, slope = float(m.params[0]), float(m.params[1])
    ci = m.conf_int()
    r2 = float(m.rsquared)
    f2 = _cohens_f2(r2)
    xh = bound.columns["x"][0]
    table = pd.DataFrame(
        {"coef": [intercept, slope],
         "ci_low": [float(ci[0, 0]), float(ci[1, 0])],
         "ci_high": [float(ci[0, 1]), float(ci[1, 1])],
         "p": [float(m.pvalues[0]), float(m.pvalues[1])]},
        index=["Intercept", xh])
    return Result(
        test_id=bound.test.id, test_name=bound.test.name, status="ok",
        statistic=("F", float(m.fvalue)), df=(float(m.df_model), float(m.df_resid)),
        p=float(m.f_pvalue),
        estimate=("slope", slope), estimate_ci=(float(ci[1, 0]), float(ci[1, 1])),
        effect=("f2", f2), effect_label=effects.f2_label(f2),
        effect_source="Cohen (1988)", n=_n(bound),
        labels={"outcome": bound.columns["outcome"][0], "x": xh},
        table=table, method="statsmodels.OLS(y, add_constant(x)).fit()",
        findings=findings,
        extra={"intercept": intercept, "slope": slope, "r2": r2,
               "adj_r2": float(m.rsquared_adj)},
        arrays={"x": x, "y": y},
    )


# --------------------------------------------------------------------------
# OLS — multiple
# --------------------------------------------------------------------------
def ols_multi(bound: Bound) -> Result:
    findings, blocks = _gather(bound)
    if blocks:
        return _blocked(bound, findings)
    data = bound.data.copy()
    pred_cols = [c for c in data.columns if c.startswith("predictors__")]
    headers = list(bound.columns.get("predictors", ()))
    pretty = _relabeller(pred_cols, headers)
    terms, refs = _terms_and_refs(bound, data, pred_cols, headers)
    m = smf.ols("outcome ~ " + " + ".join(terms), data).fit()

    table = _coef_table(m, pretty)
    vif, vif_flags = _vif(m, pretty)
    anova2 = anova_lm(m, typ=2)
    anova2.index = [pretty(ix) for ix in anova2.index]
    r2 = float(m.rsquared)
    f2 = _cohens_f2(r2)
    return Result(
        test_id=bound.test.id, test_name=bound.test.name, status="ok",
        statistic=("F", float(m.fvalue)), df=(float(m.df_model), float(m.df_resid)),
        p=float(m.f_pvalue),
        effect=("f2", f2), effect_label=effects.f2_label(f2),
        effect_source="Cohen (1988)", n=_n(bound),
        labels={"outcome": bound.columns["outcome"][0]},
        table=table,
        method="statsmodels OLS (Treatment-coded formula) + anova_lm(typ=2)",
        findings=findings + vif_flags,
        extra={"r2": r2, "adj_r2": float(m.rsquared_adj), "vif": vif,
               "reference": refs, "anova_typ2": anova2},
        arrays={"fitted": np.asarray(m.fittedvalues, float),
                "resid": np.asarray(m.resid, float)},
    )


# --------------------------------------------------------------------------
# Logistic — binary
# --------------------------------------------------------------------------
def _is_separation_warning(w) -> bool:
    names = {x.category.__name__ for x in w}
    return any("Separation" in n or "Convergence" in n for n in names)


def logistic(bound: Bound) -> Result:
    findings, blocks = _gather(bound)
    if blocks:
        return _blocked(bound, findings)
    data = bound.data.copy()
    outcome = data["outcome"]
    success, level_block = levels.pick_success(outcome, bound.params.get("success"))
    if level_block is not None:
        return _blocked(bound, findings + (level_block,))
    disp = levels.display(success)
    data["_y"] = (outcome == success).astype(float).to_numpy()
    pred_cols = [c for c in data.columns if c.startswith("predictors__")]
    headers = list(bound.columns.get("predictors", ()))
    pretty = _relabeller(pred_cols, headers)
    terms, refs = _terms_and_refs(bound, data, pred_cols, headers)

    # Separate the two failure causes (this is what made B3 misleading): genuine
    # separation (a predictor separates the outcome) vs any OTHER fit failure —
    # a singular/collinear design raises LinAlgError, which is NOT separation.
    separated = False
    fit_error = None
    m = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            m = smf.logit("_y ~ " + " + ".join(terms), data).fit(disp=0)
        except Exception as e:
            if "PerfectSeparation" in type(e).__name__:
                separated = True
            else:
                fit_error = e
    if fit_error is not None:
        f = Finding("block", "The model could not be fitted "
                    f"({type(fit_error).__name__}: {fit_error}). Check for a "
                    "constant, duplicated or perfectly collinear predictor.",
                    code="FIT")
        return _blocked(bound, findings + (f,), extra={"success": disp})
    if m is not None:
        converged = bool(m.mle_retvals.get("converged", True))
        maxb = float(np.max(np.abs(m.params)))
        if not converged or maxb > 15 or _is_separation_warning(caught):
            separated = True
    if separated:
        f = Finding("block", "The model shows perfect (or near-perfect) "
                    "separation: a predictor separates the outcome, so the odds "
                    "ratios are not estimable. Remove or combine that predictor, "
                    "or collect more varied data.", code="separation")
        return _blocked(bound, findings + (f,), extra={"success": disp})

    ci = m.conf_int()
    odds, table_rows = {}, {}
    for name in m.model.exog_names:
        label = pretty(name)
        coef = float(m.params[name])
        lo, hi = float(ci.loc[name, 0]), float(ci.loc[name, 1])
        table_rows[label] = {"coef": coef, "OR": math.exp(coef),
                             "OR_ci_low": math.exp(lo), "OR_ci_high": math.exp(hi),
                             "p": float(m.pvalues[name])}
        if name not in ("Intercept", "const"):
            odds[label] = math.exp(coef)
    table = pd.DataFrame(table_rows).T[["coef", "OR", "OR_ci_low", "OR_ci_high", "p"]]
    return Result(
        test_id=bound.test.id, test_name=bound.test.name, status="ok",
        statistic=("LLR chi2", float(m.llr)), df=(float(m.df_model),),
        p=float(m.llr_pvalue), n=_n(bound),
        labels={"outcome": bound.columns["outcome"][0]},
        table=table, method="statsmodels Logit(...).fit(disp=0)",
        findings=findings,
        extra={"or": odds, "pseudo_r2": float(m.prsquared),
               "llr_p": float(m.llr_pvalue), "success": disp, "reference": refs},
    )


RUNNERS = {"ols_simple": ols_simple, "ols_multi": ols_multi, "logistic": logistic}
