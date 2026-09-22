"""Plain-English result sentences — a pure function of ``Result`` (PLAN §6, D15).

NO-AI is an ASSET here: every branch is a deterministic, unit-testable golden
string; nothing fluent can drift. This layer only *reads* the ``Result`` a runner
already filled (statistic, df, p, estimate, effect, groups, higher, posthoc,
findings) — it never computes a statistic.

Safe-wording rules baked in (PLAN §6, each has a golden):
  1  "statistically significant" only — never "proved".
  2  a non-significant result adds "Absence of evidence is not evidence of
     absence." and, when n < 15, the S21 low-power caveat.
  3  no causal verbs (never "caused / led to / improved").
  4  direction comes from ``r.higher`` / ``r.groups``, never argument order.
  5  rank tests say "values tended to be higher", never "medians differ".
  6  ANOVA / Kruskal say "at least one group differs"; pairs come from post-hoc.
  7  an odds ratio is "odds", never "probability / times more likely".
  8  a reference level is always named.
  9  APA p-values (via ``fmt.p``).
  10 every result sentence carries n used and n dropped.
  11 the exact variant name (``r.test_name``).
  12 effect magnitude with its source.
  13 a negligible-effect note at huge n.
  17 "two-sided, α = .05" once per paragraph.
  18 a blocked result: the sentence IS the block reason.

Formatting split (PLAN §6): a *statistic / effect / estimate* value always shows
two decimals (APA), so ``_f2`` is used, NOT ``fmt.num`` (which collapses an
int-valued float to an int); *degrees of freedom* use ``fmt.num`` so an integer
df prints as ``t(19)`` and a Welch df as ``t(17.34)``. Proportions render through
``fmt.pct`` — the runners in ``props.py`` carry FRACTIONS (``k/n``), so
``pct(0.65) -> "65.0%"`` is the correct, consistent rendering.

Pure module: imports only ``fmt`` (and reads DataFrames handed in on the Result
without importing pandas). No I/O, no state — keeps the L3 gate green.
"""
from __future__ import annotations

import math

from . import fmt


# --------------------------------------------------------------------------
# tiny formatting helpers
# --------------------------------------------------------------------------
def _f2(x) -> str:
    """A statistic/effect/estimate value: always two decimals (APA). A non-finite
    value (a perfect-fit f² = inf, an empty-cell odds ratio) renders as the em
    dash via ``fmt``, never a bare "inf"/"nan"; finite values are unchanged."""
    x = float(x)
    if not math.isfinite(x):
        return fmt.NON_FINITE
    return f"{x:.2f}"


def _perfect_fit(r) -> bool:
    """Perfect fit: Cohen's f² (the regression effect) is non-finite — r² has
    reached 1.0, so F is astronomically large / infinite and not meaningful. Keyed
    on the f² effect specifically: a non-finite Cramér's V / odds ratio in another
    test does NOT make its own statistic meaningless."""
    return (r.effect is not None and r.effect[0] == "f2"
            and not math.isfinite(float(r.effect[1])))


def _perfect_fit_note(r) -> str:
    return ("Perfect fit — the model fits the data exactly, so F is not "
            "meaningful." if _perfect_fit(r) else "")


def _alpha(r) -> str:
    a = f"{r.alpha:.2f}"
    return a[1:] if a.startswith("0.") else a


def _two_sided(r) -> str:
    return f"two-sided, α = {_alpha(r)}"


def _stat_clause(r) -> str:
    name, val = r.statistic
    # On a perfect fit F is astronomical / infinite and not meaningful — render
    # the value as the em dash (never "inf", never a 30-digit number); the
    # regression sentence adds a note. Every other test keeps its .2f value.
    sval = fmt.NON_FINITE if _perfect_fit(r) else _f2(val)
    if len(r.df) == 2:
        return f"{name}({fmt.num(r.df[0])}, {fmt.num(r.df[1])}) = {sval}"
    if len(r.df) == 1:
        return f"{name}({fmt.num(r.df[0])}) = {sval}"
    return f"{name} = {sval}"


def _paren(r, with_stat: bool = True) -> str:
    parts = []
    if with_stat and r.statistic is not None:
        parts.append(_stat_clause(r))
    parts.append(fmt.p(r.p))
    return f"({', '.join(parts)}; {_two_sided(r)})"


def _n_clause(r) -> str:
    used = r.n["used"]
    dropped = r.n.get("dropped", 0)
    total = r.n.get("total", used)
    base = f"{used} cases were analysed (n = {used})"
    if dropped:
        return base + f"; {dropped} of {total} rows were excluded (missing data)."
    return base + "; no rows were excluded."


def _effect_clause(r, lead: str = "The effect was") -> str:
    if r.effect is None:
        return ""
    name, val = r.effect
    if not math.isfinite(float(val)):
        return f"{lead} not estimable."
    core = f"{name} = {_f2(val)}"
    if r.effect_ci is not None:
        core += f", {fmt.ci(*r.effect_ci)}"
    src = f"; per {r.effect_source}" if r.effect_source else ""
    return f"{lead} {r.effect_label} ({core}{src})."


def _sig(r) -> bool:
    return r.p is not None and r.p < r.alpha


def _absence(r) -> str:
    """Rule 2: a non-significant comparison names the asymmetry of evidence."""
    return "" if _sig(r) else "Absence of evidence is not evidence of absence."


def _s21(r) -> str:
    if not _sig(r) and r.n.get("used", 99) < 15:
        return (f"With n = {r.n['used']} (below 15) the test has low statistical "
                "power, so a non-significant result is not evidence that there "
                "is no effect.")
    return ""


def _join(*clauses: str) -> str:
    return " ".join(c for c in clauses if c)


def _blocked_sentence(r) -> str:
    for f in r.findings:
        if f.severity == "block":
            return f.text
    return "This test could not be run on the chosen columns."


def _posthoc_clause(r) -> str:
    if r.posthoc is None:
        return ""
    df = r.posthoc
    pcol = "p_holm" if "p_holm" in df.columns else "p"
    pairs = [f"{row['group1']} and {row['group2']} ({fmt.p(row[pcol])})"
             for _, row in df.iterrows() if row[pcol] < r.alpha]
    if pairs:
        return (f"Post-hoc comparisons ({r.posthoc_name}) showed significant "
                f"differences between {'; '.join(pairs)}.")
    return (f"Post-hoc comparisons ({r.posthoc_name}) found no pairwise "
            "differences after adjustment.")


def _omega_clause(r) -> str:
    """NICE ω²: only for one-way ANOVA, and only when the runner attached it. A
    negative unbiased estimate is reported as 0 with a note (report.py parity)."""
    if r.test_id != "anova_1w":
        return ""
    omega = r.extra.get("omega_sq")
    if omega is None:
        return ""
    if omega < 0:
        return "ω² = 0.00 (negative, reported as 0)."
    return f"ω² = {_f2(omega)}."


# ==========================================================================
# F-2G — t_1s / t_ind / t_paired
# ==========================================================================
def sentence_2g(r) -> str:
    if r.status == "blocked":
        return _blocked_sentence(r)
    sig = _sig(r)
    if r.test_id == "t_1s":
        subj = (f"between the mean of {r.labels.get('outcome', 'the outcome')} "
                "and the hypothesised value μ₀")
        if r.extra.get("mu0") is not None:
            subj += f" = {fmt.num(r.extra['mu0'])}"
    elif r.test_id == "t_ind":
        subj = (f"in {r.labels.get('outcome', 'the outcome')} between "
                f"{r.groups[0]} and {r.groups[1]}")
    else:  # t_paired
        subj = f"between {r.labels['before']} and {r.labels['after']}"
    verb = ("found a statistically significant difference" if sig
            else "did not find a statistically significant difference")
    line1 = f"{r.test_name} {verb} {subj} {_paren(r)}."

    est = f"{_f2(r.estimate[1])}, {fmt.ci(*r.estimate_ci)}"
    if sig and r.test_id == "t_ind":
        direction = f"Values were higher in {r.higher}; the {r.estimate[0]} was {est}."
    elif sig and r.test_id == "t_paired":
        direction = f"Values were higher for {r.higher}; the {r.estimate[0]} was {est}."
    else:
        direction = f"The {r.estimate[0]} was {est}."

    huge = ""
    if r.n.get("used", 0) >= 500 and r.effect_label == "negligible":
        huge = (f"Although the sample is large (n = {r.n['used']}), the effect "
                "is negligible, so the difference is unlikely to be practically "
                "important.")
    return _join(line1, direction, _effect_clause(r), huge, _absence(r),
                 _n_clause(r), _s21(r))


# ==========================================================================
# F-KG — anova_1w / anova_2w / rm_anova / kruskal / friedman
# ==========================================================================
def sentence_kg(r) -> str:
    if r.status == "blocked":
        return _blocked_sentence(r)
    if r.test_id == "anova_2w":
        return _sentence_2w(r)
    sig = _sig(r)
    if r.test_id in ("rm_anova", "friedman"):
        subj, tail = "across the repeated measurements", " (at least one differs)"
    else:
        subj, tail = "among the groups", " (at least one group differs)"
    if sig:
        verb = f"found a statistically significant difference {subj}{tail}"
    else:
        verb = f"did not find a statistically significant difference {subj}"
    line1 = f"{r.test_name} {verb} {_paren(r)}."
    variant = " ".join(r.variant_notes) if r.variant_notes else ""
    return _join(line1, _effect_clause(r), _omega_clause(r), _posthoc_clause(r),
                 variant, _absence(r), _n_clause(r), _s21(r))


def _sentence_2w(r) -> str:
    sig = _sig(r)
    A, B = r.labels["factor_a"], r.labels["factor_b"]
    was = "was" if sig else "was not"
    ename, eval_ = r.effect
    line1 = (f"{r.test_name}: the interaction between {A} and {B} {was} "
             f"statistically significant {_paren(r)}, {ename} = {_f2(eval_)} "
             f"({r.effect_label}).")
    df_resid = r.df[1]
    mains = []
    for key, lbl in (("factor_a", A), ("factor_b", B)):
        F = r.table.loc[key, "F"]
        dfk = r.table.loc[key, "df"]
        pk = r.table.loc[key, "p"]
        m_was = "was" if pk < r.alpha else "was not"
        mains.append(f"The main effect of {lbl} {m_was} statistically "
                     f"significant (F({fmt.num(dfk)}, {fmt.num(df_resid)}) = "
                     f"{_f2(F)}, {fmt.p(pk)}).")
    interp = ""
    if sig:
        interp = ("Because the interaction is statistically significant, "
                  "interpret it (see the interaction plot) rather than the main "
                  "effects; Type III sums of squares (Sum contrasts) are also "
                  "reported.")
    return _join(line1, mains[0], mains[1], interp, _n_clause(r))


# ==========================================================================
# F-RANK2 — mwu / wilcoxon
# ==========================================================================
def sentence_rank2(r) -> str:
    if r.status == "blocked":
        return _blocked_sentence(r)
    sig = _sig(r)
    verb = ("found a statistically significant difference" if sig
            else "did not find a statistically significant difference")
    if r.test_id == "mwu":
        line1 = f"{r.test_name} {verb} between {r.groups[0]} and {r.groups[1]} {_paren(r)}."
        direction = f"Values tended to be higher in {r.higher}." if sig else ""
        est = f"The {r.estimate[0]} was {_f2(r.estimate[1])}."
        method = f"Method: {r.variant_notes[0]}." if r.variant_notes else ""
        return _join(line1, direction, est, method, _effect_clause(r),
                     _absence(r), _n_clause(r), _s21(r))
    # wilcoxon
    if r.test_name.endswith("(one-sample)"):
        subj = "from the hypothesised median"
        if r.extra.get("mu0") is not None:
            subj += f" μ₀ = {fmt.num(r.extra['mu0'])}"
        direction = ""
    else:
        if r.labels.get("before") and r.labels.get("after"):
            subj = f"between {r.labels['before']} and {r.labels['after']}"
        else:
            subj = "between the paired measurements"
        direction = (f"Values tended to be higher for {r.higher}."
                     if sig and r.higher else "")
    line1 = f"{r.test_name} {verb} {subj} {_paren(r)}."
    est = f"The {r.estimate[0]} was {_f2(r.estimate[1])}."
    note = f"{r.variant_notes[0]}." if r.variant_notes else ""
    return _join(line1, direction, est, note, _effect_clause(r), _absence(r),
                 _n_clause(r), _s21(r))


# ==========================================================================
# F-ASSOC — pearson / spearman / kendall / corr_matrix
# ==========================================================================
def sentence_assoc(r) -> str:
    if r.status == "blocked":
        return _blocked_sentence(r)
    if r.test_id == "corr_matrix":
        return _sentence_corr_matrix(r)
    sig = _sig(r)
    x, y = r.labels.get("x", "x"), r.labels.get("y", "y")
    coef_name, coef = r.effect
    if sig:
        direction = "positive" if coef >= 0 else "negative"
        line1 = (f"{r.test_name} found a statistically significant {direction} "
                 f"correlation between {x} and {y} {_paren(r)}.")
    else:
        line1 = (f"{r.test_name} did not find a statistically significant "
                 f"correlation between {x} and {y} {_paren(r)}.")
    inner = f"{coef_name} = {_f2(coef)}"
    if r.effect_ci is not None:
        inner += f", {fmt.ci(*r.effect_ci)}"
    if r.test_id == "pearson":
        inner += f"; r² = {_f2(r.extra['r2'])}"
    strength = f"The correlation was {r.effect_label} ({inner})."
    ordinal = ""
    if any(f.suggest_test == "spearman" for f in r.findings):
        ordinal = ("An ordinal variable is involved; Spearman's rank "
                   "correlation may be more appropriate.")
    return _join(line1, strength, ordinal, _absence(r), _n_clause(r), _s21(r))


def _sentence_corr_matrix(r) -> str:
    ph = r.extra["p_holm"]
    k = len(ph.columns)
    sig = total = 0
    for i in range(k):
        for j in range(i + 1, k):
            total += 1
            if ph.iloc[i, j] < r.alpha:
                sig += 1
    method = r.extra.get("method", "pearson")
    was = "was" if sig == 1 else "were"
    line = (f"{r.test_name}: {method} correlations were computed among {k} "
            f"variables. {sig} of {total} pairwise correlations {was} "
            f"statistically significant after Holm adjustment ({_two_sided(r)}). "
            "See the correlation matrix and heatmap.")
    return _join(line, _n_clause(r))


# ==========================================================================
# F-CAT — chi2_ind / chi2_gof / fisher / mcnemar / cochran_q / prop_1 / prop_2
# ==========================================================================
def _pct_ci(ci) -> str:
    lo, hi = ci
    return f"95% CI [{fmt.pct(lo)}, {fmt.pct(hi)}]"


def _assoc_vars(r) -> str:
    """S4: name the two crosstab variables when the runner attached them
    (``labels['row']`` / ``labels['col']``); else today's generic phrase."""
    row, col = r.labels.get("row"), r.labels.get("col")
    return f"{row} and {col}" if row and col else "the two variables"


def _odds_clause(r) -> str:
    """The 2×2 odds-ratio sentence. With the runner's ``extra['or']`` orientation
    it names both variables and the compared levels (S4); otherwise it falls back
    to today's bare 'The odds ratio was …'. Empty when there is no OR (R×C)."""
    if r.estimate is None:
        return ""
    if not math.isfinite(float(r.estimate[1])):
        return ("The odds ratio could not be estimated because a cell in the "
                "table was empty.")
    orv = _f2(r.estimate[1])
    ci = fmt.ci(*r.estimate_ci)
    o = r.extra.get("or")
    if o and r.labels.get("row") and r.labels.get("col"):
        return (f"The odds of {r.labels['col']} = '{o['col_level']}' were {orv} "
                f"times as high for {r.labels['row']} = '{o['row_level']}' as for "
                f"'{o['row_ref']}' (odds ratio = {orv}, {ci}).")
    return f"The odds ratio was {orv}, {ci}."


def sentence_cat(r) -> str:
    if r.status == "blocked":
        return _blocked_sentence(r)
    tid = r.test_id
    sig = _sig(r)

    if tid == "chi2_ind":
        verb = ("found a statistically significant association" if sig
                else "did not find a statistically significant association")
        line1 = f"{r.test_name} {verb} between {_assoc_vars(r)} {_paren(r)}."
        return _join(line1, _odds_clause(r), _effect_clause(r), _absence(r),
                     _n_clause(r), _s21(r))

    if tid == "chi2_gof":
        verb = ("found a statistically significant difference" if sig
                else "did not find a statistically significant difference")
        line1 = (f"{r.test_name} {verb} between the observed counts and the "
                 f"expected proportions {_paren(r)}.")
        return _join(line1, _effect_clause(r), _absence(r), _n_clause(r), _s21(r))

    if tid == "fisher":
        verb = ("found a statistically significant association" if sig
                else "did not find a statistically significant association")
        paren = f"({fmt.p(r.p)}; {_two_sided(r)})"
        line1 = f"{r.test_name} {verb} between {_assoc_vars(r)} {paren}."
        if r.estimate is not None:  # 2×2
            tail = [_odds_clause(r), _effect_clause(r)]
        else:  # R×C
            tail = [" ".join(r.variant_notes) + "."] if r.variant_notes else []
        return _join(line1, *tail, _absence(r), _n_clause(r), _s21(r))

    if tid == "mcnemar":
        verb = ("found a statistically significant change" if sig
                else "did not find a statistically significant change")
        show_stat = r.statistic is not None and r.statistic[0] == "χ²"
        paren = _paren(r) if show_stat else f"({fmt.p(r.p)}; {_two_sided(r)})"
        line1 = f"{r.test_name} {verb} between the two measurements {paren}."
        change = f"{r.variant_notes[0]}." if r.variant_notes else ""
        if r.estimate is not None:
            orc = f"The odds ratio was {_f2(r.estimate[1])}, {fmt.ci(*r.estimate_ci)}."
        else:
            orc = ("The odds ratio was not estimable (one direction of change "
                   "had no cases).")
        return _join(line1, change, orc, _absence(r), _n_clause(r), _s21(r))

    if tid == "cochran_q":
        success = r.labels.get("success", "success")
        verb = ("found a statistically significant difference" if sig
                else "did not find a statistically significant difference")
        line1 = (f"{r.test_name} {verb} in the proportion of '{success}' across "
                 f"the repeated conditions {_paren(r)}.")
        return _join(line1, _posthoc_clause(r), _absence(r), _n_clause(r),
                     _s21(r))

    if tid == "prop_1":
        success = r.labels.get("success", "success")
        verb = ("found a statistically significant difference" if sig
                else "did not find a statistically significant difference")
        hyp = "the hypothesised value"
        if r.extra.get("p0") is not None:
            hyp += f" p₀ = {fmt.num(r.extra['p0'])}"
        line1 = (f"{r.test_name} {verb} between the proportion of '{success}' "
                 f"and {hyp} {_paren(r)}.")
        prop = (f"The observed proportion of '{success}' was "
                f"{fmt.pct(r.estimate[1])}, {_pct_ci(r.estimate_ci)}.")
        return _join(line1, prop, _effect_clause(r), _absence(r), _n_clause(r),
                     _s21(r))

    # prop_2
    success = r.labels.get("success", "success")
    g1, g2 = r.groups
    verb = ("found a statistically significant difference" if sig
            else "did not find a statistically significant difference")
    line1 = (f"{r.test_name} {verb} in the proportion of '{success}' between "
             f"{g1} and {g2} {_paren(r)}.")
    if sig:
        diff = (f"The proportion of '{success}' was higher in {r.higher}; the "
                f"difference was {fmt.pct(r.estimate[1])}, {_pct_ci(r.estimate_ci)}.")
    else:
        diff = (f"The difference in the proportion of '{success}' was "
                f"{fmt.pct(r.estimate[1])}, {_pct_ci(r.estimate_ci)}.")
    return _join(line1, diff, _effect_clause(r), _absence(r), _n_clause(r),
                 _s21(r))


# ==========================================================================
# F-REG — ols_simple / ols_multi / logistic
# ==========================================================================
def _ref_clause(r) -> str:
    ref = r.extra.get("reference") or {}
    if not ref:
        return ""
    return ("Reference level(s): "
            + ", ".join(f"{h} = '{lv}'" for h, lv in ref.items()) + ".")


def sentence_reg(r) -> str:
    if r.status == "blocked":
        return _blocked_sentence(r)
    tid = r.test_id
    sig = _sig(r)

    if tid == "ols_simple":
        x = r.labels.get("x", "the predictor")
        outcome = r.labels.get("outcome", "the outcome")
        lead = ("found that" if sig else "did not find that")
        line1 = (f"{r.test_name} {lead} {x} statistically significantly "
                 f"predicted {outcome} {_paren(r)}.")
        r2 = r.extra["r2"]
        r2clause = (f"The model explained {fmt.pct(r2)} of the variance in "
                    f"{outcome} (R² = {_f2(r2)}).")
        slope = (f"Each one-unit increase in {x} was associated with a change "
                 f"of {_f2(r.estimate[1])} in {outcome} "
                 f"({fmt.ci(*r.estimate_ci)}).")
        return _join(line1, r2clause, slope, _perfect_fit_note(r),
                     _effect_clause(r), _absence(r), _n_clause(r), _s21(r))

    if tid == "ols_multi":
        lead = ("found a" if sig else "did not find a")
        line1 = f"{r.test_name} {lead} statistically significant model {_paren(r)}."
        r2, adj = r.extra["r2"], r.extra["adj_r2"]
        r2clause = (f"The model explained {fmt.pct(r2)} of the variance "
                    f"(R² = {_f2(r2)}, adjusted R² = {_f2(adj)}).")
        coefs = [f"{name}: b = {_f2(row['coef'])}, "
                 f"{fmt.ci(row['ci_low'], row['ci_high'])}, {fmt.p(row['p'])}"
                 for name, row in r.table.iterrows()
                 if name not in ("Intercept", "const")]
        coef_clause = f"Coefficients — {'; '.join(coefs)}."
        return _join(line1, r2clause, coef_clause, _ref_clause(r),
                     _perfect_fit_note(r), _effect_clause(r), _absence(r),
                     _n_clause(r), _s21(r))

    # logistic (ok path; separation is a blocked Result handled above)
    outcome = r.labels.get("outcome", "the outcome")
    success = r.extra.get("success", "the event")
    lead = ("found a" if sig else "did not find a")
    line1 = (f"{r.test_name} {lead} statistically significant model predicting "
             f"'{success}' in {outcome} {_paren(r)}.")
    ors = [f"{name}: each one-unit increase multiplied the odds of '{success}' "
           f"by {_f2(row['OR'])} ({fmt.ci(row['OR_ci_low'], row['OR_ci_high'])}, "
           f"{fmt.p(row['p'])})"
           for name, row in r.table.iterrows()
           if name not in ("Intercept", "const")]
    or_clause = f"Odds ratios — {'; '.join(ors)}."
    pseudo = f"McFadden pseudo-R² = {_f2(r.extra['pseudo_r2'])}."
    return _join(line1, or_clause, _ref_clause(r), pseudo, _absence(r),
                 _n_clause(r), _s21(r))


# ==========================================================================
# F-CHK — normality / homogeneity
# ==========================================================================
def sentence_chk(r) -> str:
    if r.status == "blocked":
        return _blocked_sentence(r)
    if r.test_id == "normality":
        lines = []
        for chk in r.checks:
            if chk.statistic is None:
                lines.append(f"{chk.name}: not run ({chk.note}).")
            else:
                verdict = ("consistent with a normal distribution" if chk.passed
                           else "departs from a normal distribution")
                lines.append(f"{chk.name}: statistic = {_f2(chk.statistic)}, "
                             f"{fmt.p(chk.p)} — {verdict}.")
        return _join(f"{r.test_name} assessed normality.", " ".join(lines),
                     _n_clause(r))
    # homogeneity
    chk = r.checks[0]
    verdict = ("no evidence that the variances differ" if chk.passed
               else "the variances differ")
    line = (f"{r.test_name}: across {len(r.groups)} groups, {chk.name} gave "
            f"statistic = {_f2(chk.statistic)}, {fmt.p(chk.p)} — {verdict}.")
    return _join(line, _n_clause(r))


# ==========================================================================
# describe (utility)
# ==========================================================================
def sentence_describe(r) -> str:
    if r.status == "blocked":
        return _blocked_sentence(r)
    return _join(
        f"{r.test_name} were computed for the selected columns. See the table "
        "for n, mean, SD, median, IQR and range (numeric) and counts or "
        "percentages (categorical).", _n_clause(r))


# ==========================================================================
# dispatch + wiring map
# ==========================================================================
_DISPATCH = {
    "describe": sentence_describe,
    "t_1s": sentence_2g, "t_ind": sentence_2g, "t_paired": sentence_2g,
    "anova_1w": sentence_kg, "anova_2w": sentence_kg, "rm_anova": sentence_kg,
    "kruskal": sentence_kg, "friedman": sentence_kg,
    "mwu": sentence_rank2, "wilcoxon": sentence_rank2,
    "pearson": sentence_assoc, "spearman": sentence_assoc,
    "kendall": sentence_assoc, "corr_matrix": sentence_assoc,
    "chi2_ind": sentence_cat, "chi2_gof": sentence_cat, "fisher": sentence_cat,
    "mcnemar": sentence_cat, "cochran_q": sentence_cat, "prop_1": sentence_cat,
    "prop_2": sentence_cat,
    "ols_simple": sentence_reg, "ols_multi": sentence_reg, "logistic": sentence_reg,
    "normality": sentence_chk, "homogeneity": sentence_chk,
}


def render(r) -> str:
    """The single entry point wired into every ``TestSpec.sentence``."""
    fn = _DISPATCH.get(r.test_id)
    if fn is None:                       # pragma: no cover - registry is closed
        return (_blocked_sentence(r) if r.status == "blocked"
                else f"{r.test_name}: result computed.")
    return fn(r)


# {test_id: sentence fn} — registry._wire_sentences() swaps each spec.sentence.
SENTENCES = {tid: render for tid in _DISPATCH}
