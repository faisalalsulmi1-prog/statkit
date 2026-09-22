"""Structural sanity checks S1-S21 (PLAN §5.1).

``check(bound)`` gates whether a test can run, reading only what the Contract and
the canonical frame already carry -- level counts, group sizes, N accounting,
layout -- so it can never drift from the menu or the binder (PLAN §2). It returns
one ``Finding`` per rule that FIRES, each carrying its ``code`` ("S4") and a
plain-English reason; a rule that passes yields nothing. Severity is
``block`` (Run disabled), ``flag`` (a warning + often a suggested test) or
``info``. Data-driven assumption checks (D1-D14, Levene/Shapiro/expected counts)
are ``advise()`` -- a later chunk.

A few rules (S9 numeric-as-group, S13 ordinal-as-interval) compare the binding
against a column's *original* inferred kinds, so ``check`` optionally takes a
``profiles`` map (header -> ColumnProfile); without it those two rules are skipped.
"""
from __future__ import annotations

import pandas as pd

from . import levels
from .model import Bound, Finding, Kind

# parametric<->parametric and rank<->rank swaps for the wrong group arity (S2).
_TO_K = {"t_ind": "anova_1w", "mwu": "kruskal"}
_TO_2 = {"anova_1w": "t_ind", "kruskal": "mwu"}

# canonical columns that are grouping/label variables, never "value" columns (S8).
_LABEL_COLS = frozenset(
    {"group", "row", "col", "category", "factor_a", "factor_b", "subject", "condition"})
_CAT_ROLE_COLS = ("group", "row", "col", "category", "factor_a", "factor_b")
_REGRESSION = frozenset({"ols_simple", "ols_multi", "logistic"})


def check(bound: Bound, profiles: dict | None = None) -> tuple[Finding, ...]:
    out: list[Finding] = []
    for rule in _RULES:
        f = rule(bound, profiles)
        if f is not None:
            out.append(f)
    return tuple(out)


def gate(bound: Bound, profiles: dict | None = None) -> tuple[Finding, ...]:
    """Structural check ALWAYS runs; merged with any pre-existing (bind/advise)
    findings, dedup by code, structural findings authoritative on a code clash.

    This closes the fail-open in the runners' old ``bound.findings or check(bound)``
    idiom: a bind-attached advisory (ORDCODE/DUP) or an advise D-code no longer
    SKIPS the structural S-checks. When the bound carries no findings the result
    is ``check(bound, profiles)`` exactly, so every golden on a plain bound stays
    byte-identical."""
    structural = check(bound, profiles)
    if not bound.findings:
        return structural
    seen = {f.code for f in structural if f.code}
    merged = list(structural)
    for f in bound.findings:
        if f.code:
            if f.code in seen:            # a code check already produced: check wins
                continue
            seen.add(f.code)
        merged.append(f)                  # codeless findings all kept
    return tuple(merged)


# --------------------------------------------------------------------------
# small accessors
# --------------------------------------------------------------------------
def _has(bound, col):
    return col in bound.data.columns


def _levels(bound, col):
    return int(bound.data[col].dropna().nunique())


def _sizes(bound, col):
    return bound.data[col].dropna().value_counts()


def _role_levels(bound, name):
    for roles in bound.test.contract.roles_by_layout.values():
        for r in roles:
            if r.name == name:
                return r.levels
    return (1, None)


def _header_for(bound, col):
    """The STUDENT's header for a canonical frame column, for messages.

    A multi-column role lives in the frame as ``f"{role}__{i}"`` while the user's
    headers sit under ``bound.columns[role]``; a single-column role's frame name
    is the role name itself. Fall back to the raw column name if no header was
    recorded (so a message is never blank)."""
    for role in ("variables", "measures", "predictors", "groups"):
        pre = role + "__"
        if col.startswith(pre):
            i = int(col[len(pre):])
            hdrs = bound.columns.get(role, ())
            return hdrs[i] if i < len(hdrs) else col
    return bound.columns.get(col, (col,))[0]


# --------------------------------------------------------------------------
# the rules (one per S-code)
# --------------------------------------------------------------------------
def _s1(bound, _p):
    seen: dict = {}
    for role, headers in bound.columns.items():
        for h in headers:
            if h in seen and seen[h] != role:
                return Finding("block", f"The column '{h}' is used in two roles "
                               f"({seen[h]} and {role}); pick a different column "
                               "for one of them.", code="S1")
            seen[h] = role
    return None


def _s2(bound, _p):
    if not _has(bound, "group"):
        return None
    lo, hi = _role_levels(bound, "group")
    L = _levels(bound, "group")
    tid = bound.test.id
    if lo >= 3 and L == 2 and tid in _TO_2:
        return Finding("block", f"This test needs 3 or more groups but the chosen "
                       f"column has 2; use a two-group test instead.",
                       suggest_test=_TO_2[tid], code="S2")
    if hi == 2 and 2 < L <= 20 and tid in _TO_K:
        return Finding("block", f"This test compares 2 groups but the chosen column "
                       f"has {L}; use a test for 3+ groups instead.",
                       suggest_test=_TO_K[tid], code="S2")
    return None


def _s3(bound, _p):
    for col in _CAT_ROLE_COLS:
        if _has(bound, col) and _levels(bound, col) > 20:
            hdr = bound.columns.get(col, (col,))[0]
            return Finding("block", f"'{hdr}' has more than 20 categories "
                           f"({_levels(bound, col)}) — is this the right column?",
                           code="S3")
    return None


def _s4(bound, _p):
    if bound.test.goal != "compare" or not _has(bound, "group"):
        return None
    sizes = _sizes(bound, "group")
    small = sizes[sizes < 2]
    if len(small):
        names = ", ".join(levels.display(k) for k in small.index)
        return Finding("block", f"Group(s) with fewer than 2 values: {names}. "
                       "A comparison needs at least 2 per group.", code="S4")
    return None


def _s5(bound, _p):
    if bound.test.pairing == "paired":
        return None
    if bound.n_used < bound.test.contract.min_n:
        return Finding("block", f"Only {bound.n_used} usable rows; this test needs "
                       f"at least {bound.test.contract.min_n}.", code="S5")
    return None


def _s6(bound, _p):
    if bound.test.pairing == "paired" and bound.n_used < 3:
        unit = "values" if bound.layout == "one_sample" else "complete pairs"
        return Finding("block", f"Only {bound.n_used} {unit}; a paired test "
                       "needs at least 3.", code="S6")
    return None


def _s7(bound, _p):
    if not bound.test.contract.same_level_set:
        return None
    cols = list(bound.data.columns)
    sets = {c: frozenset(bound.data[c].dropna().unique()) for c in cols}
    bad_size = [c for c, s in sets.items() if len(s) != 2]
    distinct = set(sets.values())
    if bad_size or len(distinct) > 1:
        shown = "; ".join(f"{c}={sorted(s)}" for c, s in sets.items())
        return Finding("block", "The paired columns must share the same two "
                       f"outcome values. Found: {shown}.", code="S7")
    return None


def _s8(bound, _p):
    if bound.test.id == "describe":            # describe tolerates constant columns
        return None
    for col in bound.data.columns:
        if col in _LABEL_COLS:
            continue
        if _levels(bound, col) <= 1 and bound.data[col].notna().any():
            return Finding("block", f"Column '{_header_for(bound, col)}' is constant "
                           "(no variation); the test cannot run on it.", code="S8")
    return None


def _s9(bound, profiles):
    if profiles is None or not _has(bound, "group"):
        return None
    hdr = bound.columns.get("group", (None,))[0]
    prof = profiles.get(hdr)
    if (prof is not None and prof.kinds and prof.kinds[0] == Kind.NUMERIC
            and bound.kinds.get("group") in (Kind.CATEGORICAL, Kind.ORDINAL, Kind.BINARY)
            and _levels(bound, "group") > 10):
        return Finding("flag", f"'{hdr}' is a numeric column used as a grouping "
                       "variable with many distinct values — is that intended?",
                       code="S9")
    return None


def _s10(bound, _p):
    if bound.test.id != "pearson":
        return None
    for role in ("x", "y"):
        if (bound.kinds.get(role) == Kind.ORDINAL and _has(bound, role)
                and _levels(bound, role) <= 5):
            return Finding("flag", "One variable is ordinal with few levels; a rank "
                           "correlation is usually more appropriate.",
                           suggest_test="spearman", code="S10")
    return None


def _s11(bound, _p):
    if bound.test.id not in _REGRESSION:
        return None
    preds = bound.columns.get("predictors") or bound.columns.get("x") or ()
    p = len(preds)
    if p > 15:
        return Finding("block", f"{p} predictors is too many for a stable fit "
                       "(limit 15).", code="S11")
    if bound.n_used < p + 10:
        return Finding("block", f"Only {bound.n_used} usable rows for {p} "
                       f"predictor(s); need at least {p + 10}.", code="S11")
    return None


def _s12(bound, _p):
    if bound.test.id != "logistic" or not _has(bound, "outcome"):
        return None
    if bound.kinds.get("outcome") == Kind.NUMERIC:
        return Finding("block", "Logistic regression needs a yes/no outcome; this "
                       "outcome is numeric.", suggest_test="ols_simple", code="S12")
    if _levels(bound, "outcome") > 2:
        return Finding("block", "Logistic regression needs a 2-level outcome; "
                       "multinomial outcomes are not supported.", code="S12")
    return None


def _s13(bound, profiles):
    if profiles is None:
        return None
    for role, header in bound.columns.items():
        if bound.kinds.get(role) != Kind.NUMERIC:
            continue
        prof = profiles.get(header[0]) if header else None
        if prof is not None and Kind.ORDINAL in prof.kinds:
            return Finding("flag", f"'{header[0]}' is ordinal but is being treated as "
                           "an interval (numeric) scale.", code="S13")
    return None


def _s14(bound, _p):
    if bound.test.id != "chi2_gof":
        return None
    exp = bound.params.get("expected") or ()
    if not exp:
        return None
    ncat = _levels(bound, "category") if _has(bound, "category") else len(exp)
    if len(exp) != ncat:
        return Finding("block", f"Expected proportions ({len(exp)}) must match the "
                       f"number of categories ({ncat}).", code="S14")
    if abs(sum(exp) - 1.0) > 0.001:
        return Finding("block", f"Expected proportions must sum to 1 (they sum to "
                       f"{sum(exp):.3f}).", code="S14")
    return None


def _s15(bound, _p):
    if bound.layout == "table" and bound.n_used > 1_000_000:
        return Finding("block", f"The table's total count ({bound.n_used:,}) is too "
                       "large to expand.", code="S15")
    return None


def _s16(bound, _p):
    if bound.n_total > 0 and bound.n_used / bound.n_total < 0.7:
        reasons = ", ".join(f"{v} {k}" for k, v in bound.dropped.items())
        return Finding("flag", f"Only {bound.n_used} of {bound.n_total} rows are "
                       f"usable ({reasons}).", code="S16")
    return None


def _s17(bound, _p):
    if (bound.test.id == "chi2_ind" and _has(bound, "row") and _has(bound, "col")
            and _levels(bound, "row") == 2 and _levels(bound, "col") == 2):
        return Finding("info", "Both variables are binary; Fisher's exact test is an "
                       "alternative for small counts.", suggest_test="fisher",
                       code="S17")
    return None


def _s18(bound, _p):
    if bound.layout == "wide" and bound.test.pairing == "independent":
        return Finding("info", "These columns were read as separate groups. If they "
                       "are the same people/units measured twice, choose a paired "
                       "test at the goal step.", code="S18")
    return None


def _s19(bound, _p):
    if bound.test.id != "anova_2w" or not (_has(bound, "factor_a") and _has(bound, "factor_b")):
        return None
    tab = pd.crosstab(bound.data["factor_a"], bound.data["factor_b"])
    if (tab.to_numpy() == 0).any():
        return Finding("block", "One or more Factor A × Factor B combinations has no "
                       "data; a two-way ANOVA needs every cell filled.", code="S19")
    counts = tab.to_numpy().ravel()
    if len(set(counts)) > 1:
        return Finding("flag", "The design is unbalanced; Type II sums of squares "
                       "are used.", code="S19")
    return None


def _s20(bound, _p):
    n = bound.dropped.get("incomplete subject", 0)
    if n > 0:
        return Finding("flag", f"{n} subject(s) with incomplete measurements were "
                       "dropped (listwise).", code="S20")
    return None


def _s21(bound, _p):
    if bound.n_used < 15:
        return Finding("flag", f"Low power (n = {bound.n_used}); a non-significant "
                       "result is not evidence of no effect.", code="S21")
    return None


def _s22(bound, _p):
    """A grouping column left with too few levels AFTER listwise deletion.

    A whole group can vanish when all its outcomes are missing (or a factor level
    is empty), leaving fewer distinct levels than the role needs. Without this the
    runner would `a, b = groups` on one group and raise IndexError/ValueError;
    this states the reason and fails closed instead."""
    for col in ("group", "row", "col", "factor_a", "factor_b", "condition"):
        if not _has(bound, col):
            continue
        lo, _hi = _role_levels(bound, col)
        if lo < 2:
            continue
        L = _levels(bound, col)
        if L < lo:
            names = ", ".join(levels.display(v) for v in bound.data[col].dropna().unique())
            return Finding("block", "After removing rows with missing values, only "
                           f"{L} level(s) of '{_header_for(bound, col)}' remain "
                           f"({names}); this test needs at least {lo}.", code="S22")
    return None


_RULES = (
    _s1, _s2, _s3, _s4, _s5, _s6, _s7, _s8, _s9, _s10, _s11,
    _s12, _s13, _s14, _s15, _s16, _s17, _s18, _s19, _s20, _s21, _s22,
)
