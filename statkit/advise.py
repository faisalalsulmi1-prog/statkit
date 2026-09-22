"""Data-driven advisories D1-D14 + robust-alternative agreement (PLAN §5.2, Chunk 11).

Where ``check(bound)`` (S1-S21) is STRUCTURAL — it reads only what the contract
and the level counts imply — ``advise(bound)`` is DATA-DRIVEN: it looks at the
actual numbers (Brown-Forsythe, Shapiro, expected counts, separation, outliers,
group balance, ...) and returns plain, non-alarming ``Finding``s, each carrying
its D-code and a reason. Nothing here blocks except the two genuinely
uncomputable cases (D3 tiny expected counts → Fisher, D4 no within-group
variation); everything else is *info* or *flag* so the student stays in control.

``robust_agreement(bound, result)`` is the DoD's background cross-check: for a
parametric test whose assumptions look shaky it runs the rank / exact
alternative (Welch↔Mann-Whitney, ANOVA↔Kruskal, paired-t↔Wilcoxon,
Pearson↔Spearman) — and RM-ANOVA↔Friedman always — then reports whether the two
AGREE on the significance decision: a reassurance (info) when they do, a caution
(flag) when they do not. It is deterministic and adds no I/O; it never shows a
second p-value as the result.

``advised(runner)`` is the single wiring seam (``registry._wire`` applies it to
all 27 runners): it attaches the advisories to the bound BEFORE the runner
computes — so a D3/D4 block gates the run and every advisory rides into the
Result exactly as the structural findings do (snag 10.3) — and dedups by D-code
so the runner-emitted D3 (cat.py) / D6 (regress.py) are never doubled.

Only stdlib + numpy + scipy + pandas + our own assumptions/check/model. No I/O.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import scipy.stats as ss

from . import assumptions, fmt, levels
from .check import gate
from .model import Finding

_ALPHA = 0.05


# --------------------------------------------------------------------------
# small shared accessors
# --------------------------------------------------------------------------
def _group_arrays(bound) -> dict:
    """{level_label: outcome array} for a canonical long outcome+group frame."""
    df = bound.data
    order = list(dict.fromkeys(df["group"].tolist()))
    return {str(g): df.loc[df["group"] == g, "outcome"].to_numpy(float) for g in order}


def _residuals(bound) -> np.ndarray:
    return np.concatenate([a - a.mean() for a in _group_arrays(bound).values() if len(a)])


def _shapiro_fails(*arrays) -> bool:
    """True if Shapiro rejects normality on any array inside its valid n-window."""
    for a in arrays:
        ch = assumptions.shapiro_normality(a)
        if ch.p is not None and ch.p < _ALPHA:
            return True
    return False


def _dedup(findings) -> tuple:
    """Keep the FIRST finding for each non-empty code; codeless findings all kept."""
    seen: set = set()
    out = []
    for f in findings:
        if f.code:
            if f.code in seen:
                continue
            seen.add(f.code)
        out.append(f)
    return tuple(out)


# ==========================================================================
# D1 — Brown-Forsythe: unequal variances
# ==========================================================================
def _d1(bound):
    tid = bound.test.id
    if tid not in ("t_ind", "anova_1w"):
        return ()
    arrs = list(_group_arrays(bound).values())
    if len(arrs) < 2 or any(len(a) < 2 for a in arrs):
        return ()
    bf = assumptions.brown_forsythe(*arrs)
    if bf.p is None or not math.isfinite(bf.p) or bf.p >= _ALPHA:
        return ()
    if tid == "t_ind":
        if bool(bound.params.get("equal_var", False)):        # Student's chosen
            return (Finding("flag", "The groups have unequal variances "
                            f"(Brown-Forsythe {fmt.p(bf.p)}); Welch's t-test, which "
                            "does not assume equal variances, is safer.",
                            suggest_params=(("equal_var", False),), code="D1"),)
        return (Finding("info", "The groups have unequal variances (Brown-Forsythe "
                        f"{fmt.p(bf.p)}); Welch's method already accounts for this.",
                        code="D1"),)
    if not bool(bound.params.get("welch", False)):            # classic ANOVA chosen
        return (Finding("flag", "The groups have unequal variances (Brown-Forsythe "
                        f"{fmt.p(bf.p)}); Welch's ANOVA is more reliable here.",
                        suggest_params=(("welch", True),), code="D1"),)
    return (Finding("info", "The groups have unequal variances (Brown-Forsythe "
                    f"{fmt.p(bf.p)}); Welch's ANOVA already accounts for this.",
                    code="D1"),)


# ==========================================================================
# D2 — Shapiro: non-normality (with a sample-size aware message)
# ==========================================================================
def _d2_targets(bound):
    """(arrays to test for normality, the relevant min n, rank-alternative id)."""
    tid = bound.test.id
    if tid == "t_1s":
        x = bound.data["outcome"].to_numpy(float)
        return [x], len(x), "wilcoxon"
    if tid == "t_ind":
        arrs = list(_group_arrays(bound).values())
        return arrs, min((len(a) for a in arrs), default=0), "mwu"
    if tid == "t_paired":
        d = bound.data["before"].to_numpy(float) - bound.data["after"].to_numpy(float)
        return [d], len(d), "wilcoxon"
    if tid == "anova_1w":
        arrs = list(_group_arrays(bound).values())
        return [_residuals(bound)], min((len(a) for a in arrs), default=0), "kruskal"
    return [], 0, None


def _d2(bound):
    arrays, min_n, alt = _d2_targets(bound)
    if not arrays or not _shapiro_fails(*arrays):
        return ()
    if min_n < 30:
        return (Finding("info", "The data depart from normality and the sample is "
                        f"small (n = {min_n}); a rank-based test is a robust "
                        "alternative.", suggest_test=alt, code="D2"),)
    if min_n >= 200:
        return (Finding("info", "The data depart from normality, but at this sample "
                        f"size (n = {min_n}) the normality test flags even trivial "
                        "deviations — read the Q-Q plot rather than the p-value.",
                        code="D2"),)
    return (Finding("info", "The data depart from normality, but the sample is large "
                    f"(n = {min_n}); the test is robust to this (central limit "
                    "theorem).", code="D2"),)


# ==========================================================================
# D3 — chi-square expected counts (also owned, for the severe block, by cat.py)
# ==========================================================================
def _d3(bound):
    if bound.test.id != "chi2_ind":
        return ()
    df = bound.data
    if "row" not in df.columns or "col" not in df.columns or len(df) == 0:
        return ()
    tab = pd.crosstab(df["row"], df["col"])
    obs = tab.to_numpy(float)
    if obs.size == 0 or obs.sum() == 0:
        return ()
    _, _, _, expected = ss.chi2_contingency(obs, correction=False)
    r_, c_ = obs.shape
    if (expected < 1).any() or (r_ == 2 and c_ == 2 and (expected < 5).any()):
        return (Finding("block", "Some expected counts are too small for a reliable "
                        "chi-square test; use Fisher's exact test instead.",
                        suggest_test="fisher", code="D3"),)
    frac_low = float((expected < 5).mean())
    if frac_low > 0.20:
        if r_ * c_ <= 25 and obs.sum() <= 2000:
            return (Finding("flag", f"{frac_low * 100:.0f}% of cells have an expected "
                            "count below 5; Fisher's exact test is more reliable.",
                            suggest_test="fisher", code="D3"),)
        return (Finding("flag", f"{frac_low * 100:.0f}% of cells have an expected "
                        "count below 5; interpret the chi-square with caution (the "
                        "table is too large for an exact test).", code="D3"),)
    return ()


# ==========================================================================
# D4 — no variation within ANY group
# ==========================================================================
def _d4(bound):
    if bound.test.id not in ("t_ind", "anova_1w", "mwu", "kruskal"):
        return ()
    if "group" not in bound.data.columns:
        return ()
    arrs = list(_group_arrays(bound).values())
    if len(arrs) < 2 or any(len(a) < 1 for a in arrs):
        return ()
    if all(float(np.nanstd(a)) == 0.0 for a in arrs):
        return (Finding("block", "Every group has the same value throughout (no "
                        "variation within any group); the test cannot run.",
                        code="D4"),)
    return ()


# ==========================================================================
# D5 — logistic separation pre-check (the fit itself blocks; this is a warning)
# ==========================================================================
def _d5(bound):
    if bound.test.id != "logistic" or "outcome" not in bound.data.columns:
        return ()
    df = bound.data
    out = df["outcome"]
    headers = list(bound.columns.get("predictors", ()))
    culprits = []
    for i, c in enumerate([c for c in df.columns if c.startswith("predictors")]):
        s = df[c]
        if s.dropna().nunique() > 10:                    # only few-level predictors
            continue
        tab = pd.crosstab(s, out)
        if tab.size and (tab.to_numpy() == 0).any():
            culprits.append(headers[i] if i < len(headers) else c)
    if culprits:
        names = ", ".join(f"'{c}'" for c in culprits)
        return (Finding("flag", f"A predictor ({names}) perfectly predicts the outcome "
                        "in the cross-tabulation (a zero cell); watch for separation, "
                        "which makes the odds ratios unstable.", code="D5"),)
    return ()


# ==========================================================================
# D6 is owned by regress.py (VIF on the fitted design); advise defers to it.
# ==========================================================================


# ==========================================================================
# D7 — Mann-Whitney: ties present and both groups small
# ==========================================================================
def _d7(bound):
    if bound.test.id != "mwu" or "group" not in bound.data.columns:
        return ()
    arrs = list(_group_arrays(bound).values())
    if len(arrs) != 2:
        return ()
    a, b = arrs
    if len(a) > 8 or len(b) > 8:
        return ()
    combined = np.concatenate([a, b])
    if len(np.unique(combined)) < len(combined):
        return (Finding("info", "Tied values are present and both groups are small, so "
                        "an exact p-value is not available; the asymptotic method with "
                        "a tie correction is used.", code="D7"),)
    return ()


# ==========================================================================
# D8 — Wilcoxon: zero differences excluded
# ==========================================================================
def _d8(bound):
    if bound.test.id != "wilcoxon":
        return ()
    df = bound.data
    if "after" in df.columns:
        d = df["before"].to_numpy(float) - df["after"].to_numpy(float)
    elif "outcome" in df.columns:
        d = df["outcome"].to_numpy(float) - float(bound.params.get("mu0", 0.0))
    else:
        return ()
    k = int((d == 0).sum())
    if k > 0:
        return (Finding("info", f"{k} zero difference(s) were excluded "
                        "(zero_method='wilcox').", code="D8"),)
    return ()


# ==========================================================================
# D9 — outliers, shown never removed
# ==========================================================================
def _d9_targets(bound):
    tid = bound.test.id                                  # numeric-outcome tests only
    if tid in ("t_ind", "anova_1w", "mwu", "kruskal") and "group" in bound.data.columns:
        return _group_arrays(bound)
    if tid == "t_1s" and "outcome" in bound.data.columns:
        return {"": bound.data["outcome"].to_numpy(float)}
    return {}


def _d9(bound):
    outs = []
    for label, a in _d9_targets(bound).items():
        if len(a) < 4:
            continue
        q1, q3 = np.percentile(a, 25), np.percentile(a, 75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        for v in a:
            if v < lo or v > hi:
                outs.append((label, float(v)))
    if not outs:
        return ()
    shown = ", ".join(f"{v:g}" + (f" (group {lab})" if lab else "")
                      for lab, v in outs[:10])
    return (Finding("info", "Possible outlier(s) more than 3×IQR from the quartiles: "
                    f"{shown}. These are shown, not removed.", code="D9"),)


# ==========================================================================
# D10 — very unequal group sizes
# ==========================================================================
def _d10(bound):
    if bound.test.id not in ("t_ind", "anova_1w", "mwu", "kruskal"):
        return ()
    if "group" not in bound.data.columns:
        return ()
    sizes = [len(a) for a in _group_arrays(bound).values()]
    if len(sizes) < 2 or min(sizes) == 0:
        return ()
    if max(sizes) / min(sizes) > 3:
        return (Finding("info", f"The groups are very unequal in size ({max(sizes)} "
                        f"vs {min(sizes)}); results can be sensitive to this "
                        "imbalance.", code="D10"),)
    return ()


# ==========================================================================
# D11 — GOF / one-sample proportion: small expected counts
# ==========================================================================
def _d11(bound):
    tid = bound.test.id
    if tid == "chi2_gof" and "category" in bound.data.columns:
        cats = list(dict.fromkeys(bound.data["category"].tolist()))
        if not cats:
            return ()
        observed = bound.data["category"].value_counts().reindex(cats).to_numpy(float)
        total = float(observed.sum())
        props = bound.params.get("expected") or ()
        expected = (np.asarray(props, float) * total if props
                    else np.full(len(cats), total / len(cats)))
        if (expected < 5).any():
            return (Finding("flag", "Some expected counts are below 5; the chi-square "
                            "goodness-of-fit approximation may be unreliable.",
                            code="D11"),)
        return ()
    if tid == "prop_1" and "outcome" in bound.data.columns:
        n = int(bound.data["outcome"].dropna().shape[0])
        p0 = float(bound.params.get("p0", 0.5))
        if n * p0 < 5 or n * (1 - p0) < 5:
            return (Finding("flag", "The expected number of successes or failures under "
                            f"p₀ is below 5 (n = {n}); the normal approximation is weak "
                            "(the exact binomial test itself is still valid).",
                            code="D11"),)
    return ()


# ==========================================================================
# D12 — two-sample proportion: small np / n(1-p)
# ==========================================================================
def _d12(bound):
    if bound.test.id != "prop_2":
        return ()
    df = bound.data
    if "outcome" not in df.columns or "group" not in df.columns:
        return ()
    success, _ = levels.pick_success(df["outcome"].dropna(),
                                     bound.params.get("success"))
    if success is None:                       # unresolvable level: the run blocks
        return ()
    groups = list(dict.fromkeys(df["group"].dropna().tolist()))
    for g in groups[:2]:
        sub = df.loc[df["group"] == g, "outcome"].dropna()
        k = int((sub == success).sum())
        n = int(len(sub))
        if k < 5 or (n - k) < 5:
            return (Finding("flag", "A group has fewer than 5 successes or failures; "
                            "the z-test approximation is weak — Fisher's exact test is "
                            "more reliable.", suggest_test="fisher", code="D12"),)
    return ()


_RULES = (_d1, _d2, _d3, _d4, _d5, _d7, _d8, _d9, _d10, _d11, _d12)


def advise(bound) -> tuple:
    """Data-driven advisories for a bound (everything computable before running).

    The robust-alternative agreement (D13 / D14 / ROBUST) needs the primary
    p-value, so it lives in ``robust_agreement`` and is added post-run.
    """
    out: list = []
    for rule in _RULES:
        out.extend(rule(bound))
    return _dedup(tuple(out))


# ==========================================================================
# robust-alternative agreement (background cross-check)
# ==========================================================================
# test_id -> (rank/exact alternative's display name, the Finding code it carries)
_PARTNER = {
    "t_1s": ("Wilcoxon signed-rank test", "ROBUST"),
    "t_ind": ("Mann-Whitney U test", "ROBUST"),
    "t_paired": ("Wilcoxon signed-rank test", "ROBUST"),
    "anova_1w": ("Kruskal-Wallis test", "ROBUST"),
    "rm_anova": ("Friedman test", "D13"),
    "pearson": ("Spearman rank correlation", "D14"),
}


def _shaky(bound) -> bool:
    """Do this parametric test's assumptions look shaky enough to cross-check?

    RM-ANOVA always cross-checks (D13). The others cross-check when normality is
    rejected (or, for the two-plus-group means tests, variances are unequal)."""
    tid = bound.test.id
    if tid == "rm_anova":
        return True
    if tid == "pearson":
        x = bound.data["x"].to_numpy(float)
        y = bound.data["y"].to_numpy(float)
        return len(x) < 30 and _shapiro_fails(x, y)
    if tid == "t_1s":
        return _shapiro_fails(bound.data["outcome"].to_numpy(float))
    if tid == "t_paired":
        return _shapiro_fails(bound.data["before"].to_numpy(float)
                              - bound.data["after"].to_numpy(float))
    if tid in ("t_ind", "anova_1w"):
        arrs = list(_group_arrays(bound).values())
        if len(arrs) < 2 or any(len(a) < 2 for a in arrs):
            return False
        if _shapiro_fails(*arrs):
            return True
        bf = assumptions.brown_forsythe(*arrs)
        return bf.p is not None and math.isfinite(bf.p) and bf.p < _ALPHA
    return False


def _robust_pvalue(bound):
    """The rank/exact alternative's p-value, computed from the canonical frame.

    Deterministic (no RNG) and self-contained; returns None when the alternative
    is itself undefined (e.g. all-zero Wilcoxon differences)."""
    tid = bound.test.id
    try:
        if tid == "t_ind":
            a, b = list(_group_arrays(bound).values())
            return float(ss.mannwhitneyu(a, b, alternative="two-sided",
                                         method="auto").pvalue)
        if tid == "anova_1w":
            return float(ss.kruskal(*_group_arrays(bound).values()).pvalue)
        if tid == "t_paired":
            d = bound.data["before"].to_numpy(float) - bound.data["after"].to_numpy(float)
            d = d[d != 0]
            if len(d) == 0:
                return None
            return float(ss.wilcoxon(d, zero_method="wilcox",
                                     alternative="two-sided").pvalue)
        if tid == "t_1s":
            d = bound.data["outcome"].to_numpy(float) - float(bound.params.get("mu0", 0.0))
            d = d[d != 0]
            if len(d) == 0:
                return None
            return float(ss.wilcoxon(d, zero_method="wilcox",
                                     alternative="two-sided").pvalue)
        if tid == "rm_anova":
            cols = [bound.data[c].to_numpy(float) for c in bound.data.columns]
            return float(ss.friedmanchisquare(*cols).pvalue)
        if tid == "pearson":
            x = bound.data["x"].to_numpy(float)
            y = bound.data["y"].to_numpy(float)
            return float(ss.spearmanr(x, y).pvalue)
    except (ValueError, ZeroDivisionError):
        return None
    return None


def robust_agreement(bound, result) -> tuple:
    """One reassurance/caution advisory comparing the primary decision with its
    rank/exact alternative. Empty when the test has no partner, the result is
    blocked, or the assumptions look fine (RM-ANOVA always cross-checks)."""
    tid = bound.test.id
    if (tid not in _PARTNER or result is None or result.status != "ok"
            or result.p is None or not _shaky(bound)):
        return ()
    rp = _robust_pvalue(bound)
    if rp is None or not math.isfinite(rp):
        return ()
    name, code = _PARTNER[tid]
    suggest = "spearman" if tid == "pearson" else None
    agree = (result.p < _ALPHA) == (rp < _ALPHA)
    lead = ("Sphericity was not tested; a Friedman test" if tid == "rm_anova"
            else f"A {name}") + ", run in the background as a robustness check,"
    if agree:
        return (Finding("info", f"{lead} agrees with this conclusion.",
                        suggest_test=suggest, code=code),)
    return (Finding("flag", f"{lead} reaches a different conclusion ({fmt.p(rp)}); "
                    "interpret this result with caution.",
                    suggest_test=suggest, code=code),)


# ==========================================================================
# the wiring seam
# ==========================================================================
def advised(runner):
    """Wrap a runner so its Result carries the data-driven advisories, deduped.

    Before the runner computes, the structural findings and ``advise(bound)`` are
    merged onto ``bound.findings`` (so a D3/D4 block gates the run and every
    advisory rides into the Result, snag 10.3). After it computes, the robust
    alternative agreement is appended. Every D-code is deduped, so the
    runner-emitted D3 (cat.py) / D6 (regress.py) are never doubled."""
    def wrapped(bound):
        base = gate(bound)          # structural check ALWAYS runs, merged (S-F)
        # S-N: a structural BLOCK means the frame is not analysable (un-coded text,
        # no variation, wrong arity); the data-driven rules would then meet a frame
        # they cannot read (e.g. a rank D-rule's to_numpy(float) on kept Likert
        # text). Skip advise() entirely when a block already fired -- the block is
        # authoritative and the Run is disabled anyway.
        adv = () if any(f.severity == "block" for f in base) else advise(bound)
        bound.findings = _dedup(base + adv)
        result = runner(bound)
        if result.status == "ok" and result.p is not None:
            result.findings = _dedup(tuple(result.findings)
                                     + robust_agreement(bound, result))
        else:
            result.findings = _dedup(tuple(result.findings))
        # Chart data (Chunk 13): stash the canonical role-named frame + the
        # role->headers map so charts.py can draw from a Result alone (spec.chart
        # takes only a Result). This is the one place both bound and result meet.
        # In-memory references only (L3-safe); nothing else reads these keys.
        result.arrays.setdefault("_frame", bound.data)
        result.arrays.setdefault("_cols", bound.columns)
        return result

    wrapped.__name__ = getattr(runner, "__name__", "advised")
    wrapped.__qualname__ = getattr(runner, "__qualname__", wrapped.__name__)
    wrapped.__doc__ = runner.__doc__
    wrapped.__wrapped__ = runner
    return wrapped
