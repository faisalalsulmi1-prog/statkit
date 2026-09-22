"""Effect sizes — ours, because scipy/statsmodels ship none (PLAN D7).

Correct-by-default: pooled SD with the right ddof, Hedges' small-sample
correction, uncorrected Cramer's V (matching scipy's contingency.association,
with the thresholds scaled instead — PLAN §3), Chen (2010) OR thresholds.

Every function is a pure function of arrays/scalars; the runner decides sign
by choosing argument order, and reads the magnitude label. Point estimates use
the first-minus-second convention (`cohens_d(a, b)` is positive when `a` is
larger). Labels are sign-independent.

Only stdlib + numpy + scipy (L3 allowlist). No I/O, no state.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import chi2_contingency

# ---------------------------------------------------------------- means family


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def pooled_sd(a, b) -> float:
    """Classic pooled SD (equal-variance assumption), ddof = n1 + n2 - 2."""
    a, b = _arr(a), _arr(b)
    n1, n2 = len(a), len(b)
    sp2 = ((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2)
    return math.sqrt(sp2)


def cohens_d(a, b) -> float:
    """Cohen's d for two independent samples: (mean(a) - mean(b)) / pooled SD."""
    a, b = _arr(a), _arr(b)
    return (a.mean() - b.mean()) / pooled_sd(a, b)


def hedges_correction(df: int) -> float:
    """Hedges' small-sample bias correction J = 1 - 3/(4*df - 1), df = n1+n2-2
    (Hedges & Olkin 1985). Multiply Cohen's d by this to get Hedges' g."""
    return 1.0 - 3.0 / (4.0 * df - 1.0)


def hedges_g(a, b) -> float:
    """Hedges' g = Cohen's d x J (bias-corrected d for small samples)."""
    a, b = _arr(a), _arr(b)
    return cohens_d(a, b) * hedges_correction(len(a) + len(b) - 2)


def cohens_d_onesample(x, mu0: float) -> float:
    """One-sample d: (mean - mu0) / SD."""
    x = _arr(x)
    return (x.mean() - mu0) / x.std(ddof=1)


def cohens_dz(diff) -> float:
    """Paired d_z: mean(differences) / SD(differences)."""
    d = _arr(diff)
    return d.mean() / d.std(ddof=1)


# ---------------------------------------------------------------- ANOVA family


def _ss(groups) -> tuple[float, float, float]:
    gs = [_arr(g) for g in groups]
    grand = np.concatenate(gs).mean()
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in gs)
    ss_w = sum(((g - g.mean()) ** 2).sum() for g in gs)
    return ss_b, ss_w, ss_b + ss_w


def eta_squared(*groups) -> float:
    """One-way eta^2 = SS_between / SS_total."""
    ss_b, _, ss_t = _ss(groups)
    return ss_b / ss_t


def omega_squared(*groups) -> float:
    """One-way omega^2 = (SS_b - df_b*MS_w) / (SS_total + MS_w). Less biased
    than eta^2; can be negative (report as ~0 upstream if desired)."""
    gs = [_arr(g) for g in groups]
    k = len(gs)
    n = sum(len(g) for g in gs)
    ss_b, ss_w, ss_t = _ss(gs)
    ms_w = ss_w / (n - k)
    return (ss_b - (k - 1) * ms_w) / (ss_t + ms_w)


def partial_eta_squared(f: float, df1: int, df2: int) -> float:
    """Partial eta^2 from an F-ratio: f*df1 / (f*df1 + df2). Works for a
    two-way ANOVA term and for repeated-measures (PLAN §3 rm_anova)."""
    return (f * df1) / (f * df1 + df2)


def epsilon_squared(h: float, k: int, n: int) -> float:
    """Kruskal-Wallis epsilon^2 = (H - k + 1) / (n - k)."""
    return (h - k + 1) / (n - k)


def kendalls_w(chi2: float, n: int, k: int) -> float:
    """Friedman -> Kendall's W = chi2 / (n * (k - 1))."""
    return chi2 / (n * (k - 1))


# ---------------------------------------------------------------- rank-biserial


def rank_biserial_u(u: float, n1: int, n2: int) -> float:
    """Mann-Whitney rank-biserial r = 1 - 2U/(n1*n2)."""
    return 1.0 - 2.0 * u / (n1 * n2)


def rank_biserial_wilcoxon(w_plus: float, w_minus: float) -> float:
    """Matched-pairs rank-biserial r = (W+ - W-) / (W+ + W-)."""
    return (w_plus - w_minus) / (w_plus + w_minus)


# ---------------------------------------------------------------- contingency


def _chi2(table) -> tuple[float, int]:
    t = np.asarray(table, dtype=float)
    chi2, _, _, _ = chi2_contingency(t, correction=False)
    return chi2, int(t.sum())


def phi(table) -> float:
    """phi = sqrt(chi2 / N) for a 2x2 table (uncorrected chi2)."""
    chi2, n = _chi2(table)
    return math.sqrt(chi2 / n)


def cramers_v(table) -> float:
    """Cramer's V = sqrt(chi2 / (N * (min(r,c) - 1))), UNCORRECTED (PLAN §3:
    matches scipy.stats.contingency.association(method='cramer',
    correction=False); the small/medium/large thresholds are scaled instead --
    see cramers_v_label)."""
    t = np.asarray(table, dtype=float)
    chi2, n = _chi2(t)
    r, c = t.shape
    return math.sqrt(chi2 / (n * (min(r, c) - 1)))


def cohens_w(chi2: float, n: int) -> float:
    """Goodness-of-fit effect size w = sqrt(chi2 / N)."""
    return math.sqrt(chi2 / n)


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h for two proportions = 2*asin(sqrt(p1)) - 2*asin(sqrt(p2))."""
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def cohens_f2(r2: float) -> float:
    """Cohen's f^2 = R^2 / (1 - R^2)."""
    return r2 / (1.0 - r2)


# ---------------------------------------------------------------- magnitude labels
# Thresholds from PLAN §3. A value below the "small" cut is "negligible".


def _magnitude(value: float, small: float, medium: float, large: float) -> str:
    v = abs(value)
    if v < small:
        return "negligible"
    if v < medium:
        return "small"
    if v < large:
        return "medium"
    return "large"


def d_label(x: float) -> str:
    """d / g / d_z / Cohen's h: .2 / .5 / .8 (Cohen 1988)."""
    return _magnitude(x, 0.2, 0.5, 0.8)


h_label = d_label   # Cohen's h shares d's thresholds (PLAN §3)


def r_label(x: float) -> str:
    """r / rho / tau / rank-biserial / Kendall's W: .1 / .3 / .5 (Cohen 1988)."""
    return _magnitude(x, 0.1, 0.3, 0.5)


w_kendall_label = r_label   # Kendall's W shares .1/.3/.5 (Tomczak & Tomczak 2014)


def eta_label(x: float) -> str:
    """eta^2 / omega^2 / epsilon^2 / partial-eta^2: .01 / .06 / .14."""
    return _magnitude(x, 0.01, 0.06, 0.14)


def phi_w_label(x: float) -> str:
    """phi / Cohen's w: .1 / .3 / .5."""
    return _magnitude(x, 0.1, 0.3, 0.5)


def cramers_v_label(v: float, r: int, c: int) -> str:
    """Cramer's V label with thresholds divided by sqrt(min(r,c) - 1) (PLAN §3),
    so a 2x2 uses .1/.3/.5 and larger tables use smaller cuts."""
    s = math.sqrt(min(r, c) - 1)
    return _magnitude(v, 0.1 / s, 0.3 / s, 0.5 / s)


def or_label(odds_ratio: float) -> str:
    """Odds-ratio label, Chen, Cohen & Chen (2010): 1.68 / 3.47 / 6.71.
    Strength is symmetric about 1, so OR<1 is folded to 1/OR."""
    v = max(odds_ratio, 1.0 / odds_ratio)
    return _magnitude(v, 1.68, 3.47, 6.71)


def f2_label(x: float) -> str:
    """Cohen's f^2: .02 / .15 / .35."""
    return _magnitude(x, 0.02, 0.15, 0.35)
