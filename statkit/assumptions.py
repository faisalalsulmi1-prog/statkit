"""Assumption checks — normality and homogeneity of variance (PLAN §3, §9.1).

Correct-by-default:
  * Shapiro-Wilk is SKIPPED at n<3 and n>5000 (scipy warns / is unreliable
    outside that window); a skipped check returns passed=None with a note.
  * Homogeneity uses levene(center="median"), which IS the Brown-Forsythe test
    -- named correctly here, not "Levene".
  * Normality also offered via Lilliefors (KS with ESTIMATED parameters), which
    is NOT a naive KS against a fixed N(0,1).

Each function returns a model.Check. Input arrays are assumed already cleaned
(listwise deletion happens in bind() -- PLAN D14).

Only stdlib + numpy + scipy + statsmodels (L3 allowlist). No I/O.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import bartlett, kurtosis, levene, shapiro, skew
from statsmodels.stats.diagnostic import lilliefors

from .model import Check


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _check(name, stat, p, alpha) -> Check:
    """A finished assumption Check. A non-finite p (constant / degenerate data
    makes scipy emit nan) is NOT a failed test -- bool(nan >= alpha) is False,
    which would silently claim "not met". It is "not computable"."""
    p = float(p)
    if not math.isfinite(p):
        return Check(name, None, None, None, "not computable on this data")
    return Check(name, float(stat), p, bool(p >= alpha), "")


def shapiro_normality(x, alpha: float = 0.05) -> Check:
    """Shapiro-Wilk normality. Run only for 3 <= n <= 5000; outside that window
    the check is skipped (passed=None) with a note, never silently wrong."""
    x = _arr(x)
    n = len(x)
    if n < 3:
        return Check("Shapiro-Wilk", None, None, None,
                     f"n={n}: Shapiro-Wilk not run (needs n >= 3)")
    if n > 5000:
        return Check("Shapiro-Wilk", None, None, None,
                     f"n={n}: Shapiro-Wilk not run (n > 5000)")
    res = shapiro(x)
    return _check("Shapiro-Wilk", res.statistic, res.pvalue, alpha)


def lilliefors_normality(x, alpha: float = 0.05) -> Check:
    """Lilliefors normality: the Kolmogorov-Smirnov statistic against a normal
    with mean/SD ESTIMATED from the data (statsmodels lilliefors), not a naive
    KS against a fixed N(0,1)."""
    x = _arr(x)
    n = len(x)
    if n < 4:
        return Check("Lilliefors (KS, estimated parameters)", None, None, None,
                     f"n={n}: Lilliefors not run (needs n >= 4)")
    stat, p = lilliefors(x, dist="norm", pvalmethod="table")
    return _check("Lilliefors (KS, estimated parameters)", stat, p, alpha)


def brown_forsythe(*groups, alpha: float = 0.05) -> Check:
    """Homogeneity of variance via Brown-Forsythe = Levene centred on the MEDIAN
    (robust to non-normality). Named correctly, not "Levene"."""
    arrs = [_arr(g) for g in groups]
    res = levene(*arrs, center="median")
    return _check("Brown-Forsythe (Levene, center=median)",
                  res.statistic, res.pvalue, alpha)


def bartlett_variance(*groups, alpha: float = 0.05) -> Check:
    """Bartlett's test of equal variances (sensitive to non-normality; reported
    alongside Brown-Forsythe, PLAN §3)."""
    arrs = [_arr(g) for g in groups]
    res = bartlett(*arrs)
    return _check("Bartlett", res.statistic, res.pvalue, alpha)


def skew_kurtosis(x) -> tuple[float, float]:
    """(skewness, excess kurtosis) for the normality panel (PLAN §3)."""
    x = _arr(x)
    return float(skew(x)), float(kurtosis(x))   # scipy kurtosis is excess (Fisher)
