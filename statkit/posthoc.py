"""Post-hoc pairwise procedures (PLAN §3, §9.1, DoD §10).

  * Tukey HSD      -> thin wrapper over scipy.stats.tukey_hsd.
  * Games-Howell   -> hand-coded from scipy.stats.studentized_range + Welch df
                      (unequal variances; no dependency on a possibly-missing
                      pairwise_tukeyhsd(use_var=...) kwarg -- PLAN D3).
  * Dunn           -> hand-coded from global mid-ranks + scipy.stats.norm, with
                      the standard tie correction, Holm-adjusted via
                      statsmodels multipletests(method="holm").

Each returns a pandas DataFrame, one row per pair, in dict-insertion order.
Point estimates use the group1-minus-group2 convention (matching scipy Tukey).

Only stdlib + numpy + scipy + statsmodels + pandas (L3 allowlist). No I/O.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, studentized_range, tukey_hsd as _scipy_tukey
from statsmodels.stats.multitest import multipletests


def _pairs(names):
    return list(combinations(range(len(names)), 2))


def tukey_hsd(groups: dict, alpha: float = 0.05) -> pd.DataFrame:
    """Tukey HSD (equal variances). Wraps scipy.stats.tukey_hsd; meandiff and CI
    follow scipy's mean_i - mean_j convention."""
    names = list(groups)
    res = _scipy_tukey(*[np.asarray(groups[n], float) for n in names])
    ci = res.confidence_interval(1 - alpha)
    rows = []
    for i, j in _pairs(names):
        rows.append({
            "group1": names[i], "group2": names[j],
            "meandiff": float(res.statistic[i, j]),
            "ci_low": float(ci.low[i, j]), "ci_high": float(ci.high[i, j]),
            "p": float(res.pvalue[i, j]),
        })
    return pd.DataFrame(rows, columns=["group1", "group2", "meandiff", "ci_low", "ci_high", "p"])


def games_howell(groups: dict, alpha: float = 0.05) -> pd.DataFrame:
    """Games-Howell (unequal variances). For each pair uses the studentized
    range with the pair's own Welch-Satterthwaite df:
        SE = sqrt( (s_i^2/n_i + s_j^2/n_j) / 2 )
        q  = |mean_i - mean_j| / SE
        df = (s_i^2/n_i + s_j^2/n_j)^2 /
             ( (s_i^2/n_i)^2/(n_i-1) + (s_j^2/n_j)^2/(n_j-1) )
        p  = studentized_range.sf(q, k, df)
        CI = (mean_i - mean_j) +/- studentized_range.ppf(1-alpha, k, df) * SE
    For k=2 this reduces exactly to Welch's two-sided t-test."""
    names = list(groups)
    k = len(names)
    stat = {n: np.asarray(groups[n], float) for n in names}
    m = {n: stat[n].mean() for n in names}
    v = {n: stat[n].var(ddof=1) for n in names}
    nn = {n: len(stat[n]) for n in names}

    rows = []
    for i, j in _pairs(names):
        a, b = names[i], names[j]
        va_n, vb_n = v[a] / nn[a], v[b] / nn[b]
        diff = m[a] - m[b]
        se = np.sqrt((va_n + vb_n) / 2.0)
        q = abs(diff) / se
        df = (va_n + vb_n) ** 2 / (va_n ** 2 / (nn[a] - 1) + vb_n ** 2 / (nn[b] - 1))
        crit = studentized_range.ppf(1 - alpha, k, df)
        rows.append({
            "group1": a, "group2": b, "meandiff": float(diff), "se": float(se),
            "statistic": float(q), "df": float(df),
            "ci_low": float(diff - crit * se), "ci_high": float(diff + crit * se),
            "p": float(studentized_range.sf(q, k, df)),
        })
    return pd.DataFrame(
        rows,
        columns=["group1", "group2", "meandiff", "se", "statistic", "df", "ci_low", "ci_high", "p"],
    )


def dunn(groups: dict) -> pd.DataFrame:
    """Dunn's test (post-hoc for Kruskal-Wallis). Ranks ALL observations
    together (mid-ranks for ties), then for each pair:
        z = (Rbar_i - Rbar_j) / sqrt( sigma2 * (1/n_i + 1/n_j) )
        sigma2 = N(N+1)/12 - sum(t^3 - t) / (12*(N-1))     [tie correction]
        p = 2 * norm.sf(|z|)
    Adjusted with Holm (statsmodels multipletests, method='holm')."""
    names = list(groups)
    arrs = [np.asarray(groups[n], float) for n in names]
    allv = np.concatenate(arrs)
    ranks = rankdata(allv)
    N = len(allv)

    _, counts = np.unique(allv, return_counts=True)
    tie = float(np.sum(counts ** 3 - counts))
    sigma2 = N * (N + 1) / 12.0 - tie / (12.0 * (N - 1))

    mean_rank, size, start = {}, {}, 0
    for n, a in zip(names, arrs):
        mean_rank[n] = ranks[start:start + len(a)].mean()
        size[n] = len(a)
        start += len(a)

    rows = []
    for i, j in _pairs(names):
        a, b = names[i], names[j]
        se = np.sqrt(sigma2 * (1.0 / size[a] + 1.0 / size[b]))
        z = (mean_rank[a] - mean_rank[b]) / se
        rows.append({"group1": a, "group2": b, "z": float(z), "p": float(2 * norm.sf(abs(z)))})

    df = pd.DataFrame(rows, columns=["group1", "group2", "z", "p"])
    df["p_holm"] = multipletests(df["p"].to_numpy(), method="holm")[1]
    return df
