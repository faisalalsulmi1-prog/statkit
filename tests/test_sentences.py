"""Golden strings for the plain-English layer (PLAN §6, Chunk 12).

RED-first: every ``Result`` here is built BY HAND (no stats run) for a specific
(test_id × branch) cell, and asserted against an EXACT string that reads well to
a non-statistician. The expected strings are the authored spec — the source of
truth — not recomputed from the code (PLAN §6 golden discipline; the banned-verb
list is only a SMOKE check, these literals are the real guard).

Two guards ride on top of the goldens (collected in ``GOLDENS``):
  * banned-verb smoke over EVERY sentence (ok + blocked);
  * every OK sentence carries ``n =`` and its exact ``test_name``; every OK
    sentence WITH a p-value carries an APA p-token.
"""
from __future__ import annotations

import re

import pandas as pd
import pytest

from statkit import sentences
from statkit.model import Check, Finding, Result

ALPHA = "two-sided, α = .05"

# Every (Result, expected sentence) built below is appended here so the property
# guards run over the whole corpus (PLAN §6).
GOLDENS: list[tuple[Result, str]] = []


def _g(r: Result, expected: str) -> None:
    GOLDENS.append((r, expected))


def _say(r: Result) -> str:
    return sentences.render(r)


# ==========================================================================
# F-2G — t_1s / t_ind / t_paired
# ==========================================================================
def test_t_1s_significant():
    r = Result(
        test_id="t_1s", test_name="One-sample t-test", status="ok",
        statistic=("t", 2.45), df=(19.0,), p=0.024,
        estimate=("mean − μ₀", 3.2), estimate_ci=(0.5, 5.9),
        effect=("Cohen's d", 0.45), effect_label="small",
        effect_source="Cohen (1988)",
        n={"total": 20, "used": 20, "dropped": 0},
        labels={"outcome": "Score"})
    expected = (
        "One-sample t-test found a statistically significant difference "
        "between the mean of Score and the hypothesised value μ₀ (t(19) = "
        "2.45, p = .024; two-sided, α = .05). The mean − μ₀ was 3.20, 95% CI "
        "[0.50, 5.90]. The effect was small (Cohen's d = 0.45; per Cohen "
        "(1988)). 20 cases were analysed (n = 20); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_t_1s_not_significant_low_n_caveat():
    r = Result(
        test_id="t_1s", test_name="One-sample t-test", status="ok",
        statistic=("t", 1.10), df=(11.0,), p=0.295,
        estimate=("mean − μ₀", 1.4), estimate_ci=(-1.4, 4.2),
        effect=("Cohen's d", 0.32), effect_label="small",
        effect_source="Cohen (1988)",
        n={"total": 14, "used": 12, "dropped": 2},
        labels={"outcome": "Score"})
    expected = (
        "One-sample t-test did not find a statistically significant "
        "difference between the mean of Score and the hypothesised value μ₀ "
        "(t(11) = 1.10, p = .295; two-sided, α = .05). The mean − μ₀ was "
        "1.40, 95% CI [-1.40, 4.20]. The effect was small (Cohen's d = 0.32; "
        "per Cohen (1988)). Absence of evidence is not evidence of absence. "
        "12 cases were analysed (n = 12); 2 of 14 rows were excluded (missing "
        "data). With n = 12 (below 15) the test has low statistical power, so "
        "a non-significant result is not evidence that there is no effect.")
    assert _say(r) == expected
    _g(r, expected)


def test_t_ind_significant_welch_direction():
    r = Result(
        test_id="t_ind", test_name="Welch's independent-samples t-test",
        status="ok", statistic=("t", 3.10), df=(17.34,), p=0.006,
        estimate=("mean difference (A − B)", 4.5), estimate_ci=(1.5, 7.5),
        effect=("Hedges' g", 1.20), effect_ci=(0.4, 2.0), effect_label="large",
        effect_source="Cohen (1988)",
        n={"total": 12, "used": 12, "dropped": 0, "A": 6, "B": 6},
        groups=("A", "B"), higher="A", labels={"outcome": "Yield"})
    expected = (
        "Welch's independent-samples t-test found a statistically significant "
        "difference in Yield between A and B (t(17.34) = 3.10, p = .006; "
        "two-sided, α = .05). Values were higher in A; the mean difference (A "
        "− B) was 4.50, 95% CI [1.50, 7.50]. The effect was large (Hedges' g "
        "= 1.20, 95% CI [0.40, 2.00]; per Cohen (1988)). 12 cases were "
        "analysed (n = 12); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_t_ind_not_significant():
    r = Result(
        test_id="t_ind", test_name="Welch's independent-samples t-test",
        status="ok", statistic=("t", 0.80), df=(40.0,), p=0.428,
        estimate=("mean difference (A − B)", 1.1), estimate_ci=(-1.6, 3.8),
        effect=("Cohen's d", 0.24), effect_ci=(-0.35, 0.83), effect_label="small",
        effect_source="Cohen (1988)",
        n={"total": 42, "used": 42, "dropped": 0, "A": 21, "B": 21},
        groups=("A", "B"), higher="A", labels={"outcome": "Yield"})
    expected = (
        "Welch's independent-samples t-test did not find a statistically "
        "significant difference in Yield between A and B (t(40) = 0.80, p = "
        ".428; two-sided, α = .05). The mean difference (A − B) was 1.10, 95% "
        "CI [-1.60, 3.80]. The effect was small (Cohen's d = 0.24, 95% CI "
        "[-0.35, 0.83]; per Cohen (1988)). Absence of evidence is not "
        "evidence of absence. 42 cases were analysed (n = 42); no rows were "
        "excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_t_ind_huge_n_negligible_effect():
    r = Result(
        test_id="t_ind", test_name="Welch's independent-samples t-test",
        status="ok", statistic=("t", 2.30), df=(1998.0,), p=0.021,
        estimate=("mean difference (A − B)", 0.15), estimate_ci=(0.02, 0.28),
        effect=("Cohen's d", 0.10), effect_ci=(0.01, 0.19), effect_label="negligible",
        effect_source="Cohen (1988)",
        n={"total": 2000, "used": 2000, "dropped": 0, "A": 1000, "B": 1000},
        groups=("A", "B"), higher="A", labels={"outcome": "Yield"})
    expected = (
        "Welch's independent-samples t-test found a statistically significant "
        "difference in Yield between A and B (t(1998) = 2.30, p = .021; "
        "two-sided, α = .05). Values were higher in A; the mean difference (A "
        "− B) was 0.15, 95% CI [0.02, 0.28]. The effect was negligible "
        "(Cohen's d = 0.10, 95% CI [0.01, 0.19]; per Cohen (1988)). Although "
        "the sample is large (n = 2000), the effect is negligible, so the "
        "difference is unlikely to be practically important. 2000 cases were "
        "analysed (n = 2000); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_t_paired_significant():
    r = Result(
        test_id="t_paired", test_name="Paired-samples t-test", status="ok",
        statistic=("t", 4.02), df=(29.0,), p=0.0004,
        estimate=("mean difference (Pre − Post)", 5.6), estimate_ci=(2.8, 8.4),
        effect=("Cohen's dz", 0.73), effect_label="medium",
        effect_source="Cohen (1988)",
        n={"total": 30, "used": 30, "dropped": 0, "pairs": 30},
        labels={"before": "Pre", "after": "Post"}, higher="Pre")
    expected = (
        "Paired-samples t-test found a statistically significant difference "
        "between Pre and Post (t(29) = 4.02, p < .001; two-sided, α = .05). "
        "Values were higher for Pre; the mean difference (Pre − Post) was "
        "5.60, 95% CI [2.80, 8.40]. The effect was medium (Cohen's dz = 0.73; "
        "per Cohen (1988)). 30 cases were analysed (n = 30); no rows were "
        "excluded.")
    assert _say(r) == expected
    _g(r, expected)


# ==========================================================================
# F-KG — anova_1w / anova_2w / rm_anova / kruskal / friedman
# ==========================================================================
def test_anova_1w_significant_with_tukey_pairs():
    ph = pd.DataFrame(
        [{"group1": "A", "group2": "B", "meandiff": 1.0, "ci_low": -1.0,
          "ci_high": 3.0, "p": 0.412},
         {"group1": "A", "group2": "C", "meandiff": 5.0, "ci_low": 3.0,
          "ci_high": 7.0, "p": 0.001},
         {"group1": "B", "group2": "C", "meandiff": 4.0, "ci_low": 2.0,
          "ci_high": 6.0, "p": 0.003}])
    r = Result(
        test_id="anova_1w", test_name="One-way ANOVA", status="ok",
        statistic=("F", 9.87), df=(2.0, 15.0), p=0.002,
        effect=("η²", 0.57), effect_label="large", effect_source="Cohen (1988)",
        n={"total": 18, "used": 18, "dropped": 0}, groups=("A", "B", "C"),
        labels={"outcome": "Yield"}, posthoc=ph, posthoc_name="Tukey HSD")
    expected = (
        "One-way ANOVA found a statistically significant difference among the "
        "groups (at least one group differs) (F(2, 15) = 9.87, p = .002; "
        "two-sided, α = .05). The effect was large (η² = 0.57; per Cohen "
        "(1988)). Post-hoc comparisons (Tukey HSD) showed significant "
        "differences between A and C (p = .001); B and C (p = .003). 18 cases "
        "were analysed (n = 18); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_anova_1w_not_significant():
    r = Result(
        test_id="anova_1w", test_name="One-way ANOVA", status="ok",
        statistic=("F", 1.20), df=(2.0, 27.0), p=0.316,
        effect=("η²", 0.08), effect_label="medium", effect_source="Cohen (1988)",
        n={"total": 30, "used": 30, "dropped": 0}, groups=("A", "B", "C"),
        labels={"outcome": "Yield"})
    expected = (
        "One-way ANOVA did not find a statistically significant difference "
        "among the groups (F(2, 27) = 1.20, p = .316; two-sided, α = .05). "
        "The effect was medium (η² = 0.08; per Cohen (1988)). Absence of "
        "evidence is not evidence of absence. 30 cases were analysed (n = "
        "30); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_anova_1w_significant_no_pair_survives():
    ph = pd.DataFrame(
        [{"group1": "A", "group2": "B", "meandiff": 1.0, "ci_low": -0.1,
          "ci_high": 2.1, "p": 0.061},
         {"group1": "A", "group2": "C", "meandiff": 1.2, "ci_low": -0.1,
          "ci_high": 2.5, "p": 0.070},
         {"group1": "B", "group2": "C", "meandiff": 0.2, "ci_low": -1.0,
          "ci_high": 1.4, "p": 0.900}])
    r = Result(
        test_id="anova_1w", test_name="Welch's ANOVA", status="ok",
        statistic=("F", 3.55), df=(2.0, 18.42), p=0.049,
        effect=("η²", 0.15), effect_label="large", effect_source="Cohen (1988)",
        n={"total": 24, "used": 24, "dropped": 0}, groups=("A", "B", "C"),
        labels={"outcome": "Yield"}, posthoc=ph, posthoc_name="Games-Howell")
    expected = (
        "Welch's ANOVA found a statistically significant difference among the "
        "groups (at least one group differs) (F(2, 18.42) = 3.55, p = .049; "
        "two-sided, α = .05). The effect was large (η² = 0.15; per Cohen "
        "(1988)). Post-hoc comparisons (Games-Howell) found no pairwise "
        "differences after adjustment. 24 cases were analysed (n = 24); no "
        "rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_kruskal_significant_with_dunn():
    ph = pd.DataFrame(
        [{"group1": "Low", "group2": "Mid", "z": 1.2, "p": 0.230, "p_holm": 0.230},
         {"group1": "Low", "group2": "High", "z": 3.4, "p": 0.0007, "p_holm": 0.002},
         {"group1": "Mid", "group2": "High", "z": 2.1, "p": 0.036, "p_holm": 0.071}])
    r = Result(
        test_id="kruskal", test_name="Kruskal-Wallis H test", status="ok",
        statistic=("H", 12.4), df=(2.0,), p=0.002,
        effect=("eta-squared (H)", 0.28), effect_label="small",
        effect_source="Tomczak & Tomczak (2014)",
        n={"total": 45, "used": 45, "dropped": 0}, groups=("Low", "Mid", "High"),
        posthoc=ph, posthoc_name="Dunn's test (Holm-adjusted)")
    expected = (
        "Kruskal-Wallis H test found a statistically significant difference "
        "among the groups (at least one group differs) (H(2) = 12.40, p = "
        ".002; two-sided, α = .05). The effect was small (eta-squared (H) = "
        "0.28; per Tomczak & Tomczak (2014)). Post-hoc comparisons (Dunn's "
        "test (Holm-adjusted)) showed significant differences between Low and "
        "High (p = .002). 45 cases were analysed (n = 45); no rows were "
        "excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_friedman_not_significant():
    r = Result(
        test_id="friedman", test_name="Friedman test", status="ok",
        statistic=("χ²", 2.10), df=(2.0,), p=0.350,
        effect=("Kendall's W", 0.05), effect_label="negligible",
        effect_source="Tomczak & Tomczak (2014)",
        n={"total": 21, "used": 21, "dropped": 0, "subjects": 21, "measures": 3})
    expected = (
        "Friedman test did not find a statistically significant difference "
        "across the repeated measurements (χ²(2) = 2.10, p = .350; two-sided, "
        "α = .05). The effect was negligible (Kendall's W = 0.05; per Tomczak "
        "& Tomczak (2014)). Absence of evidence is not evidence of absence. "
        "21 cases were analysed (n = 21); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_rm_anova_significant_with_pairs_and_sphericity_note():
    ph = pd.DataFrame(
        [{"group1": "T1", "group2": "T2", "statistic": 2.1, "p": 0.05, "p_holm": 0.05},
         {"group1": "T1", "group2": "T3", "statistic": 4.2, "p": 0.001, "p_holm": 0.003},
         {"group1": "T2", "group2": "T3", "statistic": 3.0, "p": 0.008, "p_holm": 0.016}])
    r = Result(
        test_id="rm_anova", test_name="Repeated-measures ANOVA", status="ok",
        statistic=("F", 15.6), df=(2.0, 38.0), p=0.0004,
        effect=("partial η²", 0.45), effect_label="large",
        effect_source="Cohen (1988)",
        n={"total": 20, "used": 20, "dropped": 0, "subjects": 20},
        groups=("T1", "T2", "T3"), posthoc=ph,
        posthoc_name="pairwise paired t (Holm)",
        variant_notes=("Sphericity was not tested (no Greenhouse–Geisser / "
                       "Huynh–Feldt correction in the API).",))
    expected = (
        "Repeated-measures ANOVA found a statistically significant difference "
        "across the repeated measurements (at least one differs) (F(2, 38) = "
        "15.60, p < .001; two-sided, α = .05). The effect was large (partial "
        "η² = 0.45; per Cohen (1988)). Post-hoc comparisons (pairwise paired "
        "t (Holm)) showed significant differences between T1 and T3 (p = "
        ".003); T2 and T3 (p = .016). Sphericity was not tested (no "
        "Greenhouse–Geisser / Huynh–Feldt correction in the API). 20 cases "
        "were analysed (n = 20); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_anova_2w_interaction_significant():
    table = pd.DataFrame(
        [{"label": "Fertiliser", "sum_sq": 40.0, "df": 1.0, "F": 8.0, "p": 0.008,
          "partial_eta_sq": 0.25},
         {"label": "Water", "sum_sq": 10.0, "df": 1.0, "F": 2.0, "p": 0.170,
          "partial_eta_sq": 0.08},
         {"label": "Fertiliser × Water", "sum_sq": 30.0, "df": 1.0, "F": 6.0,
          "p": 0.020, "partial_eta_sq": 0.20},
         {"label": "Residual", "sum_sq": 120.0, "df": 24.0, "F": float("nan"),
          "p": float("nan"), "partial_eta_sq": float("nan")}],
        index=["factor_a", "factor_b", "interaction", "Residual"])
    r = Result(
        test_id="anova_2w", test_name="Two-way ANOVA (Type II)", status="ok",
        statistic=("F", 6.0), df=(1.0, 24.0), p=0.020,
        effect=("partial η²", 0.20), effect_label="large",
        effect_source="Cohen (1988)",
        n={"total": 28, "used": 28, "dropped": 0}, table=table,
        labels={"outcome": "Yield", "factor_a": "Fertiliser", "factor_b": "Water",
                "interaction": "Fertiliser × Water"},
        variant_notes=("The interaction is significant; Type III sums of "
                       "squares (Sum contrasts) are also reported.",))
    expected = (
        "Two-way ANOVA (Type II): the interaction between Fertiliser and "
        "Water was statistically significant (F(1, 24) = 6.00, p = .020; "
        "two-sided, α = .05), partial η² = 0.20 (large). The main effect of "
        "Fertiliser was statistically significant (F(1, 24) = 8.00, p = "
        ".008). The main effect of Water was not statistically significant "
        "(F(1, 24) = 2.00, p = .170). Because the interaction is "
        "statistically significant, interpret it (see the interaction plot) "
        "rather than the main effects; Type III sums of squares (Sum "
        "contrasts) are also reported. 28 cases were analysed (n = 28); no "
        "rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_anova_2w_interaction_not_significant():
    table = pd.DataFrame(
        [{"label": "Fertiliser", "sum_sq": 40.0, "df": 1.0, "F": 8.0, "p": 0.008,
          "partial_eta_sq": 0.25},
         {"label": "Water", "sum_sq": 10.0, "df": 1.0, "F": 2.0, "p": 0.170,
          "partial_eta_sq": 0.08},
         {"label": "Fertiliser × Water", "sum_sq": 3.0, "df": 1.0, "F": 0.6,
          "p": 0.446, "partial_eta_sq": 0.02},
         {"label": "Residual", "sum_sq": 120.0, "df": 24.0, "F": float("nan"),
          "p": float("nan"), "partial_eta_sq": float("nan")}],
        index=["factor_a", "factor_b", "interaction", "Residual"])
    r = Result(
        test_id="anova_2w", test_name="Two-way ANOVA (Type II)", status="ok",
        statistic=("F", 0.6), df=(1.0, 24.0), p=0.446,
        effect=("partial η²", 0.02), effect_label="small",
        effect_source="Cohen (1988)",
        n={"total": 28, "used": 28, "dropped": 0}, table=table,
        labels={"outcome": "Yield", "factor_a": "Fertiliser", "factor_b": "Water",
                "interaction": "Fertiliser × Water"})
    expected = (
        "Two-way ANOVA (Type II): the interaction between Fertiliser and "
        "Water was not statistically significant (F(1, 24) = 0.60, p = .446; "
        "two-sided, α = .05), partial η² = 0.02 (small). The main effect of "
        "Fertiliser was statistically significant (F(1, 24) = 8.00, p = "
        ".008). The main effect of Water was not statistically significant "
        "(F(1, 24) = 2.00, p = .170). 28 cases were analysed (n = 28); no "
        "rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


# ==========================================================================
# F-RANK2 — mwu / wilcoxon
# ==========================================================================
def test_mwu_significant_exact():
    r = Result(
        test_id="mwu", test_name="Mann-Whitney U test", status="ok",
        statistic=("U", 4.0), p=0.032,
        estimate=("Hodges-Lehmann median difference", 3.5),
        effect=("rank-biserial correlation", 0.72), effect_label="large",
        effect_source="Cohen (1988)",
        n={"total": 12, "used": 12, "dropped": 0, "A": 6, "B": 6},
        groups=("A", "B"), higher="A", variant_notes=("exact method",))
    expected = (
        "Mann-Whitney U test found a statistically significant difference "
        "between A and B (U = 4.00, p = .032; two-sided, α = .05). Values "
        "tended to be higher in A. The Hodges-Lehmann median difference was "
        "3.50. Method: exact method. The effect was large (rank-biserial "
        "correlation = 0.72; per Cohen (1988)). 12 cases were analysed (n = "
        "12); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_mwu_not_significant_asymptotic_ties():
    r = Result(
        test_id="mwu", test_name="Mann-Whitney U test", status="ok",
        statistic=("U", 180.0), p=0.610,
        estimate=("Hodges-Lehmann median difference", 0.5),
        effect=("rank-biserial correlation", 0.08), effect_label="negligible",
        effect_source="Cohen (1988)",
        n={"total": 40, "used": 40, "dropped": 0, "A": 20, "B": 20},
        groups=("A", "B"), higher="A",
        variant_notes=("asymptotic method; ties present, tie-corrected",))
    expected = (
        "Mann-Whitney U test did not find a statistically significant "
        "difference between A and B (U = 180.00, p = .610; two-sided, α = "
        ".05). The Hodges-Lehmann median difference was 0.50. Method: "
        "asymptotic method; ties present, tie-corrected. The effect was "
        "negligible (rank-biserial correlation = 0.08; per Cohen (1988)). "
        "Absence of evidence is not evidence of absence. 40 cases were "
        "analysed (n = 40); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_wilcoxon_significant_with_zero_diffs():
    r = Result(
        test_id="wilcoxon", test_name="Wilcoxon signed-rank test", status="ok",
        statistic=("W", 12.0), p=0.014,
        estimate=("median of differences", 2.0),
        effect=("matched-pairs rank-biserial correlation", 0.68),
        effect_label="large", effect_source="Cohen (1988)",
        n={"total": 18, "used": 18, "dropped": 0, "pairs": 18, "nonzero_pairs": 16},
        variant_notes=("2 zero difference(s) excluded (zero_method='wilcox')",))
    expected = (
        "Wilcoxon signed-rank test found a statistically significant "
        "difference between the paired measurements (W = 12.00, p = .014; "
        "two-sided, α = .05). The median of differences was 2.00. 2 zero "
        "difference(s) excluded (zero_method='wilcox'). The effect was large "
        "(matched-pairs rank-biserial correlation = 0.68; per Cohen (1988)). "
        "18 cases were analysed (n = 18); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_wilcoxon_one_sample_not_significant():
    r = Result(
        test_id="wilcoxon", test_name="Wilcoxon signed-rank test (one-sample)",
        status="ok", statistic=("W", 40.0), p=0.220,
        estimate=("median difference from μ₀", -1.0),
        effect=("matched-pairs rank-biserial correlation", 0.20),
        effect_label="small", effect_source="Cohen (1988)",
        n={"total": 14, "used": 14, "dropped": 0, "pairs": 14, "nonzero_pairs": 14})
    expected = (
        "Wilcoxon signed-rank test (one-sample) did not find a statistically "
        "significant difference from the hypothesised median (W = 40.00, p = "
        ".220; two-sided, α = .05). The median difference from μ₀ was -1.00. "
        "The effect was small (matched-pairs rank-biserial correlation = "
        "0.20; per Cohen (1988)). Absence of evidence is not evidence of "
        "absence. 14 cases were analysed (n = 14); no rows were excluded. "
        "With n = 14 (below 15) the test has low statistical power, so a "
        "non-significant result is not evidence that there is no effect.")
    assert _say(r) == expected
    _g(r, expected)


# ==========================================================================
# F-ASSOC — pearson / spearman / kendall / corr_matrix
# ==========================================================================
def test_pearson_significant_positive():
    r = Result(
        test_id="pearson", test_name="Pearson correlation", status="ok",
        statistic=("r", 0.82), p=0.0001,
        effect=("r", 0.82), effect_ci=(0.55, 0.94), effect_label="large",
        effect_source="Cohen (1988)",
        n={"total": 20, "used": 20, "dropped": 0},
        labels={"x": "Height", "y": "Weight"}, extra={"r2": 0.6724})
    expected = (
        "Pearson correlation found a statistically significant positive "
        "correlation between Height and Weight (r = 0.82, p < .001; "
        "two-sided, α = .05). The correlation was large (r = 0.82, 95% CI "
        "[0.55, 0.94]; r² = 0.67). 20 cases were analysed (n = 20); no rows "
        "were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_spearman_significant_negative():
    r = Result(
        test_id="spearman", test_name="Spearman rank correlation", status="ok",
        statistic=("rho", -0.45), p=0.030,
        effect=("rho", -0.45), effect_ci=(-0.72, -0.05), effect_label="medium",
        effect_source="Cohen (1988)",
        n={"total": 24, "used": 24, "dropped": 0},
        labels={"x": "Rank", "y": "Errors"}, extra={"r2": 0.2025})
    expected = (
        "Spearman rank correlation found a statistically significant negative "
        "correlation between Rank and Errors (rho = -0.45, p = .030; "
        "two-sided, α = .05). The correlation was medium (rho = -0.45, 95% CI "
        "[-0.72, -0.05]). 24 cases were analysed (n = 24); no rows were "
        "excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_kendall_not_significant_with_ordinal_flag():
    r = Result(
        test_id="kendall", test_name="Kendall's τ-b", status="ok",
        statistic=("tau-b", 0.12), p=0.310,
        effect=("tau-b", 0.12), effect_ci=(-0.11, 0.34), effect_label="small",
        effect_source="Cohen (1988)",
        n={"total": 22, "used": 22, "dropped": 0},
        labels={"x": "Satisfaction", "y": "Visits"}, extra={"r2": 0.0144},
        findings=(Finding("flag", "One variable is ordinal.",
                          suggest_test="spearman", code="S10"),))
    expected = (
        "Kendall's τ-b did not find a statistically significant correlation "
        "between Satisfaction and Visits (tau-b = 0.12, p = .310; two-sided, "
        "α = .05). The correlation was small (tau-b = 0.12, 95% CI [-0.11, "
        "0.34]). An ordinal variable is involved; Spearman's rank correlation "
        "may be more appropriate. Absence of evidence is not evidence of "
        "absence. 22 cases were analysed (n = 22); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_corr_matrix_summary():
    idx = ["A", "B", "C"]
    R = pd.DataFrame([[1.0, 0.6, 0.1], [0.6, 1.0, 0.2], [0.1, 0.2, 1.0]],
                     index=idx, columns=idx)
    ph = pd.DataFrame([[1.0, 0.004, 0.700], [0.004, 1.0, 0.500],
                       [0.700, 0.500, 1.0]], index=idx, columns=idx)
    r = Result(
        test_id="corr_matrix", test_name="Correlation matrix", status="ok",
        n={"total": 30, "used": 30, "dropped": 0}, table=R,
        extra={"p_holm": ph, "method": "pearson"})
    expected = (
        "Correlation matrix: pearson correlations were computed among 3 "
        "variables. 1 of 3 pairwise correlations was statistically "
        "significant after Holm adjustment (two-sided, α = .05). See the "
        "correlation matrix and heatmap. 30 cases were analysed (n = 30); no "
        "rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


# ==========================================================================
# F-CAT — chi2_ind / chi2_gof / fisher / mcnemar / cochran_q / prop_1 / prop_2
# ==========================================================================
def test_chi2_ind_2x2_significant_with_or():
    r = Result(
        test_id="chi2_ind", test_name="Chi-square test of independence",
        status="ok", statistic=("χ²", 6.51), df=(1.0,), p=0.011,
        estimate=("odds ratio", 3.40), estimate_ci=(1.32, 8.75),
        effect=("phi", 0.29), effect_label="small", effect_source="Cohen (1988)",
        n={"total": 78, "used": 78, "dropped": 0})
    expected = (
        "Chi-square test of independence found a statistically significant "
        "association between the two variables (χ²(1) = 6.51, p = .011; "
        "two-sided, α = .05). The odds ratio was 3.40, 95% CI [1.32, 8.75]. "
        "The effect was small (phi = 0.29; per Cohen (1988)). 78 cases were "
        "analysed (n = 78); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_chi2_ind_rxc_not_significant():
    r = Result(
        test_id="chi2_ind", test_name="Chi-square test of independence",
        status="ok", statistic=("χ²", 5.20), df=(4.0,), p=0.267,
        effect=("Cramér's V", 0.11), effect_label="small",
        effect_source="Cohen (1988)",
        n={"total": 120, "used": 120, "dropped": 0})
    expected = (
        "Chi-square test of independence did not find a statistically "
        "significant association between the two variables (χ²(4) = 5.20, p = "
        ".267; two-sided, α = .05). The effect was small (Cramér's V = 0.11; "
        "per Cohen (1988)). Absence of evidence is not evidence of absence. "
        "120 cases were analysed (n = 120); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_chi2_gof_significant():
    r = Result(
        test_id="chi2_gof", test_name="Chi-square goodness-of-fit test",
        status="ok", statistic=("χ²", 11.0), df=(3.0,), p=0.012,
        effect=("Cohen's w", 0.33), effect_label="medium",
        effect_source="Cohen (1988)",
        n={"total": 100, "used": 100, "dropped": 0})
    expected = (
        "Chi-square goodness-of-fit test found a statistically significant "
        "difference between the observed counts and the expected proportions "
        "(χ²(3) = 11.00, p = .012; two-sided, α = .05). The effect was medium "
        "(Cohen's w = 0.33; per Cohen (1988)). 100 cases were analysed (n = "
        "100); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_fisher_2x2_significant_with_or():
    r = Result(
        test_id="fisher", test_name="Fisher's exact test", status="ok",
        statistic=("odds ratio (sample)", 7.0), p=0.028,
        estimate=("odds ratio", 6.20), estimate_ci=(1.10, 51.3),
        effect=("phi", 0.42), effect_label="medium", effect_source="Cohen (1988)",
        n={"total": 22, "used": 22, "dropped": 0})
    expected = (
        "Fisher's exact test found a statistically significant association "
        "between the two variables (p = .028; two-sided, α = .05). The odds "
        "ratio was 6.20, 95% CI [1.10, 51.30]. The effect was medium (phi = "
        "0.42; per Cohen (1988)). 22 cases were analysed (n = 22); no rows "
        "were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_fisher_rxc_not_significant():
    r = Result(
        test_id="fisher", test_name="Fisher-Freeman-Halton test (Monte Carlo p)",
        status="ok", statistic=("table probability", 0.03), p=0.140,
        n={"total": 30, "used": 30, "dropped": 0},
        variant_notes=("Monte Carlo estimate (99,999 resamples, seeded for "
                       "reproducibility)",))
    expected = (
        "Fisher-Freeman-Halton test (Monte Carlo p) did not find a "
        "statistically significant association between the two variables (p = "
        ".140; two-sided, α = .05). Monte Carlo estimate (99,999 resamples, "
        "seeded for reproducibility). Absence of evidence is not evidence of "
        "absence. 30 cases were analysed (n = 30); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_mcnemar_significant_with_or():
    r = Result(
        test_id="mcnemar", test_name="McNemar's test (exact)", status="ok",
        statistic=("smaller discordant count", 3.0), p=0.021,
        estimate=("odds ratio", 4.33), estimate_ci=(1.24, 15.12),
        n={"total": 50, "used": 50, "dropped": 0, "discordant": 16},
        variant_notes=("16 of 50 changed (32.0%)",))
    expected = (
        "McNemar's test (exact) found a statistically significant change "
        "between the two measurements (p = .021; two-sided, α = .05). 16 of "
        "50 changed (32.0%). The odds ratio was 4.33, 95% CI [1.24, 15.12]. "
        "50 cases were analysed (n = 50); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_mcnemar_not_significant_or_not_estimable():
    r = Result(
        test_id="mcnemar", test_name="McNemar's test (exact)", status="ok",
        statistic=("smaller discordant count", 0.0), p=0.250,
        estimate=None, estimate_ci=None,
        n={"total": 40, "used": 40, "dropped": 0, "discordant": 3},
        variant_notes=("3 of 40 changed (7.5%)",
                       "odds ratio not estimable (one direction of change has "
                       "zero cases)"))
    expected = (
        "McNemar's test (exact) did not find a statistically significant "
        "change between the two measurements (p = .250; two-sided, α = .05). "
        "3 of 40 changed (7.5%). The odds ratio was not estimable (one "
        "direction of change had no cases). Absence of evidence is not "
        "evidence of absence. 40 cases were analysed (n = 40); no rows were "
        "excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_cochran_q_significant_with_posthoc():
    ph = pd.DataFrame(
        [{"group1": "V1", "group2": "V2", "p": 0.02, "p_holm": 0.04},
         {"group1": "V1", "group2": "V3", "p": 0.20, "p_holm": 0.20},
         {"group1": "V2", "group2": "V3", "p": 0.005, "p_holm": 0.015}])
    r = Result(
        test_id="cochran_q", test_name="Cochran's Q test", status="ok",
        statistic=("Q", 9.5), df=(2.0,), p=0.009,
        n={"total": 30, "used": 30, "dropped": 0, "subjects": 30, "measures": 3},
        labels={"success": "Yes"},
        posthoc=ph, posthoc_name="pairwise McNemar (Holm-adjusted)")
    expected = (
        "Cochran's Q test found a statistically significant difference in the "
        "proportion of 'Yes' across the repeated conditions (Q(2) = 9.50, p = "
        ".009; two-sided, α = .05). Post-hoc comparisons (pairwise McNemar "
        "(Holm-adjusted)) showed significant differences between V1 and V2 (p "
        "= .040); V2 and V3 (p = .015). 30 cases were analysed (n = 30); no "
        "rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_prop_1_significant():
    r = Result(
        test_id="prop_1",
        test_name="One-sample proportion test (exact binomial)", status="ok",
        p=0.013, estimate=("proportion 'Yes'", 0.65), estimate_ci=(0.52, 0.76),
        effect=("Cohen's h", 0.31), effect_label="small",
        effect_source="Cohen (1988)",
        n={"total": 60, "used": 60, "dropped": 0, "successes": 39},
        labels={"success": "Yes"},
        variant_notes=("tested against p₀ = 0.5; 95% Wilson interval",))
    expected = (
        "One-sample proportion test (exact binomial) found a statistically "
        "significant difference between the proportion of 'Yes' and the "
        "hypothesised value (p = .013; two-sided, α = .05). The observed "
        "proportion of 'Yes' was 65.0%, 95% CI [52.0%, 76.0%]. The effect was "
        "small (Cohen's h = 0.31; per Cohen (1988)). 60 cases were analysed "
        "(n = 60); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_prop_2_significant_direction():
    r = Result(
        test_id="prop_2", test_name="Two-sample proportion z-test", status="ok",
        statistic=("z", 2.55), p=0.011,
        estimate=("difference in proportion 'Yes' (G1 − G2)", 0.18),
        estimate_ci=(0.04, 0.32),
        effect=("Cohen's h", 0.37), effect_label="small",
        effect_source="Cohen (1988)",
        n={"total": 120, "used": 120, "dropped": 0, "G1": 60, "G2": 60},
        groups=("G1", "G2"), higher="G1", labels={"success": "Yes"})
    expected = (
        "Two-sample proportion z-test found a statistically significant "
        "difference in the proportion of 'Yes' between G1 and G2 (z = 2.55, p "
        "= .011; two-sided, α = .05). The proportion of 'Yes' was higher in "
        "G1; the difference was 18.0%, 95% CI [4.0%, 32.0%]. The effect was "
        "small (Cohen's h = 0.37; per Cohen (1988)). 120 cases were analysed "
        "(n = 120); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


# ==========================================================================
# F-REG — ols_simple / ols_multi / logistic
# ==========================================================================
def test_ols_simple_significant():
    table = pd.DataFrame(
        {"coef": [2.0, 1.5], "ci_low": [0.5, 1.1], "ci_high": [3.5, 1.9],
         "p": [0.02, 0.0001]}, index=["Intercept", "Study hours"])
    r = Result(
        test_id="ols_simple", test_name="Simple linear regression", status="ok",
        statistic=("F", 60.0), df=(1.0, 28.0), p=0.0001,
        estimate=("slope", 1.5), estimate_ci=(1.1, 1.9),
        effect=("f2", 2.14), effect_label="large", effect_source="Cohen (1988)",
        n={"total": 30, "used": 30, "dropped": 0}, table=table,
        labels={"outcome": "Exam score", "x": "Study hours"},
        extra={"intercept": 2.0, "slope": 1.5, "r2": 0.6818, "adj_r2": 0.671})
    expected = (
        "Simple linear regression found that Study hours statistically "
        "significantly predicted Exam score (F(1, 28) = 60.00, p < .001; "
        "two-sided, α = .05). The model explained 68.2% of the variance in "
        "Exam score (R² = 0.68). Each one-unit increase in Study hours was "
        "associated with a change of 1.50 in Exam score (95% CI [1.10, "
        "1.90]). The effect was large (f2 = 2.14; per Cohen (1988)). 30 cases "
        "were analysed (n = 30); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_ols_multi_significant_with_reference():
    table = pd.DataFrame(
        {"coef": [1.0, 0.8, -2.0], "ci_low": [0.2, 0.5, -3.4],
         "ci_high": [1.8, 1.1, -0.6], "p": [0.02, 0.0001, 0.006]},
        index=["Intercept", "Study hours", "Region"])
    r = Result(
        test_id="ols_multi", test_name="Multiple linear regression",
        status="ok", statistic=("F", 25.0), df=(2.0, 47.0), p=0.0001,
        effect=("f2", 1.06), effect_label="large", effect_source="Cohen (1988)",
        n={"total": 50, "used": 50, "dropped": 0}, table=table,
        labels={"outcome": "Exam score"},
        extra={"r2": 0.5152, "adj_r2": 0.50, "vif": {},
               "reference": {"Region": "North"}})
    expected = (
        "Multiple linear regression found a statistically significant model "
        "(F(2, 47) = 25.00, p < .001; two-sided, α = .05). The model "
        "explained 51.5% of the variance (R² = 0.52, adjusted R² = 0.50). "
        "Coefficients — Study hours: b = 0.80, 95% CI [0.50, 1.10], p < .001; "
        "Region: b = -2.00, 95% CI [-3.40, -0.60], p = .006. Reference "
        "level(s): Region = 'North'. The effect was large (f2 = 1.06; per "
        "Cohen (1988)). 50 cases were analysed (n = 50); no rows were "
        "excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_logistic_significant_odds_ratios():
    table = pd.DataFrame(
        {"coef": [-1.0, 0.7], "OR": [0.37, 2.01], "OR_ci_low": [0.10, 1.35],
         "OR_ci_high": [1.30, 3.00], "p": [0.12, 0.0006]},
        index=["Intercept", "Age"])
    r = Result(
        test_id="logistic", test_name="Binary logistic regression", status="ok",
        statistic=("LLR chi2", 14.2), df=(1.0,), p=0.0002,
        n={"total": 90, "used": 90, "dropped": 0},
        labels={"outcome": "Outcome"}, table=table,
        extra={"or": {"Age": 2.01}, "pseudo_r2": 0.18, "llr_p": 0.0002,
               "success": "Yes", "reference": {}})
    expected = (
        "Binary logistic regression found a statistically significant model "
        "predicting 'Yes' in Outcome (LLR chi2(1) = 14.20, p < .001; "
        "two-sided, α = .05). Odds ratios — Age: each one-unit increase "
        "multiplied the odds of 'Yes' by 2.01 (95% CI [1.35, 3.00], p < "
        ".001). McFadden pseudo-R² = 0.18. 90 cases were analysed (n = 90); "
        "no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_logistic_blocked_separation():
    r = Result(
        test_id="logistic", test_name="Binary logistic regression",
        status="blocked", n={"total": 40, "used": 40, "dropped": 0},
        findings=(Finding(
            "block", "The model shows perfect (or near-perfect) separation: a "
            "predictor separates the outcome, so the odds ratios are not "
            "estimable. Remove or combine that predictor, or collect more "
            "varied data.", code="separation"),))
    expected = (
        "The model shows perfect (or near-perfect) separation: a predictor "
        "separates the outcome, so the odds ratios are not estimable. Remove or "
        "combine that predictor, or collect more varied data.")
    assert _say(r) == expected
    _g(r, expected)


# ==========================================================================
# F-CHK — normality / homogeneity
# ==========================================================================
def test_normality_pass_and_fail():
    checks = (
        Check("Shapiro-Wilk (Score)", 0.97, 0.61, True, ""),
        Check("Lilliefors (Score)", 0.08, 0.42, True, ""),
        Check("Shapiro-Wilk (Time)", 0.88, 0.003, False, ""),
        Check("Lilliefors (Time)", 0.15, 0.010, False, ""),
    )
    r = Result(
        test_id="normality", test_name="Normality check", status="ok",
        statistic=("W", 0.97), p=0.61, checks=checks,
        n={"total": 40, "used": 40, "dropped": 0})
    expected = (
        "Normality check assessed normality. Shapiro-Wilk (Score): statistic "
        "= 0.97, p = .610 — consistent with a normal distribution. Lilliefors "
        "(Score): statistic = 0.08, p = .420 — consistent with a normal "
        "distribution. Shapiro-Wilk (Time): statistic = 0.88, p = .003 — "
        "departs from a normal distribution. Lilliefors (Time): statistic = "
        "0.15, p = .010 — departs from a normal distribution. 40 cases were "
        "analysed (n = 40); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_normality_skipped():
    checks = (
        Check("Shapiro-Wilk", None, None, None, "n>5000: Shapiro not run"),
        Check("Lilliefors", 0.01, 0.20, True, ""),
    )
    r = Result(
        test_id="normality", test_name="Normality check", status="ok",
        p=None, checks=checks, n={"total": 6000, "used": 6000, "dropped": 0})
    expected = (
        "Normality check assessed normality. Shapiro-Wilk: not run (n>5000: "
        "Shapiro not run). Lilliefors: statistic = 0.01, p = .200 — "
        "consistent with a normal distribution. 6000 cases were analysed (n = "
        "6000); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_homogeneity_pass():
    r = Result(
        test_id="homogeneity", test_name="Equality of variances", status="ok",
        statistic=("W", 0.90), p=0.410,
        checks=(Check("Brown-Forsythe (Levene, center=median)", 0.90, 0.410,
                      True, ""),),
        n={"total": 45, "used": 45, "dropped": 0}, groups=("A", "B", "C"))
    expected = (
        "Equality of variances: across 3 groups, Brown-Forsythe (Levene, "
        "center=median) gave statistic = 0.90, p = .410 — no evidence that "
        "the variances differ. 45 cases were analysed (n = 45); no rows were "
        "excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_homogeneity_fail():
    r = Result(
        test_id="homogeneity", test_name="Equality of variances", status="ok",
        statistic=("W", 5.60), p=0.006,
        checks=(Check("Brown-Forsythe (Levene, center=median)", 5.60, 0.006,
                      False, ""),),
        n={"total": 60, "used": 60, "dropped": 0}, groups=("A", "B"))
    expected = (
        "Equality of variances: across 2 groups, Brown-Forsythe (Levene, "
        "center=median) gave statistic = 5.60, p = .006 — the variances "
        "differ. 60 cases were analysed (n = 60); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


# ==========================================================================
# describe (utility)
# ==========================================================================
def test_describe():
    r = Result(
        test_id="describe", test_name="Descriptive statistics (Table 1)",
        status="ok", n={"total": 50, "used": 48, "dropped": 2})
    expected = (
        "Descriptive statistics (Table 1) were computed for the selected "
        "columns. See the table for n, mean, SD, median, IQR and range "
        "(numeric) and counts or percentages (categorical). 48 cases were "
        "analysed (n = 48); 2 of 50 rows were excluded (missing data).")
    assert _say(r) == expected
    _g(r, expected)


# ==========================================================================
# blocked results (rule 18: the sentence IS the block reason)
# ==========================================================================
def test_blocked_generic_block_reason():
    r = Result(
        test_id="t_ind", test_name="Welch's independent-samples t-test",
        status="blocked", n={"total": 5, "used": 5, "dropped": 0},
        findings=(Finding("block", "The 'Group' column has 3 levels, but this "
                          "test compares exactly 2 groups.",
                          suggest_test="anova_1w", code="S2"),))
    expected = ("The 'Group' column has 3 levels, but this test compares "
                "exactly 2 groups.")
    assert _say(r) == expected
    _g(r, expected)


def test_blocked_chi2_routes_to_fisher():
    r = Result(
        test_id="chi2_ind", test_name="Chi-square test of independence",
        status="blocked", n={"total": 20, "used": 20, "dropped": 0},
        findings=(Finding("block", "Some expected counts are too small for a "
                          "reliable chi-square test; use Fisher's exact test "
                          "instead.", suggest_test="fisher", code="D3"),))
    expected = ("Some expected counts are too small for a reliable chi-square "
                "test; use Fisher's exact test instead.")
    assert _say(r) == expected
    _g(r, expected)


# ==========================================================================
# ADDITIONAL BRANCH CELLS (second sig/non-sig cell per test + sub-branches)
# ==========================================================================
def test_t_1s_significant_below_mu0():
    r = Result(
        test_id="t_1s", test_name="One-sample t-test", status="ok",
        statistic=("t", -3.10), df=(24.0,), p=0.005,
        estimate=("mean − μ₀", -2.4), estimate_ci=(-4.0, -0.8),
        effect=("Cohen's d", 0.62), effect_label="medium",
        effect_source="Cohen (1988)",
        n={"total": 25, "used": 25, "dropped": 0}, labels={"outcome": "Change"})
    expected = (
        "One-sample t-test found a statistically significant difference "
        "between the mean of Change and the hypothesised value μ₀ (t(24) = "
        "-3.10, p = .005; two-sided, α = .05). The mean − μ₀ was -2.40, 95% "
        "CI [-4.00, -0.80]. The effect was medium (Cohen's d = 0.62; per "
        "Cohen (1988)). 25 cases were analysed (n = 25); no rows were "
        "excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_t_ind_student_variant_significant():
    r = Result(
        test_id="t_ind", test_name="Student's independent-samples t-test",
        status="ok", statistic=("t", 2.90), df=(30.0,), p=0.007,
        estimate=("mean difference (Drug − Placebo)", 3.0), estimate_ci=(0.9, 5.1),
        effect=("Cohen's d", 0.95), effect_ci=(0.25, 1.65), effect_label="large",
        effect_source="Cohen (1988)",
        n={"total": 32, "used": 32, "dropped": 0, "Drug": 16, "Placebo": 16},
        groups=("Drug", "Placebo"), higher="Drug", labels={"outcome": "Pain"})
    expected = (
        "Student's independent-samples t-test found a statistically "
        "significant difference in Pain between Drug and Placebo (t(30) = "
        "2.90, p = .007; two-sided, α = .05). Values were higher in Drug; the "
        "mean difference (Drug − Placebo) was 3.00, 95% CI [0.90, 5.10]. The "
        "effect was large (Cohen's d = 0.95, 95% CI [0.25, 1.65]; per Cohen "
        "(1988)). 32 cases were analysed (n = 32); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_t_paired_not_significant():
    r = Result(
        test_id="t_paired", test_name="Paired-samples t-test", status="ok",
        statistic=("t", 1.20), df=(19.0,), p=0.245,
        estimate=("mean difference (Pre − Post)", 1.2), estimate_ci=(-0.9, 3.3),
        effect=("Cohen's dz", 0.27), effect_label="small",
        effect_source="Cohen (1988)",
        n={"total": 20, "used": 20, "dropped": 0, "pairs": 20},
        labels={"before": "Pre", "after": "Post"}, higher="Pre")
    expected = (
        "Paired-samples t-test did not find a statistically significant "
        "difference between Pre and Post (t(19) = 1.20, p = .245; two-sided, "
        "α = .05). The mean difference (Pre − Post) was 1.20, 95% CI [-0.90, "
        "3.30]. The effect was small (Cohen's dz = 0.27; per Cohen (1988)). "
        "Absence of evidence is not evidence of absence. 20 cases were "
        "analysed (n = 20); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_rm_anova_not_significant():
    r = Result(
        test_id="rm_anova", test_name="Repeated-measures ANOVA", status="ok",
        statistic=("F", 1.10), df=(2.0, 38.0), p=0.343,
        effect=("partial η²", 0.05), effect_label="small",
        effect_source="Cohen (1988)",
        n={"total": 20, "used": 20, "dropped": 0, "subjects": 20},
        groups=("T1", "T2", "T3"),
        variant_notes=("Sphericity was not tested (no Greenhouse–Geisser / "
                       "Huynh–Feldt correction in the API).",))
    expected = (
        "Repeated-measures ANOVA did not find a statistically significant "
        "difference across the repeated measurements (F(2, 38) = 1.10, p = "
        ".343; two-sided, α = .05). The effect was small (partial η² = 0.05; "
        "per Cohen (1988)). Sphericity was not tested (no Greenhouse–Geisser "
        "/ Huynh–Feldt correction in the API). Absence of evidence is not "
        "evidence of absence. 20 cases were analysed (n = 20); no rows were "
        "excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_kruskal_not_significant():
    r = Result(
        test_id="kruskal", test_name="Kruskal-Wallis H test", status="ok",
        statistic=("H", 3.20), df=(2.0,), p=0.202,
        effect=("eta-squared (H)", 0.03), effect_label="small",
        effect_source="Tomczak & Tomczak (2014)",
        n={"total": 45, "used": 45, "dropped": 0}, groups=("A", "B", "C"))
    expected = (
        "Kruskal-Wallis H test did not find a statistically significant "
        "difference among the groups (H(2) = 3.20, p = .202; two-sided, α = "
        ".05). The effect was small (eta-squared (H) = 0.03; per Tomczak & "
        "Tomczak (2014)). Absence of evidence is not evidence of absence. 45 "
        "cases were analysed (n = 45); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_friedman_significant_with_posthoc():
    ph = pd.DataFrame(
        [{"group1": "M1", "group2": "M2", "W": 5.0, "p": 0.02, "p_holm": 0.04},
         {"group1": "M1", "group2": "M3", "W": 1.0, "p": 0.0007, "p_holm": 0.002},
         {"group1": "M2", "group2": "M3", "W": 10.0, "p": 0.30, "p_holm": 0.30}])
    r = Result(
        test_id="friedman", test_name="Friedman test", status="ok",
        statistic=("χ²", 14.0), df=(2.0,), p=0.0009,
        effect=("Kendall's W", 0.35), effect_label="medium",
        effect_source="Tomczak & Tomczak (2014)",
        n={"total": 20, "used": 20, "dropped": 0, "subjects": 20, "measures": 3},
        posthoc=ph, posthoc_name="pairwise Wilcoxon signed-rank (Holm-adjusted)")
    expected = (
        "Friedman test found a statistically significant difference across "
        "the repeated measurements (at least one differs) (χ²(2) = 14.00, p < "
        ".001; two-sided, α = .05). The effect was medium (Kendall's W = "
        "0.35; per Tomczak & Tomczak (2014)). Post-hoc comparisons (pairwise "
        "Wilcoxon signed-rank (Holm-adjusted)) showed significant differences "
        "between M1 and M2 (p = .040); M1 and M3 (p = .002). 20 cases were "
        "analysed (n = 20); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_anova_1w_welch_significant_with_games_howell_pairs():
    ph = pd.DataFrame(
        [{"group1": "A", "group2": "B", "meandiff": 4.0, "se": 1.0,
          "statistic": 4.0, "df": 10.0, "ci_low": 1.0, "ci_high": 7.0, "p": 0.01},
         {"group1": "A", "group2": "C", "meandiff": 1.0, "se": 1.0,
          "statistic": 1.0, "df": 10.0, "ci_low": -2.0, "ci_high": 4.0, "p": 0.55}])
    r = Result(
        test_id="anova_1w", test_name="Welch's ANOVA", status="ok",
        statistic=("F", 8.10), df=(2.0, 12.34), p=0.006,
        effect=("η²", 0.40), effect_label="large", effect_source="Cohen (1988)",
        n={"total": 21, "used": 21, "dropped": 0}, groups=("A", "B", "C"),
        labels={"outcome": "Yield"}, posthoc=ph, posthoc_name="Games-Howell")
    expected = (
        "Welch's ANOVA found a statistically significant difference among the "
        "groups (at least one group differs) (F(2, 12.34) = 8.10, p = .006; "
        "two-sided, α = .05). The effect was large (η² = 0.40; per Cohen "
        "(1988)). Post-hoc comparisons (Games-Howell) showed significant "
        "differences between A and B (p = .010). 21 cases were analysed (n = "
        "21); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_mwu_significant_asymptotic_no_ties():
    r = Result(
        test_id="mwu", test_name="Mann-Whitney U test", status="ok",
        statistic=("U", 60.0), p=0.014,
        estimate=("Hodges-Lehmann median difference", 2.5),
        effect=("rank-biserial correlation", 0.40), effect_label="medium",
        effect_source="Cohen (1988)",
        n={"total": 30, "used": 30, "dropped": 0, "New": 15, "Old": 15},
        groups=("New", "Old"), higher="New", variant_notes=("asymptotic method",))
    expected = (
        "Mann-Whitney U test found a statistically significant difference "
        "between New and Old (U = 60.00, p = .014; two-sided, α = .05). "
        "Values tended to be higher in New. The Hodges-Lehmann median "
        "difference was 2.50. Method: asymptotic method. The effect was "
        "medium (rank-biserial correlation = 0.40; per Cohen (1988)). 30 "
        "cases were analysed (n = 30); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_wilcoxon_paired_not_significant():
    r = Result(
        test_id="wilcoxon", test_name="Wilcoxon signed-rank test", status="ok",
        statistic=("W", 55.0), p=0.400,
        estimate=("median of differences", 0.5),
        effect=("matched-pairs rank-biserial correlation", 0.12),
        effect_label="small", effect_source="Cohen (1988)",
        n={"total": 20, "used": 20, "dropped": 0, "pairs": 20, "nonzero_pairs": 20})
    expected = (
        "Wilcoxon signed-rank test did not find a statistically significant "
        "difference between the paired measurements (W = 55.00, p = .400; "
        "two-sided, α = .05). The median of differences was 0.50. The effect "
        "was small (matched-pairs rank-biserial correlation = 0.12; per Cohen "
        "(1988)). Absence of evidence is not evidence of absence. 20 cases "
        "were analysed (n = 20); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_wilcoxon_one_sample_significant():
    r = Result(
        test_id="wilcoxon", test_name="Wilcoxon signed-rank test (one-sample)",
        status="ok", statistic=("W", 8.0), p=0.008,
        estimate=("median difference from μ₀", 3.0),
        effect=("matched-pairs rank-biserial correlation", 0.70),
        effect_label="large", effect_source="Cohen (1988)",
        n={"total": 16, "used": 16, "dropped": 0, "pairs": 16, "nonzero_pairs": 16})
    expected = (
        "Wilcoxon signed-rank test (one-sample) found a statistically "
        "significant difference from the hypothesised median (W = 8.00, p = "
        ".008; two-sided, α = .05). The median difference from μ₀ was 3.00. "
        "The effect was large (matched-pairs rank-biserial correlation = "
        "0.70; per Cohen (1988)). 16 cases were analysed (n = 16); no rows "
        "were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_pearson_not_significant():
    r = Result(
        test_id="pearson", test_name="Pearson correlation", status="ok",
        statistic=("r", 0.20), p=0.400,
        effect=("r", 0.20), effect_ci=(-0.27, 0.60), effect_label="small",
        effect_source="Cohen (1988)",
        n={"total": 20, "used": 20, "dropped": 0},
        labels={"x": "Age", "y": "Score"}, extra={"r2": 0.04})
    expected = (
        "Pearson correlation did not find a statistically significant "
        "correlation between Age and Score (r = 0.20, p = .400; two-sided, α "
        "= .05). The correlation was small (r = 0.20, 95% CI [-0.27, 0.60]; "
        "r² = 0.04). Absence of evidence is not evidence of absence. 20 cases "
        "were analysed (n = 20); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_spearman_significant_small_n_no_ci():
    r = Result(
        test_id="spearman", test_name="Spearman rank correlation", status="ok",
        statistic=("rho", 0.90), p=0.037,
        effect=("rho", 0.90), effect_ci=None, effect_label="large",
        effect_source="Cohen (1988)",
        n={"total": 4, "used": 4, "dropped": 0},
        labels={"x": "A", "y": "B"}, extra={"r2": 0.81})
    expected = (
        "Spearman rank correlation found a statistically significant positive "
        "correlation between A and B (rho = 0.90, p = .037; two-sided, α = "
        ".05). The correlation was large (rho = 0.90). 4 cases were analysed "
        "(n = 4); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_kendall_significant_positive():
    r = Result(
        test_id="kendall", test_name="Kendall's τ-b", status="ok",
        statistic=("tau-b", 0.55), p=0.001,
        effect=("tau-b", 0.55), effect_ci=(0.25, 0.76), effect_label="large",
        effect_source="Cohen (1988)",
        n={"total": 25, "used": 25, "dropped": 0},
        labels={"x": "Grade", "y": "Hours"}, extra={"r2": 0.3025})
    expected = (
        "Kendall's τ-b found a statistically significant positive correlation "
        "between Grade and Hours (tau-b = 0.55, p = .001; two-sided, α = "
        ".05). The correlation was large (tau-b = 0.55, 95% CI [0.25, 0.76]). "
        "25 cases were analysed (n = 25); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_corr_matrix_none_significant():
    idx = ["A", "B", "C", "D"]
    R = pd.DataFrame([[1.0, 0.1, 0.1, 0.1], [0.1, 1.0, 0.1, 0.1],
                      [0.1, 0.1, 1.0, 0.1], [0.1, 0.1, 0.1, 1.0]],
                     index=idx, columns=idx)
    ph = pd.DataFrame([[1.0, 0.5, 0.5, 0.5], [0.5, 1.0, 0.5, 0.5],
                       [0.5, 0.5, 1.0, 0.5], [0.5, 0.5, 0.5, 1.0]],
                      index=idx, columns=idx)
    r = Result(
        test_id="corr_matrix", test_name="Correlation matrix", status="ok",
        n={"total": 25, "used": 25, "dropped": 0}, table=R,
        extra={"p_holm": ph, "method": "spearman"})
    expected = (
        "Correlation matrix: spearman correlations were computed among 4 "
        "variables. 0 of 6 pairwise correlations were statistically "
        "significant after Holm adjustment (two-sided, α = .05). See the "
        "correlation matrix and heatmap. 25 cases were analysed (n = 25); no "
        "rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_chi2_ind_2x2_not_significant():
    r = Result(
        test_id="chi2_ind", test_name="Chi-square test of independence",
        status="ok", statistic=("χ²", 1.10), df=(1.0,), p=0.294,
        estimate=("odds ratio", 1.50), estimate_ci=(0.70, 3.21),
        effect=("phi", 0.12), effect_label="small", effect_source="Cohen (1988)",
        n={"total": 80, "used": 80, "dropped": 0})
    expected = (
        "Chi-square test of independence did not find a statistically "
        "significant association between the two variables (χ²(1) = 1.10, p = "
        ".294; two-sided, α = .05). The odds ratio was 1.50, 95% CI [0.70, "
        "3.21]. The effect was small (phi = 0.12; per Cohen (1988)). Absence "
        "of evidence is not evidence of absence. 80 cases were analysed (n = "
        "80); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_chi2_ind_rxc_significant():
    r = Result(
        test_id="chi2_ind", test_name="Chi-square test of independence",
        status="ok", statistic=("χ²", 20.0), df=(4.0,), p=0.0005,
        effect=("Cramér's V", 0.30), effect_label="medium",
        effect_source="Cohen (1988)",
        n={"total": 110, "used": 110, "dropped": 0})
    expected = (
        "Chi-square test of independence found a statistically significant "
        "association between the two variables (χ²(4) = 20.00, p < .001; "
        "two-sided, α = .05). The effect was medium (Cramér's V = 0.30; per "
        "Cohen (1988)). 110 cases were analysed (n = 110); no rows were "
        "excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_chi2_gof_not_significant():
    r = Result(
        test_id="chi2_gof", test_name="Chi-square goodness-of-fit test",
        status="ok", statistic=("χ²", 2.00), df=(3.0,), p=0.572,
        effect=("Cohen's w", 0.14), effect_label="small",
        effect_source="Cohen (1988)",
        n={"total": 100, "used": 100, "dropped": 0})
    expected = (
        "Chi-square goodness-of-fit test did not find a statistically "
        "significant difference between the observed counts and the expected "
        "proportions (χ²(3) = 2.00, p = .572; two-sided, α = .05). The effect "
        "was small (Cohen's w = 0.14; per Cohen (1988)). Absence of evidence "
        "is not evidence of absence. 100 cases were analysed (n = 100); no "
        "rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_fisher_2x2_not_significant():
    r = Result(
        test_id="fisher", test_name="Fisher's exact test", status="ok",
        statistic=("odds ratio (sample)", 2.0), p=0.320,
        estimate=("odds ratio", 1.90), estimate_ci=(0.55, 6.80),
        effect=("phi", 0.15), effect_label="small", effect_source="Cohen (1988)",
        n={"total": 24, "used": 24, "dropped": 0})
    expected = (
        "Fisher's exact test did not find a statistically significant "
        "association between the two variables (p = .320; two-sided, α = "
        ".05). The odds ratio was 1.90, 95% CI [0.55, 6.80]. The effect was "
        "small (phi = 0.15; per Cohen (1988)). Absence of evidence is not "
        "evidence of absence. 24 cases were analysed (n = 24); no rows were "
        "excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_fisher_rxc_significant():
    r = Result(
        test_id="fisher", test_name="Fisher-Freeman-Halton test (Monte Carlo p)",
        status="ok", statistic=("table probability", 0.01), p=0.004,
        n={"total": 28, "used": 28, "dropped": 0},
        variant_notes=("Monte Carlo estimate (99,999 resamples, seeded for "
                       "reproducibility)",))
    expected = (
        "Fisher-Freeman-Halton test (Monte Carlo p) found a statistically "
        "significant association between the two variables (p = .004; "
        "two-sided, α = .05). Monte Carlo estimate (99,999 resamples, seeded "
        "for reproducibility). 28 cases were analysed (n = 28); no rows were "
        "excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_mcnemar_chi2_variant_significant():
    r = Result(
        test_id="mcnemar",
        test_name="McNemar's test (χ² with continuity correction)", status="ok",
        statistic=("χ²", 5.80), p=0.016,
        estimate=("odds ratio", 2.50), estimate_ci=(1.15, 5.43),
        n={"total": 100, "used": 100, "dropped": 0, "discordant": 40},
        variant_notes=("40 of 100 changed (40.0%)",))
    expected = (
        "McNemar's test (χ² with continuity correction) found a statistically "
        "significant change between the two measurements (χ² = 5.80, p = "
        ".016; two-sided, α = .05). 40 of 100 changed (40.0%). The odds ratio "
        "was 2.50, 95% CI [1.15, 5.43]. 100 cases were analysed (n = 100); no "
        "rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_cochran_q_not_significant():
    r = Result(
        test_id="cochran_q", test_name="Cochran's Q test", status="ok",
        statistic=("Q", 1.50), df=(2.0,), p=0.472,
        n={"total": 30, "used": 30, "dropped": 0, "subjects": 30, "measures": 3},
        labels={"success": "Yes"})
    expected = (
        "Cochran's Q test did not find a statistically significant difference "
        "in the proportion of 'Yes' across the repeated conditions (Q(2) = "
        "1.50, p = .472; two-sided, α = .05). Absence of evidence is not "
        "evidence of absence. 30 cases were analysed (n = 30); no rows were "
        "excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_prop_1_not_significant():
    r = Result(
        test_id="prop_1",
        test_name="One-sample proportion test (exact binomial)", status="ok",
        p=0.210, estimate=("proportion 'Yes'", 0.55), estimate_ci=(0.42, 0.67),
        effect=("Cohen's h", 0.10), effect_label="negligible",
        effect_source="Cohen (1988)",
        n={"total": 60, "used": 60, "dropped": 0, "successes": 33},
        labels={"success": "Yes"},
        variant_notes=("tested against p₀ = 0.5; 95% Wilson interval",))
    expected = (
        "One-sample proportion test (exact binomial) did not find a "
        "statistically significant difference between the proportion of 'Yes' "
        "and the hypothesised value (p = .210; two-sided, α = .05). The "
        "observed proportion of 'Yes' was 55.0%, 95% CI [42.0%, 67.0%]. The "
        "effect was negligible (Cohen's h = 0.10; per Cohen (1988)). Absence "
        "of evidence is not evidence of absence. 60 cases were analysed (n = "
        "60); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_prop_2_not_significant_negative_difference():
    r = Result(
        test_id="prop_2", test_name="Two-sample proportion z-test", status="ok",
        statistic=("z", 1.10), p=0.271,
        estimate=("difference in proportion 'Pass' (G1 − G2)", -0.08),
        estimate_ci=(-0.22, 0.06),
        effect=("Cohen's h", 0.16), effect_label="negligible",
        effect_source="Cohen (1988)",
        n={"total": 100, "used": 100, "dropped": 0, "G1": 50, "G2": 50},
        groups=("G1", "G2"), higher="G2", labels={"success": "Pass"})
    expected = (
        "Two-sample proportion z-test did not find a statistically "
        "significant difference in the proportion of 'Pass' between G1 and G2 "
        "(z = 1.10, p = .271; two-sided, α = .05). The difference in the "
        "proportion of 'Pass' was -8.0%, 95% CI [-22.0%, 6.0%]. The effect "
        "was negligible (Cohen's h = 0.16; per Cohen (1988)). Absence of "
        "evidence is not evidence of absence. 100 cases were analysed (n = "
        "100); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_ols_simple_not_significant():
    r = Result(
        test_id="ols_simple", test_name="Simple linear regression", status="ok",
        statistic=("F", 0.90), df=(1.0, 28.0), p=0.351,
        estimate=("slope", 0.30), estimate_ci=(-0.35, 0.95),
        effect=("f2", 0.03), effect_label="small", effect_source="Cohen (1988)",
        n={"total": 30, "used": 30, "dropped": 0},
        labels={"outcome": "Score", "x": "Sleep"},
        extra={"intercept": 1.0, "slope": 0.30, "r2": 0.031, "adj_r2": 0.0})
    expected = (
        "Simple linear regression did not find that Sleep statistically "
        "significantly predicted Score (F(1, 28) = 0.90, p = .351; two-sided, "
        "α = .05). The model explained 3.1% of the variance in Score (R² = "
        "0.03). Each one-unit increase in Sleep was associated with a change "
        "of 0.30 in Score (95% CI [-0.35, 0.95]). The effect was small (f2 = "
        "0.03; per Cohen (1988)). Absence of evidence is not evidence of "
        "absence. 30 cases were analysed (n = 30); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_ols_multi_not_significant_no_reference():
    table = pd.DataFrame(
        {"coef": [1.0, 0.2, -0.5], "ci_low": [0.1, -0.1, -1.5],
         "ci_high": [1.9, 0.5, 0.5], "p": [0.02, 0.18, 0.30]},
        index=["Intercept", "Study hours", "Extra"])
    r = Result(
        test_id="ols_multi", test_name="Multiple linear regression",
        status="ok", statistic=("F", 1.50), df=(2.0, 47.0), p=0.234,
        effect=("f2", 0.06), effect_label="small", effect_source="Cohen (1988)",
        n={"total": 50, "used": 50, "dropped": 0}, table=table,
        labels={"outcome": "Score"},
        extra={"r2": 0.06, "adj_r2": 0.02, "vif": {}, "reference": {}})
    expected = (
        "Multiple linear regression did not find a statistically significant "
        "model (F(2, 47) = 1.50, p = .234; two-sided, α = .05). The model "
        "explained 6.0% of the variance (R² = 0.06, adjusted R² = 0.02). "
        "Coefficients — Study hours: b = 0.20, 95% CI [-0.10, 0.50], p = "
        ".180; Extra: b = -0.50, 95% CI [-1.50, 0.50], p = .300. The effect "
        "was small (f2 = 0.06; per Cohen (1988)). Absence of evidence is not "
        "evidence of absence. 50 cases were analysed (n = 50); no rows were "
        "excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_logistic_not_significant_with_reference():
    table = pd.DataFrame(
        {"coef": [-0.5, 0.1, 0.3], "OR": [0.61, 1.11, 1.35],
         "OR_ci_low": [0.20, 0.95, 0.70], "OR_ci_high": [1.80, 1.30, 2.60],
         "p": [0.40, 0.18, 0.35]}, index=["Intercept", "Age", "Plan"])
    r = Result(
        test_id="logistic", test_name="Binary logistic regression", status="ok",
        statistic=("LLR chi2", 3.0), df=(2.0,), p=0.223,
        n={"total": 80, "used": 80, "dropped": 0}, labels={"outcome": "Churn"},
        table=table,
        extra={"or": {"Age": 1.11, "Plan": 1.35}, "pseudo_r2": 0.04,
               "llr_p": 0.223, "success": "Yes", "reference": {"Plan": "Basic"}})
    expected = (
        "Binary logistic regression did not find a statistically significant "
        "model predicting 'Yes' in Churn (LLR chi2(2) = 3.00, p = .223; "
        "two-sided, α = .05). Odds ratios — Age: each one-unit increase "
        "multiplied the odds of 'Yes' by 1.11 (95% CI [0.95, 1.30], p = "
        ".180); Plan: each one-unit increase multiplied the odds of 'Yes' by "
        "1.35 (95% CI [0.70, 2.60], p = .350). Reference level(s): Plan = "
        "'Basic'. McFadden pseudo-R² = 0.04. Absence of evidence is not "
        "evidence of absence. 80 cases were analysed (n = 80); no rows were "
        "excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_normality_single_variable_pass():
    checks = (
        Check("Shapiro-Wilk", 0.98, 0.72, True, ""),
        Check("Lilliefors", 0.06, 0.55, True, ""),
    )
    r = Result(
        test_id="normality", test_name="Normality check", status="ok",
        statistic=("W", 0.98), p=0.72, checks=checks,
        n={"total": 30, "used": 30, "dropped": 0})
    expected = (
        "Normality check assessed normality. Shapiro-Wilk: statistic = 0.98, "
        "p = .720 — consistent with a normal distribution. Lilliefors: "
        "statistic = 0.06, p = .550 — consistent with a normal distribution. "
        "30 cases were analysed (n = 30); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_describe_no_rows_dropped():
    r = Result(
        test_id="describe", test_name="Descriptive statistics (Table 1)",
        status="ok", n={"total": 40, "used": 40, "dropped": 0})
    expected = (
        "Descriptive statistics (Table 1) were computed for the selected "
        "columns. See the table for n, mean, SD, median, IQR and range "
        "(numeric) and counts or percentages (categorical). 40 cases were "
        "analysed (n = 40); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_blocked_min_n():
    r = Result(
        test_id="t_1s", test_name="One-sample t-test", status="blocked",
        n={"total": 2, "used": 2, "dropped": 0},
        findings=(Finding("block", "Only 2 usable rows remain after removing "
                          "missing values; this test needs at least 3.",
                          code="S5"),))
    expected = ("Only 2 usable rows remain after removing missing values; this "
                "test needs at least 3.")
    assert _say(r) == expected
    _g(r, expected)


def test_blocked_constant_column():
    r = Result(
        test_id="anova_1w", test_name="One-way ANOVA", status="blocked",
        n={"total": 30, "used": 30, "dropped": 0},
        findings=(Finding("block", "A bound column has no variation (all values "
                          "are identical), so the test cannot run.",
                          code="S8"),))
    expected = ("A bound column has no variation (all values are identical), so "
                "the test cannot run.")
    assert _say(r) == expected
    _g(r, expected)


def test_blocked_too_few_pairs():
    r = Result(
        test_id="t_paired", test_name="Paired-samples t-test", status="blocked",
        n={"total": 4, "used": 2, "dropped": 2},
        findings=(Finding("block", "Fewer than 3 complete pairs remain; the "
                          "paired test cannot run.", code="S6"),))
    expected = ("Fewer than 3 complete pairs remain; the paired test cannot "
                "run.")
    assert _say(r) == expected
    _g(r, expected)


# ---- dropped-rows n-clause across families + more blocked utilities --------
def test_t_ind_not_significant_low_n_with_dropped():
    r = Result(
        test_id="t_ind", test_name="Welch's independent-samples t-test",
        status="ok", statistic=("t", 0.90), df=(10.0,), p=0.390,
        estimate=("mean difference (A − B)", 1.5), estimate_ci=(-2.2, 5.2),
        effect=("Hedges' g", 0.35), effect_ci=(-0.6, 1.3), effect_label="small",
        effect_source="Cohen (1988)",
        n={"total": 15, "used": 12, "dropped": 3, "A": 6, "B": 6},
        groups=("A", "B"), higher="A", labels={"outcome": "Score"})
    expected = (
        "Welch's independent-samples t-test did not find a statistically "
        "significant difference in Score between A and B (t(10) = 0.90, p = "
        ".390; two-sided, α = .05). The mean difference (A − B) was 1.50, 95% "
        "CI [-2.20, 5.20]. The effect was small (Hedges' g = 0.35, 95% CI "
        "[-0.60, 1.30]; per Cohen (1988)). Absence of evidence is not "
        "evidence of absence. 12 cases were analysed (n = 12); 3 of 15 rows "
        "were excluded (missing data). With n = 12 (below 15) the test has "
        "low statistical power, so a non-significant result is not evidence "
        "that there is no effect.")
    assert _say(r) == expected
    _g(r, expected)


def test_anova_1w_significant_with_dropped():
    ph = pd.DataFrame(
        [{"group1": "A", "group2": "B", "meandiff": 1.0, "ci_low": -1.0,
          "ci_high": 3.0, "p": 0.500},
         {"group1": "A", "group2": "C", "meandiff": 4.0, "ci_low": 1.5,
          "ci_high": 6.5, "p": 0.010},
         {"group1": "B", "group2": "C", "meandiff": 2.0, "ci_low": -0.5,
          "ci_high": 4.5, "p": 0.200}])
    r = Result(
        test_id="anova_1w", test_name="One-way ANOVA", status="ok",
        statistic=("F", 5.0), df=(2.0, 27.0), p=0.014,
        effect=("η²", 0.27), effect_label="large", effect_source="Cohen (1988)",
        n={"total": 35, "used": 30, "dropped": 5}, groups=("A", "B", "C"),
        labels={"outcome": "Yield"}, posthoc=ph, posthoc_name="Tukey HSD")
    expected = (
        "One-way ANOVA found a statistically significant difference among the "
        "groups (at least one group differs) (F(2, 27) = 5.00, p = .014; "
        "two-sided, α = .05). The effect was large (η² = 0.27; per Cohen "
        "(1988)). Post-hoc comparisons (Tukey HSD) showed significant "
        "differences between A and C (p = .010). 30 cases were analysed (n = "
        "30); 5 of 35 rows were excluded (missing data).")
    assert _say(r) == expected
    _g(r, expected)


def test_kruskal_significant_with_dropped_and_dunn():
    ph = pd.DataFrame(
        [{"group1": "A", "group2": "B", "z": 1.0, "p": 0.30, "p_holm": 0.50},
         {"group1": "A", "group2": "C", "z": 3.2, "p": 0.001, "p_holm": 0.006},
         {"group1": "B", "group2": "C", "z": 2.2, "p": 0.02, "p_holm": 0.040}])
    r = Result(
        test_id="kruskal", test_name="Kruskal-Wallis H test", status="ok",
        statistic=("H", 10.0), df=(2.0,), p=0.007,
        effect=("eta-squared (H)", 0.20), effect_label="small",
        effect_source="Tomczak & Tomczak (2014)",
        n={"total": 50, "used": 45, "dropped": 5}, groups=("A", "B", "C"),
        posthoc=ph, posthoc_name="Dunn's test (Holm-adjusted)")
    expected = (
        "Kruskal-Wallis H test found a statistically significant difference "
        "among the groups (at least one group differs) (H(2) = 10.00, p = "
        ".007; two-sided, α = .05). The effect was small (eta-squared (H) = "
        "0.20; per Tomczak & Tomczak (2014)). Post-hoc comparisons (Dunn's "
        "test (Holm-adjusted)) showed significant differences between A and C "
        "(p = .006); B and C (p = .040). 45 cases were analysed (n = 45); 5 "
        "of 50 rows were excluded (missing data).")
    assert _say(r) == expected
    _g(r, expected)


def test_chi2_ind_2x2_significant_with_dropped():
    r = Result(
        test_id="chi2_ind", test_name="Chi-square test of independence",
        status="ok", statistic=("χ²", 8.0), df=(1.0,), p=0.005,
        estimate=("odds ratio", 2.80), estimate_ci=(1.35, 5.80),
        effect=("phi", 0.25), effect_label="small", effect_source="Cohen (1988)",
        n={"total": 130, "used": 120, "dropped": 10})
    expected = (
        "Chi-square test of independence found a statistically significant "
        "association between the two variables (χ²(1) = 8.00, p = .005; "
        "two-sided, α = .05). The odds ratio was 2.80, 95% CI [1.35, 5.80]. "
        "The effect was small (phi = 0.25; per Cohen (1988)). 120 cases were "
        "analysed (n = 120); 10 of 130 rows were excluded (missing data).")
    assert _say(r) == expected
    _g(r, expected)


def test_wilcoxon_paired_significant_with_dropped():
    r = Result(
        test_id="wilcoxon", test_name="Wilcoxon signed-rank test", status="ok",
        statistic=("W", 15.0), p=0.020,
        estimate=("median of differences", 1.5),
        effect=("matched-pairs rank-biserial correlation", 0.55),
        effect_label="large", effect_source="Cohen (1988)",
        n={"total": 25, "used": 22, "dropped": 3, "pairs": 22, "nonzero_pairs": 22})
    expected = (
        "Wilcoxon signed-rank test found a statistically significant "
        "difference between the paired measurements (W = 15.00, p = .020; "
        "two-sided, α = .05). The median of differences was 1.50. The effect "
        "was large (matched-pairs rank-biserial correlation = 0.55; per Cohen "
        "(1988)). 22 cases were analysed (n = 22); 3 of 25 rows were excluded "
        "(missing data).")
    assert _say(r) == expected
    _g(r, expected)


def test_pearson_significant_with_dropped():
    r = Result(
        test_id="pearson", test_name="Pearson correlation", status="ok",
        statistic=("r", 0.60), p=0.001,
        effect=("r", 0.60), effect_ci=(0.30, 0.80), effect_label="large",
        effect_source="Cohen (1988)",
        n={"total": 35, "used": 30, "dropped": 5},
        labels={"x": "X", "y": "Y"}, extra={"r2": 0.36})
    expected = (
        "Pearson correlation found a statistically significant positive "
        "correlation between X and Y (r = 0.60, p = .001; two-sided, α = "
        ".05). The correlation was large (r = 0.60, 95% CI [0.30, 0.80]; r² = "
        "0.36). 30 cases were analysed (n = 30); 5 of 35 rows were excluded "
        "(missing data).")
    assert _say(r) == expected
    _g(r, expected)


def test_mwu_significant_with_dropped():
    r = Result(
        test_id="mwu", test_name="Mann-Whitney U test", status="ok",
        statistic=("U", 20.0), p=0.041,
        estimate=("Hodges-Lehmann median difference", 2.0),
        effect=("rank-biserial correlation", 0.44), effect_label="medium",
        effect_source="Cohen (1988)",
        n={"total": 40, "used": 36, "dropped": 4, "X": 18, "Y": 18},
        groups=("X", "Y"), higher="X", variant_notes=("asymptotic method",))
    expected = (
        "Mann-Whitney U test found a statistically significant difference "
        "between X and Y (U = 20.00, p = .041; two-sided, α = .05). Values "
        "tended to be higher in X. The Hodges-Lehmann median difference was "
        "2.00. Method: asymptotic method. The effect was medium "
        "(rank-biserial correlation = 0.44; per Cohen (1988)). 36 cases were "
        "analysed (n = 36); 4 of 40 rows were excluded (missing data).")
    assert _say(r) == expected
    _g(r, expected)


def test_prop_2_significant_with_dropped():
    r = Result(
        test_id="prop_2", test_name="Two-sample proportion z-test", status="ok",
        statistic=("z", 3.0), p=0.003,
        estimate=("difference in proportion 'Yes' (A − B)", 0.25),
        estimate_ci=(0.09, 0.41),
        effect=("Cohen's h", 0.50), effect_label="medium",
        effect_source="Cohen (1988)",
        n={"total": 210, "used": 200, "dropped": 10, "A": 100, "B": 100},
        groups=("A", "B"), higher="A", labels={"success": "Yes"})
    expected = (
        "Two-sample proportion z-test found a statistically significant "
        "difference in the proportion of 'Yes' between A and B (z = 3.00, p = "
        ".003; two-sided, α = .05). The proportion of 'Yes' was higher in A; "
        "the difference was 25.0%, 95% CI [9.0%, 41.0%]. The effect was "
        "medium (Cohen's h = 0.50; per Cohen (1988)). 200 cases were analysed "
        "(n = 200); 10 of 210 rows were excluded (missing data).")
    assert _say(r) == expected
    _g(r, expected)


def test_cochran_q_significant_no_pair_survives():
    ph = pd.DataFrame(
        [{"group1": "V1", "group2": "V2", "p": 0.06, "p_holm": 0.09},
         {"group1": "V1", "group2": "V3", "p": 0.08, "p_holm": 0.12},
         {"group1": "V2", "group2": "V3", "p": 0.20, "p_holm": 0.20}])
    r = Result(
        test_id="cochran_q", test_name="Cochran's Q test", status="ok",
        statistic=("Q", 6.5), df=(2.0,), p=0.039,
        n={"total": 30, "used": 30, "dropped": 0, "subjects": 30, "measures": 3},
        labels={"success": "Yes"},
        posthoc=ph, posthoc_name="pairwise McNemar (Holm-adjusted)")
    expected = (
        "Cochran's Q test found a statistically significant difference in the "
        "proportion of 'Yes' across the repeated conditions (Q(2) = 6.50, p = "
        ".039; two-sided, α = .05). Post-hoc comparisons (pairwise McNemar "
        "(Holm-adjusted)) found no pairwise differences after adjustment. 30 "
        "cases were analysed (n = 30); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_blocked_describe_no_usable_rows():
    r = Result(
        test_id="describe", test_name="Descriptive statistics (Table 1)",
        status="blocked", n={"total": 10, "used": 0, "dropped": 10},
        findings=(Finding("block", "No usable rows to summarise (all values "
                          "were missing).", code="S5"),))
    expected = "No usable rows to summarise (all values were missing)."
    assert _say(r) == expected
    _g(r, expected)


def test_blocked_normality_too_small():
    r = Result(
        test_id="normality", test_name="Normality check", status="blocked",
        n={"total": 2, "used": 2, "dropped": 0},
        findings=(Finding("block", "The variable has fewer than 3 values; "
                          "normality cannot be assessed.", code="S5"),))
    expected = ("The variable has fewer than 3 values; normality cannot be "
                "assessed.")
    assert _say(r) == expected
    _g(r, expected)


def test_blocked_homogeneity_small_group():
    r = Result(
        test_id="homogeneity", test_name="Equality of variances",
        status="blocked", n={"total": 10, "used": 10, "dropped": 0},
        findings=(Finding("block", "Group(s) with fewer than 2 values: C. "
                          "Equality of variances needs at least 2 per group.",
                          code="RUN"),))
    expected = ("Group(s) with fewer than 2 values: C. Equality of variances "
                "needs at least 2 per group.")
    assert _say(r) == expected
    _g(r, expected)


# ==========================================================================
# BATCH 2A — new branch cells (S4 named vars + odds, S12 μ₀/p₀/paired labels,
# NICE ω²).  Each Result mirrors the shape its runner now attaches; the sentence
# falls back to the older wording whenever the key is absent (covered above).
# ==========================================================================
def test_chi2_ind_2x2_named_variables_and_odds():
    r = Result(
        test_id="chi2_ind", test_name="Chi-square test of independence",
        status="ok", statistic=("χ²", 6.51), df=(1.0,), p=0.011,
        estimate=("odds ratio", 3.40), estimate_ci=(1.32, 8.75),
        effect=("phi", 0.29), effect_label="small", effect_source="Cohen (1988)",
        n={"total": 78, "used": 78, "dropped": 0},
        labels={"row": "Treatment", "col": "Response"},
        extra={"or": {"col_level": "Yes", "row_level": "Drug",
                      "row_ref": "Placebo"}})
    expected = (
        "Chi-square test of independence found a statistically significant "
        "association between Treatment and Response (χ²(1) = 6.51, p = .011; "
        "two-sided, α = .05). The odds of Response = 'Yes' were 3.40 times as "
        "high for Treatment = 'Drug' as for 'Placebo' (odds ratio = 3.40, "
        "95% CI [1.32, 8.75]). The effect was small (phi = 0.29; per Cohen "
        "(1988)). 78 cases were analysed (n = 78); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_fisher_2x2_named_variables_and_odds():
    r = Result(
        test_id="fisher", test_name="Fisher's exact test", status="ok",
        statistic=None, p=0.028,
        estimate=("odds ratio", 6.20), estimate_ci=(1.10, 51.3),
        effect=("phi", 0.42), effect_label="medium", effect_source="Cohen (1988)",
        n={"total": 22, "used": 22, "dropped": 0},
        labels={"row": "Exposure", "col": "Disease"},
        extra={"or": {"col_level": "Present", "row_level": "Exposed",
                      "row_ref": "Unexposed"}})
    expected = (
        "Fisher's exact test found a statistically significant association "
        "between Exposure and Disease (p = .028; two-sided, α = .05). The odds "
        "of Disease = 'Present' were 6.20 times as high for Exposure = "
        "'Exposed' as for 'Unexposed' (odds ratio = 6.20, 95% CI [1.10, "
        "51.30]). The effect was medium (phi = 0.42; per Cohen (1988)). 22 "
        "cases were analysed (n = 22); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_t_1s_states_mu0_value():
    r = Result(
        test_id="t_1s", test_name="One-sample t-test", status="ok",
        statistic=("t", 2.45), df=(19.0,), p=0.024,
        estimate=("mean − μ₀", 3.2), estimate_ci=(0.5, 5.9),
        effect=("Cohen's d", 0.45), effect_label="small",
        effect_source="Cohen (1988)",
        n={"total": 20, "used": 20, "dropped": 0},
        labels={"outcome": "Score"}, extra={"mu0": 100.0})
    expected = (
        "One-sample t-test found a statistically significant difference between "
        "the mean of Score and the hypothesised value μ₀ = 100 (t(19) = 2.45, "
        "p = .024; two-sided, α = .05). The mean − μ₀ was 3.20, 95% CI "
        "[0.50, 5.90]. The effect was small (Cohen's d = 0.45; per Cohen "
        "(1988)). 20 cases were analysed (n = 20); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_prop_1_states_p0_value():
    r = Result(
        test_id="prop_1",
        test_name="One-sample proportion test (exact binomial)", status="ok",
        p=0.013, estimate=("proportion 'Yes'", 0.65), estimate_ci=(0.52, 0.76),
        effect=("Cohen's h", 0.31), effect_label="small",
        effect_source="Cohen (1988)",
        n={"total": 60, "used": 60, "dropped": 0, "successes": 39},
        labels={"success": "Yes"}, extra={"p0": 0.5, "success": "Yes"})
    expected = (
        "One-sample proportion test (exact binomial) found a statistically "
        "significant difference between the proportion of 'Yes' and the "
        "hypothesised value p₀ = 0.50 (p = .013; two-sided, α = .05). The "
        "observed proportion of 'Yes' was 65.0%, 95% CI [52.0%, 76.0%]. The "
        "effect was small (Cohen's h = 0.31; per Cohen (1988)). 60 cases were "
        "analysed (n = 60); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_wilcoxon_paired_named_and_direction():
    r = Result(
        test_id="wilcoxon", test_name="Wilcoxon signed-rank test", status="ok",
        statistic=("W", 12.0), p=0.014,
        estimate=("median of differences", 2.0),
        effect=("matched-pairs rank-biserial correlation", 0.68),
        effect_label="large", effect_source="Cohen (1988)",
        n={"total": 18, "used": 18, "dropped": 0, "pairs": 18,
           "nonzero_pairs": 18},
        labels={"before": "Pre", "after": "Post"}, groups=("Pre", "Post"),
        higher="Pre")
    expected = (
        "Wilcoxon signed-rank test found a statistically significant difference "
        "between Pre and Post (W = 12.00, p = .014; two-sided, α = .05). Values "
        "tended to be higher for Pre. The median of differences was 2.00. The "
        "effect was large (matched-pairs rank-biserial correlation = 0.68; per "
        "Cohen (1988)). 18 cases were analysed (n = 18); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_wilcoxon_one_sample_states_mu0_value():
    r = Result(
        test_id="wilcoxon", test_name="Wilcoxon signed-rank test (one-sample)",
        status="ok", statistic=("W", 8.0), p=0.008,
        estimate=("median difference from μ₀", 3.0),
        effect=("matched-pairs rank-biserial correlation", 0.70),
        effect_label="large", effect_source="Cohen (1988)",
        n={"total": 16, "used": 16, "dropped": 0, "pairs": 16,
           "nonzero_pairs": 16}, extra={"mu0": 50.0})
    expected = (
        "Wilcoxon signed-rank test (one-sample) found a statistically "
        "significant difference from the hypothesised median μ₀ = 50 (W = 8.00, "
        "p = .008; two-sided, α = .05). The median difference from μ₀ was 3.00. "
        "The effect was large (matched-pairs rank-biserial correlation = 0.70; "
        "per Cohen (1988)). 16 cases were analysed (n = 16); no rows were "
        "excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_anova_1w_reports_omega_squared_positive():
    ph = pd.DataFrame(
        [{"group1": "A", "group2": "B", "meandiff": 1.0, "ci_low": -1.0,
          "ci_high": 3.0, "p": 0.412},
         {"group1": "A", "group2": "C", "meandiff": 5.0, "ci_low": 3.0,
          "ci_high": 7.0, "p": 0.001},
         {"group1": "B", "group2": "C", "meandiff": 4.0, "ci_low": 2.0,
          "ci_high": 6.0, "p": 0.003}])
    r = Result(
        test_id="anova_1w", test_name="One-way ANOVA", status="ok",
        statistic=("F", 9.87), df=(2.0, 15.0), p=0.002,
        effect=("η²", 0.57), effect_label="large", effect_source="Cohen (1988)",
        n={"total": 18, "used": 18, "dropped": 0}, groups=("A", "B", "C"),
        labels={"outcome": "Yield"}, posthoc=ph, posthoc_name="Tukey HSD",
        extra={"omega_sq": 0.52})
    expected = (
        "One-way ANOVA found a statistically significant difference among the "
        "groups (at least one group differs) (F(2, 15) = 9.87, p = .002; "
        "two-sided, α = .05). The effect was large (η² = 0.57; per Cohen "
        "(1988)). ω² = 0.52. Post-hoc comparisons (Tukey HSD) showed "
        "significant differences between A and C (p = .001); B and C "
        "(p = .003). 18 cases were analysed (n = 18); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


def test_anova_1w_reports_omega_squared_negative_as_zero():
    r = Result(
        test_id="anova_1w", test_name="One-way ANOVA", status="ok",
        statistic=("F", 0.40), df=(2.0, 27.0), p=0.673,
        effect=("η²", 0.03), effect_label="negligible",
        effect_source="Cohen (1988)",
        n={"total": 30, "used": 30, "dropped": 0}, groups=("A", "B", "C"),
        labels={"outcome": "Yield"}, extra={"omega_sq": -0.05})
    expected = (
        "One-way ANOVA did not find a statistically significant difference "
        "among the groups (F(2, 27) = 0.40, p = .673; two-sided, α = .05). The "
        "effect was negligible (η² = 0.03; per Cohen (1988)). ω² = 0.00 "
        "(negative, reported as 0). Absence of evidence is not evidence of "
        "absence. 30 cases were analysed (n = 30); no rows were excluded.")
    assert _say(r) == expected
    _g(r, expected)


# ==========================================================================
# F2 — a non-estimable odds ratio SAYS so (empty 2×2 cell -> infinite OR),
# never a bare number or an em dash.
# ==========================================================================
def test_fisher_odds_ratio_not_estimable_when_cell_empty():
    r = Result(
        test_id="fisher", test_name="Fisher's exact test", status="ok",
        statistic=None, p=0.028,
        estimate=("odds ratio", float("inf")), estimate_ci=(1.50, float("inf")),
        effect=("phi", 0.50), effect_label="medium", effect_source="Cohen (1988)",
        n={"total": 22, "used": 22, "dropped": 0},
        labels={"row": "Exposure", "col": "Disease"},
        extra={"or": {"col_level": "Present", "row_level": "Exposed",
                      "row_ref": "Unexposed"}})
    expected = (
        "Fisher's exact test found a statistically significant association "
        "between Exposure and Disease (p = .028; two-sided, α = .05). The odds "
        "ratio could not be estimated because a cell in the table was empty. "
        "The effect was medium (phi = 0.50; per Cohen (1988)). 22 cases were "
        "analysed (n = 22); no rows were excluded.")
    s = _say(r)
    assert s == expected
    assert "inf" not in s and "—" not in s
    _g(r, expected)


def test_effect_clause_non_finite_reads_not_estimable():
    r = Result(
        test_id="chi2_ind", test_name="Chi-square test of independence",
        status="ok", statistic=("χ²", 6.51), df=(1.0,), p=0.011,
        effect=("Cramér's V", float("inf")), effect_label="large",
        effect_source="Cohen (1988)",
        n={"total": 40, "used": 40, "dropped": 0})
    s = _say(r)
    assert "not estimable" in s
    assert "inf" not in s and "—" not in s


# ==========================================================================
# S-E — a perfect-fit OLS (Cohen's f² = inf, F astronomically large) SAYS so,
# never a bare "inf" nor a 30-digit F in the sentence.
# ==========================================================================
def test_ols_simple_perfect_fit_reads_not_meaningful():
    table = pd.DataFrame(
        {"coef": [2.0, 1.5], "ci_low": [1.5, 1.5], "ci_high": [1.5, 1.5],
         "p": [0.0, 0.0]}, index=["Intercept", "X"])
    r = Result(
        test_id="ols_simple", test_name="Simple linear regression", status="ok",
        statistic=("F", 1.5449491690281546e31), df=(1.0, 12.0), p=0.0,
        estimate=("slope", 1.5), estimate_ci=(1.5, 1.5),
        effect=("f2", float("inf")), effect_label="large",
        effect_source="Cohen (1988)",
        n={"total": 14, "used": 14, "dropped": 0}, table=table,
        labels={"outcome": "Y", "x": "X"},
        extra={"intercept": 2.0, "slope": 1.5, "r2": 1.0, "adj_r2": 1.0})
    expected = (
        "Simple linear regression found that X statistically significantly "
        "predicted Y (F(1, 12) = —, p < .001; two-sided, α = .05). The model "
        "explained 100.0% of the variance in Y (R² = 1.00). Each one-unit "
        "increase in X was associated with a change of 1.50 in Y (95% CI [1.50, "
        "1.50]). Perfect fit — the model fits the data exactly, so F is not "
        "meaningful. The effect was not estimable. 14 cases were analysed "
        "(n = 14); no rows were excluded.")
    s = _say(r)
    assert s == expected
    assert "inf" not in s
    assert not re.search(r"\d{12,}", s), "astronomical F leaked into the sentence"
    _g(r, expected)


# ==========================================================================
# PROPERTY GUARDS over the whole golden corpus (PLAN §6)
# ==========================================================================
_BANNED = re.compile(r"caused|proves|medians differ|times more likely")


def _corpus():
    # GOLDENS is populated as the per-cell tests above run (pytest runs a module
    # in definition order). When only a guard is selected in isolation the
    # per-cell tests have not run, so skip rather than fail misleadingly — the
    # meaningful run is the full module / full suite, where GOLDENS is full.
    if not GOLDENS:
        pytest.skip("golden corpus is populated by the per-cell tests; run the "
                    "whole module (the full suite always does)")
    return GOLDENS


def test_corpus_is_large_enough():
    # DoD: ~100 exact-string goldens. Each _g() call is one; every family cell
    # above contributes. Guard against silently shrinking the corpus.
    assert len(_corpus()) >= 85, len(GOLDENS)


def test_no_sentence_uses_a_banned_verb():
    # SMOKE check only (a finite banned list passes vacuously); the exact
    # goldens above are the real guard.
    for r, s in _corpus():
        assert not _BANNED.search(s), (r.test_id, s)


def test_every_ok_sentence_carries_n_and_test_name():
    for r, s in _corpus():
        if r.status != "ok":
            continue
        assert "n = " in s, (r.test_id, s)
        assert r.test_name in s, (r.test_id, s)


def test_every_ok_sentence_with_a_p_value_reports_it():
    for r, s in _corpus():
        if r.status != "ok" or r.p is None:
            continue
        assert ("p = " in s or "p < " in s), (r.test_id, s)


def test_no_causal_or_overclaiming_wording():
    # Rule 1/3: never "proved"; no bare causal verbs. Belt-and-braces over the
    # smoke list above.
    for r, s in _corpus():
        low = s.lower()
        assert "proved" not in low, (r.test_id, s)
        assert "caused" not in low, (r.test_id, s)
