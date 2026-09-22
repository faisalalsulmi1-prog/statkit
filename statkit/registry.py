"""The closed test registry — StatKit's scope document as data (PLAN §3).

One `TestSpec` per menu entry: 24 inferential tests + 3 utilities = 27. Each
carries a COMPLETE `Contract` (which layouts it accepts, the roles per layout,
its params, pairing, level constraints) so the menu, the column pickers and the
structural checks all read from one declarative source and cannot drift.

Chunk 1 is the skeleton: every runner/sentence/chart is the `_todo` stub that
raises NotImplementedError. The *contracts* are the real deliverable here and
are meant to be correct now; the statistics land in later chunks (PLAN §10).

Notation map from §3 (`role:kinds[levels]` in LONG layouts; wide = one numeric
column per group/condition; table = a count grid / category+count expanded to
long by `bind.table_to_long`):
  * a single categorical role's [lo..hi] = allowed DISTINCT LEVELS -> Role.levels
  * a multi-column role's [lo..hi]      = allowed COLUMN COUNT     -> Role.min/max
"""
from .model import Contract, Kind, Param, Role, TestSpec

# Kind shorthands, used only in this module for readable contracts.
N, O, C, B, ID, DT = (
    Kind.NUMERIC, Kind.ORDINAL, Kind.CATEGORICAL, Kind.BINARY, Kind.ID, Kind.DATE,
)


def _todo(*_args, **_kwargs):
    """Stub runner/sentence/chart for Chunk 1. Fails loudly so no test can
    mistake an unimplemented statistic for a real (empty) Result."""
    raise NotImplementedError("runner not implemented yet (Chunk 1 skeleton)")


def _spec(id, name, goal, pairing, contract, family, library, citation,
          menu_help, aliases=()):
    return TestSpec(
        id=id, name=name, goal=goal, pairing=pairing, contract=contract,
        run=_todo, sentence=_todo, chart=_todo, family=family,
        library=library, citation=citation, menu_help=menu_help, aliases=aliases,
    )


SPECS: tuple[TestSpec, ...] = (
    # --- utilities -------------------------------------------------------
    _spec(
        "describe", "Descriptive statistics (Table 1)", "describe", "any",
        Contract(
            roles_by_layout={"long": (
                Role("variables", (N, O, C, B), "Variables to summarise",
                     min=1, max=None),
                Role("group", (C, B, O), "Split by group (optional)",
                     min=0, max=1, levels=(2, 20)),
            )},
            canonical="long", pairing="any", min_n=1,
        ),
        "(table only)", "pandas (describe / value_counts)",
        "APA (2020) reporting guidelines",
        "Summarise every column: n, mean, SD, median, IQR, min/max, and "
        "counts/percentages for categories — optionally by group.",
        aliases=("descriptives", "summary statistics", "table 1"),
    ),
    _spec(
        "normality", "Normality check", "assumptions", "any",
        Contract(
            roles_by_layout={"long": (
                Role("variables", (N,), "Numeric variables", min=1, max=None),
                Role("group", (C, B, O), "Split by group (optional)",
                     min=0, max=1, levels=(2, 20)),
            )},
            canonical="long", pairing="any", min_n=3,
        ),
        "F-CHK", "scipy.stats.shapiro / statsmodels lilliefors + skew/kurtosis",
        "Shapiro & Wilk (1965); Lilliefors (1967)",
        "Check whether a numeric variable is approximately normal (W, p, plus "
        "histogram and Q–Q plot).",
        aliases=("shapiro", "shapiro-wilk", "qq plot"),
    ),
    _spec(
        "homogeneity", "Equality of variances", "assumptions", "any",
        Contract(
            roles_by_layout={"long": (
                Role("outcome", (N,), "Outcome (numeric)"),
                Role("group", (C, B, O), "Group", levels=(2, 20)),
            )},
            canonical="long", pairing="any", min_n=3,
        ),
        "F-CHK", "scipy.stats.levene(center='median') [Brown-Forsythe] + bartlett",
        "Brown & Forsythe (1974); Bartlett (1937)",
        "Check whether groups have equal spread before an ANOVA or Student's t.",
        aliases=("levene", "brown-forsythe", "equal variances",
                 "homogeneity of variance"),
    ),
    # --- compare: two groups --------------------------------------------
    _spec(
        "t_1s", "One-sample t-test", "compare", "any",
        Contract(
            roles_by_layout={"long": (Role("outcome", (N,), "Outcome (numeric)"),)},
            canonical="long", pairing="any",
            params=(Param("mu0", "float", 0.0, "Hypothesised mean (μ₀)",
                          help="The value to compare the sample mean against."),),
            min_n=3,
        ),
        "F-2G", "scipy.stats.ttest_1samp(x, mu0)", "Student (1908)",
        "Is the mean of one numeric column different from a fixed value?",
        aliases=("one-sample t-test", "single-sample t-test"),
    ),
    _spec(
        "t_ind", "Welch's independent-samples t-test", "compare", "independent",
        Contract(
            roles_by_layout={
                "long": (
                    Role("outcome", (N,), "Outcome (numeric)"),
                    Role("group", (B, C, O), "Group (2 levels)", levels=(2, 2)),
                ),
                "wide": (Role("groups", (N,), "The two groups' columns",
                              min=2, max=2),),
            },
            canonical="long", pairing="independent",
            params=(Param("equal_var", "bool", False,
                          "Assume equal variances (Student's t — not recommended)"),),
            min_n=3,
        ),
        "F-2G",
        "scipy.stats.ttest_ind(a, b, equal_var=equal_var, alternative='two-sided')",
        "Welch (1947); Delacre et al. (2017)",
        "Compare the means of two independent groups (Welch by default).",
        aliases=("welch", "independent t-test", "unpaired t-test",
                 "two-sample t-test", "student's t-test"),
    ),
    _spec(
        "t_paired", "Paired-samples t-test", "compare", "paired",
        Contract(
            roles_by_layout={
                "wide": (
                    Role("before", (N,), "Before / condition 1"),
                    Role("after", (N,), "After / condition 2"),
                ),
                "long": (
                    Role("subject", (ID, C), "Subject / unit id"),
                    Role("condition", (C, B), "Condition (2 levels)", levels=(2, 2)),
                    Role("outcome", (N,), "Outcome (numeric)"),
                ),
            },
            canonical="wide", pairing="paired", min_n=3,
        ),
        "F-2G", "scipy.stats.ttest_rel(a, b)", "Student (1908)",
        "Compare two measurements on the same people/units (e.g. before vs after).",
        aliases=("paired t-test", "dependent t-test", "matched t-test"),
    ),
    # --- compare: k groups ----------------------------------------------
    _spec(
        "anova_1w", "One-way ANOVA", "compare", "independent",
        Contract(
            roles_by_layout={
                "long": (
                    Role("outcome", (N,), "Outcome (numeric)"),
                    Role("group", (C, B, O), "Group (3–20 levels)", levels=(3, 20)),
                ),
                "wide": (Role("groups", (N,), "The groups' columns",
                              min=3, max=20),),
            },
            canonical="long", pairing="independent",
            params=(Param("welch", "bool", False,
                          "Use Welch's ANOVA (unequal variances)"),),
            min_n=3,
        ),
        "F-KG",
        "scipy.stats.f_oneway / statsmodels anova_oneway(use_var='unequal', welch_correction=True)",
        "Fisher (1925); Tukey (1949); Games & Howell (1976)",
        "Compare the means of three or more independent groups.",
        aliases=("one-way anova", "anova", "welch anova", "games-howell"),
    ),
    _spec(
        "anova_2w", "Two-way ANOVA", "compare", "independent",
        Contract(
            roles_by_layout={"long": (
                Role("outcome", (N,), "Outcome (numeric)"),
                Role("factor_a", (C, B, O), "Factor A (2–10 levels)", levels=(2, 10)),
                Role("factor_b", (C, B, O), "Factor B (2–10 levels)", levels=(2, 10)),
            )},
            canonical="long", pairing="independent", min_n=3,
        ),
        "F-KG",
        "statsmodels ols('outcome ~ C(factor_a, Sum)*C(factor_b, Sum)') + anova_lm(typ=2)",
        "Fisher (1925)",
        "Test the effect of two categorical factors (and their interaction) on a "
        "numeric outcome.",
        aliases=("two-way anova", "factorial anova"),
    ),
    _spec(
        "rm_anova", "Repeated-measures ANOVA", "compare", "paired",
        Contract(
            roles_by_layout={
                "wide": (Role("measures", (N,), "The repeated-measure columns",
                              min=3, max=20),),
                "long": (
                    Role("subject", (ID, C), "Subject id"),
                    Role("condition", (C, B, O),
                         "Within-subject condition (3–20 levels)", levels=(3, 20)),
                    Role("outcome", (N,), "Outcome (numeric)"),
                ),
            },
            canonical="wide", pairing="paired", min_n=3,
        ),
        "F-KG",
        "statsmodels AnovaRM(data, 'outcome', 'subject', within=['condition']).fit()",
        "Fisher (1925)",
        "Compare three or more measurements taken on the same people/units.",
        aliases=("repeated-measures anova", "within-subjects anova"),
    ),
    # --- compare: rank-based --------------------------------------------
    _spec(
        "mwu", "Mann-Whitney U test", "compare", "independent",
        Contract(
            roles_by_layout={
                "long": (
                    Role("outcome", (N, O), "Outcome (numeric or ordinal)"),
                    Role("group", (B, C, O), "Group (2 levels)", levels=(2, 2)),
                ),
                "wide": (Role("groups", (N, O), "The two groups' columns",
                              min=2, max=2),),
            },
            canonical="long", pairing="independent", min_n=3,
        ),
        "F-RANK2",
        "scipy.stats.mannwhitneyu(a, b, alternative='two-sided', method='auto')",
        "Mann & Whitney (1947)",
        "Non-parametric comparison of two independent groups (ranks, not means).",
        aliases=("mann-whitney", "mann-whitney u", "wilcoxon rank-sum", "u test"),
    ),
    _spec(
        "wilcoxon", "Wilcoxon signed-rank test", "compare", "paired",
        Contract(
            roles_by_layout={
                "wide": (
                    Role("before", (N, O), "Before / condition 1"),
                    Role("after", (N, O), "After / condition 2"),
                ),
                "long": (
                    Role("subject", (ID, C), "Subject / unit id"),
                    Role("condition", (C, B), "Condition (2 levels)", levels=(2, 2)),
                    Role("outcome", (N, O), "Outcome (numeric or ordinal)"),
                ),
                # one-sample mode: a lone outcome column vs mu0 (from the t_1s
                # -> Wilcoxon suggestion). bind recognises it structurally.
                "one_sample": (
                    Role("outcome", (N, O), "Outcome (numeric or ordinal)"),
                ),
            },
            canonical="wide", pairing="paired",
            params=(Param("mu0", "float", 0.0,
                          "Hypothesised median (one-sample mode only)"),),
            min_n=3,
        ),
        "F-RANK2",
        "scipy.stats.wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')",
        "Wilcoxon (1945)",
        "Non-parametric comparison of two paired measurements (ranks).",
        aliases=("wilcoxon signed-rank", "signed-rank test"),
    ),
    _spec(
        "kruskal", "Kruskal-Wallis H test", "compare", "independent",
        Contract(
            roles_by_layout={
                "long": (
                    Role("outcome", (N, O), "Outcome (numeric or ordinal)"),
                    Role("group", (C, B, O), "Group (3–20 levels)", levels=(3, 20)),
                ),
                "wide": (Role("groups", (N, O), "The groups' columns",
                              min=3, max=20),),
            },
            canonical="long", pairing="independent", min_n=3,
        ),
        "F-KG", "scipy.stats.kruskal(*groups)",
        "Kruskal & Wallis (1952); Dunn (1964); Holm (1979)",
        "Non-parametric comparison of three or more independent groups.",
        aliases=("kruskal-wallis", "kruskal-wallis h", "dunn's test"),
    ),
    _spec(
        "friedman", "Friedman test", "compare", "paired",
        Contract(
            roles_by_layout={
                "wide": (Role("measures", (N, O), "The repeated-measure columns",
                              min=3, max=20),),
                "long": (
                    Role("subject", (ID, C), "Subject id"),
                    Role("condition", (C, B, O),
                         "Within-subject condition (3–20 levels)", levels=(3, 20)),
                    Role("outcome", (N, O), "Outcome (numeric or ordinal)"),
                ),
            },
            canonical="wide", pairing="paired", min_n=3,
        ),
        "F-KG", "scipy.stats.friedmanchisquare(*cols)",
        "Friedman (1937); Holm (1979)",
        "Non-parametric comparison of three or more paired measurements.",
        aliases=("friedman test",),
    ),
    # --- relate ----------------------------------------------------------
    _spec(
        "pearson", "Pearson correlation", "relate", "any",
        Contract(
            roles_by_layout={"long": (
                Role("x", (N, O, B), "First variable (x)"),
                Role("y", (N, O, B), "Second variable (y)"),
            )},
            canonical="long", pairing="any", min_n=3,
        ),
        "F-ASSOC", "scipy.stats.pearsonr(x, y) + .confidence_interval()",
        "Pearson (1895)",
        "Measure the linear association between two numeric variables (r, r², CI).",
        aliases=("point-biserial", "pearson r", "pearson correlation"),
    ),
    _spec(
        "spearman", "Spearman rank correlation", "relate", "any",
        Contract(
            roles_by_layout={"long": (
                Role("x", (N, O), "First variable (x)"),
                Role("y", (N, O), "Second variable (y)"),
            )},
            canonical="long", pairing="any", min_n=3,
        ),
        "F-ASSOC", "scipy.stats.spearmanr(x, y)",
        "Spearman (1904); Bonett & Wright (2000)",
        "Measure the monotonic (rank) association between two variables (ρ, CI).",
        aliases=("spearman rho", "spearman's rho", "rank correlation"),
    ),
    _spec(
        "kendall", "Kendall's τ-b", "relate", "any",
        Contract(
            roles_by_layout={"long": (
                Role("x", (N, O, B), "First variable (x)"),
                Role("y", (N, O, B), "Second variable (y)"),
            )},
            canonical="long", pairing="any", min_n=3,
        ),
        "F-ASSOC", "scipy.stats.kendalltau(x, y)", "Kendall (1938)",
        "Rank association robust to ties, good for ordinal data (τ-b, CI).",
        aliases=("kendall tau", "kendall's tau-b", "tau-b"),
    ),
    _spec(
        "corr_matrix", "Correlation matrix", "relate", "any",
        Contract(
            roles_by_layout={"long": (
                Role("variables", (N, O), "Variables (2–30)", min=2, max=30),
            )},
            canonical="long", pairing="any",
            params=(Param("method", "choice", "pearson", "Correlation method",
                          choices=("pearson", "spearman")),),
            min_n=3,
        ),
        "F-ASSOC-M", "scipy.stats.pearsonr / spearmanr, pairwise-complete",
        "Holm (1979)",
        "All pairwise correlations among several variables, with a heatmap.",
        aliases=("correlation matrix", "corr matrix"),
    ),
    # --- counts ----------------------------------------------------------
    _spec(
        "chi2_ind", "Chi-square test of independence", "counts", "independent",
        Contract(
            roles_by_layout={
                "long": (
                    Role("row", (C, B, O), "Row variable", levels=(2, 20)),
                    Role("col", (C, B, O), "Column variable", levels=(2, 20)),
                ),
                "table": (Role("counts", (N,), "Count grid", min=1, max=None),),
            },
            canonical="long", pairing="independent", min_n=3,
        ),
        "F-CAT", "scipy.stats.chi2_contingency(xtab, correction=False)",
        "Pearson (1900); Cramér (1946)",
        "Are two categorical variables associated? (no Yates correction).",
        aliases=("chi-square", "chi-squared",
                 "chi-square test of independence", "contingency test"),
    ),
    _spec(
        "chi2_gof", "Chi-square goodness-of-fit", "counts", "any",
        Contract(
            roles_by_layout={
                "long": (Role("category", (C, B, O), "Category", levels=(2, 20)),),
                "table": (
                    Role("category", (C, B, O), "Category labels", levels=(2, 20)),
                    Role("counts", (N,), "Counts"),
                ),
            },
            canonical="long", pairing="any",
            params=(Param("expected", "proportions", (),
                          "Expected proportions (default: equal)",
                          of_role="category"),),
            min_n=3,
        ),
        "F-CAT", "scipy.stats.chisquare(observed, f_exp=N*p)",
        "Pearson (1900); Cohen (1988)",
        "Do observed category counts match expected proportions?",
        aliases=("goodness of fit", "chi-square goodness-of-fit", "gof"),
    ),
    _spec(
        "fisher", "Fisher's exact test", "counts", "independent",
        Contract(
            roles_by_layout={
                "long": (
                    Role("row", (C, B, O), "Row variable", levels=(2, 20)),
                    Role("col", (C, B, O), "Column variable", levels=(2, 20)),
                ),
                "table": (Role("counts", (N,), "Count grid", min=1, max=None),),
            },
            canonical="long", pairing="independent", min_n=3,
        ),
        "F-CAT", "scipy.stats.fisher_exact(xtab) (R×C: Freeman-Halton)",
        "Fisher (1922); Freeman & Halton (1951)",
        "Exact association test for small count tables (2×2 or small R×C).",
        aliases=("fisher exact", "fisher's exact", "fisher-freeman-halton"),
    ),
    _spec(
        "mcnemar", "McNemar's test", "counts", "paired",
        Contract(
            roles_by_layout={"wide": (
                Role("before", (B, C), "Before (2 levels)", levels=(2, 2)),
                Role("after", (B, C), "After (2 levels)", levels=(2, 2)),
            )},
            canonical="wide", pairing="paired", same_level_set=True, min_n=3,
        ),
        "F-CAT",
        "statsmodels mcnemar(table, exact=(b+c<25), correction=True)",
        "McNemar (1947)",
        "Did a yes/no outcome change on the same people/units (before vs after)?",
        aliases=("mcnemar's test", "mcnemar test"),
    ),
    _spec(
        "cochran_q", "Cochran's Q test", "counts", "paired",
        Contract(
            roles_by_layout={"wide": (
                Role("measures", (B,), "Binary repeated measures (3–20)",
                     min=3, max=20),
            )},
            canonical="wide", pairing="paired", same_level_set=True, min_n=3,
        ),
        "F-CAT", "statsmodels cochrans_q(X)",
        "Cochran (1950); Holm (1979)",
        "Did a yes/no outcome differ across three or more repeated conditions?",
        aliases=("cochran q", "cochran's q"),
    ),
    _spec(
        "prop_1", "One-sample proportion test", "counts", "any",
        Contract(
            roles_by_layout={"long": (
                Role("outcome", (B, C), "Outcome (2 levels)", levels=(2, 2)),
            )},
            canonical="long", pairing="any",
            params=(
                Param("success", "level", None, "Success level", of_role="outcome"),
                Param("p0", "float", 0.5, "Hypothesised proportion (p₀)"),
            ),
            min_n=3,
        ),
        "F-CAT",
        "scipy.stats.binomtest(k, n, p0) + statsmodels proportion_confint(method='wilson')",
        "Wilson (1927)",
        "Is a single proportion different from a fixed value? (exact binomial).",
        aliases=("one-sample proportion", "binomial test", "one-sample z"),
    ),
    _spec(
        "prop_2", "Two-sample proportion z-test", "counts", "independent",
        Contract(
            roles_by_layout={"long": (
                Role("outcome", (B, C), "Outcome (2 levels)", levels=(2, 2)),
                Role("group", (B, C), "Group (2 levels)", levels=(2, 2)),
            )},
            canonical="long", pairing="independent",
            params=(Param("success", "level", None, "Success level",
                          of_role="outcome"),),
            min_n=3,
        ),
        "F-CAT",
        "statsmodels proportions_ztest + confint_proportions_2indep(method='newcomb')",
        "Newcombe (1998)",
        "Compare a yes/no proportion between two independent groups.",
        aliases=("two-sample proportion", "proportion z-test", "two-proportion z"),
    ),
    # --- predict ---------------------------------------------------------
    _spec(
        "ols_simple", "Simple linear regression", "predict", "any",
        Contract(
            roles_by_layout={"long": (
                Role("outcome", (N,), "Outcome (numeric)"),
                Role("x", (N, O, B), "Predictor (x)"),
            )},
            canonical="long", pairing="any", min_n=3,
        ),
        "F-REG", "statsmodels OLS(y, add_constant(x)).fit()", "Cohen (1988)",
        "Predict a numeric outcome from one predictor (slope, R², CI).",
        aliases=("simple linear regression", "simple regression",
                 "linear regression"),
    ),
    _spec(
        "ols_multi", "Multiple linear regression", "predict", "any",
        Contract(
            roles_by_layout={"long": (
                Role("outcome", (N,), "Outcome (numeric)"),
                Role("predictors", (N, O, B, C), "Predictors (1–15)",
                     min=1, max=15),
            )},
            canonical="long", pairing="any",
            params=(Param("reference", "level", None,
                          "Reference level (per categorical predictor)",
                          of_role="predictors"),),
            min_n=3,
        ),
        "F-REG", "statsmodels OLS with get_dummies(drop_first=True, dtype=float)",
        "Cohen (1988)",
        "Predict a numeric outcome from several predictors (coefficients, R²).",
        aliases=("multiple linear regression", "multiple regression"),
    ),
    _spec(
        "logistic", "Binary logistic regression", "predict", "any",
        Contract(
            roles_by_layout={"long": (
                Role("outcome", (B, C), "Outcome (2 levels)", levels=(2, 2)),
                Role("predictors", (N, O, B, C), "Predictors (1–15)",
                     min=1, max=15),
            )},
            canonical="long", pairing="any",
            params=(
                Param("success", "level", None, "Success (event) level",
                      of_role="outcome"),
                Param("reference", "level", None,
                      "Reference level (per categorical predictor)",
                      of_role="predictors"),
            ),
            min_n=3,
        ),
        "F-REG", "statsmodels Logit(y01, X).fit(disp=0)", "McFadden (1974)",
        "Predict a yes/no outcome from one or more predictors (odds ratios).",
        aliases=("logistic regression", "binary logistic", "logit"),
    ),
)

REGISTRY: dict[str, TestSpec] = {s.id: s for s in SPECS}


# --------------------------------------------------------------------------
# Wiring: point every spec's `run` at its real runner (PLAN §10).
#
# TestSpec is frozen, so we set `run` via object.__setattr__ after construction.
# The runner modules import only model/check/assumptions/effects/... never this
# module, so importing them here at the BOTTOM (after REGISTRY exists) cannot
# form a cycle. A missing runner is a build error, not a silent `_todo`.
# --------------------------------------------------------------------------
def _wire() -> None:
    from . import advise, assess, assoc, cat, means, props, ranks, regress
    runners: dict[str, object] = {}
    for module in (means, ranks, cat, props, assoc, regress, assess):
        runners.update(module.RUNNERS)
    missing = set(REGISTRY) - set(runners)
    if missing:  # pragma: no cover - guarded against a half-wired registry
        raise RuntimeError(f"no runner for registry id(s): {sorted(missing)}")
    # Chunk 11: every runner is wrapped so its Result carries the data-driven
    # advisories (D1-D14) + the robust-alternative agreement, deduped by D-code.
    # advise.advised keeps __wrapped__ pointing at the raw runner.
    for tid, fn in runners.items():
        object.__setattr__(REGISTRY[tid], "run", advise.advised(fn))


def _wire_sentences() -> None:
    """Point every spec's `sentence` at the real plain-English renderer.

    `sentences` imports only model/fmt (never this module), so importing it here
    at the BOTTOM cannot form a cycle. A spec left as the `_todo` stub is a build
    error, asserted by tests/test_sentence_wiring.py."""
    from . import sentences
    missing = set(REGISTRY) - set(sentences.SENTENCES)
    if missing:  # pragma: no cover - guarded against a half-wired registry
        raise RuntimeError(f"no sentence for registry id(s): {sorted(missing)}")
    for tid, fn in sentences.SENTENCES.items():
        object.__setattr__(REGISTRY[tid], "sentence", fn)


def _wire_charts() -> None:
    """Point every spec's `chart` at the real chart dispatcher (Chunk 13).

    `charts` imports only model-free helpers + matplotlib/scipy/pandas (never this
    module), so importing it here at the BOTTOM cannot form a cycle. Every spec
    gets a callable `chart` (Result -> ((caption, png_bytes), ...)); tests with no
    §7 chart return () rather than raising, so the app never crashes. A spec left
    as the `_todo` stub is a build error, asserted by tests/test_registry.py."""
    from . import charts
    missing = set(REGISTRY) - set(charts.CHARTS)
    if missing:  # pragma: no cover - guarded against a half-wired registry
        raise RuntimeError(f"no chart for registry id(s): {sorted(missing)}")
    for tid, fn in charts.CHARTS.items():
        object.__setattr__(REGISTRY[tid], "chart", fn)


_wire()
_wire_sentences()
_wire_charts()
