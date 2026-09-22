"""RED list for statkit.fmt (PLAN §6 + §9.1 fmt list).

These pin the number/statistic formatting rules the whole sentence + report
layer will read: APA p-values (no leading zero, never "p = .000"), the
int-collapsing num(), and the CI string. Expected values are literals from
PLAN §6 / §9.1, not recomputed from the code.
"""
import math

from statkit import fmt

INF = float("inf")
NAN = float("nan")
DASH = "—"  # em dash — the deterministic non-finite render


# --- p(): APA p-value (§6, §9.1) -----------------------------------------
def test_p_below_point_001_is_lt():
    assert fmt.p(0.0004) == "p < .001"


def test_p_exact_value_drops_leading_zero():
    assert fmt.p(0.032) == "p = .032"


def test_p_one_keeps_three_decimals_and_leading_digit():
    assert fmt.p(1.0) == "p = 1.000"


def test_p_zero_never_prints_p_equals_000():
    # A raw 0.0 formats to "0.000"; the < .001 branch must catch it first.
    assert fmt.p(0.0) == "p < .001"


def test_p_at_the_point_001_boundary_is_equals():
    # 0.001 is NOT < .001, so it prints exactly; still never ".000".
    assert fmt.p(0.001) == "p = .001"


# --- num(): int-valued collapses, else d decimals (§9.1) ------------------
def test_num_int_valued_float_prints_as_int():
    assert fmt.num(2.0) == "2"


def test_num_rounds_to_two_decimals():
    assert fmt.num(2.345) == "2.35"


def test_num_non_integer_keeps_trailing_zero():
    assert fmt.num(1.2) == "1.20"


def test_num_negative_zero_is_plain_zero():
    assert fmt.num(-0.0) == "0"


# --- ci(): the 95% CI string (§6) ----------------------------------------
def test_ci_formats_two_decimals_each_bound():
    assert fmt.ci(1.2, 3.4) == "95% CI [1.20, 3.40]"


# --- stat(): "name = value" (§6) -----------------------------------------
def test_stat_pairs_name_with_num_value():
    assert fmt.stat("t", 2.31) == "t = 2.31"


# --- pct(): proportion -> percent string (§6; semantics pinned here) ------
def test_pct_treats_input_as_a_proportion():
    assert fmt.pct(0.523) == "52.3%"


# The Chunk-12 pct() in-vs-percent-points decision (STATE parked question):
# props.py runners carry FRACTIONS — prop_1 estimate = k/n, its Wilson CI as
# fractions, prop_2 estimate = p1 - p2 (a fraction difference that can be
# negative). So pct(x) = x*100 is the correct, consistent rendering for exactly
# what a proportion Result carries. These pin that so the sentence layer and any
# future edit stay honest.
def test_pct_renders_a_prop_1_estimate_fraction():
    # prop_1: estimate ("proportion 'Yes'", 0.65) -> "65.0%"
    assert fmt.pct(0.65) == "65.0%"


def test_pct_renders_a_prop_1_wilson_bound_fraction():
    assert fmt.pct(0.52) == "52.0%" and fmt.pct(0.76) == "76.0%"


def test_pct_renders_a_negative_prop_2_difference_fraction():
    # prop_2: estimate can be a negative fraction (p1 - p2) -> "-8.0%"
    assert fmt.pct(-0.08) == "-8.0%"


# --- non-finite safety (F2): inf/-inf/nan can never crash or leak "inf"/"nan"
# The real app hits these: an empty 2x2 cell -> Fisher's infinite odds ratio,
# a perfect-fit regression -> Cohen's f2 = inf. Every formatter renders a
# non-finite numeric as the em dash so the downstream sentence layer is
# deterministic; none may raise. Contextual wording is a later worker's job.
def test_num_positive_infinity_is_em_dash():
    assert fmt.num(INF) == DASH


def test_num_negative_infinity_is_em_dash():
    assert fmt.num(-INF) == DASH


def test_num_nan_is_em_dash():
    assert fmt.num(NAN) == DASH


def test_p_non_finite_is_em_dash():
    # A real p is always finite; nan/inf/-inf are a guard, never "p = nan"
    # and never the misleading "p < .001" that -inf would otherwise hit.
    assert fmt.p(NAN) == DASH
    assert fmt.p(INF) == DASH
    assert fmt.p(-INF) == DASH


def test_ci_non_finite_upper_bound_is_em_dash():
    # Fisher's infinite odds ratio: the finite bound still renders.
    assert fmt.ci(1.5, INF) == "95% CI [1.50, " + DASH + "]"


def test_ci_both_bounds_non_finite_are_em_dashes():
    assert fmt.ci(-INF, INF) == "95% CI [" + DASH + ", " + DASH + "]"


def test_pct_non_finite_is_em_dash():
    assert fmt.pct(NAN) == DASH
    assert fmt.pct(INF) == DASH


def test_stat_non_finite_value_is_em_dash_via_num():
    # stat delegates to num, so a non-finite statistic reads "f2 = —".
    assert fmt.stat("f2", INF) == "f2 = " + DASH


def test_non_finite_formatters_never_raise():
    for x in (INF, -INF, NAN):
        fmt.num(x)
        fmt.p(x)
        fmt.pct(x)
        fmt.stat("s", x)
        fmt.ci(x, 1.0)
        fmt.ci(1.0, x)
        fmt.ci(x, x)


# --- finite regression: guarding must not drift any existing output --------
def test_finite_outputs_unchanged_after_non_finite_guards():
    # Literals pinned from the pre-fix code (verified independently), so this
    # proves the non-finite guards left every finite path byte-identical.
    assert fmt.num(2.0) == "2"
    assert fmt.num(2.345) == "2.35"
    assert fmt.num(1.2) == "1.20"
    assert fmt.num(-0.0) == "0"
    assert fmt.p(0.032) == "p = .032"
    assert fmt.p(0.0) == "p < .001"
    assert fmt.p(1.0) == "p = 1.000"
    assert fmt.ci(1.2, 3.4) == "95% CI [1.20, 3.40]"
    assert fmt.ci(-1.5, 2.5, 90) == "90% CI [-1.50, 2.50]"
    assert fmt.pct(0.523) == "52.3%"
    assert fmt.pct(-0.08) == "-8.0%"
    assert fmt.stat("t", 2.31) == "t = 2.31"
