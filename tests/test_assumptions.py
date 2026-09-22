"""Golden tests for statkit.assumptions (PLAN §9.1 assumptions RED list, §3).

Key requirements verified here:
  * Shapiro-Wilk is SKIPPED at n<3 and n>5000 (returns a Check with passed=None).
  * lilliefors is NOT naive KS: a mean-5 normal sample passes lilliefors
    (params estimated) but is rejected catastrophically by kstest vs N(0,1).
  * homogeneity via Brown-Forsythe = levene(center="median"), NAMED correctly.
"""
import numpy as np
import pytest
from scipy.stats import bartlett, kstest, levene

from statkit import assumptions
from statkit.model import Check


# ------------------------- Shapiro skip rules -------------------------

def test_shapiro_skipped_below_n3():
    c = assumptions.shapiro_normality([1.0, 2.0])
    assert isinstance(c, Check)
    assert c.passed is None
    assert c.statistic is None and c.p is None
    assert "not run" in c.note.lower()
    assert "shapiro" in c.name.lower()


def test_shapiro_skipped_above_n5000():
    x = list(np.linspace(0.0, 1.0, 5001))   # 5001 > 5000
    c = assumptions.shapiro_normality(x)
    assert c.passed is None
    assert c.statistic is None and c.p is None
    assert "5000" in c.note


def test_shapiro_runs_at_n3_and_n5000_boundaries():
    # n=3 and n=5000 are INSIDE the allowed 3<=n<=5000 window: they must run.
    lo = assumptions.shapiro_normality([1.0, 2.0, 4.0])
    assert lo.passed is not None and lo.statistic is not None and lo.p is not None
    rng = np.random.default_rng(0)
    hi = assumptions.shapiro_normality(rng.normal(size=5000))
    assert hi.passed is not None and hi.p is not None


def test_shapiro_passes_normal_fails_skewed():
    rng = np.random.default_rng(7)
    normal = rng.normal(0, 1, 60)
    c_norm = assumptions.shapiro_normality(normal)
    assert c_norm.passed is True          # p >= .05

    skewed = [1.0] * 20 + [100.0]         # blatantly non-normal
    c_sk = assumptions.shapiro_normality(skewed)
    assert c_sk.passed is False           # p < .05


# ------------------------- lilliefors != naive KS -------------------------

def test_lilliefors_is_not_naive_ks_on_mean5_sample():
    rng = np.random.default_rng(42)
    x = rng.normal(5.0, 1.0, 80)
    c = assumptions.lilliefors_normality(x)
    # lilliefors estimates mu/sigma -> normal sample passes
    assert c.passed is True
    assert c.p > 0.05
    # a naive KS against the standard normal N(0,1) rejects it catastrophically
    assert kstest(x, "norm").pvalue < 1e-10
    # and the module's lilliefors p is nowhere near that KS p (they are different tests)
    assert c.p > 0.5


# ------------------------- homogeneity: Brown-Forsythe naming -------------------------

def test_brown_forsythe_named_and_is_levene_median():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [2.0, 4.0, 6.0, 8.0, 10.0]
    d = [1.0, 1.0, 1.0, 2.0, 20.0]
    c = assumptions.brown_forsythe(a, b, d)
    assert isinstance(c, Check)
    assert c.name.startswith("Brown-Forsythe")
    # must be levene with center="median" (NOT mean-centred classic Levene)
    ref = levene(a, b, d, center="median")
    assert c.statistic == pytest.approx(ref.statistic, rel=1e-12)
    assert c.p == pytest.approx(ref.pvalue, rel=1e-12)
    # and it is genuinely different from the mean-centred variant on this data
    assert c.statistic != pytest.approx(levene(a, b, d, center="mean").statistic)


def test_bartlett_wraps_scipy():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [2.0, 4.0, 6.0, 8.0, 10.0]
    c = assumptions.bartlett_variance(a, b)
    assert c.name.startswith("Bartlett")
    ref = bartlett(a, b)
    assert c.statistic == pytest.approx(ref.statistic, rel=1e-12)
    assert c.p == pytest.approx(ref.pvalue, rel=1e-12)


# ------------------------- S-O: non-finite p is "not computable" -----------

def test_non_finite_p_is_not_computable_not_failed():
    # A nan p (constant / degenerate data) is NOT a failed check -- bool(nan>=alpha)
    # is False, which would silently claim "not met". It must read "not computable".
    const = [3.0] * 5
    for c in (assumptions.shapiro_normality(const),
              assumptions.lilliefors_normality(const),
              assumptions.brown_forsythe([1.0] * 4, [2.0] * 4),
              assumptions.bartlett_variance([1.0] * 4, [2.0] * 4)):
        assert c.passed is None
        assert c.statistic is None and c.p is None
        assert "not computable" in c.note.lower()


# ------------------------- skew / kurtosis helper -------------------------

def test_skew_kurtosis_symmetric_is_zero_skew():
    skew, kurt = assumptions.skew_kurtosis([1.0, 2.0, 3.0, 4.0, 5.0])
    assert skew == pytest.approx(0.0, abs=1e-9)
    assert isinstance(kurt, float)
