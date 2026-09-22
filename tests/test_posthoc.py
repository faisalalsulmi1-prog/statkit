"""Golden tests for statkit.posthoc (PLAN §9.1 posthoc RED list, §3, DoD §10).

Goldens & their sources (R is NOT installed on this machine, so PLAN's
"constants from R" is replaced by PUBLISHED worked examples + orthogonal
internal checks; both routes must agree):

  * Tukey HSD  -> asserted against scipy.stats.tukey_hsd directly (that IS an
    independent oracle, present in scipy 1.18.1).

  * Games-Howell -> PUBLISHED: rstatix `games_howell_test(len ~ dose)` on R's
    ToothGrowth (datanovia rstatix reference docs). Published estimates
    9.13 / 15.5 / 6.37, 95% CIs [5.69,12.6] / [12.3,18.7] / [3.19,9.54],
    adjusted p 3.76e-7 / ~0 / 5.57e-5. ToothGrowth group means checksummed to
    the known 10.605 / 19.735 / 26.100 so a bad transcription fails loudly.
    ORTHOGONAL: for k=2 groups Games-Howell reduces EXACTLY to Welch's t-test
    (scipy.stats.ttest_ind(equal_var=False)) -- a fully independent code path.

  * Dunn -> PUBLISHED: FSA::dunnTest via the rcompanion handbook (F_08),
    Pooh/Piglet/Tigger Likert data (heavy ties -> tests the tie correction).
    Published z / unadjusted p: 3.7702412 / 1.630898e-4, 0.4813074 / 0.6302980,
    -3.2889338 / 1.0056766e-3. ORTHOGONAL: for k=2 groups Dunn's z equals the
    tie-corrected Mann-Whitney normal-approximation z (no continuity), via
    scipy.stats.mannwhitneyu -- an independent route. Holm adjustment
    hand-computed (step-down) as an independent golden.
"""
import numpy as np
import pytest
from scipy.stats import mannwhitneyu, rankdata, ttest_ind, tukey_hsd

from statkit import posthoc

# R ToothGrowth `len` grouped by dose (means checksummed in the test).
TOOTH = {
    "0.5": [4.2, 11.5, 7.3, 5.8, 6.4, 10, 11.2, 11.2, 5.2, 7,
            15.2, 21.5, 17.6, 9.7, 14.5, 10, 8.2, 9.4, 16.5, 9.7],
    "1": [16.5, 16.5, 15.2, 17.3, 22.5, 17.3, 13.6, 14.5, 18.8, 15.5,
          19.7, 23.3, 23.6, 26.4, 20, 25.2, 25.8, 21.2, 14.5, 27.3],
    "2": [23.6, 18.5, 33.9, 25.5, 26.4, 32.5, 26.7, 21.5, 23.3, 29.5,
          25.5, 26.4, 22.4, 24.5, 24.8, 30.9, 26.4, 27.3, 29.4, 23],
}

DUNN = {
    "Pooh":   [3, 5, 4, 4, 4, 4, 4, 4, 5, 5],
    "Piglet": [2, 4, 2, 2, 1, 2, 3, 2, 2, 3],
    "Tigger": [4, 4, 4, 4, 5, 3, 5, 4, 4, 3],
}


def _row(df, g1, g2):
    m = df[(df.group1 == g1) & (df.group2 == g2)]
    assert len(m) == 1, f"expected one row for ({g1},{g2})"
    return m.iloc[0]


# ------------------------- Tukey HSD (oracle = scipy) -------------------------

def test_tukey_matches_scipy():
    groups = {"a": [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10], "c": [2, 4, 6, 8, 10]}
    df = posthoc.tukey_hsd(groups)
    assert list(df.columns) == ["group1", "group2", "meandiff", "ci_low", "ci_high", "p"]
    assert len(df) == 3            # 3 pairs

    ref = tukey_hsd(*groups.values())
    ci = ref.confidence_interval(0.95)
    names = list(groups)
    for i in range(3):
        for j in range(i + 1, 3):
            r = _row(df, names[i], names[j])
            assert r.meandiff == pytest.approx(ref.statistic[i, j])
            assert r.p == pytest.approx(ref.pvalue[i, j])
            assert r.ci_low == pytest.approx(ci.low[i, j])
            assert r.ci_high == pytest.approx(ci.high[i, j])


# ------------------------- Games-Howell -------------------------

def test_tooth_growth_checksum():
    # guards the transcription of the published dataset
    assert np.mean(TOOTH["0.5"]) == pytest.approx(10.605, abs=1e-3)
    assert np.mean(TOOTH["1"]) == pytest.approx(19.735, abs=1e-3)
    assert np.mean(TOOTH["2"]) == pytest.approx(26.100, abs=1e-3)


def test_games_howell_published_rstatix_toothgrowth():
    df = posthoc.games_howell(TOOTH)
    assert list(df.columns) == [
        "group1", "group2", "meandiff", "se", "statistic", "df", "ci_low", "ci_high", "p"]
    assert len(df) == 3

    r01 = _row(df, "0.5", "1")
    assert abs(r01.meandiff) == pytest.approx(9.13, abs=0.01)      # rstatix estimate
    assert r01.p == pytest.approx(3.763e-07, rel=1e-2)            # published p.adj 3.76e-7
    assert {round(abs(r01.ci_low), 2), round(abs(r01.ci_high), 2)} == {5.69, 12.57}

    r02 = _row(df, "0.5", "2")
    assert abs(r02.meandiff) == pytest.approx(15.495, abs=0.01)    # published ~15.5
    assert r02.p == pytest.approx(1.368e-13, rel=1e-2)            # published ~0

    r12 = _row(df, "1", "2")
    assert abs(r12.meandiff) == pytest.approx(6.365, abs=0.01)     # published ~6.37
    assert r12.p == pytest.approx(5.569e-05, rel=1e-2)           # published 5.57e-5
    assert {round(abs(r12.ci_low), 2), round(abs(r12.ci_high), 2)} == {3.19, 9.54}


def test_games_howell_k2_equals_welch_orthogonal():
    two = {"A": TOOTH["0.5"], "B": TOOTH["1"]}
    df = posthoc.games_howell(two)
    assert len(df) == 1
    gh_p = df.iloc[0].p
    welch = ttest_ind(np.asarray(TOOTH["0.5"], float),
                      np.asarray(TOOTH["1"], float), equal_var=False)
    assert gh_p == pytest.approx(welch.pvalue, rel=1e-9)


# ------------------------- Dunn -------------------------

def test_dunn_published_rcompanion_values():
    df = posthoc.dunn(DUNN)
    assert list(df.columns) == ["group1", "group2", "z", "p", "p_holm"]
    assert len(df) == 3

    # pair order = dict insertion: Pooh-Piglet, Pooh-Tigger, Piglet-Tigger
    r_pp = _row(df, "Pooh", "Piglet")
    assert r_pp.z == pytest.approx(3.7702412, abs=1e-6)      # published |Z|, sign per order
    assert r_pp.p == pytest.approx(1.630898e-4, rel=1e-5)

    r_pt = _row(df, "Pooh", "Tigger")
    assert r_pt.z == pytest.approx(0.4813074, abs=1e-6)
    assert r_pt.p == pytest.approx(0.6302980448, rel=1e-6)

    r_pigt = _row(df, "Piglet", "Tigger")
    assert r_pigt.z == pytest.approx(-3.2889338, abs=1e-6)
    assert r_pigt.p == pytest.approx(1.0056766e-3, rel=1e-5)


def test_dunn_holm_adjustment_hand_computed():
    # raw p (dict order) = [1.630898e-4, 0.6302980448, 1.0056766e-3]
    # step-down Holm (m=3): sort asc -> *3, *2, *1 with running max
    #   1.630898e-4*3 = 4.892694e-4
    #   1.0056766e-3*2 = 2.0113532e-3  (running max)
    #   0.6302980448*1 = 0.6302980448  (running max)
    df = posthoc.dunn(DUNN)
    r_pp = _row(df, "Pooh", "Piglet")
    r_pt = _row(df, "Pooh", "Tigger")
    r_pigt = _row(df, "Piglet", "Tigger")
    assert r_pp.p_holm == pytest.approx(4.892694e-4, rel=1e-4)
    assert r_pigt.p_holm == pytest.approx(2.0113532e-3, rel=1e-4)
    assert r_pt.p_holm == pytest.approx(0.6302980448, rel=1e-6)
    # Holm never decreases a p-value
    assert (df.p_holm >= df.p - 1e-12).all()


def test_dunn_k2_equals_tie_corrected_mwu_z_orthogonal():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8], float)
    y = np.array([3, 4, 5, 6, 9, 10, 11, 2], float)
    df = posthoc.dunn({"x": x, "y": y})
    z_dunn = df.iloc[0].z

    # independent route: tie-corrected MWU normal approx z, no continuity
    allv = np.concatenate([x, y])
    N = len(allv)
    _, cnts = np.unique(allv, return_counts=True)
    tie = np.sum(cnts ** 3 - cnts)
    U, _ = mannwhitneyu(x, y, alternative="two-sided", use_continuity=False, method="asymptotic")
    mu = len(x) * len(y) / 2
    sig = np.sqrt(len(x) * len(y) / 12 * ((N + 1) - tie / (N * (N - 1))))
    z_mwu = (U - mu) / sig
    assert abs(z_dunn) == pytest.approx(abs(z_mwu), rel=1e-9)
