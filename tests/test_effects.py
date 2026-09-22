"""Golden tests for statkit.effects (PLAN §9.1 effects RED list, §3 thresholds).

Every expected value is either hand-computed from the definition (a known-good
literal, independent of the module's code) or cross-checked against an
orthogonal scipy route. Effect sizes are ours because scipy/statsmodels ship
none (PLAN D7).
"""
import numpy as np
import pytest
from scipy.stats import f_oneway
from scipy.stats.contingency import association

from statkit import effects

# Two samples with hand-computable pooled SD (both SDs = sqrt(2.5) = 1.58114).
A = [1.0, 2.0, 3.0, 4.0, 5.0]   # mean 3
B = [3.0, 4.0, 5.0, 6.0, 7.0]   # mean 5


# ------------------------- Cohen's d / Hedges' g -------------------------

def test_pooled_sd_hand_value():
    assert effects.pooled_sd(A, B) == pytest.approx(1.58114, abs=1e-4)


def test_cohens_d_hand_value():
    # (5 - 3) / 1.58114 = 1.26491 ; first-minus-second convention
    assert effects.cohens_d(B, A) == pytest.approx(1.26491, abs=1e-4)
    assert effects.cohens_d(A, B) == pytest.approx(-1.26491, abs=1e-4)


def test_hedges_correction_at_df8():
    # J = 1 - 3/(4*df - 1), df = n1+n2-2 = 8  ->  1 - 3/31 = 0.903226
    assert effects.hedges_correction(8) == pytest.approx(0.903226, abs=1e-5)


def test_hedges_g_is_d_times_J():
    # 1.26491 * 0.903226 = 1.14250
    assert effects.hedges_g(B, A) == pytest.approx(1.14250, abs=1e-4)


def test_cohens_d_onesample():
    x = [2.0, 4.0, 6.0, 8.0, 10.0]   # mean 6, sd sqrt(10)=3.16228
    # (6 - 4) / 3.16228 = 0.63246
    assert effects.cohens_d_onesample(x, 4.0) == pytest.approx(0.63246, abs=1e-4)


def test_cohens_dz_paired():
    diff = [1.0, 2.0, 3.0, 4.0, 5.0]   # mean 3, sd 1.58114
    assert effects.cohens_dz(diff) == pytest.approx(3.0 / 1.58114, abs=1e-4)


# ------------------------- ANOVA family -------------------------
# One-way with SS_between=54, SS_within=6, SS_total=60 (hand-computed).
G1 = [1.0, 2.0, 3.0]
G2 = [4.0, 5.0, 6.0]
G3 = [7.0, 8.0, 9.0]


def test_eta_squared_hand_and_orthogonal_to_F():
    eta2 = effects.eta_squared(G1, G2, G3)
    assert eta2 == pytest.approx(0.9, abs=1e-9)          # 54/60
    # orthogonal: eta^2 = df_b*F / (df_b*F + df_w) with F from scipy
    F, _ = f_oneway(G1, G2, G3)
    df_b, df_w = 2, 6
    assert eta2 == pytest.approx(df_b * F / (df_b * F + df_w), rel=1e-9)


def test_omega_squared_hand_value():
    # (54 - 2*(6/6)) / (60 + 1) = 52/61 = 0.852459
    assert effects.omega_squared(G1, G2, G3) == pytest.approx(0.852459, abs=1e-5)


def test_partial_eta_squared_from_F():
    # f*df1 / (f*df1 + df2)
    assert effects.partial_eta_squared(4.0, 2, 12) == pytest.approx(8 / 20, abs=1e-9)


def test_epsilon_squared_kruskal():
    # H=7.2 (Kruskal on the three groups), (H - k + 1)/(N - k) = 5.2/6
    assert effects.epsilon_squared(7.2, 3, 9) == pytest.approx(5.2 / 6, abs=1e-9)


def test_kendalls_w():
    # chi2 / (n*(k-1)) = 8 / (5*3) = 0.53333
    assert effects.kendalls_w(8.0, 5, 4) == pytest.approx(8 / 15, abs=1e-9)


# ------------------------- rank-biserial -------------------------

def test_rank_biserial_from_U():
    # 1 - 2U/(n1 n2) = 1 - 4/25 = 0.84
    assert effects.rank_biserial_u(2.0, 5, 5) == pytest.approx(0.84, abs=1e-9)


def test_rank_biserial_wilcoxon():
    # (W+ - W-)/(W+ + W-)
    assert effects.rank_biserial_wilcoxon(18.0, 3.0) == pytest.approx(15 / 21, abs=1e-9)


# ------------------------- contingency effect sizes -------------------------
T2 = np.array([[10, 20], [30, 15]])
T3 = np.array([[10, 20, 30], [6, 9, 17], [8, 8, 4]])


def test_phi_2x2():
    # phi = sqrt(chi2/N), uncorrected chi2
    assert effects.phi(T2) == pytest.approx(0.327327, abs=1e-5)


def test_cramers_v_matches_scipy_uncorrected():
    # PLAN §3: uncorrected Cramer's V (scipy association is the orthogonal oracle)
    v2 = effects.cramers_v(T2)
    assert v2 == pytest.approx(association(T2, method="cramer", correction=False), rel=1e-9)
    v3 = effects.cramers_v(T3)
    assert v3 == pytest.approx(association(T3, method="cramer", correction=False), rel=1e-9)


def test_cohens_w_gof():
    # sqrt(chi2/N) = sqrt(10/50) = 0.447214
    assert effects.cohens_w(10.0, 50) == pytest.approx(0.447214, abs=1e-5)


def test_cohens_h():
    # 2asin(sqrt(.6)) - 2asin(sqrt(.4)) = 0.402715
    assert effects.cohens_h(0.6, 0.4) == pytest.approx(0.402715, abs=1e-5)


def test_cohens_f2():
    # R2/(1-R2) = 0.36/0.64 = 0.5625
    assert effects.cohens_f2(0.36) == pytest.approx(0.5625, abs=1e-9)


# ------------------------- magnitude labels (PLAN §3 thresholds) -------------------------

def test_d_labels():
    assert effects.d_label(0.1) == "negligible"
    assert effects.d_label(0.3) == "small"
    assert effects.d_label(0.6) == "medium"
    assert effects.d_label(0.9) == "large"
    assert effects.d_label(-0.9) == "large"      # sign-independent


def test_r_labels():
    assert effects.r_label(0.05) == "negligible"
    assert effects.r_label(0.2) == "small"
    assert effects.r_label(0.4) == "medium"
    assert effects.r_label(0.6) == "large"


def test_eta_labels():
    assert effects.eta_label(0.005) == "negligible"
    assert effects.eta_label(0.03) == "small"
    assert effects.eta_label(0.10) == "medium"
    assert effects.eta_label(0.20) == "large"


def test_cramers_v_label_scales_thresholds():
    # 2x2: min(r,c)-1 = 1 -> thresholds .1/.3/.5 like phi
    assert effects.cramers_v_label(0.05, 2, 2) == "negligible"
    assert effects.cramers_v_label(0.2, 2, 2) == "small"
    # 3x3: divide by sqrt(2) -> small threshold .0707, medium .2121, large .3536
    assert effects.cramers_v_label(0.05, 3, 3) == "negligible"
    assert effects.cramers_v_label(0.10, 3, 3) == "small"     # .10 > .0707
    assert effects.cramers_v_label(0.25, 3, 3) == "medium"    # .25 > .2121
    assert effects.cramers_v_label(0.40, 3, 3) == "large"     # .40 > .3536


def test_or_label_folds_below_one():
    # Chen (2010) OR thresholds 1.68/3.47/6.71, strength symmetric about 1
    assert effects.or_label(1.5) == "negligible"
    assert effects.or_label(2.0) == "small"
    assert effects.or_label(4.0) == "medium"
    assert effects.or_label(7.0) == "large"
    assert effects.or_label(1 / 4.0) == "medium"    # 0.25 folds to 4.0


def test_f2_labels():
    assert effects.f2_label(0.01) == "negligible"
    assert effects.f2_label(0.05) == "small"
    assert effects.f2_label(0.20) == "medium"
    assert effects.f2_label(0.40) == "large"
