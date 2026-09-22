"""RED-first advisories D1-D14 + robust-alternative agreement (Chunk 11, PLAN §5.2).

``advise(bound)`` is the DATA-DRIVEN companion to the structural ``check(bound)``
(S1-S21): it inspects the canonical frame and produces plain, non-alarming
``Finding``s (Brown-Forsythe, Shapiro, expected counts, separation, outliers,
group balance, ...), each carrying its D-code. ``robust_agreement(bound, result)``
runs the parametric test's rank/exact alternative IN THE BACKGROUND and reports
whether the two AGREE on the significance decision -- a reassurance (info) or a
caution (flag), never a second p shown as the result. ``advised(runner)`` is the
single wiring seam: it attaches the advisories to the bound BEFORE the runner
computes (snag 10.3) and dedups every D-code so the runner-emitted D3/D6 are
never doubled.

Seams under test (pre-agreed): ``advise.advise``, ``advise.robust_agreement`` and
the wired ``REGISTRY[id].run`` (dedup + carried advisories).
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from statkit import advise, bind
from statkit.infer import Dataset
from statkit.model import ColumnProfile, Finding, Kind, Result
from statkit.registry import REGISTRY

N, O, CAT, B, ID = (Kind.NUMERIC, Kind.ORDINAL, Kind.CATEGORICAL, Kind.BINARY, Kind.ID)


# --------------------------------------------------------------------------
# dataset / bound builders (mirror tests/test_means.py)
# --------------------------------------------------------------------------
def _isna(v):
    return v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v))


def _profile(name, kinds, values):
    non_null = [v for v in values if not _isna(v)]
    if kinds[0] in (N, ID, Kind.DATE, Kind.EMPTY):
        levels = ()
    else:
        seen, levels = set(), []
        for v in non_null:
            s = str(v)
            if s not in seen:
                seen.add(s)
                levels.append(s)
        levels = tuple(levels)
    return ColumnProfile(
        name=name, kinds=tuple(kinds), n_total=len(values),
        n_missing=len(values) - len(non_null), n_levels=len(levels), levels=levels)


def _ds(cols):
    df_cols, profiles, n = {}, [], None
    for name, (values, kinds) in cols.items():
        n = len(values) if n is None else n
        non_null = [v for v in values if not _isna(v)]
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                      for v in non_null)
        if kinds[0] in (N, O, B, ID) and numeric:
            df_cols[name] = pd.array(
                [None if _isna(v) else float(v) for v in values], dtype="float64")
        else:
            df_cols[name] = pd.array(
                [pd.NA if _isna(v) else str(v) for v in values], dtype="string")
        profiles.append(_profile(name, kinds, values))
    return Dataset(df=pd.DataFrame(df_cols), profiles=tuple(profiles),
                   excel_rows=tuple(range(2, 2 + (n or 0))))


def _bound(test_id, cols, columns, layout=None, params=None):
    return bind.bind(REGISTRY[test_id], _ds(cols), columns, layout=layout,
                     params=params)


def _codes(findings):
    return [f.code for f in findings]


def _by(findings, code):
    return [f for f in findings if f.code == code]


def _fake_result(test_id, p):
    return Result(test_id=test_id, test_name=test_id, status="ok", p=p)


# ==========================================================================
# D1 — Brown-Forsythe (unequal variances)
# ==========================================================================
def test_d1_t_ind_unequal_variance_info_when_welch():
    # very different spreads, similar means; Welch (default) -> info reassurance.
    a = [10.0, 10.1, 9.9, 10.0, 10.2, 9.8, 10.1, 9.9]
    b = [1.0, 20.0, 3.0, 18.0, 5.0, 16.0, 2.0, 19.0]
    bnd = _bound("t_ind", {"Y": (a + b, (N,)),
                           "G": (["A"] * 8 + ["B"] * 8, (CAT,))},
                 {"outcome": ("Y",), "group": ("G",)})
    d1 = _by(advise.advise(bnd), "D1")
    assert d1 and d1[0].severity == "info"
    assert "variance" in d1[0].text.lower()


def test_d1_anova_unequal_variance_flags_welch():
    a = [10.0, 10.1, 9.9, 10.0, 10.2, 9.8]
    b = [10.0, 10.2, 9.8, 10.1, 9.9, 10.0]
    c = [1.0, 20.0, 3.0, 18.0, 5.0, 16.0]
    bnd = _bound("anova_1w", {"Y": (a + b + c, (N,)),
                              "G": (["A"] * 6 + ["B"] * 6 + ["C"] * 6, (CAT,))},
                 {"outcome": ("Y",), "group": ("G",)})
    d1 = _by(advise.advise(bnd), "D1")
    assert d1 and d1[0].severity == "flag"
    assert d1[0].suggest_params  # suggests welch=True


def _assert_no_zero_p(text):
    # B2: a genuinely tiny p must never print as the meaningless p = 0.000 / .000 / 0.
    assert "p = 0.000" not in text, text
    assert "p = .000" not in text, text
    assert "p = 0" not in text, text          # bare zero (also catches "p = 0.000")


def test_d1_anova_tiny_p_prints_p_lt_001_not_zero():
    # variances wildly unequal with real n -> Brown-Forsythe p is genuinely < .001,
    # which raw f"{p:.3f}" renders as the meaningless "p = 0.000" in the report's
    # advisory. It must route through fmt.p and read "p < .001".
    a = [10.0, 10.1, 9.9, 10.0, 10.05, 9.95] * 3
    b = [10.0, 10.2, 9.8, 10.1, 9.9, 10.0] * 3
    c = [1.0, 40.0, 3.0, 38.0, 5.0, 36.0] * 3          # huge spread
    bnd = _bound("anova_1w", {"Y": (a + b + c, (N,)),
                              "G": (["A"] * 18 + ["B"] * 18 + ["C"] * 18, (CAT,))},
                 {"outcome": ("Y",), "group": ("G",)})
    d1 = _by(advise.advise(bnd), "D1")
    assert d1 and d1[0].severity == "flag"
    assert "p < .001" in d1[0].text
    _assert_no_zero_p(d1[0].text)


# ==========================================================================
# D2 — Shapiro non-normal, small n -> suggest rank alternative
# ==========================================================================
def test_d2_non_normal_small_n_suggests_rank():
    a = [1.0, 1.0, 1.0, 1.0, 2.0, 50.0]          # strong right skew
    b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    bnd = _bound("t_ind", {"Y": (a + b, (N,)),
                           "G": (["A"] * 6 + ["B"] * 6, (CAT,))},
                 {"outcome": ("Y",), "group": ("G",)})
    d2 = _by(advise.advise(bnd), "D2")
    assert d2 and d2[0].severity == "info"
    assert d2[0].suggest_test == "mwu"


# ==========================================================================
# D3 — chi-square expected counts (block severe + dedup; flag moderate)
# ==========================================================================
def test_d3_chi2_small_expected_blocks_to_fisher():
    row = ["A"] * 8 + ["B"] * 8
    col = ["P"] * 7 + ["Q"] * 1 + ["P"] * 1 + ["Q"] * 7   # 2x2, expected all 4 < 5
    bnd = _bound("chi2_ind", {"R": (row, (CAT,)), "C": (col, (CAT,))},
                 {"row": ("R",), "col": ("C",)})
    d3 = _by(advise.advise(bnd), "D3")
    assert d3 and d3[0].severity == "block"
    assert d3[0].suggest_test == "fisher"
    # wired: exactly one D3 even though cat.py also owns the severe block.
    r = REGISTRY["chi2_ind"].run(bnd)
    assert r.status == "blocked"
    assert len(_by(r.findings, "D3")) == 1


def test_d3_chi2_moderate_low_expected_flags_fisher():
    # 3x3 with several 1<=expected<5 cells but none <1 -> flag, not block.
    row = (["A"] * 6 + ["B"] * 6 + ["C"] * 6)
    col = (["P"] * 2 + ["Q"] * 2 + ["S"] * 2) * 3
    bnd = _bound("chi2_ind", {"R": (row, (CAT,)), "C": (col, (CAT,))},
                 {"row": ("R",), "col": ("C",)})
    d3 = _by(advise.advise(bnd), "D3")
    assert d3 and d3[0].severity == "flag"
    assert d3[0].suggest_test == "fisher"


# ==========================================================================
# D4 — zero variance in every group -> block
# ==========================================================================
def test_d4_every_group_constant_blocks():
    y = [5.0, 5.0, 5.0, 10.0, 10.0, 10.0, 15.0, 15.0, 15.0]
    g = ["A"] * 3 + ["B"] * 3 + ["C"] * 3
    bnd = _bound("anova_1w", {"Y": (y, (N,)), "G": (g, (CAT,))},
                 {"outcome": ("Y",), "group": ("G",)})
    d4 = _by(advise.advise(bnd), "D4")
    assert d4 and d4[0].severity == "block"


# ==========================================================================
# D5 — logistic separation pre-check
# ==========================================================================
def test_d5_logistic_separation_precheck_flag():
    out = [0.0] * 6 + [1.0] * 6
    pred = ["A"] * 6 + ["B"] * 6                  # perfectly separates the outcome
    bnd = _bound("logistic", {"Out": (out, (B, CAT)), "Pred": (pred, (CAT,))},
                 {"outcome": ("Out",), "predictors": ("Pred",)})
    d5 = _by(advise.advise(bnd), "D5")
    assert d5 and d5[0].severity == "flag"
    assert "separation" in d5[0].text.lower()


# ==========================================================================
# D6 — VIF (owned by regress.py); advise must not double it
# ==========================================================================
def test_d6_vif_flag_single_when_wired():
    x1 = [float(v) for v in range(1, 15)]
    x2 = [2.0 * v + (0.01 if int(v) % 2 else -0.01) for v in x1]   # collinear
    y = [v + (0.2 if int(v) % 2 else -0.2) for v in x1]
    bnd = _bound("ols_multi", {"Y": (y, (N,)), "X1": (x1, (N,)), "X2": (x2, (N,))},
                 {"outcome": ("Y",), "predictors": ("X1", "X2")})
    r = REGISTRY["ols_multi"].run(bnd)
    assert r.status == "ok"
    assert len(_by(r.findings, "D6")) == 1        # runner emits it; advise dedups


# ==========================================================================
# D7 — MWU ties present and both n<=8
# ==========================================================================
def test_d7_mwu_ties_small_n_info():
    a = [1.0, 2.0, 2.0, 3.0, 4.0]
    b = [2.0, 3.0, 3.0, 4.0, 5.0]
    bnd = _bound("mwu", {"Y": (a + b, (N,)),
                         "G": (["A"] * 5 + ["B"] * 5, (CAT,))},
                 {"outcome": ("Y",), "group": ("G",)})
    d7 = _by(advise.advise(bnd), "D7")
    assert d7 and d7[0].severity == "info"


# ==========================================================================
# D8 — Wilcoxon zero differences excluded
# ==========================================================================
def test_d8_wilcoxon_zero_diffs_info():
    before = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    after = [5.0, 6.0, 7.0, 3.0, 2.0, 1.0, 0.0]      # first three diffs are zero
    bnd = _bound("wilcoxon", {"Bef": (before, (N,)), "Aft": (after, (N,))},
                 {"before": ("Bef",), "after": ("Aft",)})
    d8 = _by(advise.advise(bnd), "D8")
    assert d8 and d8[0].severity == "info"
    assert "zero" in d8[0].text.lower()


# ==========================================================================
# D9 — outliers surfaced, never removed
# ==========================================================================
def test_d9_outliers_info_lists_value():
    a = [10.0, 11.0, 12.0, 13.0, 14.0, 100.0]        # 100 is a gross outlier
    b = [20.0, 21.0, 22.0, 23.0, 24.0, 25.0]
    bnd = _bound("t_ind", {"Y": (a + b, (N,)),
                           "G": (["A"] * 6 + ["B"] * 6, (CAT,))},
                 {"outcome": ("Y",), "group": ("G",)})
    d9 = _by(advise.advise(bnd), "D9")
    assert d9 and d9[0].severity == "info"
    assert "100" in d9[0].text


# ==========================================================================
# D10 — group size ratio > 3:1
# ==========================================================================
def test_d10_group_ratio_info():
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    bnd = _bound("t_ind", {"Y": (a + b, (N,)),
                           "G": (["A"] * 3 + ["B"] * 12, (CAT,))},
                 {"outcome": ("Y",), "group": ("G",)})
    d10 = _by(advise.advise(bnd), "D10")
    assert d10 and d10[0].severity == "info"


# ==========================================================================
# D11 — goodness-of-fit expected count < 5
# ==========================================================================
def test_d11_gof_low_expected_flag():
    cat = (["A"] * 3 + ["B"] * 3 + ["C"] * 3 + ["D"] * 3 + ["E"] * 3)  # exp 3 each
    bnd = _bound("chi2_gof", {"Cat": (cat, (CAT,))}, {"category": ("Cat",)})
    d11 = _by(advise.advise(bnd), "D11")
    assert d11 and d11[0].severity == "flag"


# ==========================================================================
# D12 — two-sample proportion, small np
# ==========================================================================
def test_d12_prop2_small_expected_flag():
    out = (["Yes"] * 2 + ["No"] * 8) + (["Yes"] * 6 + ["No"] * 4)     # g1 k=2 < 5
    grp = ["G1"] * 10 + ["G2"] * 10
    bnd = _bound("prop_2", {"Out": (out, (B, CAT)), "Grp": (grp, (B, CAT))},
                 {"outcome": ("Out",), "group": ("Grp",)},
                 params={"success": "Yes"})
    d12 = _by(advise.advise(bnd), "D12")
    assert d12 and d12[0].severity == "flag"
    assert d12[0].suggest_test == "fisher"


# B3 — a NUMERIC 0/1 outcome whose success is chosen by its display string "0".
# _d12's flag is symmetric in the two levels (it checks both successes<5 AND
# failures<5), so the flag alone cannot prove the chosen level was resolved. The
# teeth: (a) it still fires on a genuinely rare level and stays quiet when every
# level is plentiful, and (b) the local ad-hoc success picker is GONE, so _d12
# must route through levels.pick_success, which resolves "0" against float levels.
def test_d12_uses_chosen_success_display_string():
    rare = [0.0] * 3 + [1.0] * 8 + [0.0] * 9 + [1.0] * 9      # G1: only 3 zeros
    grp = ["G1"] * 11 + ["G2"] * 18
    bnd = _bound("prop_2", {"Out": (rare, (B,)), "Grp": (grp, (B, CAT))},
                 {"outcome": ("Out",), "group": ("Grp",)}, params={"success": "0"})
    assert _by(advise.advise(bnd), "D12")                     # zeros are rare in G1

    ok = [0.0] * 9 + [1.0] * 9 + [0.0] * 9 + [1.0] * 9        # every level >= 5
    grp2 = ["G1"] * 18 + ["G2"] * 18
    bnd2 = _bound("prop_2", {"Out": (ok, (B,)), "Grp": (grp2, (B, CAT))},
                  {"outcome": ("Out",), "group": ("Grp",)}, params={"success": "0"})
    assert not _by(advise.advise(bnd2), "D12")

    assert not hasattr(advise, "_pick_success")              # delegation to levels
    assert not hasattr(advise, "_POSITIVE")


# ==========================================================================
# D13 — RM-ANOVA always cross-checks Friedman (agree / disagree)
# ==========================================================================
def _rm_bound():
    c1 = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    c2 = [3.0, 4.0, 3.0, 4.0, 3.0, 4.0]
    c3 = [5.0, 6.0, 5.0, 6.0, 5.0, 6.0]      # every subject c1<c2<c3 -> Friedman sig
    return _bound("rm_anova", {"C1": (c1, (N,)), "C2": (c2, (N,)), "C3": (c3, (N,))},
                  {"measures": ("C1", "C2", "C3")})


def test_d13_rm_anova_friedman_agree_info():
    bnd = _rm_bound()
    fs = advise.robust_agreement(bnd, _fake_result("rm_anova", 0.001))
    d13 = _by(fs, "D13")
    assert d13 and d13[0].severity == "info"
    assert "friedman" in d13[0].text.lower()


def test_d13_rm_anova_friedman_disagree_flag():
    bnd = _rm_bound()
    fs = advise.robust_agreement(bnd, _fake_result("rm_anova", 0.6))
    d13 = _by(fs, "D13")
    assert d13 and d13[0].severity == "flag"


# ==========================================================================
# D14 — Pearson cross-checks Spearman when non-normal (agree / disagree)
# ==========================================================================
def _pearson_bound():
    x = [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 100]      # non-normal, monotone
    y = [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 200]
    return _bound("pearson", {"X": (x, (N,)), "Y": (y, (N,))},
                  {"x": ("X",), "y": ("Y",)})


def test_d14_pearson_spearman_agree_info():
    bnd = _pearson_bound()
    fs = advise.robust_agreement(bnd, _fake_result("pearson", 0.001))
    d14 = _by(fs, "D14")
    assert d14 and d14[0].severity == "info"
    assert d14[0].suggest_test == "spearman"


def test_d14_pearson_spearman_disagree_flag():
    bnd = _pearson_bound()
    fs = advise.robust_agreement(bnd, _fake_result("pearson", 0.6))
    d14 = _by(fs, "D14")
    assert d14 and d14[0].severity == "flag"


# ==========================================================================
# Robust-alternative agreement, general parametric case (Welch <-> MWU)
# ==========================================================================
def _shaky_t_ind_bound():
    a = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 100.0]      # non-normal (Shapiro fails)
    b = [10.0, 11.0, 10.0, 11.0, 10.0, 11.0, 10.0, 200.0]
    return _bound("t_ind", {"Y": (a + b, (N,)),
                            "G": (["A"] * 8 + ["B"] * 8, (CAT,))},
                  {"outcome": ("Y",), "group": ("G",)})


def test_robust_general_agree_is_reassurance():
    bnd = _shaky_t_ind_bound()
    fs = advise.robust_agreement(bnd, _fake_result("t_ind", 0.001))
    rob = _by(fs, "ROBUST")
    assert rob and rob[0].severity == "info"
    assert "mann-whitney" in rob[0].text.lower()


def test_robust_general_disagree_is_caution():
    bnd = _shaky_t_ind_bound()
    fs = advise.robust_agreement(bnd, _fake_result("t_ind", 0.6))
    rob = _by(fs, "ROBUST")
    assert rob and rob[0].severity == "flag"
    assert "caution" in rob[0].text.lower()


def test_robust_disagree_tiny_p_prints_p_lt_001_not_zero():
    # perfectly separated non-normal groups -> the Mann-Whitney alternative's p is
    # genuinely < .001 (significant) while the primary result is not (p = 0.6), so
    # the caution shows the alternative's p. Raw f"{rp:.3f}" would print the
    # meaningless "p = 0.000"; it must route through fmt.p and read "p < .001".
    a = ([1.0, 2.0] * 6) + [1.0, 100.0]                # 14 values, all <= 100
    b = ([200.0, 201.0] * 6) + [200.0, 300.0]          # 14 values, all >= 200
    bnd = _bound("t_ind", {"Y": (a + b, (N,)),
                           "G": (["A"] * 14 + ["B"] * 14, (CAT,))},
                 {"outcome": ("Y",), "group": ("G",)})
    fs = advise.robust_agreement(bnd, _fake_result("t_ind", 0.6))
    rob = _by(fs, "ROBUST")
    assert rob and rob[0].severity == "flag"
    assert "p < .001" in rob[0].text
    _assert_no_zero_p(rob[0].text)


# ==========================================================================
# Wiring / dedup guarantees
# ==========================================================================
def test_wired_run_carries_advisories():
    bnd = _shaky_t_ind_bound()
    r = REGISTRY["t_ind"].run(bnd)
    assert r.status == "ok"
    # a shaky Welch run carries the robust-alternative agreement advisory.
    assert _by(r.findings, "ROBUST")


def test_wired_run_has_no_duplicate_codes():
    for bnd in (_shaky_t_ind_bound(), _pearson_bound(), _rm_bound()):
        r = REGISTRY[bnd.test.id].run(bnd)
        codes = [c for c in _codes(r.findings) if c]
        assert len(codes) == len(set(codes)), (bnd.test.id, codes)


def test_advised_dedups_a_preexisting_code():
    # a finding already on the bound with a code advise would also emit is not doubled.
    bnd = _shaky_t_ind_bound()
    r = REGISTRY["t_ind"].run(bnd)
    for code in {c for c in _codes(r.findings) if c}:
        assert len(_by(r.findings, code)) == 1


def test_advised_runs_structural_check_even_when_bound_carries_a_finding():
    # S-F: a bind-attached advisory must NOT stop advised() from running the
    # structural check -- a constant outcome still surfaces its S8 block.
    bnd = _bound("t_ind", {"Y": ([5.0] * 6, (N,)),
                           "G": (["A"] * 3 + ["B"] * 3, (CAT,))},
                 {"outcome": ("Y",), "group": ("G",)})
    bnd.findings = (Finding("flag", "1 duplicate ignored", code="DUP"),)
    r = REGISTRY["t_ind"].run(bnd)
    assert r.status == "blocked"
    assert "S8" in _codes(r.findings)          # structural block no longer skipped
    assert "DUP" in _codes(r.findings)         # pre-existing finding preserved


def test_advised_skips_data_rules_when_structurally_blocked(monkeypatch):
    # S-N: a structural BLOCK (constant outcome -> S8) must stop advised() from
    # even CALLING advise(): the data-driven rules would meet a frame that a block
    # exists precisely because it cannot be analysed (un-coded text, no variation).
    # Pre-fix advise() ran unconditionally, so a raising advise propagates instead
    # of the run returning a clean blocked Result.
    bnd = _bound("t_ind", {"Y": ([5.0] * 6, (N,)),
                           "G": (["A"] * 3 + ["B"] * 3, (CAT,))},
                 {"outcome": ("Y",), "group": ("G",)})

    def _boom(_b):
        raise RuntimeError("advise must not run on a structurally blocked bound")
    monkeypatch.setattr(advise, "advise", _boom)

    r = REGISTRY["t_ind"].run(bnd)             # RED: RuntimeError propagates (advise ran)
    assert r.status == "blocked"
    codes = _codes(r.findings)
    assert "S8" in codes                       # the structural block still surfaces
    assert not any(c and c.startswith("D") for c in codes)   # no data-rule findings
