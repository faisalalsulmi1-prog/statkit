"""RED-first tests for check.py -- the structural sanity checks S1-S21 (PLAN §5.1).

check(bound) reads the Contract x the canonical frame x level/size counts and
returns a Finding per rule that FIRES (block / flag / info) with its code. One
test per code constructs the minimal Bound that trips it and asserts the code +
severity appear. (D1-D14 are advise(), Chunk 11 -- out of scope here.)
"""
from __future__ import annotations

import pandas as pd
import pytest

from statkit import check as chk
from statkit import means, ranks
from statkit.model import Bound, ColumnProfile, Finding, Kind
from statkit.registry import REGISTRY

N, O, C, B = (Kind.NUMERIC, Kind.ORDINAL, Kind.CATEGORICAL, Kind.BINARY)


def _bound(test_id, data=None, columns=None, kinds=None, params=None,
           layout=None, n_total=None, n_used=None, dropped=None):
    spec = REGISTRY[test_id]
    data = pd.DataFrame(data if data is not None else {})
    n_used = len(data) if n_used is None else n_used
    n_total = n_used if n_total is None else n_total
    return Bound(
        test=spec, layout=layout or spec.contract.canonical,
        columns=columns or {}, kinds=kinds or {}, params=params or {},
        data=data, n_total=n_total, n_used=n_used, dropped=dropped or {},
        dropped_rows=(),
    )


def _codes(findings):
    return {f.code for f in findings}


def _by_code(findings, code):
    return next(f for f in findings if f.code == code)


def _prof(name, kinds, levels=()):
    return ColumnProfile(name=name, kinds=tuple(kinds), n_total=0, n_missing=0,
                         n_levels=len(levels), levels=tuple(levels))


# --- S1 same column in two roles ------------------------------------------
def test_s1_same_column_two_roles():
    b = _bound("t_ind",
               data={"outcome": [1, 2, 3], "group": ["A", "B", "A"]},
               columns={"outcome": ("Score",), "group": ("Score",)}, n_used=30)
    f = _by_code(chk.check(b), "S1")
    assert f.severity == "block"


# --- S2 group level count wrong for the test's arity ----------------------
def test_s2_three_groups_for_two_group_test_blocks_and_suggests():
    b = _bound("t_ind",
               data={"outcome": list(range(9)),
                     "group": ["A", "B", "C"] * 3},
               columns={"outcome": ("Y",), "group": ("G",)}, n_used=30)
    f = _by_code(chk.check(b), "S2")
    assert f.severity == "block"
    assert f.suggest_test in ("anova_1w", "kruskal")


# --- S3 categorical role with >20 levels ----------------------------------
def test_s3_too_many_levels():
    b = _bound("chi2_ind",
               data={"row": [f"r{i}" for i in range(21)],
                     "col": ["x"] * 21},
               columns={"row": ("R",), "col": ("Col",)}, n_used=21)
    assert _by_code(chk.check(b), "S3").severity == "block"


# --- S4 a group with n<2 --------------------------------------------------
def test_s4_singleton_group():
    b = _bound("t_ind",
               data={"outcome": [1, 2, 3, 4], "group": ["A", "B", "B", "B"]},
               columns={"outcome": ("Y",), "group": ("G",)}, n_used=30)
    assert _by_code(chk.check(b), "S4").severity == "block"


# --- S5 n_used < min_n (non-paired) ---------------------------------------
def test_s5_too_few_rows():
    b = _bound("pearson", data={"x": [1, 2], "y": [3, 4]},
               columns={"x": ("X",), "y": ("Y",)}, n_used=2)
    assert _by_code(chk.check(b), "S5").severity == "block"


# --- S6 paired complete pairs < 3 -----------------------------------------
def test_s6_too_few_pairs():
    b = _bound("t_paired", data={"before": [1.0, 2.0], "after": [2.0, 3.0]},
               columns={"before": ("Pre",), "after": ("Post",)}, n_used=2)
    assert _by_code(chk.check(b), "S6").severity == "block"


# --- S7 same_level_set violated (McNemar) ---------------------------------
def test_s7_level_sets_differ():
    b = _bound("mcnemar",
               data={"before": ["Yes", "No", "Yes", "No"],
                     "after": ["Pass", "Fail", "Pass", "Fail"]},
               columns={"before": ("Pre",), "after": ("Post",)}, n_used=30)
    assert _by_code(chk.check(b), "S7").severity == "block"


# --- S8 constant column ---------------------------------------------------
def test_s8_constant_column():
    b = _bound("pearson", data={"x": [5, 5, 5, 5], "y": [1, 2, 3, 4]},
               columns={"x": ("X",), "y": ("Y",)}, n_used=4)
    assert _by_code(chk.check(b), "S8").severity == "block"


# --- S9 numeric column overridden to a group with >10 distinct ------------
def test_s9_numeric_used_as_group():
    prof = {"G": _prof("G", (N, C)), "Y": _prof("Y", (N,))}
    b = _bound("anova_1w",
               data={"outcome": list(range(12)),
                     "group": [float(i) for i in range(12)]},
               columns={"outcome": ("Y",), "group": ("G",)},
               kinds={"outcome": N, "group": C}, n_used=30)
    assert _by_code(chk.check(b, prof), "S9").severity == "flag"


# --- S10 Pearson with an ordinal (<=5 levels) side ------------------------
def test_s10_pearson_ordinal_side():
    b = _bound("pearson",
               data={"x": [1, 2, 3, 4, 5, 1, 2], "y": [2.0, 3, 1, 5, 4, 2, 3]},
               columns={"x": ("Likert",), "y": ("Score",)},
               kinds={"x": O, "y": N}, n_used=30)
    f = _by_code(chk.check(b), "S10")
    assert f.severity == "flag" and f.suggest_test == "spearman"


# --- S11 regression n_used < p + 10 ---------------------------------------
def test_s11_regression_underpowered():
    b = _bound("ols_multi",
               data={"outcome": [1.0] * 5, "predictors__0": [1.0] * 5,
                     "predictors__1": [2.0] * 5},
               columns={"outcome": ("Y",), "predictors": ("A", "B")}, n_used=5)
    assert _by_code(chk.check(b), "S11").severity == "block"


# --- S12 logistic outcome not 2 levels ------------------------------------
def test_s12_logistic_multinomial():
    b = _bound("logistic",
               data={"outcome": ["a", "b", "c", "a", "b", "c"],
                     "predictors__0": [1.0, 2, 3, 4, 5, 6]},
               columns={"outcome": ("Y",), "predictors": ("X",)},
               kinds={"outcome": C, "predictors": N}, n_used=30)
    assert _by_code(chk.check(b), "S12").severity == "block"


# --- S13 ordinal used as numeric ------------------------------------------
def test_s13_ordinal_as_numeric():
    prof = {"Likert": _prof("Likert", (N, O, C)), "X": _prof("X", (N,))}
    b = _bound("ols_simple",
               data={"outcome": [1.0, 2, 3, 4], "x": [2.0, 3, 4, 5]},
               columns={"outcome": ("Likert",), "x": ("X",)},
               kinds={"outcome": N, "x": N}, n_used=30)
    assert _by_code(chk.check(b, prof), "S13").severity == "flag"


# --- S14 GOF expected proportions don't sum to 1 --------------------------
def test_s14_gof_bad_expected():
    b = _bound("chi2_gof", data={"category": ["a", "b", "c"] * 4},
               columns={"category": ("Cat",)},
               params={"expected": (0.3, 0.3, 0.3)}, n_used=30)
    assert _by_code(chk.check(b), "S14").severity == "block"


# --- S15 table total count > 1,000,000 ------------------------------------
def test_s15_huge_table():
    b = _bound("chi2_ind", data={"row": [], "col": []}, layout="table",
               columns={"counts": ("A", "B")}, n_total=2_000_000, n_used=2_000_000)
    assert _by_code(chk.check(b), "S15").severity == "block"


# --- S16 large drop fraction ----------------------------------------------
def test_s16_low_retention():
    b = _bound("pearson", data={"x": [1.0] * 6, "y": [2.0] * 6},
               columns={"x": ("X",), "y": ("Y",)},
               n_total=10, n_used=6, dropped={"missing value": 4})
    assert _by_code(chk.check(b), "S16").severity == "flag"


# --- S17 chi-square with both roles binary --------------------------------
def test_s17_chi2_both_binary():
    b = _bound("chi2_ind",
               data={"row": ["Y", "N"] * 5, "col": ["P", "Q"] * 5},
               columns={"row": ("R",), "col": ("Col",)}, n_used=30)
    f = _by_code(chk.check(b), "S17")
    assert f.severity == "info" and f.suggest_test == "fisher"


# --- S18 independent test bound from a wide sheet -------------------------
def test_s18_independent_from_wide():
    b = _bound("t_ind", data={"outcome": [1, 2, 3, 4], "group": ["A", "A", "B", "B"]},
               columns={"groups": ("A", "B")}, layout="wide", n_used=30)
    assert _by_code(chk.check(b), "S18").severity == "info"


# --- S19 two-way with an empty factor combination -------------------------
def test_s19_two_way_empty_cell():
    b = _bound("anova_2w",
               data={"outcome": [1, 2, 3, 4],
                     "factor_a": ["X", "X", "Y", "Y"],
                     "factor_b": ["P", "Q", "P", "P"]},
               columns={"outcome": ("Y",), "factor_a": ("A",), "factor_b": ("B",)},
               n_used=30)
    assert _by_code(chk.check(b), "S19").severity == "block"


# --- S20 incomplete subjects dropped --------------------------------------
def test_s20_incomplete_subjects():
    b = _bound("rm_anova",
               data={"measures__0": [1.0] * 5, "measures__1": [2.0] * 5,
                     "measures__2": [3.0] * 5},
               columns={"subject": ("S",), "condition": ("C",), "outcome": ("Y",)},
               layout="long", n_total=7, n_used=5,
               dropped={"incomplete subject": 2})
    f = _by_code(chk.check(b), "S20")
    assert f.severity == "flag" and "2" in f.text


# --- S21 low N ------------------------------------------------------------
def test_s21_low_power():
    b = _bound("t_ind", data={"outcome": list(range(10)),
                              "group": ["A", "B"] * 5},
               columns={"outcome": ("Y",), "group": ("G",)}, n_used=10)
    assert _by_code(chk.check(b), "S21").severity == "flag"


# --- S22 a grouping level has no usable values (would crash the runner) ----
def test_s22_blocks_when_a_group_level_has_no_usable_values():
    # After listwise deletion the second group's rows all dropped (its outcomes
    # were missing), so the canonical frame carries only ONE level. Today the
    # runner does `a, b = groups` and raises IndexError; S22 must turn that into
    # a stated block, and the runner must then return status="blocked".
    cases = {
        "t_ind": {"outcome": [1.0, 2.0, 3.0, 4.0], "group": ["A", "A", "A", "A"]},
        "mwu": {"outcome": [1.0, 2.0, 3.0, 4.0], "group": ["A", "A", "A", "A"]},
        "prop_2": {"outcome": ["yes", "no", "yes", "no"],
                   "group": ["A", "A", "A", "A"]},
    }
    for tid, data in cases.items():
        b = _bound(tid, data=data,
                   columns={"outcome": ("Score",), "group": ("Arm",)},
                   n_total=8, n_used=4)
        f = _by_code(chk.check(b), "S22")
        assert f.severity == "block", tid
        assert "Arm" in f.text, tid                # names the STUDENT's header
        assert REGISTRY[tid].run(b).status == "blocked", tid


# --- S8 skips describe + names the student's header, not the role name ------
def test_s8_skips_describe_and_names_the_header():
    # describe deliberately tolerates a constant column: no S8 block.
    d = _bound("describe", data={"variables__0": [2020, 2020, 2020, 2020]},
               columns={"variables": ("Year",)}, n_used=30)
    assert "S8" not in _codes(chk.check(d))
    # a real inferential test still blocks a constant column, but the message
    # names the header ('Score'), never the canonical role name.
    t = _bound("t_ind",
               data={"outcome": [5.0, 5.0, 5.0, 5.0], "group": ["A", "B", "A", "B"]},
               columns={"outcome": ("Score",), "group": ("Arm",)}, n_used=30)
    f = _by_code(chk.check(t), "S8")
    assert f.severity == "block"
    assert "Score" in f.text
    assert "outcome" not in f.text and "variables__0" not in f.text
    # a MULTI-column role: S8 maps "variables__1" back to its student header.
    cm = _bound("corr_matrix",
                data={"variables__0": [1.0, 2.0, 3.0, 4.0],
                      "variables__1": [7.0, 7.0, 7.0, 7.0]},
                columns={"variables": ("Age", "Constant")}, n_used=30)
    f2 = _by_code(chk.check(cm), "S8")
    assert "Constant" in f2.text and "variables__1" not in f2.text


# --- a clean bound raises nothing structural ------------------------------
def test_clean_bound_has_no_blocks():
    b = _bound("t_ind",
               data={"outcome": list(range(40)), "group": ["A", "B"] * 20},
               columns={"outcome": ("Y",), "group": ("G",)}, n_total=40, n_used=40)
    findings = chk.check(b)
    assert not any(f.severity == "block" for f in findings)


# ==========================================================================
# S-F — gate(): the structural check ALWAYS runs, merged with pre-existing
# (bind/advise) findings, so a bind-attached advisory can never suppress an
# S-block (the fail-open the runners' `findings or check` idiom had).
# ==========================================================================
def test_gate_equals_check_when_bound_has_no_findings():
    # invariant: gate(bound) == check(bound) exactly on a plain bound, so every
    # golden that runs a runner on a plain bound stays byte-identical.
    b = _bound("t_ind", data={"outcome": list(range(10)), "group": ["A", "B"] * 5},
               columns={"outcome": ("Y",), "group": ("G",)}, n_used=10)
    assert not b.findings
    assert "S21" in _codes(chk.check(b))            # (meaningful: a finding exists)
    assert chk.gate(b) == chk.check(b)


def test_gate_runs_structural_check_even_when_bound_carries_a_finding():
    # a constant outcome trips S8; a pre-seeded non-structural DUP must NOT hide it.
    b = _bound("t_ind",
               data={"outcome": [5.0, 5.0, 5.0, 5.0], "group": ["A", "B", "A", "B"]},
               columns={"outcome": ("Score",), "group": ("Arm",)}, n_used=30)
    b.findings = (Finding("flag", "1 duplicate ignored", code="DUP"),)
    codes = _codes(chk.gate(b))
    assert "S8" in codes            # structural block still produced
    assert "DUP" in codes           # pre-existing finding preserved


def test_gate_structural_finding_wins_on_a_code_clash():
    b = _bound("t_ind",
               data={"outcome": [5.0, 5.0, 5.0, 5.0], "group": ["A", "B", "A", "B"]},
               columns={"outcome": ("Score",), "group": ("Arm",)}, n_used=30)
    b.findings = (Finding("info", "impostor S8", code="S8"),)   # same code as check
    g = chk.gate(b)
    s8 = _by_code(g, "S8")
    assert s8.severity == "block"                    # the real structural S8
    assert "impostor" not in s8.text
    assert sum(1 for f in g if f.code == "S8") == 1   # deduped, not doubled


def test_preseeded_finding_no_longer_suppresses_S22_block_through_a_runner():
    # An ordinal-text sheet whose only group collapses under listwise carries a
    # bind ORDCODE finding; before gate() the runner SKIPPED S22 and crashed on
    # `a, b = groups`. Now the structural block still fires -> a clean blocked
    # Result, for both the `_findings` (ranks) and `_gate` (means) idioms.
    for runner in (ranks.mwu, means.t_ind):
        b = _bound(runner.__name__,
                   data={"outcome": [1.0, 2.0, 3.0, 4.0],
                         "group": ["A", "A", "A", "A"]},
                   columns={"outcome": ("Score",), "group": ("Arm",)},
                   n_total=8, n_used=4)
        b.findings = (Finding("info", "ordinal coded 1..3", code="ORDCODE"),)
        r = runner(b)
        assert r.status == "blocked", runner.__name__
        assert "S22" in _codes(r.findings), runner.__name__


# --- S4 / S22 message text renders a numeric-coded level as "2", not "2.0" --
def test_s4_message_renders_numeric_group_level_without_trailing_zero():
    b = _bound("t_ind",
               data={"outcome": [1.0, 2.0, 3.0], "group": [1.0, 1.0, 2.0]},
               columns={"outcome": ("Score",), "group": ("Arm",)}, n_used=30)
    f = _by_code(chk.check(b), "S4")
    assert "2.0" not in f.text and "2" in f.text


def test_s22_message_renders_numeric_group_level_without_trailing_zero():
    b = _bound("t_ind",
               data={"outcome": [1.0, 2.0, 3.0, 4.0], "group": [1.0, 1.0, 1.0, 1.0]},
               columns={"outcome": ("Score",), "group": ("Arm",)},
               n_total=8, n_used=4)
    f = _by_code(chk.check(b), "S22")
    assert "1.0" not in f.text
