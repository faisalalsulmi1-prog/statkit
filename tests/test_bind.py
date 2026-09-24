"""RED-first tests for bind.py (PLAN §2 / §6 / §9.1 bind list, Chunk 6).

bind() turns a test's Contract + a Dataset (typed df + profiles) + the user's
role->column choices into a canonical analysis frame (columns named by role),
with explicit listwise deletion and N accounting (D14). The three normalisers:
  * wide -> long   (group columns stacked)          e.g. t_ind / anova_1w wide
  * long -> wide   (subject x condition pivot)       e.g. t_paired / rm_anova long
  * table -> long  (count grid / category+count)     e.g. chi2_ind / chi2_gof table
plus the two parked contracts nailed here: wilcoxon ONE-SAMPLE (single column vs
mu0) and table_to_long for count grids.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from statkit import assoc, bind, cat, check, means, ranks
from statkit.clean import Table
from statkit.grid import Cell
from statkit.infer import Dataset
from statkit.model import ColumnProfile, Kind
from statkit.registry import REGISTRY
from statkit import infer as _inf

N, O, C, B, ID = (Kind.NUMERIC, Kind.ORDINAL, Kind.CATEGORICAL, Kind.BINARY, Kind.ID)


def _profile(name, kinds, values):
    """A minimal profile: name + kinds + level bookkeeping (enough for bind)."""
    non_null = [v for v in values if not _isna(v)]
    if kinds[0] in (Kind.NUMERIC, Kind.ID, Kind.DATE, Kind.EMPTY):
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
        n_missing=len(values) - len(non_null), n_levels=len(levels), levels=levels,
    )


def _isna(v):
    return v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v))


def _ds(cols, excel_rows=None):
    """Build a Dataset from {name: (values, kinds)}; dtype from the default kind."""
    df_cols = {}
    profiles = []
    n = None
    for name, (values, kinds) in cols.items():
        n = len(values) if n is None else n
        non_null = [v for v in values if not _isna(v)]
        numeric_vals = all(isinstance(v, (int, float)) for v in non_null)
        if kinds[0] in (Kind.NUMERIC, Kind.ORDINAL, Kind.BINARY, Kind.ID) and numeric_vals:
            df_cols[name] = pd.array(
                [None if _isna(v) else float(v) for v in values], dtype="float64")
        else:
            df_cols[name] = pd.array(
                [pd.NA if _isna(v) else str(v) for v in values], dtype="string")
        profiles.append(_profile(name, kinds, values))
    df = pd.DataFrame(df_cols)
    excel_rows = tuple(excel_rows or range(2, 2 + n))
    return Dataset(df=df, profiles=tuple(profiles), excel_rows=excel_rows)


# --------------------------------------------------------------------------
# long passthrough + listwise deletion + canonical role naming
# --------------------------------------------------------------------------
def test_long_passthrough_names_columns_by_role_and_drops_incomplete():
    ds = _ds({
        "Height": ([170, 165, None, 180], (N,)),
        "Weight": ([70, 60, 55, 80], (N,)),
    }, excel_rows=[2, 3, 4, 5])
    b = bind.bind(REGISTRY["pearson"], ds,
                  {"x": ("Height",), "y": ("Weight",)}, layout="long")
    assert b.layout == "long"
    assert list(b.data.columns) == ["x", "y"]           # named by role
    assert b.n_total == 4
    assert b.n_used == 3                                  # row 4 (Height blank) dropped
    assert b.dropped == {"missing value": 1}
    assert (4, "missing value") in b.dropped_rows        # 1-based excel row


def test_effective_kind_is_first_of_accepts_intersect_kinds():
    # A Likert column (numeric, ordinal, categorical). outcome accepts (N,) -> NUMERIC;
    # group accepts (B, C, O) -> CATEGORICAL (B absent, C first present).
    ds = _ds({
        "Score": ([3, 4, 5, 2, 1, 4], (N, O, C)),
        "Arm":   ([3, 4, 5, 2, 1, 4], (N, O, C)),
    })
    b = bind.bind(REGISTRY["t_ind"], ds,
                  {"outcome": ("Score",), "group": ("Arm",)}, layout="long")
    assert b.kinds["outcome"] == Kind.NUMERIC
    assert b.kinds["group"] == Kind.CATEGORICAL


# --------------------------------------------------------------------------
# wide -> long
# --------------------------------------------------------------------------
def test_wide_to_long_stacks_group_columns():
    ds = _ds({
        "Placebo": ([10, 12, 11, None], (N,)),
        "Drug":    ([14, 16, 15, 13], (N,)),
    })
    b = bind.bind(REGISTRY["t_ind"], ds, {"groups": ("Placebo", "Drug")}, layout="wide")
    assert b.layout == "wide"                             # the layout the user gave
    assert set(b.data.columns) == {"outcome", "group"}   # canonical long
    assert sorted(b.data["group"].unique()) == ["Drug", "Placebo"]
    # blank cell in a wide group column is layout padding, not a dropped observation
    assert b.n_used == 7
    assert (b.data["group"] == "Placebo").sum() == 3
    assert (b.data["group"] == "Drug").sum() == 4


def test_wide_to_long_k_groups_for_anova():
    ds = _ds({
        "A": ([1, 2, 3], (N,)), "B": ([4, 5, 6], (N,)), "C": ([7, 8, 9], (N,)),
    })
    b = bind.bind(REGISTRY["anova_1w"], ds, {"groups": ("A", "B", "C")}, layout="wide")
    assert set(b.data["group"].unique()) == {"A", "B", "C"}
    assert b.n_used == 9


# --------------------------------------------------------------------------
# long -> wide  (subject x condition pivot)
# --------------------------------------------------------------------------
def test_long_to_wide_pairs_before_after():
    ds = _ds({
        "Pt":   (["p1", "p1", "p2", "p2", "p3"], (ID, C)),
        "Time": (["Pre", "Post", "Pre", "Post", "Pre"], (C,)),
        "BP":   ([120, 110, 130, 118, 140], (N,)),
    })
    b = bind.bind(REGISTRY["t_paired"], ds,
                  {"subject": ("Pt",), "condition": ("Time",), "outcome": ("BP",)},
                  layout="long")
    assert set(b.data.columns) == {"before", "after"}    # canonical wide, 2 conditions
    assert b.n_total == 3                                  # 3 subjects
    assert b.n_used == 2                                   # p3 has only Pre -> dropped
    assert b.dropped == {"incomplete subject": 1}


def test_long_to_wide_k_measures_for_rm_anova():
    from synth import long_k3
    from statkit import grid, clean, infer as inf
    data, name = long_k3(subjects=6)
    ds = inf.infer(clean.clean(grid.load(data, name)[0]))
    b = bind.bind(REGISTRY["rm_anova"], ds,
                  {"subject": ("Subject",), "condition": ("Condition",),
                   "outcome": ("Score",)}, layout="long")
    assert list(b.data.columns) == ["measures__0", "measures__1", "measures__2"]
    assert b.n_used == 6                                  # 6 complete subjects


# --------------------------------------------------------------------------
# table -> long  (count grid + category+count)
# --------------------------------------------------------------------------
def test_table_to_long_expands_count_grid_dropping_totals():
    from synth import rc_grid
    from statkit import grid, clean, infer as inf
    data, name = rc_grid()
    ds = inf.infer(clean.clean(grid.load(data, name)[0]))
    counts = ("Product A", "Product B", "Product C")
    b = bind.bind(REGISTRY["chi2_ind"], ds, {"counts": counts}, layout="table")
    assert set(b.data.columns) == {"row", "col"}
    assert b.n_used == 80                                 # inner-cell total (Total row/col dropped)
    assert set(b.data["row"].unique()) == {"North", "South", "East"}
    assert set(b.data["col"].unique()) == {"Product A", "Product B", "Product C"}
    assert (b.data["row"] == "North").sum() == 28         # 12+7+9


def test_table_to_long_helper_direct():
    df = pd.DataFrame({
        "Region": ["North", "South", "Total"],
        "X": [2, 3, 5], "Y": [1, 4, 5], "Total": [3, 7, 10],
    })
    long = bind.table_to_long(df, counts=("X", "Y"), row_label_col="Region")
    assert len(long) == 10                                # 2+3+1+4
    assert set(long.columns) == {"row", "col"}
    assert (long["row"] == "Total").sum() == 0            # total row dropped


def test_table_over_a_million_is_not_expanded_but_totalled():
    # A grid summing to > 1e6 must not materialise 1e6+ rows; n_used carries the
    # total so check.S15 can block it. (Guards against an OOM on a pathological grid.)
    ds = _ds({
        "Region": (["North", "South"], (C,)),
        "A": ([600_000, 700_000], (N,)),
        "B": ([400_000, 300_000], (N,)),
    })
    b = bind.bind(REGISTRY["chi2_ind"], ds, {"counts": ("A", "B")}, layout="table")
    assert b.n_used == 2_000_000
    assert len(b.data) == 0                               # not expanded
    findings = check.check(b)
    assert any(f.code == "S15" and f.severity == "block" for f in findings)


def test_category_count_table_for_gof():
    ds = _ds({
        "Phenotype": (["Red", "Yellow", "Green"], (C,)),
        "Count":     ([3, 2, 1], (N,)),
    })
    b = bind.bind(REGISTRY["chi2_gof"], ds,
                  {"category": ("Phenotype",), "counts": ("Count",)}, layout="table")
    assert list(b.data.columns) == ["category"]
    assert b.n_used == 6
    assert (b.data["category"] == "Red").sum() == 3


# --------------------------------------------------------------------------
# wilcoxon ONE-SAMPLE (single column vs mu0) -- parked contract
# --------------------------------------------------------------------------
def test_wilcoxon_one_sample_single_column():
    ds = _ds({"Change": ([1.0, -2.0, 3.0, None, 0.5], (N, O))})
    b = bind.bind(REGISTRY["wilcoxon"], ds, {"outcome": ("Change",)},
                  layout="long", params={"mu0": 0.0})
    assert list(b.data.columns) == ["outcome"]           # single canonical column
    assert "after" not in b.data.columns                 # one-sample: no pair
    assert b.n_used == 4                                   # the blank dropped
    assert b.params["mu0"] == 0.0


def test_wilcoxon_two_sample_still_pairs():
    ds = _ds({"Pre": ([1, 2, 3], (N,)), "Post": ([2, 4, 1], (N,))})
    b = bind.bind(REGISTRY["wilcoxon"], ds,
                  {"before": ("Pre",), "after": ("Post",)}, layout="wide")
    assert set(b.data.columns) == {"before", "after"}


# --------------------------------------------------------------------------
# B6 — long -> wide records the derived condition labels on bound.columns (1A)
# --------------------------------------------------------------------------
def test_long_to_wide_records_condition_labels():
    # t_paired long, "Post" first-seen -> before names the first pivot column.
    ds = _ds({
        "Pt":   (["p1", "p1", "p2", "p2", "p3", "p3"], (ID, C)),
        "Time": (["Post", "Pre", "Post", "Pre", "Post", "Pre"], (C,)),
        "BP":   ([110, 120, 118, 130, 115, 140], (N,)),
    })
    b = bind.bind(REGISTRY["t_paired"], ds,
                  {"subject": ("Pt",), "condition": ("Time",), "outcome": ("BP",)},
                  layout="long")
    assert b.columns["before"] == ("Post",)
    assert b.columns["after"] == ("Pre",)
    # rm_anova long -> measures carries every condition label in order.
    from synth import long_k3
    from statkit import grid, clean, infer as inf
    data, name = long_k3(subjects=6)
    ds2 = inf.infer(clean.clean(grid.load(data, name)[0]))
    b2 = bind.bind(REGISTRY["rm_anova"], ds2,
                   {"subject": ("Subject",), "condition": ("Condition",),
                    "outcome": ("Score",)}, layout="long")
    assert b2.columns["measures"] == ("Baseline", "Week4", "Week8")


# --------------------------------------------------------------------------
# S3 — count normalisers reject non-integer / negative cells (1A)
# --------------------------------------------------------------------------
def test_table_to_long_rejects_non_integer_and_negative_counts():
    frac = pd.DataFrame({"Region": ["North"], "X": [33.3], "Y": [66.7]})
    with pytest.raises(bind.BindError):
        bind.table_to_long(frac, counts=("X", "Y"), row_label_col="Region")
    neg = pd.DataFrame({"Region": ["North"], "X": [-5], "Y": [3]})
    with pytest.raises(bind.BindError):
        bind.table_to_long(neg, counts=("X", "Y"), row_label_col="Region")


# --------------------------------------------------------------------------
# S5 — corr_matrix binds pairwise-complete (keeps rows with scattered NaN) (1A)
# --------------------------------------------------------------------------
def test_corr_matrix_bind_keeps_rows_with_scattered_missing():
    a = [None, 2, 3, 4, 5, 6, 7, 8, 9, 10]        # NaN in row 0
    bb = [2, 1, 4, None, 6, 5, 8, 7, 10, 9]       # NaN in row 3
    cc = [1, 3, 2, 5, 4, 7, None, 9, 8, 11]       # NaN in row 6
    ds = _ds({"a": (a, (N,)), "b": (bb, (N,)), "c": (cc, (N,))})
    b = bind.bind(REGISTRY["corr_matrix"], ds, {"variables": ("a", "b", "c")},
                  layout="long")
    assert b.n_used == 10                          # no cross-column listwise
    assert b.data.isna().any().any()               # NaN survives to the runner


# --------------------------------------------------------------------------
# F9 — duplicate subject x condition rows are silently truncated (iloc[0]); the
# second value is dropped with no trace. Keep the first (as now) but flag it.
# --------------------------------------------------------------------------
def test_long_to_wide_flags_duplicate_subject_condition():
    ds = _ds({
        "Pt":   (["p1", "p1", "p1", "p2", "p2"], (ID, C)),
        "Time": (["Pre", "Post", "Pre", "Pre", "Post"], (C,)),
        "BP":   ([120, 110, 99, 130, 118], (N,)),      # p1 has TWO "Pre" (120, 99)
    })
    b = bind.bind(REGISTRY["t_paired"], ds,
                  {"subject": ("Pt",), "condition": ("Time",), "outcome": ("BP",)},
                  layout="long")
    assert set(b.data.columns) == {"before", "after"}    # still binds
    assert b.n_used == 2                                   # both subjects complete
    dup = [f for f in b.findings if f.code == "DUP"]
    assert len(dup) == 1
    assert dup[0].severity == "flag"                      # warn-level
    assert "1" in dup[0].text                             # one duplicate ignored


def test_long_to_wide_no_duplicates_has_no_dup_finding():
    # happy path unchanged: no duplicate -> no DUP finding on the bound.
    ds = _ds({
        "Pt":   (["p1", "p1", "p2", "p2"], (ID, C)),
        "Time": (["Pre", "Post", "Pre", "Post"], (C,)),
        "BP":   ([120, 110, 130, 118], (N,)),
    })
    b = bind.bind(REGISTRY["t_paired"], ds,
                  {"subject": ("Pt",), "condition": ("Time",), "outcome": ("BP",)},
                  layout="long")
    assert not any(f.code == "DUP" for f in b.findings)


# --------------------------------------------------------------------------
# layout inference (optional layout arg)
# --------------------------------------------------------------------------
def test_layout_inferred_from_provided_roles():
    ds = _ds({"Pre": ([1, 2, 3], (N,)), "Post": ([2, 4, 1], (N,))})
    b = bind.bind(REGISTRY["t_paired"], ds, {"before": ("Pre",), "after": ("Post",)})
    assert b.layout == "wide"


# --------------------------------------------------------------------------
# NB2 — an ordinal/numeric role over a TEXT column is CODED at bind (one choke)
# so a rank/means runner's to_numpy(float) never meets a raw Likert string.
# --------------------------------------------------------------------------
def _infer_ds(cols, excel_rows=None):
    """A Dataset from the REAL infer pipeline: {name: [raw cell values]} -> typed
    df + profiles, so an ordinal-text column carries a level_order and a mixed
    numeric+text column carries (CATEGORICAL, NUMERIC) exactly as production does."""
    header = list(cols)
    n = len(next(iter(cols.values())))
    rows = [[Cell(cols[name][i]) for name in header] for i in range(n)]
    er = list(excel_rows or range(2, 2 + n))
    return _inf.infer(Table(header=header, rows=rows, excel_rows=er, header_row=1))


def test_ordinal_text_outcome_coded_and_mwu_runs():
    # 2-group sheet, Likert TEXT ordinal outcome -> outcome column becomes numeric
    # codes 1..k; group kept as labels; mwu returns a populated, non-blocked Result.
    ds = _infer_ds({
        "Arm": ["Placebo"] * 5 + ["Drug"] * 5,
        "Score": ["Disagree", "Neutral", "Agree", "Strongly agree", "Agree",
                  "Strongly disagree", "Disagree", "Neutral", "Disagree", "Agree"],
    })
    b = bind.bind(REGISTRY["mwu"], ds,
                  {"outcome": ("Score",), "group": ("Arm",)}, layout="long")
    assert pd.api.types.is_numeric_dtype(b.data["outcome"])          # coded to numbers
    assert set(b.data["outcome"].dropna().unique()) <= {1.0, 2.0, 3.0, 4.0, 5.0}
    assert b.data["group"].dropna().tolist().count("Placebo") == 5   # group NOT coded
    res = ranks.mwu(b)
    assert res.status == "ok"                                         # not blocked, no crash
    assert res.p is not None


def test_ordinal_text_outcome_kruskal_three_groups():
    ds = _infer_ds({
        "Arm": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
        "Score": ["Disagree", "Neutral", "Agree", "Agree",
                  "Neutral", "Agree", "Strongly agree", "Agree",
                  "Strongly disagree", "Disagree", "Neutral", "Disagree"],
    })
    b = bind.bind(REGISTRY["kruskal"], ds,
                  {"outcome": ("Score",), "group": ("Arm",)}, layout="long")
    assert pd.api.types.is_numeric_dtype(b.data["outcome"])
    assert set(b.data["group"].dropna().unique()) == {"A", "B", "C"}  # labels intact
    res = ranks.kruskal(b)
    assert res.status == "ok"
    assert res.p is not None


def test_ordinal_coding_finding_present_on_bound():
    ds = _infer_ds({
        "Arm": ["P", "P", "P", "D", "D", "D"],
        "Score": ["Disagree", "Neutral", "Agree", "Agree", "Neutral", "Disagree"],
    })
    b = bind.bind(REGISTRY["mwu"], ds,
                  {"outcome": ("Score",), "group": ("Arm",)}, layout="long")
    ord_notes = [f for f in b.findings if f.code == "ORDCODE"]
    assert len(ord_notes) == 1
    assert "coded" in ord_notes[0].text.lower()
    assert "'Agree'" in ord_notes[0].text                 # verbatim label named


def test_numeric_role_over_mixed_column_coerces_junk_to_missing():
    ds = _infer_ds({
        "Val": ["5", "6", "7", "8", "9", "10", "11", "12", "oops", "bad"],
    })
    prof = {p.name: p for p in ds.profiles}["Val"]
    assert prof.kinds[0] == Kind.CATEGORICAL and Kind.NUMERIC in prof.kinds  # mixed
    b = bind.bind(REGISTRY["t_1s"], ds, {"outcome": ("Val",)},
                  layout="long", params={"mu0": 0.0})
    assert pd.api.types.is_numeric_dtype(b.data["outcome"])
    assert b.n_total == 10
    assert b.n_used == 8                                   # 2 junk cells coerced -> NaN
    assert b.dropped == {"missing value": 2}


def test_categorical_role_columns_are_never_coded():
    # Pref is ORDINAL by profile (low/medium/high) yet used in a CATEGORICAL role
    # (chi2 col) -> it must stay labels, not become 1/2/3 (guard vs over-coding).
    ds = _infer_ds({
        "Sex": ["M", "F", "M", "F", "M", "F", "M", "F"],
        "Pref": ["Low", "High", "Medium", "Low", "High", "Medium", "Low", "High"],
    })
    assert {p.name: p for p in ds.profiles}["Pref"].kinds[0] == Kind.ORDINAL
    b = bind.bind(REGISTRY["chi2_ind"], ds,
                  {"row": ("Sex",), "col": ("Pref",)}, layout="long")
    assert not pd.api.types.is_numeric_dtype(b.data["col"])
    assert set(b.data["col"].dropna().unique()) == {"Low", "Medium", "High"}
    assert not any(f.code == "ORDCODE" for f in b.findings)


# --------------------------------------------------------------------------
# S-B(a) — long -> wide numeric-coded condition labels render "1"/"2", not "1.0"
# --------------------------------------------------------------------------
def test_long_to_wide_numeric_condition_labels_render_cleanly():
    ds = _ds({
        "Pt":   (["p1", "p1", "p2", "p2", "p3", "p3"], (ID, C)),
        "Time": ([1, 2, 1, 2, 1, 2], (N, B, C)),          # numeric-coded condition
        "BP":   ([120, 110, 130, 118, 140, 132], (N,)),
    })
    b = bind.bind(REGISTRY["t_paired"], ds,
                  {"subject": ("Pt",), "condition": ("Time",), "outcome": ("BP",)},
                  layout="long")
    assert b.columns["before"] == ("1",)                  # not "1.0"
    assert b.columns["after"] == ("2",)


# --------------------------------------------------------------------------
# S-I — a Yes/No TEXT column used in a correlation/regression role (accepts[0]
# NUMERIC, single-column) is coded 0/1 at the bind choke (point-biserial), so
# x/y.to_numpy(float) never meets 'No'. Categorical roles keep their labels.
# --------------------------------------------------------------------------
def test_binary_text_x_is_coded_01_for_pearson():
    scores = [55, 61, 48, 72, 66, 58, 80, 45, 70, 63,
              52, 77, 68, 49, 74, 60, 57, 83, 65, 71]
    ds = _infer_ds({"Score": scores,
                    "Passed": ["Yes" if s >= 60 else "No" for s in scores]})
    b = bind.bind(REGISTRY["pearson"], ds,
                  {"x": ("Score",), "y": ("Passed",)}, layout="long")
    assert pd.api.types.is_numeric_dtype(b.data["y"])       # pre-fix: string dtype
    assert set(b.data["y"].dropna()) == {0.0, 1.0}
    assert (b.data["y"] == 1.0).sum() == 13                 # Yes -> 1 (pick_success)
    bc = [f for f in b.findings if f.code == "BINCODE"]     # pre-fix: no BINCODE
    assert len(bc) == 1
    assert bc[0].severity == "info"
    assert all(tok in bc[0].text for tok in ("Passed", "Yes", "No"))


def test_binary_text_is_not_coded_for_categorical_roles():
    # over-coding guard: a Yes/No column in a CATEGORICAL role (chi2 row, prop_1
    # outcome) keeps its labels and yields no BINCODE. Green today; keep as guard.
    ds = _infer_ds({
        "Passed": ["Yes", "No"] * 10,
        "Sex": ["M", "F"] * 10,
    })
    b = bind.bind(REGISTRY["chi2_ind"], ds,
                  {"row": ("Passed",), "col": ("Sex",)}, layout="long")
    assert not pd.api.types.is_numeric_dtype(b.data["row"])
    assert set(b.data["row"].dropna()) == {"Yes", "No"}
    assert not any(f.code == "BINCODE" for f in b.findings)
    b2 = bind.bind(REGISTRY["prop_1"], ds, {"outcome": ("Passed",)}, layout="long")
    assert not pd.api.types.is_numeric_dtype(b2.data["outcome"])
    assert set(b2.data["outcome"].dropna()) == {"Yes", "No"}
    assert not any(f.code == "BINCODE" for f in b2.findings)


# --------------------------------------------------------------------------
# S-J(bind) — an ORDINAL column whose order is unknown (no vocab, e.g. an
# override on an unrecognised scale) must BLOCK with a real reason at bind, not
# leave the text uncoded to crash the rank runner's to_numpy(float).
# --------------------------------------------------------------------------
def test_ordinal_text_without_order_blocks_with_reason():
    arm = ["A"] * 5 + ["B"] * 5
    rating = ["Terrible", "Bad", "OK", "Great", "Bad"] * 2
    header = ["Arm", "Rating"]
    n = len(arm)
    rows = [[Cell(arm[i]), Cell(rating[i])] for i in range(n)]
    ds = _inf.infer(Table(header=header, rows=rows,
                          excel_rows=list(range(2, 2 + n)), header_row=1),
                    overrides={"Rating": Kind.ORDINAL})
    prof = {p.name: p for p in ds.profiles}["Rating"]
    assert prof.kinds == (Kind.ORDINAL,)
    assert prof.level_order is None
    b = bind.bind(REGISTRY["mwu"], ds,                      # must NOT raise
                  {"outcome": ("Rating",), "group": ("Arm",)}, layout="long")
    assert b.n_used == 10                                   # pre-fix: 0 (column blanked)
    ord_notes = [f for f in b.findings if f.code == "ORDORDER"]
    assert len(ord_notes) == 1                              # pre-fix: 0 (no finding)
    assert ord_notes[0].severity == "block"
    txt = ord_notes[0].text
    assert "Rating" in txt and "order" in txt
    assert all(lv in txt for lv in prof.levels)            # every level named
    b.findings = check.gate(b, {p.name: p for p in ds.profiles})
    # the ONLY block is ORDORDER; pre-fix the blanked column also tripped S5 (0
    # usable rows) and S22 (no group levels left): ['S5', 'S22', 'ORDORDER'].
    assert [f.code for f in b.findings if f.severity == "block"] == ["ORDORDER"]
    res = ranks.mwu(b)
    assert res.status == "blocked"                          # pre-fix: ValueError 'Terrible'
    assert res.findings[0].code != "S5"                    # pre-fix: S5 was first


# --------------------------------------------------------------------------
# S-K(b) — when a numeric role cannot bind because the sheet's columns were read
# as ID (all-distinct), the greyed reason NAMES them instead of lying "your
# sheet has none".
# --------------------------------------------------------------------------
def test_role_reason_names_id_read_numeric_columns():
    before = [40, 41, 43, 44, 46, 47, 49, 50, 52, 53,
              55, 56, 58, 59, 61, 62, 64, 65, 67, 68]
    ds = _infer_ds({"Before": before,
                    "After": [v + 5 + (i % 2) for i, v in enumerate(before)]})
    profs = {p.name: p for p in ds.profiles}
    assert profs["Before"].kinds == (ID,)
    assert profs["After"].kinds == (ID,)
    ok, why = bind.satisfies(REGISTRY["t_paired"], ds)
    assert ok is False
    assert "Before" in why and "After" in why              # pre-fix: names neither
    assert "numeric" in why.casefold()
    assert "your sheet has none" not in why                # pre-fix: present (the lie)
    # guard: with NO numeric-dtype ID column the old generic message is kept.
    ds2 = _infer_ds({"City": ["NY", "LA", "SF"] * 4, "Team": ["X", "Y"] * 6})
    ok2, why2 = bind.satisfies(REGISTRY["t_paired"], ds2)
    assert ok2 is False
    assert "your sheet has none" in why2


# --------------------------------------------------------------------------
# N12 — auto_columns fails CLOSED (None) when a required single-column role has
# no eligible column, never a partial {'x': (...), 'y': ()}.
# --------------------------------------------------------------------------
def test_auto_columns_returns_none_when_a_required_single_role_is_empty():
    ds = _ds({"Score": ([1.5, 2.5, 3.5, 4.5, 5.5, 6.5], (N,)),
              "Passed": (["Yes", "No"] * 3, (B, C))})
    assert bind.auto_columns(REGISTRY["spearman"], ds, "long") is None


# --------------------------------------------------------------------------
# S-Q(bind) — a LABEL-READING role (describe / chi2, whose accepts include
# CATEGORICAL) may use an UNORDERED ordinal AS LABELS: bind must NOT force an
# ORDORDER block or code it; it keeps the text so describe can COUNT the levels.
# --------------------------------------------------------------------------
def test_describe_keeps_counts_for_unordered_ordinal():
    arm = ["A"] * 5 + ["B"] * 5
    rating = ["Terrible", "Bad", "OK", "Great", "Bad"] * 2
    header = ["Arm", "Rating"]
    n = len(arm)
    rows = [[Cell(arm[i]), Cell(rating[i])] for i in range(n)]
    ds = _inf.infer(Table(header=header, rows=rows,
                          excel_rows=list(range(2, 2 + n)), header_row=1),
                    overrides={"Rating": Kind.ORDINAL})
    assert {p.name: p for p in ds.profiles}["Rating"].level_order is None
    b = bind.bind(REGISTRY["describe"], ds,
                  {"variables": ("Rating",), "group": ("Arm",)}, layout="long")
    assert not any(f.code == "ORDORDER" for f in b.findings)   # pre-fix: ORDORDER present
    b.findings = check.gate(b, {p.name: p for p in ds.profiles})
    res = means.describe(b)
    assert res.status == "ok"                                  # pre-fix: blocked (all-NaN)
    assert res.extra["categoricals"]["Rating"] == {
        "A": {"Terrible": 1, "Bad": 2, "OK": 1, "Great": 1},
        "B": {"Terrible": 1, "Bad": 2, "OK": 1, "Great": 1}}
    assert res.descriptives is None


# --------------------------------------------------------------------------
# S-P(bind) — a Total / % / percentage column is a SUMMARY, not a category
# count: role_columns for the 'counts' role must never offer one (a Total column
# double-counts the grid). Other roles (a numeric outcome) still see it.
# --------------------------------------------------------------------------
def test_counts_role_never_offers_total_or_percent_columns():
    ds = _ds({
        "Region": (["North", "South", "East"], (C,)),
        "Yes":    ([20, 15, 10], (N,)),
        "No":     ([25, 20, 15], (N,)),
        "Total":  ([45, 35, 25], (N,)),
        "Pct":    ([45, 35, 25], (N,)),
    })
    counts_role = next(r for r in REGISTRY["chi2_ind"].contract.roles_by_layout["table"]
                       if r.name == "counts")
    assert bind.role_columns(counts_role, ds) == ["Yes", "No"]   # pre-fix: +Total,+Pct
    assert bind.auto_columns(REGISTRY["chi2_ind"], ds, "table") == {
        "counts": ("Yes", "No")}
    b = bind.bind(REGISTRY["chi2_ind"], ds, {"counts": ("Yes", "No")}, layout="table")
    assert b.n_used == 105                                       # pre-fix: 315 (Total+Pct)
    # scope guard: the skip is ONLY for the counts role -- a numeric OUTCOME role
    # still sees a column called 'Total'.
    outcome_role = next(r for r in REGISTRY["t_ind"].contract.roles_by_layout["long"]
                        if r.name == "outcome")
    assert "Total" in bind.role_columns(outcome_role, ds)


# --------------------------------------------------------------------------
# S-N(bind) — a BINARY-effective column that does NOT have exactly two levels
# cannot be 0/1 coded: bind BLOCKS naming the levels, instead of silently
# dropping the surplus level to NaN (which shrinks n and pools two levels).
# --------------------------------------------------------------------------
def test_binary_override_on_three_level_column_blocks_naming_levels():
    scores = list(range(50, 71))                    # 21 numeric values
    grades = [["Low", "Mid", "High"][i % 3] for i in range(21)]
    header = ["Score", "Grade"]
    rows = [[Cell(scores[i]), Cell(grades[i])] for i in range(21)]
    ds = _inf.infer(Table(header=header, rows=rows,
                          excel_rows=list(range(2, 2 + 21)), header_row=1),
                    overrides={"Grade": Kind.BINARY})
    prof = {p.name: p for p in ds.profiles}["Grade"]
    assert prof.kinds == (Kind.BINARY,)
    assert prof.levels == ("Low", "Mid", "High")    # three levels, not two
    b = bind.bind(REGISTRY["pearson"], ds,
                  {"x": ("Score",), "y": ("Grade",)}, layout="long")
    blocks = [f.code for f in b.findings if f.severity == "block"]
    assert blocks == ["BINLEVELS"]                   # pre-fix: [] (silently BINCODE-d)
    assert not any(f.code == "BINCODE" for f in b.findings)
    binf = next(f for f in b.findings if f.code == "BINLEVELS")
    assert all(lv in binf.text for lv in ("Low", "Mid", "High"))   # all levels named
    assert "Grade" in binf.text
    assert b.n_used == 21                            # pre-fix: 14 (7 'High' rows -> NaN)
    b.findings = check.gate(b)
    assert assoc.pearson(b).status == "blocked"      # pre-fix: ran on 14 rows


# --------------------------------------------------------------------------
# S-M wording — when a numeric role cannot bind because a REPEATED column was
# read as an ID, the greyed reason must not claim "(every value is distinct)"
# (a repeated id is the opposite of distinct) -- it says "(identifiers, not
# measurements)" and still names the column + the Numeric fix.
# --------------------------------------------------------------------------
def test_repeated_id_greyed_reason_does_not_claim_distinct_values():
    ids = [n for n in range(1, 13) for _ in range(2)]      # 1..12 each twice (repeats)
    time = ["Pre", "Post"] * 12
    note = ["ok", "bad", "good", "fair"] * 6
    header = ["ID", "Time", "Note"]
    rows = [[Cell(ids[i]), Cell(time[i]), Cell(note[i])] for i in range(24)]
    ds = _inf.infer(Table(header=header, rows=rows,
                          excel_rows=list(range(2, 2 + 24)), header_row=1))
    assert {p.name: p for p in ds.profiles}["ID"].kinds == (Kind.ID,)
    ok, why = bind.satisfies(REGISTRY["t_ind"], ds)
    assert ok is False
    assert "'ID'" in why
    assert "Numeric" in why
    assert "every value is distinct" not in why           # pre-fix: present (a lie)


# --------------------------------------------------------------------------
# NB5 (S-P regression) -- the summary-header skip matched SUBSTRINGS, so a
# Likert answer column 'Totally agree' / 'Totally disagree' vanished from the
# count-grid picker: chi2 silently ran on half the grid with status ok. The skip
# must match WHOLE WORDS ('_' read as a space) and keep excluding real summary
# columns (Subtotal / Total / % of total / Pct / Grand total / n_total).
# --------------------------------------------------------------------------
def test_counts_grid_role_keeps_totally_agree_columns():
    likert = _infer_ds({
        "Question":         ["Q1", "Q2", "Q3"],
        "Totally disagree": [10, 12, 8],
        "Disagree":         [15, 10, 14],
        "Agree":            [14, 16, 12],
        "Totally agree":    [11, 12, 16],
    })
    four = ["Totally disagree", "Disagree", "Agree", "Totally agree"]
    grid_role = next(r for r in REGISTRY["chi2_ind"].contract.roles_by_layout["table"]
                     if r.name == "counts")
    assert grid_role.max is None                          # the multi-column GRID role
    assert bind.role_columns(grid_role, likert) == four   # pre-fix: ['Disagree', 'Agree']
    auto = bind.auto_columns(REGISTRY["chi2_ind"], likert, "table")
    assert auto == {"counts": tuple(four)}                # pre-fix: ('Disagree', 'Agree')
    b = bind.bind(REGISTRY["chi2_ind"], likert, auto, layout="table")
    assert b.n_used == 150                                # pre-fix: 81 -- chi2 on half the grid
    # guard (green pre-fix): real summary columns are STILL skipped for the grid
    summary = _ds({
        "Region":      (["North", "South", "East"], (C,)),
        "Yes":         ([20, 15, 10], (N,)),
        "No":          ([25, 20, 15], (N,)),
        "Subtotal":    ([45, 35, 25], (N,)),
        "Total":       ([45, 35, 25], (N,)),
        "% of total":  ([45, 35, 25], (N,)),
        "Pct":         ([45, 35, 25], (N,)),
        "Grand total": ([45, 35, 25], (N,)),
        "n_total":     ([45, 35, 25], (N,)),
    })
    assert bind.role_columns(grid_role, summary) == ["Yes", "No"]


# --------------------------------------------------------------------------
# NB5 (second half) -- the skip must apply ONLY to the multi-column GRID role:
# a chi2_gof category+count table whose single count column is called 'Total'
# was greyed "needs a numeric column; your sheet has none" (a lie).
# --------------------------------------------------------------------------
def test_gof_single_count_column_named_total_is_offered():
    gof = _infer_ds({"Phenotype": ["Purple", "White", "Pink", "Red"],
                     "Total":     [315, 108, 101, 32]})
    spec = REGISTRY["chi2_gof"]
    assert bind.satisfies_layout(spec, gof, "table") == (True, "")
    # pre-fix: (False, 'needs a numeric column; your sheet has none')
    assert bind.auto_columns(spec, gof, "table") == {
        "category": ("Phenotype",), "counts": ("Total",)}        # pre-fix: None
    b = bind.bind(spec, gof, {"category": ("Phenotype",), "counts": ("Total",)},
                  layout="table")
    assert b.n_used == 556


# --------------------------------------------------------------------------
# R1 (NB5 follow-on) -- the whole-word skip read 'GrandTotal' / 'RowTotal' as
# ONE word, so a CamelCase summary column slipped INTO the count grid: chi2 ran
# status ok on n=315 (true 105) over a 3x4 crosstab (df 6, not 2 -> wrong p too).
# CamelCase is split BEFORE the word match; 'TotallyAgree' (Likert) stays a real
# count column; a Total-like header is skipped for the GRID role only (chi2_gof's
# single count column keeps 'TotalRevenue', as does a numeric outcome role).
# --------------------------------------------------------------------------
def test_counts_grid_role_skips_camelcase_total_columns():
    grid = _infer_ds({
        "Region":     ["North", "South", "East"],
        "Yes":        [20, 15, 10],
        "No":         [25, 20, 15],
        "GrandTotal": [45, 35, 25],
        "RowTotal":   [45, 35, 25],
        "TOTAL":      [45, 35, 25],
    })
    grid_role = next(r for r in REGISTRY["chi2_ind"].contract.roles_by_layout["table"]
                     if r.name == "counts")
    assert bind.role_columns(grid_role, grid) == ["Yes", "No"]
    # pre-fix: ['Yes', 'No', 'GrandTotal', 'RowTotal'] (only the all-caps TOTAL caught)
    auto = bind.auto_columns(REGISTRY["chi2_ind"], grid, "table")
    assert auto == {"counts": ("Yes", "No")}
    b = bind.bind(REGISTRY["chi2_ind"], grid, auto, layout="table")
    assert b.n_used == 105                                 # pre-fix: 315, status ok, 0 blocks
    b.findings = check.gate(b)
    assert cat.chi2_ind(b).status == "ok"                  # the corrected grid still runs
    # keep-guards: CamelCase Likert answers are real count columns; 'TotalRevenue'
    # is skipped for the GRID role only.
    likert = _infer_ds({
        "Question":        ["Q1", "Q2", "Q3"],
        "TotallyDisagree": [10, 12, 8],
        "Disagree":        [15, 10, 14],
        "Agree":           [14, 16, 12],
        "TotallyAgree":    [11, 12, 16],
        "TotalRevenue":    [100, 200, 300],
    })
    four = ["TotallyDisagree", "Disagree", "Agree", "TotallyAgree"]
    assert bind.role_columns(grid_role, likert) == four    # pre-fix: four + ['TotalRevenue']
    gof_role = next(r for r in REGISTRY["chi2_gof"].contract.roles_by_layout["table"]
                    if r.name == "counts")
    assert bind.role_columns(gof_role, likert) == four + ["TotalRevenue"]
    outcome_role = next(r for r in REGISTRY["t_ind"].contract.roles_by_layout["long"]
                        if r.name == "outcome")
    assert "TotalRevenue" in bind.role_columns(outcome_role, likert)


# --------------------------------------------------------------------------
# R2 (NB5 follow-on) -- NB5 scoped the WHOLE summary skip to the grid role, which
# re-opened a percent column to chi2_gof's single count column: with '% of Total'
# BEFORE 'Count' and whole-number percents, auto_columns bound the percents and
# gof ran status ok on n=100 (true 200). A percent-like header is a RATE, never a
# count -- skipped for EVERY counts role; only the Total-like skip is grid-only.
# --------------------------------------------------------------------------
def test_gof_count_role_never_offers_a_percent_column():
    gof = _infer_ds({"Phenotype":  ["Purple", "White", "Pink", "Red"],
                     "% of Total": [50, 25, 15, 10],
                     "Count":      [100, 50, 30, 20]})
    spec = REGISTRY["chi2_gof"]
    gof_role = next(r for r in spec.contract.roles_by_layout["table"] if r.name == "counts")
    assert gof_role.max == 1                               # the SINGLE count column role
    assert bind.role_columns(gof_role, gof) == ["Count"]   # pre-fix: ['% of Total', 'Count']
    auto = bind.auto_columns(spec, gof, "table")
    assert auto == {"category": ("Phenotype",), "counts": ("Count",)}
    # pre-fix: counts=('% of Total',)
    b = bind.bind(spec, gof, auto, layout="table")
    assert b.n_used == 200                                 # pre-fix: 100 (the percents)


# --------------------------------------------------------------------------
# R3 (cosmetic) -- `\bpercent` had no closing word boundary, so a 'Percentile'
# column vanished from the count-grid picker (a missing option, not a wrong
# answer). 'percent' / 'percentage' / 'percentages' stay whole-word summaries.
# --------------------------------------------------------------------------
def test_counts_grid_role_keeps_percentile_column():
    ds = _ds({
        "Group":       (["A", "B", "C"], (C,)),
        "Percentile":  ([90, 50, 10], (N,)),
        "Yes":         ([20, 15, 10], (N,)),
        "No":          ([25, 20, 15], (N,)),
        "Percentage":  ([44, 43, 40], (N,)),
        "Percentages": ([44, 43, 40], (N,)),
        "percent_yes": ([44, 43, 40], (N,)),
    })
    grid_role = next(r for r in REGISTRY["chi2_ind"].contract.roles_by_layout["table"]
                     if r.name == "counts")
    assert bind.role_columns(grid_role, ds) == ["Percentile", "Yes", "No"]
    # pre-fix: ['Yes', 'No'] -- 'Percentile' swallowed by the open-ended `\bpercent`


# --------------------------------------------------------------------------
# R4 (row-total backstop) -- a "row total" is a ticked grid column equal to the
# row-wise sum of the OTHER ticked columns. The header-based summary skip
# (_is_summary_header) only knows total-ish WORDS, so a total column headed 'N',
# 'Overall', 'Row Sum' (sum is exact-match only), a glued 'GRANDTOTAL', a
# digit-glued 'Total2024'/'Q1Total', or a non-English 'Gesamt'/'合計'/'Итого'
# slips INTO the grid -- chi2 then runs status ok on a silently DOUBLE-COUNTED n
# (Agree+Disagree ticked a second time). _check_row_total is a numeric BindError
# backstop inside _table_bind (same class/site/handler as the _check_count
# BindError), blocking a ticked summary column (#2) only; a MISSING count column
# (#1) is out of scope (deferred).
# --------------------------------------------------------------------------
def test_count_grid_blocks_a_ticked_column_that_is_the_row_sum_of_the_others():
    spec = REGISTRY["chi2_ind"]
    grid_role = next(r for r in spec.contract.roles_by_layout["table"]
                     if r.name == "counts")
    for header in ("N", "Overall", "Row Sum", "GRANDTOTAL", "Total2024",
                   "Q1Total", "Gesamt", "合計", "Итого"):
        ds = _ds({
            "Item":     (["Q1", "Q2", "Q3"], (C,)),
            "Agree":    ([20, 15, 10], (N,)),
            "Disagree": ([25, 20, 15], (N,)),
            header:     ([45, 35, 25], (N,)),
        })
        # the seam (green pre AND post): the header-based skip does NOT catch
        # these, so the total column is offered -- only a numeric backstop bites.
        assert header in bind.role_columns(grid_role, ds)
        with pytest.raises(bind.BindError) as exc:
            bind.bind(spec, ds, {"counts": ("Agree", "Disagree", header)},
                      layout="table")
        # pre-fix: DID NOT RAISE (n_used=210, chi2_ind ok
        # χ²=0.1296296296296296 p=0.9979881129874916 -- Agree+Disagree doubled)
        assert (f"The column '{header}' equals 'Agree' + 'Disagree' on every row"
                in str(exc.value))
        assert "Untick it" in str(exc.value)
    # the true 2-column grid still binds, gates clean, and runs
    ds2 = _ds({
        "Item":     (["Q1", "Q2", "Q3"], (C,)),
        "Agree":    ([20, 15, 10], (N,)),
        "Disagree": ([25, 20, 15], (N,)),
    })
    b = bind.bind(spec, ds2, {"counts": ("Agree", "Disagree")}, layout="table")
    assert b.n_used == 105
    b.findings = check.gate(b)
    assert cat.chi2_ind(b).status == "ok"


def test_count_grid_row_sum_backstop_does_not_over_block():
    spec = REGISTRY["chi2_ind"]

    def binds(cols):
        n = len(next(iter(cols.values())))
        spec_cols = {"Row": ([f"r{i}" for i in range(n)], (C,))}
        for name, vals in cols.items():
            spec_cols[name] = (vals, (N,))
        b = bind.bind(spec, _ds(spec_cols), {"counts": tuple(cols)}, layout="table")
        return b.n_used

    # legit 3-category grid: no column equals the sum of the other two
    assert binds({"Yes": [20, 15, 10], "No": [25, 20, 15],
                  "Maybe": [30, 5, 12]}) == 152
    # two ticked columns can never be a row total (guard: >=3 live columns)
    assert binds({"A": [10, 7], "B": [10, 7]}) == 34
    # a single row: a match needs >=2 non-zero candidate rows
    assert binds({"A": [10], "B": [20], "C": [30]}) == 60
    # C == A+B only where all three are zero: <2 non-zero candidate rows
    assert binds({"A": [0, 0, 10], "B": [0, 0, 20], "C": [0, 0, 30]}) == 60
    # an all-zero column is dropped from 'live' -> a 2-column case, never blocks
    assert binds({"A": [10, 5], "B": [10, 5], "C": [0, 0]}) == 30


def test_count_grid_row_sum_backstop_skips_blank_cells_and_total_rows():
    spec = REGISTRY["chi2_ind"]
    ds = _ds({
        "Region": (["North", "South", "East", "Total"], (C,)),
        "Yes":    ([20, 15, 10, 45], (N,)),
        "No":     ([25, 20, 15, 60], (N,)),
        "N":      ([45, None, 25, 105], (N,)),
    })
    with pytest.raises(bind.BindError, match=r"'N' equals 'Yes' \+ 'No'"):
        bind.bind(spec, ds, {"counts": ("Yes", "No", "N")}, layout="table")
    # pre-fix: DID NOT RAISE (n_used=175, ok p~6.87e-06). The 'Total' row and the
    # blank South cell are skipped; two clean non-zero rows still refuse.


# --------------------------------------------------------------------------
# S-R: infinity tokens in a MIXED column are MISSING, not huge values
# --------------------------------------------------------------------------
def test_infinity_tokens_in_mixed_column_are_missing():
    # infer already rejects 'inf'/'-inf'/'Infinity' ("treated as missing"), but
    # bind's NUMERIC re-coercion used pd.to_numeric, which ACCEPTS them, so ±inf
    # survived into the frame: describe n=19 / mean nan / min -inf, OLS "ok" with
    # F = nan. An infinity token is an unmeasurable cell, i.e. missing.
    ds = _infer_ds({"Val": ["3", "5", "4", "8", "7", "9", "10", "13", "12", "14",
                            "16", "15", "18", "17", "20", "19",
                            "inf", "-inf", "Infinity", "oops"]})
    prof = {p.name: p for p in ds.profiles}["Val"]
    assert prof.kinds == (C, N)                                   # mixed: 16/20 numeric
    assert "4 text cell(s) treated as missing" in prof.notes      # infer's verdict
    role = bind._role_by_name(REGISTRY["t_1s"].contract, "outcome")
    coded, note = bind._code_series(ds.df["Val"], N, prof, role, "Val")
    assert note is None
    assert str(coded.dtype) == "Float64"                          # nullable float kept
    assert coded.isna().tolist()[-4:] == [True] * 4               # pre-fix: [inf, -inf, inf, <NA>]
    assert all(math.isfinite(v) for v in coded.dropna())
    b = bind.bind(REGISTRY["describe"], ds, {"variables": ("Val",)}, layout="long")
    row = means.describe(b).descriptives.iloc[0]
    assert (row["n"], row["min"], row["max"]) == (16, 3.0, 20.0)  # pre-fix: (19, -inf, inf)
    assert row["mean"] == pytest.approx(11.875)                   # pre-fix: nan
    b = bind.bind(REGISTRY["t_1s"], ds, {"outcome": ("Val",)},
                  layout="long", params={"mu0": 0.0})
    assert (b.n_total, b.n_used, b.dropped) == (20, 16, {"missing value": 4})  # pre-fix: 19 / {…: 1}
    assert all(math.isfinite(v) for v in b.data["outcome"])


# --------------------------------------------------------------------------
# S-S (row-label backstop) -- in a count grid the FIRST column holds the row
# labels (the categories). The counts role only knows NUMERIC, so a numeric
# row-label column (Dose 1/2/3, Year, Grade) is OFFERED -- and auto-pick takes it.
# Two faces, one cause: (a) every column ticked -> the row-label fallback
# `next(c for c in df.columns if c not in counts)` raised StopIteration (the
# app's generic "Sorry, I couldn't prepare your data" handler, no stated cause);
# (b) the first column ticked while another column is free -> that column
# silently became the row label and the label VALUES were counted (Dose 1+2+3 =
# 6 phantom units: n=126 true 120, chi2 17.22 df=4 vs 16.28 df=2, status ok).
# _check_row_label is a BindError backstop inside _table_bind (same class/site/
# handler as _check_count and R4's _check_row_total): the first column ticked as
# a count -> stated BindError, both faces.
# --------------------------------------------------------------------------
def test_count_grid_blocks_a_ticked_numeric_row_label_column():
    spec = REGISTRY["chi2_ind"]
    grid_role = next(r for r in spec.contract.roles_by_layout["table"]
                     if r.name == "counts")
    ds = _ds({
        "Dose":     ([1, 2, 3], (N,)),
        "Cured":    ([12, 20, 30], (N,)),
        "NotCured": ([28, 20, 10], (N,)),
        "Note":     (["pilot", "main", "repeat"], (C,)),
    })
    # the seam (green pre AND post): the counts role only knows NUMERIC, so the
    # numeric row-label column is offered -- and auto-pick takes all three.
    assert bind.role_columns(grid_role, ds) == ["Dose", "Cured", "NotCured"]
    assert bind.auto_columns(spec, ds, "table") == {"counts": ("Dose", "Cured", "NotCured")}
    real = _infer_ds({"Dose": [1, 2, 3], "Cured": [12, 20, 30],
                      "NotCured": [28, 20, 10], "Note": ["pilot", "main", "repeat"]})
    assert "Dose" in bind.role_columns(grid_role, real)     # the REAL infer pipeline too
    with pytest.raises(bind.BindError) as exc:
        bind.bind(spec, ds, {"counts": ("Dose", "Cured", "NotCured")}, layout="table")
    # pre-fix: DID NOT RAISE -- 'Note' silently became the row label and the Dose
    # values were COUNTED: n_used=126 (true 120), chi2_ind ok
    # χ²=17.218274291028603 p=0.0017530069407236425 df=(4.0,)  (true 16.2848 / 0.000291 / df=2)
    assert "The column 'Dose' is the first column of your table" in str(exc.value)
    assert "Untick the row-label column" in str(exc.value)


def test_count_grid_all_columns_ticked_is_a_row_label_bind_error_not_a_crash():
    spec = REGISTRY["chi2_ind"]
    ds = _ds({
        "Dose":     ([1, 2, 3], (N,)),
        "Cured":    ([12, 20, 30], (N,)),
        "NotCured": ([28, 20, 10], (N,)),
    })
    with pytest.raises(bind.BindError, match=r"Untick the row-label column"):
        bind.bind(spec, ds, {"counts": ("Dose", "Cured", "NotCured")}, layout="table")
    # pre-fix: raised StopIteration('') -- no column left to be the row label --
    # which the app's generic except-Exception handler swallowed with no stated cause.


def test_count_grid_with_the_row_label_column_unticked_still_binds():
    spec = REGISTRY["chi2_ind"]
    ds = _ds({
        "Dose":     ([1, 2, 3], (N,)),
        "Cured":    ([12, 20, 30], (N,)),
        "NotCured": ([28, 20, 10], (N,)),
        "Note":     (["pilot", "main", "repeat"], (C,)),
    })
    b = bind.bind(spec, ds, {"counts": ("Cured", "NotCured")}, layout="table")
    assert b.n_used == 120
    assert sorted(set(b.data["row"])) == [1.0, 2.0, 3.0]        # Dose IS the row label
    b.findings = check.gate(b)
    r = cat.chi2_ind(b)
    assert r.status == "ok"
    assert round(r.statistic[1], 4) == 16.2848
    assert r.statistic[1] == pytest.approx(16.284760845383758)
    assert r.p == pytest.approx(0.00029094380404415325)
    assert r.df == (2.0,)
    # the same true grid on the 3-column sheet
    ds3 = _ds({
        "Dose":     ([1, 2, 3], (N,)),
        "Cured":    ([12, 20, 30], (N,)),
        "NotCured": ([28, 20, 10], (N,)),
    })
    assert bind.bind(spec, ds3, {"counts": ("Cured", "NotCured")}, layout="table").n_used == 120
    # a label-first grid with a free trailing column is untouched (no over-block)
    ds2 = _ds({
        "Item":     (["Q1", "Q2", "Q3"], (C,)),
        "Agree":    ([20, 15, 10], (N,)),
        "Disagree": ([25, 20, 15], (N,)),
        "Note":     (["a", "b", "c"], (C,)),
    })
    assert bind.bind(spec, ds2, {"counts": ("Agree", "Disagree")}, layout="table").n_used == 105
