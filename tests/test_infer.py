"""RED list for statkit.infer (PLAN Chunk 5 / §4.4 type-inference table).

Each test pins one row of the §4.4 kinds table (the seam is the pure per-column
function ``profile_column(name, cells, excel_rows, unit, override) -> ColumnProfile``)
plus the ~30% reconciliation (approximate cells -> failed_cells), the snag-1
censored-on-text decision, override narrowing, and ``propose_merges``.

Cells are built by hand from grid.Cell; no fixtures are loaded here (that is
test_corpus.py, the master corpus gate).
"""
from datetime import date, datetime

import pytest

from statkit.clean import Table
from statkit.coerce import Convention
from statkit.grid import Cell
from statkit.infer import ORDERED_VOCABS, infer, profile_column, propose_merges
from statkit.model import Kind


def prof(vals, name="Col", unit=None, override=None, fmt="General"):
    cells = [Cell(v, fmt) for v in vals]
    er = list(range(2, 2 + len(cells)))
    return profile_column(name, cells, er, unit=unit, override=override)


def table_of(vals, name="x", fmt="General", header_row=1, **kw):
    """A one-column cleaned Table for the infer()-level tests."""
    rows = [[Cell(v, fmt)] for v in vals]
    er = list(range(2, 2 + len(vals)))
    return Table(header=[name], rows=rows, excel_rows=er, header_row=header_row, **kw)


# --- the effective-set gate (§4.4 top rows) -------------------------------
def test_all_empty_is_empty_kind():
    p = prof([None, None, None])
    assert p.kinds == (Kind.EMPTY,)
    assert p.n_missing == 3 and p.n_total == 3


def test_dates_ge_85pct_is_date():
    p = prof([datetime(2020, 1, 1), date(2020, 1, 2), datetime(2020, 3, 4)])
    assert p.kinds == (Kind.DATE,)


def test_bools_ge_85pct_is_binary():
    p = prof([True, False, True, False])
    assert p.kinds == (Kind.BINARY,)
    assert p.n_levels == 2


def test_numeric_fraction_ge_85_takes_numeric_path_and_lists_text_as_failed():
    p = prof([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, "oops"])  # 6 num / 7 = .857
    assert p.kinds[0] == Kind.NUMERIC
    assert len(p.failed_cells) == 1
    assert "oops" in [t for _, t in p.failed_cells]


def test_numeric_fraction_between_50_and_85_is_categorical_then_numeric():
    p = prof([1, 2, 3, 4, 5, 6, 7, "a", "b", "c"])  # 7/10 = .70
    assert p.kinds == (Kind.CATEGORICAL, Kind.NUMERIC)
    assert len(p.failed_cells) == 3          # mixed column still lists failures


def test_text_fraction_below_50_takes_text_path():
    p = prof(["red", "green", "blue", "red", "green", "blue", "red"])
    assert p.kinds == (Kind.CATEGORICAL,)
    assert p.n_levels == 3


# --- numeric sub-rules ----------------------------------------------------
def test_numeric_constant():
    assert prof([5.0, 5.0, 5.0, 5.0]).kinds == (Kind.NUMERIC,)


def test_numeric_two_distinct_is_numeric_binary_categorical():
    assert prof([0, 1, 0, 1, 0, 1]).kinds == (Kind.NUMERIC, Kind.BINARY, Kind.CATEGORICAL)


def test_numeric_small_integer_set_is_likert_triple():
    assert prof([1, 2, 3, 2, 1, 3, 2]).kinds == (
        Kind.NUMERIC, Kind.ORDINAL, Kind.CATEGORICAL)


def test_numeric_all_distinct_with_id_header_is_id():
    vals = [100, 205, 333, 412, 587, 690, 731, 842, 913, 1024,
            1150, 1261, 1372, 1483, 1594, 1605, 1716, 1827, 1938, 2049,
            2150, 2261, 2372, 2483]                       # 24, scattered
    assert prof(vals, name="Patient No").kinds == (Kind.ID,)


def test_numeric_all_distinct_near_contiguous_with_gaps_is_id():
    # ages 30..59 with two gaps: near-contiguous but not a perfect run -> ID
    vals = [v for v in range(30, 62) if v not in (40, 55)]   # 30 distinct, span 32
    assert prof(vals, name="Age").kinds == (Kind.ID,)


def test_numeric_perfect_run_is_ambiguous_numeric_id():
    # a perfect 1..24 sequence looks like a row index -> (NUMERIC, ID)
    assert prof(list(range(1, 25)), name="Score").kinds == (Kind.NUMERIC, Kind.ID)


def test_numeric_all_distinct_no_id_evidence_is_numeric_id():
    vals = [3.7, 15.2, 88.1, 210.5, 333.9, 512.4, 690.0, 741.6, 913.2, 1004.8,
            1150.1, 1261.7, 1372.3, 1483.9, 1594.5, 1655.0, 1716.6, 1827.2,
            1938.8, 2049.4, 2150.0, 2261.6, 2372.2, 2483.8]   # 24, huge range
    assert prof(vals, name="Measurement").kinds == (Kind.NUMERIC, Kind.ID)


def test_numeric_otherwise():
    assert prof([1.5, 2.7, 1.5, 3.3, 2.7, 4.1, 1.5]).kinds == (Kind.NUMERIC,)


# --- text sub-rules -------------------------------------------------------
def test_text_constant():
    assert prof(["A", "A", "A"]).kinds == (Kind.CATEGORICAL,)


def test_text_two_distinct_with_repeats_is_binary():
    assert prof(["Y", "N", "Y", "N", "Y"]).kinds == (Kind.BINARY, Kind.CATEGORICAL)


def test_text_two_levels_is_binary_strict_444():
    # §4.4: a two-level text column is (BINARY, CATEGORICAL). (The oracle is
    # internally inconsistent for 2-unique free-text notes -- s2_2 comments and
    # w2_2_build Notes call it categorical while h5_3/s1_1 Notes call it binary;
    # infer follows the spec and those two are flagged divergences in test_corpus.)
    assert prof(["yes", "no"]).kinds == (Kind.BINARY, Kind.CATEGORICAL)


def test_text_ordered_vocab_is_ordinal():
    p = prof(["Agree", "Disagree", "Neutral", "Agree", "Strongly agree", "Disagree"])
    assert p.kinds == (Kind.ORDINAL, Kind.CATEGORICAL)
    assert p.level_order is not None


def test_text_small_cardinality_is_categorical():
    assert prof(["a", "b", "c", "d", "e", "a", "b"]).kinds == (Kind.CATEGORICAL,)


def test_text_all_distinct_high_cardinality_is_id():
    # §4.4: unique high-cardinality text (codes, emails, IDs) -> free-text ID.
    assert prof([f"tok{i}" for i in range(25)]).kinds == (Kind.ID,)


def test_text_many_levels_with_repeats_is_categorical_flagged():
    vals = [f"lvl{i}" for i in range(21)] + [f"lvl{i}" for i in range(19)]  # 21 distinct / 40
    p = prof(vals)
    assert p.kinds == (Kind.CATEGORICAL,)
    assert any(">20" in n for n in p.notes)


def test_case_hash_header_is_id():
    # "Case#" is an identifier column despite the trailing '#' separator.
    assert prof(list(range(1, 25)), name="Case#").kinds == (Kind.ID,)


# --- censored / approximate reconciliations -------------------------------
def test_censored_on_numeric_path_counts_as_censored_not_failed():
    p = prof(["<0.01", "5", "6", "7", "8", "9", "10", "11"])
    assert p.kinds[0] == Kind.NUMERIC
    assert p.n_censored == 1 and len(p.failed_cells) == 0


def test_censored_on_text_path_is_a_level_snag1():
    p = prof([">100m", "0-50m", "50-100m", "0-50m", ">100m", "50-100m"])
    assert p.kinds[0] == Kind.CATEGORICAL
    assert p.n_censored == 0 and p.n_levels == 3


def test_approximate_cell_is_a_failed_cell_snag7():
    p = prof(["~30", "10", "20", "30", "40", "50", "60"])
    assert p.kinds[0] == Kind.NUMERIC
    assert p.n_missing == 0
    assert [t for _, t in p.failed_cells] == ["~30"]


# --- override, unit, levels, merges --------------------------------------
def test_override_narrows_to_one_kind():
    assert prof([1, 2, 3, 2, 1], override=Kind.CATEGORICAL).kinds == (Kind.CATEGORICAL,)


def test_unit_is_passed_through():
    assert prof([1, 2, 3], unit="kg").unit == "kg"


def test_levels_count_for_categorical():
    assert prof(["a", "b", "a", "c"]).n_levels == 3


def test_numeric_levels_are_lossless_no_rounding_collision():
    # S-D: fractional numeric levels must render distinctly & exactly -- under the
    # old fmt.num rounding 0.001/0.002 both became "0" and 0.125 became "0.12",
    # so the picker showed colliding labels for genuinely distinct doses.
    p = prof([0.001, 0.002, 0.125, 2.5], override=Kind.CATEGORICAL)
    assert p.levels == ("0.001", "0.002", "0.125", "2.5")
    assert len(set(p.levels)) == p.n_levels == 4      # no two labels collide


def test_propose_merges_folds_case_and_space_variants():
    merges = propose_merges(["Male", "male ", "MALE", "Female"])
    # the three male spellings collapse to one canonical; Female stays put
    canon = {merges.get(v, v) for v in ["Male", "male ", "MALE"]}
    assert len(canon) == 1


def test_ordered_vocabs_include_the_non_agree_scales():
    flat = {tok for vocab in ORDERED_VOCABS for tok in vocab}
    assert {"never", "sometimes", "always"} <= flat        # frequency
    assert {"none", "mild", "moderate", "severe"} <= flat  # severity
    assert {"low", "medium", "high"} <= flat               # low-med-high


# --- S6: convention cleanup surfaced as notes + log (batch 1F) -------------
def test_thousands_ambiguity_is_surfaced_in_notes():
    # "1,234"/"2,345": the detector reads them as 1234/2345 but the thousands
    # separator is ambiguous -- that judgement must reach the student, both on the
    # column's notes and on the dataset log.
    ds = infer(table_of(["1,234", "2,345"]))
    prof0 = ds.profiles[0]
    assert any("1,234" in n and "1234" in n for n in prof0.notes), prof0.notes
    assert any(n.startswith("Column 'x'") for n in ds.log), ds.log


def test_conventions_override_reads_eu_decimal():
    # a per-column convention override forces EU reading: "1,234" -> 1.234.
    ds = infer(table_of(["1,234", "2,345"]),
               conventions={"x": Convention(decimal=",", thousands=".")})
    assert list(ds.df["x"]) == [1.234, 2.345]


# --- S7: level merges applied (batch 1F) ----------------------------------
def test_merges_are_applied():
    tbl = table_of(["male ", "MALE", "Female", "Female", "male "], name="Sex")
    ds = infer(tbl, merges={"Sex": {"male ": "Male", "MALE": "Male"}})
    prof0 = ds.profiles[0]
    assert prof0.levels == ("Male", "Female")
    assert prof0.n_levels == 2
    assert any("merged levels in 'Sex'" in n for n in ds.log), ds.log


# --- S11 support: clean-stage drops carried onto the Dataset (batch 1F) ----
def test_dataset_carries_clean_stage_drops():
    tbl = table_of(["1", "2", "3"], header_row=3,
                   dropped_rows=[(5, "spacer"), (6, "footer total")])
    ds = infer(tbl)
    assert ds.dropped_rows == tuple(tbl.dropped_rows)
    assert ds.header_row == tbl.header_row
    assert ds.n_rows_read == len(tbl.rows)


# --- NB4: a mixed column's numeric df cells round-trip exactly (W2) ---------
def test_mixed_column_df_cells_are_lossless():
    # A CATEGORICAL-default mixed column's numeric cells must round-trip EXACTLY
    # through the built DataFrame (not fmt.num 2-dp-rounded), so bind._code_series
    # can recover the real doses. 18 fractional doses + 6 "control" -> f .75 (mixed).
    ds = infer(table_of(["0.001", "0.002", "0.005", "control"] * 6))
    assert set(ds.df.iloc[:, 0].dropna().tolist()) == {
        "0.001", "0.002", "0.005", "control"}


# --- S-K(a): near-contiguity ID rule is for INTEGER runs only --------------
def test_fractional_all_distinct_near_contiguous_is_numeric_id():
    # a 22-distinct 1-dp measurement (BMI, absorbance) is near-contiguous but must
    # stay usable as numeric -> (NUMERIC, ID), not demoted to a bare ID.
    vals = [21.3, 22.1, 23.4, 24.8, 25.2, 26.7, 27.1, 28.5, 29.9, 30.2, 19.8,
            20.4, 22.9, 23.7, 24.1, 25.9, 26.3, 27.8, 28.2, 29.4, 31.1, 32.6]
    assert prof(vals).kinds == (Kind.NUMERIC, Kind.ID)


# --- S-J: ORDINAL / BINARY overrides are LABEL kinds, not numeric-family ----
def test_ordinal_override_on_vocab_text_keeps_levels_and_order():
    p = prof(["low", "medium", "high"] * 4, override=Kind.ORDINAL)
    assert p.levels == ("low", "medium", "high")
    assert p.level_order == ("low", "medium", "high")
    assert p.n_levels == 3
    assert p.n_missing == 0
    assert p.failed_cells == ()
    assert not any("treated as missing" in n for n in p.notes)


def test_ordinal_override_on_unknown_text_keeps_levels_no_order():
    p = prof(["Terrible", "Bad", "OK", "Great"] * 5, override=Kind.ORDINAL)
    assert p.levels == ("Terrible", "Bad", "OK", "Great")
    assert p.level_order is None
    assert p.n_missing == 0
    assert not any("treated as missing" in n for n in p.notes)


def test_binary_override_on_text_keeps_labels():
    p = prof(["Yes", "No"] * 10, override=Kind.BINARY)
    assert p.levels == ("Yes", "No")
    assert not any("treated as missing" in n for n in p.notes)


def test_ordinal_override_on_numeric_codes_keeps_numbers():
    from pandas.api.types import is_numeric_dtype
    ds = infer(table_of([1, 2, 3, 2, 1, 3]), overrides={"x": Kind.ORDINAL})
    p = ds.profiles[0]
    assert p.levels == ("1", "2", "3")
    assert p.level_order is None
    assert is_numeric_dtype(ds.df["x"])


# --- S-M: a repeated integer key under an ID header is a long-layout subject id
def test_repeated_integer_id_header_is_id():
    ids = [i for i in range(1, 13) for _ in (0, 1)]   # 1..12 each twice, m=24
    assert prof(ids, name="ID").kinds == (Kind.ID,)
    assert prof(ids, name="Subject").kinds == (Kind.ID,)
    # guards:
    assert prof(ids, name="Dose").kinds == (Kind.NUMERIC,)          # no ID header = a measurement
    assert prof(list(range(1, 19)), name="ID").kinds == (Kind.NUMERIC,)  # all-distinct <20 rows: oracle h3_2 col 1 stays
    assert prof([i for i in range(1, 9) for _ in (0, 1)], name="ID").kinds == (
        Kind.NUMERIC, Kind.ORDINAL, Kind.CATEGORICAL)              # d<=10 row unchanged


def test_repeated_integer_id_column_binds_long_paired():
    # S-M end-to-end: a repeated integer ID + a 2-level condition + a numeric
    # outcome must bind t_paired as LONG (the wide bogus-run is closed).
    from statkit import bind, check
    from statkit.registry import REGISTRY
    rows = []
    for i in range(1, 13):
        rows.append([Cell(i), Cell("Pre"), Cell(40 + i)])
        rows.append([Cell(i), Cell("Post"), Cell(44 + i + (i % 3))])
    table = Table(header=["ID", "Time", "Score"], rows=rows,
                  excel_rows=list(range(2, 26)), header_row=1)
    ds = infer(table)
    spec = REGISTRY["t_paired"]
    assert bind.satisfiable_layouts(spec, ds) == ["long"]
    assert bind.auto_columns(spec, ds, "long") == {
        "subject": ("ID",), "condition": ("Time",), "outcome": ("Score",)}
    # the wide bogus-run is closed: ID no longer offered as `before`
    assert bind.role_columns(spec.contract.roles_by_layout["wide"][0], ds) == ["Score"]
    b = bind.bind(spec, ds, {"subject": ("ID",), "condition": ("Time",),
                             "outcome": ("Score",)}, layout="long")
    assert b.n_used == 12
    b.findings = check.gate(b, {p.name: p for p in ds.profiles})
    assert spec.run(b).status == "ok"


# --- satisfaction scale is an ordered vocabulary (W2) ----------------------
def test_satisfaction_scale_is_ordered_vocab():
    p = prof(["Very dissatisfied", "Dissatisfied", "Neutral",
              "Satisfied", "Very satisfied"] * 4)
    assert p.kinds == (Kind.ORDINAL, Kind.CATEGORICAL)
    assert p.level_order == ("very dissatisfied", "dissatisfied", "neutral",
                             "satisfied", "very satisfied")
