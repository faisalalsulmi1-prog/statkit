"""Chunk 3 RED list for clean.py (PLAN §4.2, §4.5 snags, §9.1).

clean.py works at the ROW/COLUMN-STRUCTURE level only: find the header, drop
title/notes/summary/subtotal/footer/repeated-header/merged-note/spacer rows,
join two-row headers, consume a units row, name/dedupe headers, honour hidden
rows, and preserve the original 1-based excel_rows for every surviving data row.
Type inference / cell coercion belong to Chunks 4/5 and are NOT asserted here.

Acceptance is measured against the committed oracles in tests/fixtures/generated/
(header_row / header_row2 / units_row / data_start_row / n_rows / n_cols / names).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/ for corpus.py

from statkit import grid as gridmod
from statkit.clean import clean
from statkit.grid import Cell, Grid

import corpus


# --------------------------------------------------------------------------
# tiny grid builder for hand-authored shapes
# --------------------------------------------------------------------------
def mkgrid(values, merged=None, hidden_rows=frozenset(), hidden_cols=frozenset()):
    rows = [[Cell(v) for v in row] for row in values]
    excel_rows = list(range(1, len(rows) + 1))
    return Grid(
        rows=rows,
        excel_rows=excel_rows,
        merged=list(merged or []),
        hidden_rows=frozenset(hidden_rows),
        hidden_cols=frozenset(hidden_cols),
        sheet="(single)",
        source="test.xlsx",
    )


# ==========================================================================
# 1. THE FIVE CANONICAL HEADER SHAPES (DoD)
# ==========================================================================
def test_shape_clean_header_row1():
    g = mkgrid([
        ["Name", "Age", "Score"],
        ["Al", 20, 3.1],
        ["Bo", 22, 4.5],
        ["Cy", 25, 2.8],
    ])
    t = clean(g)
    assert t.header_row == 1
    assert t.header == ["Name", "Age", "Score"]
    assert len(t.rows) == 3
    assert not t.ambiguous
    assert not t.two_row


def test_shape_two_title_rows_then_header():
    g = mkgrid([
        ["Annual Report 2026", None, None],
        ["prepared by lab", None, None],
        ["Name", "Age", "Score"],
        ["Al", 20, 3.1],
        ["Bo", 22, 4.5],
    ])
    t = clean(g)
    assert t.header_row == 3
    assert len(t.rows) == 2
    assert t.excel_rows == [4, 5]


def test_shape_two_row_header_lookdown():
    g = mkgrid([
        ["ID", "Name", "Age", "Ratings", None],
        [None, None, None, "Q1", "Q2"],
        ["P1", "Al", 20, 3, 4],
        ["P2", "Bo", 22, 5, 2],
    ], merged=[(1, 4, 1, 5)])
    t = clean(g)
    assert t.two_row
    assert t.header_row == 1
    assert t.header_row2 == 2
    assert t.data_start_row == 3
    assert t.header == ["ID", "Name", "Age", "Ratings – Q1", "Ratings – Q2"]
    assert len(t.rows) == 2


def test_shape_all_numeric_codes_is_ambiguous():
    g = mkgrid([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ])
    t = clean(g)
    assert t.ambiguous


def test_shape_header_at_row4():
    g = mkgrid([
        ["Lab log", None, None],
        [None, None, None],
        [None, None, None],
        ["Sample", "pH", "Temp"],
        ["S1", 7.0, 20],
        ["S2", 6.8, 21],
    ])
    t = clean(g)
    assert t.header_row == 4
    assert len(t.rows) == 2


# ==========================================================================
# 2. TWO-ROW look-UP (snag 13) + the guard that keeps a lone merged title single
# ==========================================================================
def test_two_row_header_lookup_snag13():
    g = mkgrid([
        [None, "Term 1", None, "Term 2", None],
        ["Name", "Test", "Quiz", "Test", "Quiz"],
        ["Al", 80, 90, 85, 88],
        ["Bo", 70, 75, 72, 79],
    ], merged=[(1, 2, 1, 3), (1, 4, 1, 5)])
    t = clean(g)
    assert t.two_row
    assert t.header_row == 1
    assert t.header_row2 == 2
    assert t.header == [
        "Name",
        "Term 1 – Test", "Term 1 – Quiz",
        "Term 2 – Test", "Term 2 – Quiz",
    ]


def test_lone_merged_title_is_not_two_row():
    # A single merged caption above the header (only 1 non-empty top cell) must
    # NOT be read as a two-row header (s1_3 May regression).
    g = mkgrid([
        [None, None, None, "Water Quality", None, None],
        ["Date", "Station", "Temp", "pH", "DO", "Taxa"],
        ["5/3", "S1", 12.4, 7.8, 9.6, 14],
        ["5/4", "S2", 13.1, 7.6, 8.9, 11],
    ], merged=[(1, 4, 1, 6)])
    t = clean(g)
    assert not t.two_row
    assert t.header_row == 2
    assert t.header[0] == "Date"


# ==========================================================================
# 3. UNITS ROW (snag 4) -- consumed, header stays BARE, unit carried separately
# ==========================================================================
def test_units_row_consumed_names_stay_bare():
    g = mkgrid([
        ["Site", "Temp", "pH"],
        [None, "C", "%"],
        ["A", 20, 7.0],
        ["B", 21, 6.8],
    ])
    t = clean(g)
    assert t.units_row == 2
    assert t.data_start_row == 3
    assert t.header == ["Site", "Temp", "pH"]      # NOT "Temp (C)"
    assert t.units == [None, "C", "%"]
    assert len(t.rows) == 2


def test_real_units_row_over_numeric_is_still_consumed():
    # NARROWING guard: a real units row (short tokens sitting ABOVE numeric
    # columns) must STILL be consumed after the NB1 fix -- prove the feature was
    # narrowed, not removed. Only a majority of annotated columns need be numeric
    # (w2_9_1 has 'mg/L' over inline-unit text), so the label column may be text.
    g = mkgrid([
        ["Site", "Temp", "Depth"],
        [None, "C", "m"],           # unit tokens over the two numeric columns
        ["A1", 20, 5],
        ["A2", 21, 6],
        ["A3", 19, 4],
    ])
    t = clean(g)
    assert t.units_row == 2
    assert t.data_start_row == 3
    assert t.units == [None, "C", "m"]
    assert len(t.rows) == 3
    assert t.excel_rows == [3, 4, 5]


def test_short_token_row_over_text_is_not_units():
    # NB1: a first data row of single-letter categories ('A'/'B'/'C') matches the
    # units-token SHAPE, but the columns below it hold TEXT not numbers, so it is
    # NOT a units row and must survive as the first data row.
    g = mkgrid([
        ["Grade", "Room"],
        ["A", "B"],
        ["B", "A"],
        ["A", "C"],
        ["C", "B"],
    ])
    t = clean(g)
    assert t.units_row is None
    assert len(t.rows) == 4
    assert t.excel_rows == [2, 3, 4, 5]      # first data row kept
    assert t.dropped_rows == []


# ==========================================================================
# 3b. ALL-CATEGORICAL (ALL-TEXT) SHEETS -- NB1 (first-row loss) + S-H (false
#     header ambiguity). The fixture corpus has ZERO all-categorical sheets, so
#     these two defects hid until now.
# ==========================================================================
def test_first_data_row_of_all_text_sheet_is_not_a_units_row():
    # NB1: on an all-text sheet the row below the header ('M','Yes') matched the
    # units-token shape and was silently consumed as a phantom units row -> the
    # first data row vanished with dropped_rows empty ("no rows excluded").
    header = ["Sex", "Smoker"]
    data = [["M" if k % 2 == 0 else "F", "Yes" if k % 3 == 0 else "No"]
            for k in range(19)]
    g = mkgrid([header] + data)
    t = clean(g)
    assert t.units_row is None
    assert t.header_row == 1
    assert len(t.rows) == 19                  # header + 19 data, all 19 kept
    assert t.excel_rows[0] == 2               # first data row NOT swallowed
    assert t.excel_rows == list(range(2, 21))
    assert t.dropped_rows == []


def test_all_text_sheet_with_blank_under_header_keeps_all_data():
    # NB1 variant: a blank spacer directly under the header. The blank dragged the
    # real header's below-score down so a DATA row was mis-picked as the header,
    # then the units bug ate more -- Fable saw most rows vanish. The header must
    # be row 1, the blank a spacer, and every data row a survivor.
    header = ["Group", "Rating"]
    grp = ["A", "B", "C"]
    cats = ["Good", "Fair", "Poor"]
    data = [[grp[k % 3], cats[k % 3]] for k in range(9)]
    g = mkgrid([header, [None, None]] + data)
    t = clean(g)
    assert t.header_row == 1                   # real header, not a data row
    assert t.units_row is None
    assert len(t.rows) == 9                     # all 9 data rows survive
    assert t.excel_rows == list(range(3, 12))
    assert not t.ambiguous


def test_all_text_survey_sheet_is_not_header_ambiguous():
    # S-H: every row of an all-text survey is ~all-text, so each data row became a
    # false rival 'header' -> a bogus near-tie -> ambiguous=True + a header picker
    # on a perfectly clean survey. A rival whose values recur below is a data row,
    # not a header candidate, so a clean categorical survey is NOT ambiguous.
    header = ["Sex", "Smoker", "Exercises"]
    opts = [("M", "Yes", "Daily"), ("F", "No", "Weekly"),
            ("F", "Yes", "Never"), ("M", "No", "Daily")]
    data = [list(opts[k % 4]) for k in range(16)]
    g = mkgrid([header] + data)
    t = clean(g)
    assert not t.ambiguous
    assert t.header_row == 1
    assert len(t.rows) == 16


# T1/T2 seam: 24 all-text rows cycling four 2-level columns on different periods.
_BLANK_HDR_DATA = [
    [["M", "F"][i % 2], ["Yes", "No"][(i // 2) % 2],
     ["North", "South"][(i // 4) % 2], ["Placebo", "Drug"][(i // 3) % 2]]
    for i in range(24)
]


@pytest.mark.parametrize("header,expected", [
    (["Sex", "Smoker", "Region", None],
     ["Sex", "Smoker", "Region", "Column D"]),
    ([None, "Smoker", "Region", "Arm"],
     ["Column A", "Smoker", "Region", "Arm"]),
])
def test_all_text_sheet_with_blank_header_cell_keeps_header_row(header, expected):
    # NB3: on an all-text sheet a header with one blank cell scores below a full
    # data row (fill 0.75 vs 1.0), so pre-fix the DATA row won the argmax
    # (header_row==2, header ['M','Yes','North','Placebo'], 22 rows). A data-like
    # row -- its values recurring down their own columns -- must never win header
    # detection over the real header.
    g = mkgrid([header] + _BLANK_HDR_DATA)
    t = clean(g)
    assert t.header_row == 1
    assert t.header == expected
    assert t.ambiguous is False
    assert len(t.rows) == 24
    assert t.excel_rows[0] == 2


def test_na_token_data_row_over_numeric_columns_is_not_units():
    # N13: a first data row of NA tokens (Ali,NA,n/a,na) over numeric columns
    # matches the units-token SHAPE gate, so pre-fix it was CONSUMED as a units
    # row (units_row==2, units ['Ali','NA','n/a','na'], 5 rows). NA tokens are
    # missing-value markers, never units -- the row is a subject with missing data.
    g = mkgrid([
        ["Name", "Height", "Weight", "Age"],
        ["Ali", "NA", "n/a", "na"],
        ["Bob", 170, 65, 30],
        ["Cara", 160, 55, 25],
        ["Dan", 180, 80, 40],
        ["Eve", 165, 60, 28],
        ["Finn", 175, 70, 35],
    ])
    t = clean(g)
    assert t.units_row is None
    assert all(u is None for u in t.units)
    assert len(t.rows) == 6
    assert t.excel_rows[0] == 2


# ==========================================================================
# 4. ROW-DROP RULES
# ==========================================================================
def test_repeated_header_row_dropped_snag3():
    g = mkgrid([
        ["OrderID", "Region", "Units"],
        ["O1", "West", 5],
        ["OrderID", "Region", "Units"],   # re-pasted header, row 3
        ["O2", "East", 3],
    ])
    t = clean(g)
    assert len(t.rows) == 2
    assert t.excel_rows == [2, 4]


def test_full_width_merged_note_row_dropped_snag5():
    g = mkgrid([
        ["Date", "Station", "Temp", "pH"],
        ["5/3", "S1", 12.4, 7.8],
        ["note: site flooded", None, None, None],   # merged A3:D3
        ["5/4", "S2", 13.1, 7.6],
    ], merged=[(3, 1, 3, 4)])
    t = clean(g)
    assert len(t.rows) == 2
    assert t.excel_rows == [2, 4]


def test_summary_row_end_anchored_snag12():
    g = mkgrid([
        ["Name", "Class", "Test"],
        ["Al", "A", 85],
        ["Class A Average", None, 85.0],   # group-prefixed subtotal
        ["Bo", "B", 78],
    ])
    t = clean(g)
    assert len(t.rows) == 2
    assert t.excel_rows == [2, 4]


def test_summary_row_start_anchored():
    g = mkgrid([
        ["Age Group", "Accepted", "Declined"],
        ["18-35", 87, 23],
        ["36-65", 124, 16],
        ["Total", 211, 39],
    ])
    t = clean(g)
    assert len(t.rows) == 2


def test_footer_note_and_spacer_dropped():
    g = mkgrid([
        ["Name", "Score"],
        ["Al", 3],
        ["Bo", 4],
        [None, None],
        ["Source: internal survey", None],
    ])
    t = clean(g)
    assert len(t.rows) == 2
    assert t.excel_rows == [2, 3]


def test_interior_blank_spacer_dropped_and_counted():
    g = mkgrid([
        ["Name", "Score"],
        ["Al", 3],
        [None, None],
        ["Bo", 4],
    ])
    t = clean(g)
    assert len(t.rows) == 2
    assert t.excel_rows == [2, 4]


# ==========================================================================
# 5. HEADER NAMING + COLUMNS
# ==========================================================================
def test_none_header_named_by_letter():
    g = mkgrid([
        ["A", None, "C"],
        [1, 2, 3],
        [4, 5, 6],
    ])
    t = clean(g)
    assert t.header == ["A", "Column B", "C"]


def test_duplicate_headers_suffixed():
    g = mkgrid([
        ["X", "X", "Y"],
        [1, 2, 3],
    ])
    t = clean(g)
    assert t.header == ["X", "X (2)", "Y"]


def test_whitespace_collapsed_but_verbatim():
    g = mkgrid([
        ["  Full   Name ", "Age"],
        ["Al", 20],
    ])
    t = clean(g)
    assert t.header == ["Full Name", "Age"]


def test_empty_columns_are_kept_not_dropped():
    # grid width == n_cols across the whole corpus: clean keeps every column.
    g = mkgrid([
        ["Name", "Blank", "Score"],
        ["Al", None, 3],
        ["Bo", None, 4],
    ])
    t = clean(g)
    assert len(t.header) == 3
    assert t.header == ["Name", "Blank", "Score"]


# ==========================================================================
# 6. HIDDEN ROWS + excel_rows integrity
# ==========================================================================
def test_hidden_rows_kept_by_default():
    g = mkgrid([
        ["Name", "Score"],
        ["Al", 3],
        ["Bo", 4],
        ["Cy", 5],
    ], hidden_rows={3})
    t = clean(g)
    assert len(t.rows) == 3
    assert 3 in t.hidden_rows


def test_hidden_rows_excluded_when_asked():
    g = mkgrid([
        ["Name", "Score"],
        ["Al", 3],
        ["Bo", 4],
        ["Cy", 5],
    ], hidden_rows={3})
    t = clean(g, keep_hidden=False)
    assert t.excel_rows == [2, 4]


def test_excel_rows_preserved_through_drops():
    g = mkgrid([
        ["Report", None],
        ["Name", "Score"],
        ["Al", 3],
        [None, None],
        ["Bo", 4],
        ["Total", 7],
    ])
    t = clean(g)
    assert t.header_row == 2
    assert t.excel_rows == [3, 5]


# ==========================================================================
# 7. SNAG 2 -- full-width merged title must NOT create a false AMBIGUOUS
# ==========================================================================
def test_merged_full_width_title_not_ambiguous():
    g = mkgrid([
        ["Riverside Cafe Customer Survey", None, None, None],
        [None, None, None, None],
        ["RespondentID", "Age", "Gender", "Score"],
        ["R01", 64, "Female", 3],
        ["R02", 41, "Male", 4],
    ], merged=[(1, 1, 1, 4)])
    t = clean(g)
    assert t.header_row == 3
    assert not t.ambiguous


# ==========================================================================
# 8. CORPUS -- header_row / n_rows / n_cols / names line up with the oracles
# ==========================================================================
# s1_3 June oracle mislabels col5/col6 (col5 header is 'pH' with data; col6 is
# the empty unnamed column). Assert the CORRECTED names for that one sheet.
_JUNE_CORRECTED = [
    "Date", "Station", "Time", "Temp C", "pH", "Column F",
    "DO mg/L", "Taxa Count", "Notes",
]


def _data_sheets():
    out = []
    for fid in corpus.fixture_ids():
        ex = corpus.expect(fid)
        for sh in ex["sheets"]:
            if sh.get("role") == "data":
                out.append((fid, sh["name"]))
    return out


# ==========================================================================
# 9. B4 -- a summary-worded first cell must NOT delete real data rows
# ==========================================================================
def test_rating_levels_named_average_are_kept_as_data():
    # A Rating column whose LEVELS are "Average" / "Below average" / ... plus two
    # text rows that merely START with a summary keyword ("Mean Street",
    # "Total Recall"). None is a subtotal: they recur and/or sit with text, so
    # all 9 rows must survive. (Today: 6 of 9 are silently dropped.)
    g = mkgrid([
        ["Rating", "Score", "Group"],
        ["Average", 72, "A"],
        ["Below average", 55, "B"],
        ["Above average", 88, "A"],
        ["Average", 69, "C"],
        ["Good", 91, "B"],
        ["Poor", 40, "C"],
        ["Excellent", 84, "A"],
        ["Mean Street", 63, "A"],
        ["Total Recall", 77, "B"],
    ])
    t = clean(g)
    assert len(t.rows) == 9
    assert t.dropped_rows == []


def test_labelled_aggregate_row_still_dropped():
    # A genuine subtotal ("Class A Average"): a UNIQUE compound label sitting on
    # an all-numeric row -> still dropped as a summary row (regression guard).
    g = mkgrid([
        ["Name", "Test1", "Test2"],
        ["Al", 80, 78],
        ["Bo", 70, 75],
        ["Cy", 90, 88],
        ["Class A Average", 71.2, 68.0],
    ])
    t = clean(g)
    assert len(t.rows) == 3
    assert t.excel_rows == [2, 3, 4]
    reason = [r for (e, r) in t.dropped_rows if e == 5]
    assert reason and "summary" in reason[0]


def test_restore_rows_keeps_a_summary_row():
    # The same aggregate row, but the caller pinned its 1-based excel row via
    # restore_rows -> it is exempt from summary/footer classification and kept.
    g = mkgrid([
        ["Name", "Test1", "Test2"],
        ["Al", 80, 78],
        ["Bo", 70, 75],
        ["Cy", 90, 88],
        ["Class A Average", 71.2, 68.0],
    ])
    t = clean(g, restore_rows={5})
    assert len(t.rows) == 4
    assert t.excel_rows == [2, 3, 4, 5]


# ==========================================================================
# 10. B5 -- forcing a header row must NOT erase the detector's ambiguity, so
#     the app can keep offering the header picker instead of oscillating.
# ==========================================================================
def test_forced_header_row_keeps_detector_ambiguity():
    g = mkgrid([
        ["red flag", "blue sky", "green tea"],
        ["big data", "cold brew", "hot take"],
        ["fast car", "slow lane", "old news"],
    ])
    t = clean(g, header_row=2)
    assert t.header_row == 2
    assert t.ambiguous is True
    assert len(t.candidates) > 1


@pytest.mark.parametrize("fid,sheet", _data_sheets())
def test_corpus_clean_matches_oracle(fid, sheet):
    grids = {g.sheet: g for g in corpus.load_grids(fid)}
    assert sheet in grids, f"{fid}: sheet {sheet!r} not returned by grid.load"
    ex = [s for s in corpus.expect(fid)["sheets"] if s["name"] == sheet][0]

    t = clean(grids[sheet])

    assert t.header_row == ex["header_row"], "header_row"
    assert t.two_row == ex["two_row"], "two_row"
    assert t.header_row2 == ex["header_row2"], "header_row2"
    assert t.units_row == ex["units_row"], "units_row"
    assert t.data_start_row == ex["data_start_row"], "data_start_row"
    assert len(t.rows) == ex["n_rows"], "n_rows (surviving data rows)"
    assert len(t.header) == ex["n_cols"], "n_cols"
    assert not t.ambiguous, "no false AMBIGUOUS"

    expected_names = [c["name"] for c in ex["columns"]]
    if fid == "s1_3" and sheet == "June":
        expected_names = _JUNE_CORRECTED   # oracle bug worked around (see report)
    assert t.header == expected_names, "header names"
