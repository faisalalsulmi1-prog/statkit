"""RED list for statkit.grid (PLAN §4.1 + §9.1).

grid.load(bytes, filename) -> list[Grid], one Grid per NON-EMPTY sheet in
workbook order. It reads from an in-memory buffer only (L3): the shipped tool
never sees a disk path -- Streamlit hands it the upload as bytes. These tests
therefore read the fixture bytes from disk *here* (tests are not the shipped
tool and are not scanned by test_no_ai_no_io) and hand the buffer to grid.load.

Expected values are literals independently probed by re-opening each fixture
with openpyxl / xlrd directly -- a known-good oracle, never recomputed the way
grid.py computes them.
"""
from datetime import datetime
from pathlib import Path

import pytest

from statkit import grid
from statkit.grid import UNCOMPUTED, Cell, Grid
from tests import corpus

GEN = Path(__file__).resolve().parent / "fixtures" / "generated"


def _load(basename):
    """Load a committed fixture by basename through grid.load, from bytes."""
    p = GEN / basename
    return grid.load(p.read_bytes(), basename)


# --------------------------------------------------------------------------
# xlsx: values, number formats, formula flags
# --------------------------------------------------------------------------
def test_xlsx_literal_cell_value_fmt_and_flag():
    # gen_w2_5_2: A3 'Phenotype' (General text), C4 55.0387... with a literal
    # percent number format '0.0"%"' -- both literals, formula flag False.
    g = _load("gen_w2_5_2.xlsx")[0]
    a3 = g.rows[2][0]
    assert a3.value == "Phenotype"
    assert a3.fmt == "General"
    assert a3.formula is False
    c4 = g.rows[3][2]
    assert c4.value == pytest.approx(55.03875968992248)
    assert c4.fmt == '0.0"%"'          # fmt string preserved verbatim for coerce (snag 11)
    assert c4.formula is False


def test_xlsx_formula_flag_and_cached_value():
    # gen_w2_4_1 F2 = '=E2-D2', cached -34.29, fmt '#,##0.00'
    g = _load("gen_w2_4_1.xlsx")[0]
    f2 = g.rows[1][5]
    assert f2.formula is True
    assert f2.value == pytest.approx(-34.29)
    assert f2.fmt == "#,##0.00"


def test_uncomputed_formula_becomes_sentinel():
    # gen_s3_2 E18 = '=AVERAGE(B18:D18)' with NO cached value -> UNCOMPUTED
    g = _load("gen_s3_2.xlsx")[0]
    e18 = g.rows[17][4]
    assert e18.formula is True
    assert e18.value is UNCOMPUTED


# --------------------------------------------------------------------------
# merged ranges, hidden rows/cols
# --------------------------------------------------------------------------
def test_merged_ranges_reported_1based_inclusive():
    # gen_h3_2 has a two-row full-width merged title/header band.
    g = _load("gen_h3_2.xlsx")[0]
    assert (1, 1, 1, 7) in g.merged
    assert (2, 1, 2, 7) in g.merged


def test_hidden_rows_and_cols():
    # gen_w2_4_1: rows 11,26,47 hidden; column G (7) hidden.
    g = _load("gen_w2_4_1.xlsx")[0]
    assert g.hidden_rows == frozenset({11, 26, 47})
    assert g.hidden_cols == frozenset({7})


# --------------------------------------------------------------------------
# multi-sheet: order preserved, empty sheet skipped
# --------------------------------------------------------------------------
def test_empty_sheet_skipped_order_preserved():
    # gen_w2_8_1: Sheet1 (empty 1x1), Data, Notes -> two grids, in order.
    grids = _load("gen_w2_8_1.xlsx")
    assert [g.sheet for g in grids] == ["Data", "Notes"]


def test_multi_sheet_all_kept_in_workbook_order():
    grids = _load("gen_h5_3.xlsx")
    assert [g.sheet for g in grids] == ["Beep Test", "Body Comp", "Strength 1RM"]


# --------------------------------------------------------------------------
# csv / txt: encoding, delimiter, line endings
# --------------------------------------------------------------------------
def test_csv_semicolon_delimiter_and_bom_stripped():
    grids = _load("gen_w2_3_2.csv")            # BOM + CRLF + ';' delimited
    assert len(grids) == 1
    g = grids[0]
    assert g.sheet == "(single)"
    assert g.rows[0][0].value == "ShipmentID"  # BOM gone, not split on comma
    assert len(g.rows[0]) == 8


def test_csv_comma_bom_stripped():
    g = _load("gen_s2_3.csv")[0]
    assert g.rows[0][0].value == "OrderID"


def test_csv_no_stray_carriage_returns():
    g = _load("gen_s2_3.csv")[0]              # CRLF file
    assert not any("\r" in c.value for row in g.rows for c in row if isinstance(c.value, str))


def test_txt_tab_delimited_cp1252_fallback():
    # gen_w2_10_1.txt is tab-delimited, cp1252 (a 0xB5 micro sign fails utf-8).
    g = _load("gen_w2_10_1.txt")[0]
    assert len(g.rows[0]) == 10
    assert g.rows[0][5].value == "Concentration (µg/mL)"


def test_cp1252_fallback_in_memory():
    # bytes with cp1252 smart quotes (0x93/0x94) -> decoded, not crashed.
    data = b"Name,Note\r\nAcme,\x93hi\x94\r\n"
    g = grid.load(data, "x.csv")[0]
    assert g.rows[1][1].value == "“hi”"


def test_csv_empty_field_becomes_none():
    data = b"a,b,c\r\n1,,3\r\n"
    g = grid.load(data, "x.csv")[0]
    assert g.rows[1][1].value is None


# --------------------------------------------------------------------------
# S14 -- bad / mis-encoded uploads: UTF-16 CSV, and files renamed to a
# workbook extension they are not.
# --------------------------------------------------------------------------
def test_utf16_csv_is_decoded():
    # A UTF-16 (BOM'd) CSV must decode to clean cells, not one garbage column.
    data = "Name,Score,Group\r\nAl,3,X\r\nBo,4,Y\r\n".encode("utf-16")
    g = grid.load(data, "data.csv")[0]
    assert g.rows[0][0].value == "Name"
    assert [c.value for c in g.rows[0]] == ["Name", "Score", "Group"]
    assert len(g.rows[0]) == 3


def test_csv_renamed_xlsx_raises_grid_error():
    data = b"Name,Score\r\nAl,3\r\nBo,4\r\n"
    with pytest.raises(grid.GridError):
        grid.load(data, "data.xlsx")


def test_html_renamed_xls_raises_grid_error():
    data = b"<html><body><table><tr><td>Name</td></tr></table></body></html>"
    with pytest.raises(grid.GridError):
        grid.load(data, "report.xls")


# --------------------------------------------------------------------------
# xls (legacy) via xlrd: dates, booleans, merged, hidden
# --------------------------------------------------------------------------
def test_xls_date_bool_merged_hidden():
    g = _load("gen_w2_1_1.xls")[0]
    # B3 (row 3, col 2) is a date cell -> datetime
    b3 = g.rows[2][1]
    assert b3.value == datetime(2009, 1, 12)
    # G3 (row 3, col 7) is a boolean cell -> a real bool, not int
    g3 = g.rows[2][6]
    assert g3.value is False
    # merged title band A1:I1
    assert (1, 1, 1, 9) in g.merged
    # hidden row 30 (0-based 29 in xlrd)
    assert 30 in g.hidden_rows


# --------------------------------------------------------------------------
# structural invariants
# --------------------------------------------------------------------------
def test_excel_rows_are_1based_and_contiguous():
    g = _load("gen_h2_1.xlsx")[0]
    assert g.excel_rows == list(range(1, len(g.rows) + 1))


def test_grid_is_rectangular():
    g = _load("gen_w2_3_2.csv")[0]
    widths = {len(r) for r in g.rows}
    assert len(widths) == 1


def test_source_records_filename():
    g = _load("gen_h2_1.xlsx")[0]
    assert g.source == "gen_h2_1.xlsx"


# --------------------------------------------------------------------------
# large-sheet fallback (>400k cells) -> read_only, values only, note set
# --------------------------------------------------------------------------
def test_large_sheet_fallback():
    data, fname = corpus.fixture_bytes("w2_2_build")   # 548,109 cells, built on demand
    grids = grid.load(data, fname)
    assert len(grids) == 1
    g = grids[0]
    assert len(g.rows) == 60901                # 1 header + 60,900 data rows
    assert "large" in g.note.lower()
    assert g.merged == []
    assert g.hidden_rows == frozenset()
    assert all(c.fmt == "General" for c in g.rows[0])   # fast mode drops fmt


# --------------------------------------------------------------------------
# synth corpus loaders (tests/corpus.py) -- Chunk 5 consumes these
# --------------------------------------------------------------------------
def test_corpus_lists_all_fixtures_excluding_dropped():
    ids = corpus.fixture_ids()
    assert "h2_1" in ids
    assert "w2_2_build" in ids          # the builder is listed
    assert "h1_1" not in ids            # dropped/ is excluded
    assert len(ids) == 35               # 34 files + 1 builder (STATE.md: 35 sidecars)


def test_corpus_fixture_bytes_and_name():
    data, name = corpus.fixture_bytes("h2_1")
    assert data[:2] == b"PK"            # xlsx is a zip
    assert name == "gen_h2_1.xlsx"


def test_corpus_load_grids_roundtrips_through_grid_load():
    grids = corpus.load_grids("h2_1")
    assert grids and all(isinstance(g, Grid) for g in grids)


def test_corpus_expect_sidecar_loads():
    e = corpus.expect("h2_1")
    assert e["fixture_id"] == "h2_1"
    assert "sheets" in e
