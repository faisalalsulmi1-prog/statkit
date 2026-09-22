"""RED list for statkit.coerce (PLAN Chunk 4 / §4.3 + §9.1 coerce list + line-477 grill list).

These assertions ARE the spec: per-column `detect_convention` and per-cell
`coerce_cell`, honouring snags 1/7/8/11/14. Written before the module exists
(watch them fail), then made green.

Seam under test: the two public pure functions `detect_convention(cells) ->
(Convention, [Action])` and `coerce_cell(cell, conv) -> Coerced`, plus the tiny
`resolve_censored(coerced, numeric_path)` that infer.py will call at Chunk 5 to
apply the snag-1 text/numeric-column decision. No fixtures are loaded here (that
is Chunk 5's test_corpus.py); Cells are built by hand.
"""
import datetime
import math

import pytest

from statkit.coerce import (
    UNCOMPUTED,
    Action,
    Cell,
    Coerced,
    Convention,
    coerce_cell,
    detect_convention,
    resolve_censored,
)

US = Convention(decimal=".", thousands=",")          # 1,234.5
EU = Convention(decimal=",", thousands=".")          # 1.234,5
SPACE = Convention(decimal=".", thousands=" ")       # 1 234
KG = Convention(decimal=".", thousands=",", unit="kg")


def num_cells(values, fmt="0.00"):
    return [Cell(v, fmt) for v in values]


def text_cells(values):
    return [Cell(v, "General") for v in values]


# --------------------------------------------------------------------------
# detect_convention
# --------------------------------------------------------------------------
def test_detect_ambiguous_us_thousands_gives_info():
    conv, actions = detect_convention(text_cells(["1,234", "5,678"]))
    assert conv.thousands == "," and conv.decimal == "."
    assert any(a.auto and "toggle" in a.detail.lower() for a in actions), actions


def test_detect_eu_decimal_comma():
    conv, _ = detect_convention(text_cells(["3,5", "4,2", "12,75"]))
    assert conv.decimal == "," and conv.thousands == "."


def test_detect_eu_full_form():
    conv, _ = detect_convention(text_cells(["1.234,5", "9.999,0"]))
    assert conv.decimal == "," and conv.thousands == "."


def test_detect_space_thousands():
    conv, _ = detect_convention(text_cells(["1 234", "5 678"]))
    assert conv.thousands == " "


def test_detect_unit_kg():
    conv, _ = detect_convention(text_cells(["12 kg", "13 kg", "14 kg"]))
    assert conv.unit is not None and conv.unit.lower() == "kg"


def test_detect_unit_percent():
    conv, _ = detect_convention(text_cells(["12%", "30%", "45%"]))
    assert conv.unit == "%"


def test_detect_no_unit_below_threshold_snag8():
    # ' C' on 1 of 7 numeric cells (<85%) -> NO column unit (snag 8).
    conv, _ = detect_convention(text_cells(["20.1 C", "19", "21", "22", "23", "24", "25"]))
    assert conv.unit is None


def test_detect_mixed_decimal_and_comma_flags_c_cells():
    conv, actions = detect_convention(text_cells(["3.5", "9.1", "3,5"]))
    assert conv.decimal == "." and conv.thousands == ","
    flags = [a for a in actions if not a.auto]
    assert flags, "mixed D&C must raise a FLAG"
    assert any("3,5" in a.detail for a in flags), flags


def test_detect_percent_format_true_on_unquoted_pct():
    conv, _ = detect_convention(num_cells([0.12, 0.5, 0.9], fmt="0%"))
    assert conv.percent_format is True


def test_detect_percent_format_false_on_literal_quoted_pct_snag11():
    # '0.0"%"' -> the % is a quoted literal, not a scaling format.
    conv, _ = detect_convention(num_cells([55.04, 30.0, 14.96], fmt='0.0"%"'))
    assert conv.percent_format is False


# --------------------------------------------------------------------------
# coerce_cell -- the grill's exact list (PLAN line 477)
# --------------------------------------------------------------------------
def test_us_thousands_stripped():
    assert coerce_cell(Cell("1,234"), US) == Coerced("num", num=1234.0)


def test_eu_decimal_comma():
    assert coerce_cell(Cell("3,5"), EU) == Coerced("num", num=3.5)


def test_eu_thousands_and_decimal():
    assert coerce_cell(Cell("1.234,5"), EU) == Coerced("num", num=1234.5)


def test_percent_text_is_percent_points():
    assert coerce_cell(Cell("12%"), US) == Coerced("num", num=12.0)


def test_accounting_parens_negative():
    assert coerce_cell(Cell("(12)"), US) == Coerced("num", num=-12.0)


def test_unicode_minus():
    assert coerce_cell(Cell("−12"), US) == Coerced("num", num=-12.0)


def test_unit_stripped_when_detected():
    assert coerce_cell(Cell("12 kg"), KG) == Coerced("num", num=12.0)


def test_space_thousands():
    assert coerce_cell(Cell("1 234"), SPACE) == Coerced("num", num=1234.0)


def test_currency_stripped():
    assert coerce_cell(Cell("$1,200"), US) == Coerced("num", num=1200.0)


def test_censored_low():
    assert coerce_cell(Cell("<0.01"), US) == Coerced("censored", text="<0.01")


def test_na_token_empty():
    assert coerce_cell(Cell("N/A"), US).kind == "empty"


def test_dot_is_na_empty():
    assert coerce_cell(Cell("."), US).kind == "empty"


def test_none_word_is_real_text_not_na():
    # 'none' is too likely real data in a stats tool ("no symptoms", severity
    # none, "no complication") to drop as missing -- it is a text level, not NA.
    c = coerce_cell(Cell("none"), US)
    assert c.kind == "text" and c.text == "none"
    assert coerce_cell(Cell("None"), US) == Coerced("text", text="None")


def test_bool_before_int():
    assert coerce_cell(Cell(True), US).kind == "bool"
    assert coerce_cell(Cell(False), US).kind == "bool"


def test_int_and_float_num():
    assert coerce_cell(Cell(42), US) == Coerced("num", num=42.0)
    assert coerce_cell(Cell(3.14), US) == Coerced("num", num=3.14)


def test_nan_is_empty():
    assert coerce_cell(Cell(float("nan")), US).kind == "empty"


def test_percent_number_format_scales_x100():
    # 0.12 shown as "12%" (fmt 0%) -> 12 percent points.
    conv = Convention(percent_format=True)
    assert coerce_cell(Cell(0.12, "0%"), conv) == Coerced("num", num=12.0)


def test_none_is_empty_and_uncomputed():
    assert coerce_cell(Cell(None), US).kind == "empty"
    assert coerce_cell(Cell(UNCOMPUTED), US).kind == "uncomputed"


def test_dates():
    assert coerce_cell(Cell(datetime.date(2020, 1, 2)), US).kind == "date"
    assert coerce_cell(Cell("2020-01-02"), US).kind == "date"


def test_yes_no_true_false_stay_text_not_bool():
    for w in ("yes", "no", "true", "false"):
        c = coerce_cell(Cell(w), US)
        assert c.kind == "text" and c.text == w, w


def test_plain_word_is_text_level():
    assert coerce_cell(Cell("Control"), US) == Coerced("text", text="Control")


# --------------------------------------------------------------------------
# grouping validation -- IP-like tokens must NOT parse as huge numbers
# (silent-wrong: a few '.'-grouped octets make detect pick EU thousands,
# then stripping the dots turns "35.80.194.178" into 3580194178).
# --------------------------------------------------------------------------
def test_ip_like_token_is_not_a_number():
    # Under the (mis-detected) EU convention, dotted octets that are not clean
    # 3-digit thousands groups must fail to a text cell (raw kept), not a number.
    conv = Convention(decimal=",", thousands=".")
    for ip in ("35.80.194.178", "97.237.64.17", "26.1.64.232", "192.168.1.1"):
        c = coerce_cell(Cell(ip), conv)
        assert c.kind == "text" and c.text == ip, ip


def test_grouping_validation_keeps_legit_numbers():
    # The grouping guard must not regress real grouped/plain numbers.
    assert coerce_cell(Cell("192"), US) == Coerced("num", num=192.0)
    assert coerce_cell(Cell("1234567"), US) == Coerced("num", num=1234567.0)
    assert coerce_cell(Cell("12.345.678"), EU) == Coerced("num", num=12345678.0)
    assert coerce_cell(Cell("1,234"), US) == Coerced("num", num=1234.0)
    assert coerce_cell(Cell("1.234,5"), EU) == Coerced("num", num=1234.5)
    assert coerce_cell(Cell("1 234"), SPACE) == Coerced("num", num=1234.0)
    assert coerce_cell(Cell("$1,200"), US) == Coerced("num", num=1200.0)
    assert coerce_cell(Cell("3,5"), EU) == Coerced("num", num=3.5)


# --------------------------------------------------------------------------
# snag 11 -- literal 0.0"%" must NOT be x100 (silent-wrong)
# --------------------------------------------------------------------------
def test_snag11_literal_percent_not_scaled_even_if_conv_says_percent():
    # Per-cell guard: even a percent-column convention must not scale a cell
    # whose own fmt carries only a QUOTED %.  55.04 must stay 55.04, not 5504.
    conv = Convention(percent_format=True)
    assert coerce_cell(Cell(55.04, '0.0"%"'), conv) == Coerced("num", num=55.04)


# --------------------------------------------------------------------------
# snag 7 -- leading ~ / approximate marker -> strip + note, NOT a failed cell
# --------------------------------------------------------------------------
def test_snag7_tilde_percent():
    c = coerce_cell(Cell("~30%"), US)
    assert c.kind == "num" and c.num == 30.0
    assert c.note == "approximate value"


def test_snag7_tilde_decimal():
    c = coerce_cell(Cell("~5.5"), US)
    assert c.kind == "num" and c.num == 5.5
    assert c.note == "approximate value"


def test_snag7_almost_equal_sign():
    c = coerce_cell(Cell("≈5.5"), US)  # ≈5.5
    assert c.kind == "num" and c.num == 5.5


# --------------------------------------------------------------------------
# snag 14 -- digit-less non-detect tokens -> censored, BEFORE the NA test
# --------------------------------------------------------------------------
@pytest.mark.parametrize("tok", ["<LOD", "<LOQ", "<DL", "BDL", "ND", "N.D.", "n.d.", "<0.5"])
def test_snag14_nondetect_censored(tok):
    c = coerce_cell(Cell(tok), US)
    assert c.kind == "censored", tok


def test_snag14_nondetect_beats_na_token():
    # "nd" / "n.d." are in NA_TOKENS; the non-detect branch must win (order).
    assert coerce_cell(Cell("nd"), US).kind == "censored"
    assert coerce_cell(Cell("n.d."), US).kind == "censored"


# --------------------------------------------------------------------------
# snag 8 -- strip only the DETECTED unit, never an arbitrary trailing alpha
# --------------------------------------------------------------------------
def test_snag8_undetected_unit_is_failed_cell():
    c = coerce_cell(Cell("20.1 C"), US)  # no unit detected on this column
    assert c.kind == "text" and c.text == "20.1 C"


def test_snag8_detected_unit_strips_cleanly():
    conv = Convention(decimal=".", thousands=",", unit="C")
    assert coerce_cell(Cell("20.1 C"), conv) == Coerced("num", num=20.1)


# --------------------------------------------------------------------------
# snag 1 -- a censored token on a TEXT column is a level, not missing
# --------------------------------------------------------------------------
def test_snag1_censored_on_text_column_becomes_level():
    coerced = coerce_cell(Cell(">100m"), US)
    assert coerced.kind == "censored"                      # coerce_cell alone
    demoted = resolve_censored(coerced, numeric_path=False)  # infer's decision
    assert demoted == Coerced("text", text=">100m")


def test_snag1_censored_na_token_on_text_column_is_empty():
    demoted = resolve_censored(coerce_cell(Cell("ND"), US), numeric_path=False)
    assert demoted.kind == "empty"


def test_snag1_censored_on_numeric_column_stays_censored():
    kept = resolve_censored(coerce_cell(Cell("<0.5"), US), numeric_path=True)
    assert kept.kind == "censored"


# --------------------------------------------------------------------------
# Chunk-5 integration: coerce shares grid's ONE canonical Cell/UNCOMPUTED, so a
# real grid uncomputed-formula cell is recognised by identity (was a "text" bug).
# --------------------------------------------------------------------------
def test_grid_uncomputed_sentinel_is_the_canonical_one():
    from statkit import coerce as coercemod
    from statkit import grid as gridmod

    assert coercemod.UNCOMPUTED is gridmod.UNCOMPUTED
    assert coercemod.Cell is gridmod.Cell


def test_coerce_cell_recognises_a_grid_uncomputed_formula_cell():
    from statkit import grid as gridmod

    cell = gridmod.Cell(gridmod.UNCOMPUTED, "General", formula=True)
    assert coerce_cell(cell, US).kind == "uncomputed"
