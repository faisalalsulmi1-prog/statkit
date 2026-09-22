"""RED-first tests for statkit.levels (batch 1B; B3 / S12 plumbing).

``levels`` is the ONE place a stored outcome value becomes the string a student
sees, a chosen level (a display string from a dropdown) is resolved back to the
actual stored value, and the "success" level is picked for the proportion /
logistic runners. The B3 bug was a numeric 0/1 outcome (stored as floats) versus
a "1"/"0" dropdown choice (strings) never matching — so every assertion below
turns on the numeric-vs-string boundary, not on string levels that would pass
vacuously either way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from statkit import levels
from statkit.model import Finding


# --------------------------------------------------------------------------
# display — an int-valued float prints as an int (the NICE "1.0" -> "1" fix)
# --------------------------------------------------------------------------
def test_display_int_valued_float_drops_the_point():
    assert levels.display(1.0) == "1"          # not "1.0"
    assert levels.display(0.0) == "0"
    assert levels.display(np.float64(1.0)) == "1"
    assert levels.display(2) == "2"


def test_display_non_int_float_is_lossless_not_rounded():
    # S-D: a fractional level renders EXACTLY, not rounded to 2 dp -- two distinct
    # doses must never collapse to the same label (that merged groups downstream).
    assert levels.display(2.5) == "2.5"           # not "2.50"
    assert levels.display(0.125) == "0.125"       # not "0.12"
    assert levels.display(2.0) == "2"             # int-valued float still an int
    assert levels.display(0.001) != levels.display(0.002)   # distinct, not both "0"
    assert levels.display(0.001) == "0.001"
    assert levels.display(0.002) == "0.002"


def test_display_fractional_float_has_no_binary_noise():
    # a clean lossless rendering, never "0.30000000000000004".
    assert levels.display(0.3) == "0.3"
    assert levels.display(1.1) == "1.1"


def test_display_index_relabels_losslessly():
    # DRY helper W3-W5 will use to relabel a pandas Index / list of labels.
    out = levels.display_index([1.0, 2.0])
    assert list(out) == ["1", "2"]
    idx = pd.Index([0.001, 0.002, 0.125])
    relabelled = levels.display_index(idx)
    assert isinstance(relabelled, pd.Index)
    assert list(relabelled) == ["0.001", "0.002", "0.125"]


def test_display_index_passes_text_through():
    assert list(levels.display_index(["Yes", "No"])) == ["Yes", "No"]
    assert list(levels.display_index([1.0, "mixed"])) == ["1", "mixed"]


def test_display_string_is_verbatim():
    assert levels.display("Yes") == "Yes"
    assert levels.display("1") == "1"


def test_display_missing_is_empty_string():
    assert levels.display(None) == ""
    assert levels.display(float("nan")) == ""
    assert levels.display(pd.NA) == ""


# --------------------------------------------------------------------------
# display_levels — profile levels when set, else first-seen unique displays
# --------------------------------------------------------------------------
def test_display_levels_from_series_first_seen():
    s = pd.Series([1.0, 1.0, 0.0, 1.0])
    assert levels.display_levels(s) == ["1", "0"]        # first-seen order, NICE


def test_display_levels_prefers_profile_when_present():
    class _P:
        levels = ("Low", "High")
    s = pd.Series([1.0, 0.0])
    assert levels.display_levels(s, _P()) == ["Low", "High"]


# --------------------------------------------------------------------------
# resolve — a display string back to the actual stored value
# --------------------------------------------------------------------------
def test_resolve_display_string_to_numeric_value():
    v = levels.resolve("1", pd.Series([0.0, 1.0]))
    assert v == 1.0 and isinstance(v, float)


def test_resolve_zero_display_string():
    assert levels.resolve("0", pd.Series([0.0, 1.0])) == 0.0


def test_resolve_numeric_chosen_matches_by_equality():
    assert levels.resolve(0.0, pd.Series([0.0, 1.0])) == 0.0


def test_resolve_string_level_verbatim():
    assert levels.resolve("Yes", pd.Series(["No", "Yes"])) == "Yes"


def test_resolve_absent_level_is_none():
    assert levels.resolve("maybe", pd.Series([0.0, 1.0])) is None


# --------------------------------------------------------------------------
# pick_success — chosen (resolved) / positive token / last sorted / block
# --------------------------------------------------------------------------
def test_pick_success_resolves_chosen_display_string():
    v, block = levels.pick_success(pd.Series([0.0, 1.0]), "0")
    assert v == 0.0 and block is None


def test_pick_success_unresolvable_chosen_blocks_with_level_code():
    v, block = levels.pick_success(pd.Series([0.0, 1.0]), "maybe")
    assert v is None
    assert isinstance(block, Finding) and block.severity == "block"
    assert block.code == "LEVEL"
    assert "maybe" in block.text


def test_pick_success_default_picks_positive_token():
    v, block = levels.pick_success(pd.Series(["no", "yes"]))
    assert v == "yes" and block is None


def test_pick_success_default_numeric_recognises_one_as_positive():
    # 1.0 displays as "1", which is a positive token -> success is 1.0.
    v, block = levels.pick_success(pd.Series([0.0, 1.0]))
    assert v == 1.0 and block is None


def test_pick_success_default_no_positive_token_uses_last_sorted():
    v, block = levels.pick_success(pd.Series(["apple", "banana"]))
    assert v == "banana" and block is None


# --------------------------------------------------------------------------
# POSITIVE is the shared token set and no longer carries the redundant "1.0"
# --------------------------------------------------------------------------
def test_positive_token_set_is_display_normalised():
    assert "1" in levels.POSITIVE
    assert "1.0" not in levels.POSITIVE          # display() already normalises it
    assert "yes" in levels.POSITIVE
