"""RED list for statkit.model — the domain types (PLAN §2).

Tests behaviour, not mere construction: the Kind enum's exact vocabulary, the
Bound.blocked property, per-instance mutable defaults on Result (a classic
dataclass footgun), Role frozenness and its documented defaults.
"""
import dataclasses

import pandas as pd
import pytest

from statkit import model
from statkit.model import Bound, Check, Contract, Finding, Kind, Result, Role


# --- Kind: the exact closed vocabulary (PLAN §2 / EXPECT_SCHEMA §0) -------
def test_kind_has_exactly_the_eight_members():
    assert {k.value for k in Kind} == {
        "numeric", "ordinal", "categorical", "binary",
        "id", "date", "empty", "ignore",
    }
    assert len(list(Kind)) == 8


def test_kind_is_a_str_enum():
    # Kind(str, Enum): a member IS its lowercase string, so it round-trips
    # through JSON/DataFrame column labels without a conversion layer.
    assert Kind.NUMERIC == "numeric"
    assert Kind("categorical") is Kind.CATEGORICAL


# --- Bound.blocked: true iff any finding is a block (PLAN §2) -------------
def _bare_bound(findings):
    return Bound(
        test=None, layout="long", columns={}, kinds={}, params={},
        data=pd.DataFrame(), n_total=0, n_used=0, dropped={},
        dropped_rows=(), findings=findings,
    )


def test_bound_blocked_true_when_a_block_finding_present():
    b = _bare_bound((Finding(severity="block", text="nope", code="S1"),))
    assert b.blocked is True


def test_bound_blocked_false_without_a_block_finding():
    b = _bare_bound((Finding(severity="flag", text="hmm", code="S16"),))
    assert b.blocked is False


def test_bound_blocked_false_with_no_findings():
    assert _bare_bound(()).blocked is False


# --- Result: mutable defaults are per-instance, not shared ----------------
def test_result_mutable_defaults_are_independent():
    a = Result(test_id="t_ind", test_name="A", status="ok")
    b = Result(test_id="t_ind", test_name="B", status="ok")
    a.n["used"] = 30
    a.labels["outcome"] = "Score"
    assert b.n == {} and b.labels == {}
    assert a.n is not b.n


# --- Role: frozen, with the PLAN §2 defaults -----------------------------
def test_role_defaults():
    r = Role(name="outcome", accepts=(Kind.NUMERIC,), label="Outcome")
    assert r.min == 1 and r.max == 1 and r.levels == (1, None) and r.help == ""


def test_role_is_frozen():
    r = Role(name="group", accepts=(Kind.CATEGORICAL,), label="Group")
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.min = 5  # type: ignore[misc]


# --- the domain surface the rest of the package imports -------------------
def test_all_documented_types_exist():
    for name in (
        "Kind", "ColumnProfile", "Role", "Param", "Contract",
        "Finding", "Check", "Bound", "Result", "TestSpec",
    ):
        assert hasattr(model, name), name


def test_check_carries_a_tri_state_passed():
    # passed is bool | None (None = not applicable) — sentences read it.
    c = Check(name="Shapiro", statistic=None, p=None, passed=None, note="n=2")
    assert c.passed is None


def test_contract_defaults():
    roles = {"long": (Role(name="outcome", accepts=(Kind.NUMERIC,), label="x"),)}
    c = Contract(roles_by_layout=roles, canonical="long")
    assert c.params == () and c.pairing == "independent"
    assert c.same_level_set is False and c.min_n == 3
