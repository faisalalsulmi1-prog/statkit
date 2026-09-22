"""Per-column number-convention detection and per-cell coercion (PLAN §4.3).

Turns raw sheet cells into a typed `Coerced` result carrying the numeric value
(or None), the original text, and a classification the downstream inference layer
counts as n_missing / n_censored / failed_cells. Two public pure functions:

    detect_convention(cells) -> (Convention, [Action])   # per column
    coerce_cell(cell, conv)  -> Coerced                   # per cell

plus `resolve_censored`, the tiny text/numeric-column decision infer.py applies
(snag 1). Honours the fixture-corpus snags: 1 (censored on a text column is a
level), 7 (~ approximate marker), 8 (strip only the detected unit), 11 (a quoted
`%` in the format does NOT scale x100), 14 (digit-less non-detect tokens).

Cell and the UNCOMPUTED sentinel are the CANONICAL ones from grid.py (Chunk-5
integration): coerce imports them so that a real grid uncomputed-formula cell
(`cell.value is grid.UNCOMPUTED`) is recognised here by identity, not merely by
duck typing. Before unification each module minted its own `object()` sentinel,
so a grid uncomputed cell fell through to a bogus "text" classification.
"""
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

# The one canonical Cell view + uncomputed-formula sentinel live in grid.py
# (the raw-reader layer). coerce/clean/infer all share these exact objects.
from .grid import UNCOMPUTED, Cell


@dataclass(frozen=True)
class Action:
    """A line in the "what I did to your sheet" log.

    `auto=True`  -> done silently but logged (INFO).
    `auto=False` -> surfaced to the student with a control (FLAG).
    """
    what: str
    auto: bool = True
    detail: str = ""


# --- coercion vocabulary (PLAN §4.3) -------------------------------------
# NB: "none" is deliberately NOT an NA token -- in a stats tool it is too likely
# real data (severity "none", "no symptoms", "no complication") to drop silently.
NA_TOKENS = frozenset({
    "", "na", "n/a", "n.a.", "nan", "null", "-", "--", "—", "–",
    ".", "?", "missing", "not available", "not applicable", "nd", "n.d.",
    "unknown", "#n/a", "#value!", "#div/0!", "#ref!", "#name?",
})
CENSORED_RE = re.compile(r"^[<>]=?\s*[\d.,]+\s*[A-Za-z%]*$")
# snag 14: digit-less non-detect / below-limit tokens.
NONDETECT_RE = re.compile(r"^<\s*(LOD|LOQ|DL|MDL|RL)$|^(BDL|ND|N\.D\.)$", re.IGNORECASE)
# Date TEXT: '-' or '/' separators only. A '.' separator (dd.mm.yyyy) is
# excluded -- it is ambiguous with a decimal number and, per the committed
# oracles (Wave-2 EU dd.mm.yyyy trap), such text is a categorical string, not a
# date (real datetime CELLS are still dates via coerce_cell's datetime branch).
DATE_RE = re.compile(r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}(\s+\d{1,2}:\d{2}(:\d{2})?)?$")
NUMISH_RE = re.compile(r"^[\s$€£¥(+\-−–]*[\d.,' ]+[\s%A-Za-z°/µ)]*$")

_CURRENCY_RE = re.compile(r"[$€£¥]")
_LEAD_SIGN_CUR_RE = re.compile(r"^[\s$€£¥+\-−–]+")
_TRAIL_ALPHA_RE = re.compile(r"[A-Za-z°µ/%]+$")

# decimal/thousands pattern probes (PLAN §4.3).
_US = re.compile(r"^\d{1,3}(,\d{3})+(\.\d+)?$")
_EU = re.compile(r"^\d{1,3}(\.\d{3})+(,\d+)?$")
_C = re.compile(r"^\d+,\d{1,2}$")
_D = re.compile(r"^\d+\.\d+$")
_F = re.compile(r"^\d+,\d{3,}$")
_E = re.compile(r"^\d{1,3}( \d{3})+([.,]\d+)?$")


@dataclass(frozen=True)
class Convention:
    decimal: str = "."
    thousands: str | None = ","
    unit: str | None = None
    percent_format: bool = False


@dataclass(frozen=True)
class Coerced:
    kind: Literal["empty", "num", "text", "date", "bool", "censored", "uncomputed"]
    num: float | None = None
    text: str | None = None
    note: str | None = None


# --- helpers -------------------------------------------------------------
def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _has_unquoted_percent(fmt: str | None) -> bool:
    """True iff the number format contains a `%` that would make Excel scale x100.

    A `%` inside a quoted literal (`0.0"%"`) or backslash-escaped (`0.0\\%`) is a
    display character only, not a scaling directive (snag 11).
    """
    if not fmt:
        return False
    stripped = re.sub(r"\\.", "", fmt)          # drop \x escapes
    stripped = re.sub(r'"[^"]*"', "", stripped)  # drop "quoted" literals
    return "%" in stripped


def _suffix(raw: str) -> str:
    """Trailing alpha/unit run of a numish text cell (after sign/currency/parens)."""
    s = raw.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    s = _LEAD_SIGN_CUR_RE.sub("", s)
    m = _TRAIL_ALPHA_RE.search(s)
    return m.group(0) if m else ""


def _core(raw: str, unit: str | None) -> str:
    """Numeric core of a cell for pattern classification: strip sign, currency,
    the detected unit and a `%` suffix, but KEEP internal spaces (pattern E)."""
    s = raw.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    s = _LEAD_SIGN_CUR_RE.sub("", s)
    if unit:
        s = re.sub(r"\s*" + re.escape(unit) + r"\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*%\s*$", "", s)
    return s.strip()


def _detect_unit(texts: list[str], n_numeric: int, actions: list[Action]) -> str | None:
    if not texts:
        return None
    counts: dict[str, int] = {}
    first_seen: dict[str, str] = {}
    for raw in texts:
        suf = _suffix(raw)
        if not suf:
            continue
        key = suf.casefold()
        counts[key] = counts.get(key, 0) + 1
        first_seen.setdefault(key, suf)
    if not counts:
        return None
    key, n = max(counts.items(), key=lambda kv: kv[1])
    # Fraction is over ALL value cells (numeric cells carry no suffix), so a
    # unit on a minority of cells is NOT the column's unit (snag 8): e.g. 4 bare
    # floats + 3 "20.1 C" -> 3/7 < .85 -> undetected, and "20.1 C" fails to a
    # failed cell instead of being silently stripped.
    if n / (len(texts) + n_numeric) < 0.85:
        return None
    unit = "%" if key == "%" else first_seen[key]
    actions.append(Action(f"unit '{unit}' removed", auto=True,
                          detail=f"detected on {n}/{len(texts)} cells"))
    return unit


def _detect_decimal(texts: list[str], unit: str | None,
                    actions: list[Action]) -> tuple[str, str | None]:
    us = eu = c = d = f = e = 0
    c_cells: list[str] = []
    for raw in texts:
        core = _core(raw, unit)
        if _US.match(core):
            us += 1
        if _EU.match(core):
            eu += 1
        if _C.match(core):
            c += 1
            c_cells.append(raw.strip())
        if _D.match(core):
            d += 1
        if _F.match(core):
            f += 1
        if _E.match(core):
            e += 1

    if e > 0:
        decimal = "," if (c > 0 or eu > 0) and d == 0 else "."
        return decimal, " "
    if d > 0 and c == 0:
        return ".", ","
    if c > 0 and d == 0:
        return ",", "."
    if d > 0 and c > 0:
        actions.append(Action("mixed decimal conventions", auto=False,
                              detail="unreadable cells: " + ", ".join(c_cells)))
        return ".", ","
    if eu > 0 and us == 0 and f == 0:
        return ",", "."
    if us > 0 or f > 0:
        actions.append(Action("ambiguous thousands separator", auto=True,
                              detail="read '1,234' as 1234 - toggle if it means 1.234"))
        return ".", ","
    return ".", ","


# --- public API ----------------------------------------------------------
def detect_convention(cells: list[Cell]) -> tuple[Convention, list[Action]]:
    """Infer a column's decimal/thousands/unit/percent convention (PLAN §4.3)."""
    actions: list[Action] = []

    numeric = [c for c in cells
               if _is_number(c.value)
               and not (isinstance(c.value, float) and math.isnan(c.value))]
    pct = [c for c in numeric if _has_unquoted_percent(c.fmt)]
    percent_format = bool(numeric) and len(pct) / len(numeric) >= 0.85

    texts = [c.value.strip() for c in cells
             if isinstance(c.value, str) and c.value.strip()
             and NUMISH_RE.match(c.value.strip())]

    unit = _detect_unit(texts, len(numeric), actions)
    decimal, thousands = _detect_decimal(texts, unit, actions)
    return Convention(decimal, thousands, unit, percent_format), actions


def _coerce_str(raw: str, conv: Convention) -> Coerced:
    s = raw.strip()
    if s == "":
        return Coerced("empty")

    note = None
    if s[:1] in ("~", "≈"):  # snag 7: approximate marker
        s = s[1:].strip()
        note = "approximate value"
        if s == "":
            return Coerced("empty", note=note)

    # snag 14 + censored, BEFORE the NA-token test.
    if NONDETECT_RE.match(s) or CENSORED_RE.match(s):
        return Coerced("censored", text=s, note=note)

    if s.casefold() in NA_TOKENS:
        return Coerced("empty", note=note)

    if DATE_RE.match(s):
        return Coerced("date", text=s, note=note)

    num = _try_number(s, conv)
    if num is not None:
        return Coerced("num", num=num, note=note)
    return Coerced("text", text=s, note=note)


def _valid_grouping(t: str, decimal: str, thousands: str | None) -> bool:
    """Grouping sanity on a numeric candidate BEFORE separators are stripped.

    Rejects tokens where blindly removing the thousands separator would fabricate
    a number: there may be at most one decimal separator, and if a thousands
    separator is present every group after the first must be exactly 3 digits
    (the first 1-3). So "35.80.194.178" / "192.168.1.1" (octets that aren't
    3-digit groups) fail and are kept as text, not parsed into a huge integer.
    """
    core = t.strip().lstrip("+-")
    if decimal and core.count(decimal) > 1:
        return False
    int_part = core.split(decimal, 1)[0] if decimal else core
    if thousands and thousands in int_part:
        groups = int_part.split(thousands)
        if any(not g.isdigit() for g in groups):
            return False
        if not 1 <= len(groups[0]) <= 3:
            return False
        if any(len(g) != 3 for g in groups[1:]):
            return False
    return True


def _try_number(s: str, conv: Convention) -> float | None:
    neg = False
    t = s
    if t.startswith("(") and t.endswith(")"):  # accounting parens
        neg = True
        t = t[1:-1].strip()
    t = re.sub(r"[−–]", "-", t)  # unicode minus, en-dash -> ascii
    t = _CURRENCY_RE.sub("", t)
    if conv.unit:  # snag 8: strip ONLY the detected unit, never arbitrary alpha
        t = re.sub(r"\s*" + re.escape(conv.unit) + r"\s*$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*%\s*$", "", t)  # percent points are canonical
    if not _valid_grouping(t, conv.decimal, conv.thousands):
        return None
    if conv.thousands:
        t = re.sub(re.escape(conv.thousands), "", t)
    if conv.decimal == ",":
        t = re.sub(",", ".", t)
    t = t.strip()
    try:
        val = float(t)
    except ValueError:
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return -val if neg else val


def coerce_cell(cell: Cell, conv: Convention) -> Coerced:
    """Classify and coerce one raw cell under the column's `conv` (PLAN §4.3)."""
    v = cell.value
    if v is None:
        return Coerced("empty")
    if v is UNCOMPUTED:
        return Coerced("uncomputed")
    if isinstance(v, bool):  # BEFORE int -- bool subclasses int
        return Coerced("bool", text="TRUE" if v else "FALSE")
    if isinstance(v, (int, float)):
        if isinstance(v, float) and math.isnan(v):
            return Coerced("empty")
        num = float(v)
        # snag 11: scale only when the column is a percent format AND this cell's
        # own fmt carries an UNQUOTED %.
        if conv.percent_format and _has_unquoted_percent(cell.fmt):
            num *= 100
        return Coerced("num", num=num)
    if isinstance(v, (datetime, date)):  # datetime is a subclass of date
        return Coerced("date", text=v.isoformat())
    if isinstance(v, str):
        return _coerce_str(v, conv)
    return Coerced("text", text=str(v))


def resolve_censored(coerced: Coerced, numeric_path: bool) -> Coerced:
    """Apply the snag-1 column decision to a censored cell (called by infer.py).

    On a numeric-path column (f >= .50) a censored token is honoured as
    censored/non-detect (treated as missing). On a text column it is demoted:
    the raw token is a legitimate categorical level (e.g. ">100m"), unless its
    casefold is an NA token, in which case it is empty.
    """
    if coerced.kind != "censored" or numeric_path:
        return coerced
    tok = (coerced.text or "").strip()
    if tok.casefold() in NA_TOKENS:
        return Coerced("empty", note=coerced.note)
    return Coerced("text", text=tok, note=coerced.note)
