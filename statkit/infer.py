"""Table -> Dataset: kinds, DataFrame, per-column profiles (PLAN §4.4).

The fourth ingest stage. It takes a cleaned ``Table`` (from ``clean.clean``) and,
per column, detects the number convention (``coerce.detect_convention``), coerces
every cell (``coerce.coerce_cell``), and infers the column's possible ``Kind``s in
preference order -- the first is the default the UI and reports use, the whole
tuple is what role-binding (Chunk 6) intersects with each role's ``accepts``.

The §4.4 kinds table, verbatim (see EXPECT_SCHEMA §3 for the oracle wording):

  * m == 0                         -> EMPTY
  * dates   >= .85 of m            -> DATE
  * bools   >= .85 of m            -> BINARY
  * numeric f >= .85               -> numeric path
  * .50 <= f < .85                 -> (CATEGORICAL, NUMERIC)   [mixed]
  * f < .50                        -> text path

  numeric path (on the parsed numbers, d distinct):
    d == 1                                   -> NUMERIC (constant)
    d == 2                                   -> (NUMERIC, BINARY, CATEGORICAL)
    all integers and d <= 10                 -> (NUMERIC, ORDINAL, CATEGORICAL)
    all-distinct and m >= 20 and (ID_RE | near-contiguous INTEGERS) -> ID
    all-distinct and m >= 20                 -> (NUMERIC, ID)
    all integers, values repeat, ID header   -> ID (long-layout subject id)
    otherwise                                -> NUMERIC

  text path (on the distinct string labels d):
    d == 1                                   -> CATEGORICAL (constant)
    d == 2 and levels recur (d < m)          -> (BINARY, CATEGORICAL)
    d == 2 all-distinct (d == m)             -> CATEGORICAL   (free-text, not binary)
    levels subset of an ORDERED_VOCAB        -> (ORDINAL, CATEGORICAL)
    d > 20 and d/m > .9                      -> ID (free text)
    d > 20                                   -> CATEGORICAL (+ ">20 levels" note)
    otherwise                                -> CATEGORICAL

Counting model (matches the committed oracles):
  * ``n_missing``  = blank + NA-token cells (coerce kind "empty").
  * ``n_censored`` = censored / non-detect cells, treated as missing, but only on
    the numeric family (f >= .50); on a text column a censored token is a level
    (snag 1) and does not count.
  * ``failed_cells`` = non-numeric text (and approximate ``~``/``≈`` cells -- the
    ~30% reconciliation, snag 7) in a numeric-family column, treated as missing
    and listed; empty for a pure text column.
  * censored cells are EXCLUDED from ``m`` when deciding the numeric fraction, so
    a chemistry column that is mostly real numbers plus a few ``<LOD`` stays
    numeric instead of being dragged onto the mixed path.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from . import levels
from .coerce import coerce_cell, detect_convention, resolve_censored
from .grid import Cell
from .model import ColumnProfile, Kind

APPROX_NOTE = "approximate value"

# Numeric-column header pattern that promotes an all-distinct numeric column to
# an ID (§4.4). Applied to the whitespace-stripped, casefolded header.
ID_RE = re.compile(
    r"^(id|code|no\.?|number|serial|subject|participant|patient|student|record|case|sample)"
    r"([\s_#.\-]*(id|no\.?|number|code)|\s*#)?$",   # ...or a trailing '#' (Case#)
    re.IGNORECASE,
)

# Ordered categorical vocabularies (§4.4): a text column whose levels are a
# subset of one of these is ORDINAL, and carries that vocab (filtered) as its
# level_order. casefold matching. Order within each tuple is the ordinal order.
ORDERED_VOCABS: tuple[tuple[str, ...], ...] = (
    ("strongly disagree", "disagree", "neutral", "agree", "strongly agree"),
    ("never", "rarely", "sometimes", "often", "always"),
    ("none", "mild", "moderate", "severe"),
    ("low", "medium", "high"),
    ("poor", "fair", "good", "very good", "excellent"),
    ("very dissatisfied", "dissatisfied", "neutral", "satisfied", "very satisfied"),
)
_VOCAB_SETS = tuple((frozenset(v), v) for v in ORDERED_VOCABS)

_NUMERIC_FAMILY = frozenset({Kind.NUMERIC, Kind.ORDINAL, Kind.BINARY, Kind.ID})


@dataclass
class Dataset:
    """Typed columns + their profiles, ready for binding (PLAN §4.4)."""
    df: pd.DataFrame
    profiles: tuple[ColumnProfile, ...]
    excel_rows: tuple[int, ...]
    source: str = ""
    sheet: str = ""
    log: tuple[str, ...] = ()
    # clean-stage provenance, carried through for the report's data section (S11).
    dropped_rows: tuple = ()            # (excel_row, reason), verbatim from clean
    header_row: int = 0                 # 1-based header row clean chose
    n_rows_read: int = 0               # surviving data rows profiled


# --------------------------------------------------------------------------
# per-cell bucketing
# --------------------------------------------------------------------------
def _raw(cell) -> str:
    v = cell.value
    return "" if v is None else str(v).strip()


def _is_num(v) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and not (isinstance(v, float) and math.isnan(v)))


def _format_unit(cells):
    """A unit read off the NUMBER FORMAT of numeric cells (not a cell suffix):
    a '%' anywhere in the format (quoted `0.0"%"` or unquoted `0%`) -> "%";
    a currency glyph in the format (`"$"#,##0`) -> that glyph. >= 85% of the
    numeric cells must share it. Returns None when there is no numeric majority."""
    numeric = [c for c in cells if _is_num(c.value)]
    if not numeric:
        return None
    if sum(1 for c in numeric if c.fmt and "%" in c.fmt) / len(numeric) >= 0.85:
        return "%"
    glyphs = Counter(
        g for c in numeric for g in "$€£¥" if c.fmt and g in c.fmt)
    if glyphs:
        g, n = glyphs.most_common(1)[0]
        if n / len(numeric) >= 0.85:
            return g
    return None


@dataclass
class _Buckets:
    empty: int = 0
    uncomputed: int = 0
    nums: list = field(default_factory=list)      # (excel_row, float)
    approx: list = field(default_factory=list)    # (excel_row, raw_str)
    texts: list = field(default_factory=list)     # (excel_row, label_str)
    dates: list = field(default_factory=list)     # (excel_row, iso_or_raw)
    bools: list = field(default_factory=list)     # (excel_row, "TRUE"/"FALSE")
    censored: list = field(default_factory=list)  # (excel_row, Coerced, raw_str)


def _bucket(cells, coerced, excel_rows) -> _Buckets:
    b = _Buckets()
    for er, cell, co in zip(excel_rows, cells, coerced):
        if co.kind == "empty":
            b.empty += 1
        elif co.kind == "uncomputed":
            b.uncomputed += 1
        elif co.kind == "censored":
            b.censored.append((er, co, _raw(cell)))
        elif co.kind == "num":
            if co.note == APPROX_NOTE:
                b.approx.append((er, _raw(cell)))
            else:
                b.nums.append((er, co.num))
        elif co.kind == "date":
            b.dates.append((er, co.text))
        elif co.kind == "bool":
            b.bools.append((er, co.text))
        else:  # "text"
            b.texts.append((er, co.text if co.text is not None else _raw(cell)))
    return b


# --------------------------------------------------------------------------
# levels
# --------------------------------------------------------------------------
def _levels(pool) -> tuple[str, ...]:
    """Distinct display labels in first-seen order. ``pool`` is a list of
    (key, label); keys de-duplicate (numbers and strings never collide)."""
    seen: set = set()
    out: list[str] = []
    for key, label in pool:
        if key not in seen:
            seen.add(key)
            out.append(label)
    return tuple(out)


def _num_pool(nums):
    # label via levels.display (lossless for fractional floats) so a profile's
    # numeric level labels never round two distinct doses onto one string (S-D);
    # the de-dup KEY stays the exact value, so n_levels is unchanged.
    return [(("n", v), levels.display(v)) for _, v in nums]


def _str_pool(items):
    return [(("t", s), s) for _, s in items]


# --------------------------------------------------------------------------
# kind rules
# --------------------------------------------------------------------------
def _numeric_kinds(name, values, m) -> tuple[Kind, ...]:
    d = len(set(values))
    all_int = all(float(v).is_integer() for v in values)
    if d == 1:
        return (Kind.NUMERIC,)
    if d == 2:
        return (Kind.NUMERIC, Kind.BINARY, Kind.CATEGORICAL)
    if all_int and d <= 10:
        return (Kind.NUMERIC, Kind.ORDINAL, Kind.CATEGORICAL)
    if d == len(values) and m >= 20:
        span = max(values) - min(values) + 1
        if ID_RE.match(name.strip()):
            return (Kind.ID,)
        # near-contiguous WITH gaps -> an identifier range (e.g. ages 27..55).
        # A PERFECT run (span == count, a 1..N row index) or a wide spread stays
        # ambiguous -> (NUMERIC, ID). (§4.4 near-contiguity, refined to the
        # committed oracles: Plot 1..20 / Txn 1001..1050 are (numeric, id).)
        # S-K(a): only INTEGER runs demote to a bare ID; a fractional measurement
        # (BMI, absorbance) that happens to be near-contiguous stays (NUMERIC, ID).
        if all_int and span <= 1.5 * m and span != len(values):
            return (Kind.ID,)
        return (Kind.NUMERIC, Kind.ID)
    # S-M: an integer key under an ID header whose values REPEAT (ID 1..N once per
    # condition) is a LONG-layout subject id, not a measurement -- the all-distinct
    # rules above never see it. An all-distinct ID-named column under 20 rows stays
    # (NUMERIC,) as the committed oracle h3_2 col 1 expects.
    if all_int and d < len(values) and ID_RE.match(name.strip()):
        return (Kind.ID,)
    return (Kind.NUMERIC,)


def _ordered_vocab(label_set_cf):
    for vocab_set, vocab in _VOCAB_SETS:
        if label_set_cf <= vocab_set:
            return vocab
    return None


def _text_kinds(labels, m) -> tuple[tuple[Kind, ...], tuple[str, ...] | None, list[str]]:
    """-> (kinds, level_order, notes)."""
    d = len(labels)
    label_set_cf = {s.casefold() for s in labels}
    notes: list[str] = []
    if d == 1:
        return (Kind.CATEGORICAL,), None, notes
    if d == 2:                                       # §4.4: two levels -> binary
        return (Kind.BINARY, Kind.CATEGORICAL), None, notes
    vocab = _ordered_vocab(label_set_cf)
    if vocab is not None:
        order = tuple(v for v in vocab if v in label_set_cf)
        return (Kind.ORDINAL, Kind.CATEGORICAL), order, notes
    if d > 20 and d / m > 0.9:                        # §4.4: free-text ID
        return (Kind.ID,), None, notes
    if d > 20:
        notes.append(f"FLAG >20 levels ({d})")
        return (Kind.CATEGORICAL,), None, notes
    return (Kind.CATEGORICAL,), None, notes


# --------------------------------------------------------------------------
# override
# --------------------------------------------------------------------------
def _apply_override(kind: Kind, b: _Buckets):
    """A per-column override forces exactly one kind; recompute the applicable
    levels / failed set for it (§4.4 override UI, minimal).

    S-J: only NUMERIC and ID are numeric-family FOR LEVELS -- everything text-shaped
    is dropped as failed and there are no labels. ORDINAL and BINARY are LABEL kinds
    (Likert vocab, Yes/No): keep the labels exactly as the CATEGORICAL branch does,
    and for ORDINAL carry the matching ordered vocab (filtered to labels present)."""
    if kind in (Kind.NUMERIC, Kind.ID):
        failed = b.approx + b.texts + b.dates + b.bools
        return (kind,), (), None, failed
    pool = (_num_pool(b.nums) + _str_pool(b.texts) + _str_pool(b.approx)
            + _str_pool(b.dates) + _str_pool(b.bools)
            + _str_pool([(er, r) for er, _, r in b.censored]))
    labels = _levels(pool)
    level_order = None
    if kind is Kind.ORDINAL:
        label_set_cf = {s.casefold() for s in labels}
        vocab = _ordered_vocab(label_set_cf)
        if vocab is not None:
            level_order = tuple(v for v in vocab if v in label_set_cf)
    return (kind,), labels, level_order, []


# --------------------------------------------------------------------------
# convention cleanup -> human notes / merges (S6, S7)
# --------------------------------------------------------------------------
def _conv_note(action) -> str:
    """One "what I did to read this column" line from a detect_convention Action
    (S6): e.g. "unit 'kg' removed", "mixed decimal conventions — unreadable
    cells: …", "ambiguous thousands separator — read '1,234' as 1234 …"."""
    return f"{action.what} — {action.detail}" if action.detail else action.what


def _apply_merges(cells, name, merges):
    """Map a column's string cell values through ``merges[name]`` (strip-matched,
    so "male "/"MALE" -> "Male"), returning ``(new_cells, log_lines)`` (S7).

    Non-string cells pass through untouched; only applied when the caller supplies
    a map for this column, so the default ``infer(table)`` path is unchanged.
    """
    mapping = merges.get(name)
    if not mapping:
        return cells, []
    norm = {str(k).strip(): (v, k) for k, v in mapping.items()}
    applied: dict = {}          # canonical -> [source spellings applied, first-seen]
    out = []
    for c in cells:
        v = c.value
        if isinstance(v, str) and v.strip() in norm:
            canonical, orig = norm[v.strip()]
            out.append(Cell(canonical, c.fmt))
            seen = applied.setdefault(canonical, [])
            if orig not in seen:
                seen.append(orig)
        else:
            out.append(c)
    log = [f"merged levels in '{name}': "
           + ", ".join(f"'{o}'" for o in origs) + f" → '{canonical}'"
           for canonical, origs in applied.items()]
    return out, log


# --------------------------------------------------------------------------
# the per-column profile (the tested seam)
# --------------------------------------------------------------------------
def profile_column(name, cells, excel_rows, unit=None, override=None,
                   conv=None) -> ColumnProfile:
    if conv is None:
        conv, actions = detect_convention(cells)
    else:                       # caller-supplied convention override: no re-read,
        actions = []            # the student already resolved the ambiguity (S6).
    # unit precedence: a units row (clean) > a cell suffix (detect_convention) >
    # the number format (%, currency).
    rep_unit = unit or conv.unit or _format_unit(cells)
    coerced = [coerce_cell(c, conv) for c in cells]
    b = _bucket(cells, coerced, excel_rows)
    n_total = len(cells)

    # m for the path decision: non-missing, non-uncomputed, EXCLUDING censored.
    m = len(b.nums) + len(b.approx) + len(b.dates) + len(b.bools) + len(b.texts)

    level_order = None
    notes: list[str] = [_conv_note(a) for a in actions]   # S6: cleanup surfaced
    failed: list = []
    n_censored = 0

    if override is not None:
        kinds, levels, level_order, failed = _apply_override(override, b)
        n_missing = b.empty
        if override in _NUMERIC_FAMILY:
            n_censored = len(b.censored)
        else:
            n_missing += sum(
                1 for _, co, _r in b.censored
                if resolve_censored(co, False).kind == "empty")
    elif m == 0:
        kinds, levels = (Kind.EMPTY,), ()
        n_missing = b.empty + len(b.censored)
        if b.uncomputed:
            notes.append(f"{b.uncomputed} uncomputed formula cell(s)")
    elif len(b.dates) / m >= 0.85:
        kinds, levels = (Kind.DATE,), ()
        n_missing = b.empty
        n_censored = len(b.censored)
    elif len(b.bools) / m >= 0.85:
        kinds = (Kind.BINARY,)
        levels = _levels(_str_pool(b.bools))
        n_missing = b.empty
    else:
        f = len(b.nums) / m
        if f >= 0.85:                                    # numeric path
            values = [v for _, v in b.nums]
            kinds = _numeric_kinds(name, values, m)
            levels = ()
            failed = b.approx + b.texts + b.dates + b.bools
            n_censored = len(b.censored)
            n_missing = b.empty
        elif f >= 0.50:                                  # mixed
            kinds = (Kind.CATEGORICAL, Kind.NUMERIC)
            failed = b.approx + b.texts
            pool = _num_pool(b.nums) + _str_pool(b.texts) + _str_pool(b.approx)
            levels = _levels(pool)
            n_censored = len(b.censored)
            n_missing = b.empty
        else:                                            # text path
            n_missing = b.empty
            text_items = list(b.texts)
            for er, co, raw in b.censored:               # snag 1: censored -> level
                r = resolve_censored(co, False)
                if r.kind == "empty":
                    n_missing += 1
                else:
                    text_items.append((er, r.text))
            pool = (_str_pool(text_items) + _num_pool(b.nums)
                    + _str_pool(b.approx) + _str_pool(b.dates) + _str_pool(b.bools))
            labels = _levels(pool)
            kinds, level_order, tnotes = _text_kinds(labels, len(pool))
            notes.extend(tnotes)
            levels = labels

    if failed and kinds[0] in (Kind.NUMERIC, Kind.ORDINAL, Kind.BINARY, Kind.ID,
                               Kind.CATEGORICAL):
        raws = [t for _, t in failed]
        notes.append(f"{len(failed)} text cell(s) treated as missing")
    failed_cells = tuple((er, t) for er, t in failed)[:20]
    if n_censored:
        notes.append(f"{n_censored} censored / non-detect cell(s) treated as missing")

    # an uncomputed formula cell is a missing value (needs "re-save in Excel").
    n_missing += b.uncomputed

    # levels are meaningful only for a categorical-family default kind.
    if kinds[0] in (Kind.NUMERIC, Kind.ID, Kind.DATE, Kind.EMPTY):
        levels = ()

    return ColumnProfile(
        name=name,
        kinds=kinds,
        n_total=n_total,
        n_missing=n_missing,
        n_levels=len(levels),
        levels=tuple(levels),
        level_order=level_order,
        unit=rep_unit,
        n_censored=n_censored,
        notes=tuple(notes),
        failed_cells=failed_cells,
    )


# --------------------------------------------------------------------------
# DataFrame build (dtype driven by the default Kind, never sniffed -- D9/T20)
# --------------------------------------------------------------------------
def _df_column(kind: Kind, coerced):
    if kind in (Kind.NUMERIC, Kind.ORDINAL, Kind.BINARY, Kind.ID):
        vals = [co.num if (co.kind == "num" and co.note != APPROX_NOTE) else None
                for co in coerced]
        # a text-family binary/ordinal/id (yes/no, agree, S01) has no numbers ->
        # fall back to string so labels survive.
        if any(v is not None for v in vals) or kind is Kind.NUMERIC:
            return pd.array(vals, dtype="float64")
        return pd.array([_disp(co) for co in coerced], dtype="string")
    if kind is Kind.DATE:
        raw = [co.text if co.kind == "date" else None for co in coerced]
        return pd.to_datetime(pd.Series(raw), errors="coerce", format="mixed")
    if kind is Kind.EMPTY:
        return pd.array([None] * len(coerced), dtype="float64")
    return pd.array([_disp(co) for co in coerced], dtype="string")


def _disp(co):
    if co.kind == "empty" or co.kind == "uncomputed":
        return pd.NA
    if co.kind == "num":
        # NB4: a CATEGORICAL-default mixed column's numeric cells must round-trip
        # exactly through pd.to_numeric in bind._code_series -- levels.display is
        # lossless (shortest repr for fractional, '1' for an int), fmt.num rounds.
        return levels.display(co.num)
    return co.text if co.text is not None else pd.NA


def infer(table, overrides: dict | None = None, merges: dict | None = None,
          conventions: dict | None = None) -> Dataset:
    """Full column inference over a cleaned ``Table`` (PLAN §4.4).

    ``overrides``    -- per-column forced ``Kind`` (name -> Kind).
    ``merges``       -- per-column level-merge map (name -> {source: canonical}),
                        applied before profiling (S7).
    ``conventions``  -- per-column ``Convention`` override (name -> Convention);
                        when given, that column is read with it instead of the
                        detected one, and no ambiguity note is raised (S6).
    """
    overrides = overrides or {}
    merges = merges or {}
    conventions = conventions or {}
    profiles: list[ColumnProfile] = []
    df_cols: dict[str, object] = {}
    log: list[str] = list(table.log)
    header = table.header
    units = table.units or [None] * len(header)
    for j, name in enumerate(header):
        cells = [row[j] for row in table.rows]
        unit = units[j] if j < len(units) else None
        cells, merge_log = _apply_merges(cells, name, merges)   # S7
        log.extend(merge_log)
        user_conv = conventions.get(name)
        if user_conv is not None:
            conv, actions = user_conv, []
        else:
            conv, actions = detect_convention(cells)
        coerced = [coerce_cell(c, conv) for c in cells]
        prof = profile_column(name, cells, table.excel_rows, unit=unit,
                              override=overrides.get(name), conv=user_conv)
        profiles.append(prof)
        log.extend(f"Column '{name}': {_conv_note(a)}" for a in actions)  # S6
        col = _df_column(prof.kinds[0], coerced)
        key = name if name not in df_cols else f"{name}__{j}"
        df_cols[key] = col
    df = pd.DataFrame(df_cols) if df_cols else pd.DataFrame()
    return Dataset(
        df=df,
        profiles=tuple(profiles),
        excel_rows=tuple(table.excel_rows),
        source=table.source,
        sheet=table.sheet,
        log=tuple(log),
        dropped_rows=tuple(table.dropped_rows),
        header_row=table.header_row,
        n_rows_read=len(table.rows),
    )


# --------------------------------------------------------------------------
# label-merge proposal (surface only; never auto-applied) -- §4.4
# --------------------------------------------------------------------------
def propose_merges(values) -> dict:
    """Group level spellings by strip().casefold() + whitespace collapse +
    trailing-punctuation removal; propose the most common spelling as canonical.
    Prefix merges are never proposed."""
    groups: dict[str, list[str]] = {}
    for v in values:
        key = re.sub(r"\s+", " ", str(v).strip().casefold()).rstrip(".,;:!-")
        groups.setdefault(key, []).append(v)
    out: dict[str, str] = {}
    for variants in groups.values():
        if len(set(variants)) <= 1:
            continue
        canonical = max(variants, key=variants.count)
        for v in variants:
            if v != canonical:
                out[v] = canonical
    return out
