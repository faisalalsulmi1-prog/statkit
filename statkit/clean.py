"""Grid -> Table: find the header and drop the noise (PLAN §4.2).

The second ingest stage. It takes one ``Grid`` (raw cells from ``grid.load``)
and resolves it to a ``Table``: the real header (single or two-row), a units
row consumed into per-column units, and a body of data rows with every dropped
title / spacer / summary / footer / repeated-header / merged-note row logged and
the original 1-based ``excel_rows`` preserved for every survivor. It works at the
row/column-structure level only -- type inference and cell coercion are Chunks
4/5 (``coerce.py`` / ``infer.py``).

Header scoring (§4.2 step 3), on the FIRST 30 non-empty rows, width = row width:

    score(r) = fill(r) * strfrac(r) * (0.5 + 0.5*below(r)) - 0.25*[fill(r) < 0.5]

with ``fill`` on the PRE-merged-fill grid (a merged range = one filled cell at
its top-left, snag 2), ``strfrac`` = fraction of non-empty cells that are text
and not numeric-coercible, ``below`` = mean fill of the next 5 rows.

The header is the highest-scoring row that is NOT *data-like* -- a row whose
values recur down their own columns (categorical data, not one-off labels) -- so
a full data row never steals header detection from a real header a blank cell
made shorter (snag NB3); when every window row is data-like the whole window is
the pool, preserving the argmax on all-categorical sheets.

Two behaviours here DIVERGE from PLAN's prose in favour of the committed oracles
(the project's standing "oracles win" rule, cf. the ~30% decision):

  * A units row is CONSUMED (skipped, recorded in ``units_row``) and its tokens
    become per-column ``units`` -- the header names stay BARE. PLAN §4.2 step 4a
    said to append " (unit)"; the oracles keep bare names + a ``unit`` field.
  * Empty columns are KEPT (not dropped). Grid width == oracle ``n_cols`` for all
    37 corpus sheets; infer.py marks an all-empty column ``EMPTY`` and the UI
    hides it. PLAN step 10 said to drop them.

The look-up two-row rule (snag 13) additionally requires the top row to carry
>= 2 non-empty cells, so a lone merged caption (s1_3 May) is not misread as a
two-row header.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .coerce import NA_TOKENS
from .grid import Cell, Grid

HEADER_WINDOW = 30
AMBIG_BEST = 0.4      # best score below this -> ambiguous
AMBIG_GAP = 0.15      # best - second below this -> ambiguous
JOIN = " – "     # two-row header join: space en-dash space

_SUMMARY_START = re.compile(
    r"^(total|sum|subtotal|grand total|mean|average|avg|sd|std\.?|n\s*=)\b"
)
_SUMMARY_END = re.compile(r"\b(total|subtotal|average|avg|mean|sum)\s*$")
# Footer first-cell markers, split by strength (B4): _FOOTER_ALWAYS are prose
# lead-ins that are never data; _FOOTER_COND are summary keywords that are a
# footer only on an otherwise-numeric/sparse row (a real "Total Recall" data
# row starts with "total" too).
_FOOTER_ALWAYS = re.compile(r"^(source|notes?|\*|†)")
_FOOTER_COND = re.compile(r"^(total|sum|mean|average|n\s*=)\b")
# A units token: unit-ish chars only, must start with a letter or % / ° / µ
# (never a digit), so "C", "%", "mg/L", "µS/cm", "s.u." match but "1.02"/"B1" do not.
_UNIT_TOKEN = re.compile(r"^[A-Za-z%°µ][A-Za-z%°µ·/.\s]*$")
_WS = re.compile(r"\s+")


@dataclass
class Table:
    header: list           # list[str] -- cleaned column names (verbatim, deduped)
    rows: list             # list[list[Cell]] -- surviving DATA rows only
    excel_rows: list       # list[int] -- 1-based sheet row of each surviving data row
    header_row: int        # 1-based excel row of the header (TOP row if two_row)
    two_row: bool = False
    header_row2: int | None = None      # 1-based bottom header row when two_row
    units_row: int | None = None        # 1-based units row consumed (snag 4)
    data_start_row: int | None = None    # 1-based first data row
    units: list = field(default_factory=list)      # per-column unit str | None
    candidates: list = field(default_factory=list)  # (excel_row, score) desc
    ambiguous: bool = False
    dropped_rows: list = field(default_factory=list)  # (excel_row, reason)
    hidden_rows: frozenset = frozenset()   # surviving rows that were hidden (for UI)
    hidden_cols: frozenset = frozenset()
    log: list = field(default_factory=list)  # human action lines
    sheet: str = ""
    source: str = ""


# --------------------------------------------------------------------------
# cell predicates
# --------------------------------------------------------------------------
def _num_like(s: str) -> bool:
    """True if the string would coerce to a number (crude, header-scoring only)."""
    t = s.strip().replace("−", "-").replace("–", "-")
    for ch in "$€£¥, %'":
        t = t.replace(ch, "")
    t = t.strip()
    if t.startswith("(") and t.endswith(")"):
        t = "-" + t[1:-1]
    if not t:
        return False
    try:
        float(t)
        return True
    except ValueError:
        return False


def _texty(cell: Cell) -> bool:
    v = cell.value
    return isinstance(v, str) and v.strip() != "" and not _num_like(v)


def _nonempty(cell: Cell) -> bool:
    return cell.value is not None


def _is_blank(row) -> bool:
    return not any(_nonempty(c) for c in row)


# --------------------------------------------------------------------------
# header scoring
# --------------------------------------------------------------------------
def _fill(row) -> float:
    return sum(1 for c in row if _nonempty(c)) / len(row) if row else 0.0


def _strfrac(row) -> float:
    ne = [c for c in row if _nonempty(c)]
    if not ne:
        return 0.0
    return sum(1 for c in ne if _texty(c)) / len(ne)


def _below(rows, i) -> float:
    # Skip a blank spacer run sitting directly under row i before measuring the
    # data block: a single blank line between a header and its data must not
    # demote the header (else, on an all-text sheet where the header and data
    # rows tie on strfrac, a DATA row wins and the first rows vanish).
    j = i + 1
    while j < len(rows) and _is_blank(rows[j]):
        j += 1
    nxt = rows[j:j + 5]
    if not nxt:
        return 0.0
    return sum(_fill(r) for r in nxt) / len(nxt)


def _score(rows, i) -> float:
    f = _fill(rows[i])
    s = f * _strfrac(rows[i]) * (0.5 + 0.5 * _below(rows, i))
    if f < 0.5:
        s -= 0.25
    return s


def _recurs_in_column(rows, i) -> bool:
    """True if a strict majority of row i's values reappear in their own column
    ELSEWHERE -- the mark of a DATA row (categorical values 'M'/'F', 'Yes'/'No'
    repeat down a column; a header's labels appear only once, as the header).
    Checked over the whole column, not just below, so a data row near the bottom
    of the sheet is still recognised (its values recur among the rows above)."""
    row = rows[i]
    ne = [(c, row[c].value) for c in range(len(row)) if _nonempty(row[c])]
    if not ne:
        return False
    recurring = 0
    for c, v in ne:
        vs = str(v).strip().casefold()
        if any(k != i and c < len(rows[k]) and _nonempty(rows[k][c])
               and str(rows[k][c].value).strip().casefold() == vs
               for k in range(len(rows))):
            recurring += 1
    return recurring * 2 > len(ne)


def _detect_header(grid):
    rows = grid.rows
    nonempty = [i for i, r in enumerate(rows) if any(_nonempty(c) for c in r)]
    window = nonempty[:HEADER_WINDOW]
    scored = [(i, _score(rows, i)) for i in window]
    ordered = sorted(scored, key=lambda t: (-t[1], t[0]))
    candidates = [(grid.excel_rows[i], sc) for i, sc in ordered]

    # Value recurrence, computed ONCE over ALL rows (exact, O(cells)): a header's
    # labels appear once; a DATA row's values repeat down their columns. keys[i]
    # is row i casefolded per cell (None for empty); row_counts counts identical
    # rows (a re-pasted header copy is identical, not "recurring"); col_counts[c]
    # counts each value seen in column c.
    keys = []
    col_counts: dict = {}
    for r in rows:
        key = tuple(str(c.value).strip().casefold() if _nonempty(c) else None
                    for c in r)
        keys.append(key)
        for c, v in enumerate(key):
            if v is not None:
                col_counts.setdefault(c, Counter())[v] += 1
    row_counts = Counter(keys)

    def _data_like(i):
        # A strict majority of row i's cells carry a value that recurs in a
        # NON-identical row of the same column (identical rows subtracted, so a
        # real header with a re-pasted copy is not mistaken for data).
        ne = [(c, v) for c, v in enumerate(keys[i]) if v is not None]
        if not ne:
            return False
        self_n = row_counts[keys[i]]
        recurring = sum(1 for c, v in ne if col_counts[c][v] - self_n >= 1)
        return recurring * 2 > len(ne)

    # argmax over rows that are NOT data-like -- a data row must never win header
    # detection over a real header, even when a blank header cell lowers its fill
    # (snag NB3); fall back to all rows when every window row is data-like.
    # earliest row wins a tie.
    pool = [t for t in scored if not _data_like(t[0])] or scored
    best_i, best = max(pool, key=lambda t: (t[1], -t[0]))
    # Only a nearly-all-text row (strfrac >= 0.9) is a plausible rival HEADER; a
    # text-heavy DATA row must not manufacture a false near-tie (h6_2, s2_2). On
    # an all-text survey EVERY row is ~all-text, so also exclude any row whose
    # values recur below -- a data row, not a header candidate (S-H) -- and any
    # copy of the chosen header (a re-pasted header is never a rival).
    rivals = [sc for i, sc in scored
              if i != best_i and _strfrac(rows[i]) >= 0.9
              and not _recurs_in_column(rows, i) and keys[i] != keys[best_i]]
    second = max(rivals) if rivals else -1.0
    ambiguous = best < AMBIG_BEST or (best - second) < AMBIG_GAP
    return best_i, candidates, ambiguous


# --------------------------------------------------------------------------
# merged ranges
# --------------------------------------------------------------------------
def _hspan_ge2(merged, exrow) -> bool:
    """A horizontal merge (>=2 cols) covering 1-based excel row ``exrow``."""
    return any(m[0] <= exrow <= m[2] and (m[3] - m[1]) >= 1 for m in merged)


def _merge_filled(grid):
    """Copy of grid.rows with every merged range filled from its top-left cell."""
    rows = [[Cell(c.value, c.fmt, c.formula) for c in r] for r in grid.rows]
    idx = {e: i for i, e in enumerate(grid.excel_rows)}
    for (r1, c1, r2, c2) in grid.merged:
        if r1 not in idx:
            continue
        src = grid.rows[idx[r1]][c1 - 1]
        for rr in range(r1, r2 + 1):
            if rr not in idx:
                continue
            row = rows[idx[rr]]
            for cc in range(c1, c2 + 1):
                if cc - 1 < len(row):
                    row[cc - 1] = Cell(src.value, src.fmt, src.formula)
    return rows


# --------------------------------------------------------------------------
# header names
# --------------------------------------------------------------------------
def _col_letter(n: int) -> str:   # 1-based -> A, B, ... Z, AA
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _txt(cell: Cell) -> str:
    v = cell.value
    if v is None:
        return ""
    return _WS.sub(" ", str(v)).strip()


def _name_columns(top_cells, bottom_cells, width):
    raw = []
    for c in range(width):
        top = _txt(top_cells[c]) if c < len(top_cells) else ""
        bot = _txt(bottom_cells[c]) if bottom_cells and c < len(bottom_cells) else ""
        if bottom_cells:
            parts = [p for p in (top, bot) if p]
            raw.append(JOIN.join(parts) if parts else None)
        else:
            raw.append(top or None)
    # None -> "Column <Letter>"
    named = [nm if nm else f"Column {_col_letter(c + 1)}" for c, nm in enumerate(raw)]
    # dedupe: first stays, later get " (2)", " (3)"
    seen: dict = {}
    out = []
    for nm in named:
        if nm in seen:
            seen[nm] += 1
            out.append(f"{nm} ({seen[nm]})")
        else:
            seen[nm] = 1
            out.append(nm)
    return out


# --------------------------------------------------------------------------
# units row (snag 4)
# --------------------------------------------------------------------------
def _is_units_row(rows, down) -> bool:
    # 1. SHAPE: every non-empty cell is a short (<=6 char) unit-ish token.
    row = rows[down]
    positions = [c for c in range(len(row)) if _nonempty(row[c])]
    if not positions:
        return False
    for c in positions:
        v = row[c].value
        if not isinstance(v, str):
            return False
        t = v.strip()
        if not t or len(t) > 6 or not _UNIT_TOKEN.match(t):
            return False
        if t.casefold() in NA_TOKENS:      # a missing-value marker is never a unit
            return False
    # 2. A real units row ANNOTATES NUMBERS: a majority of the columns it marks
    # must be numeric-majority in the data below (so an all-categorical sheet,
    # whose first data row also matches the token shape, is NOT a units row and
    # keeps its first row), and a token must not itself recur as a data value in
    # its own column below (a real "kg"/"%" never reappears; "Yes"/"M" does).
    below = rows[down + 1: down + 6]
    if not below:
        return False
    numeric_cols = 0
    for c in positions:
        col = [r[c] for r in below if c < len(r) and _nonempty(r[c])]
        if not col:
            continue
        numeric = sum(1 for cell in col if _num_or_na(cell))
        token = row[c].value.strip().casefold()
        recurs = any(str(cell.value).strip().casefold() == token for cell in col)
        if numeric * 2 > len(col) and not recurs:
            numeric_cols += 1
    return numeric_cols >= 1 and numeric_cols * 2 > len(positions)


# --------------------------------------------------------------------------
# data-row classification
# --------------------------------------------------------------------------
def _num_or_na(cell) -> bool:
    """True if the cell would coerce to a number or reads as a missing-value token."""
    v = cell.value
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    s = str(v).strip()
    return _num_like(s) or s.casefold() in NA_TOKENS


def _others_num_or_na(ne) -> bool:
    """Every non-first non-empty cell is numeric-coercible or an NA token."""
    return all(_num_or_na(c) for c in ne[1:])


def _region_label_counts(region) -> dict:
    """casefolded first-cell label -> how many region rows carry it (B4)."""
    counts: dict = {}
    for _e, row in region:
        ne = [c for c in row if _nonempty(c)]
        if ne and isinstance(ne[0].value, str):
            lab = ne[0].value.strip().casefold()
            counts[lab] = counts.get(lab, 0) + 1
    return counts


def _is_summary(row, label_counts) -> bool:
    ne = [c for c in row if _nonempty(c)]
    if not ne or not isinstance(ne[0].value, str):
        return False
    t = ne[0].value.strip().casefold()
    if not (_SUMMARY_START.match(t) or _SUMMARY_END.search(t)):
        return False
    # A subtotal label does not recur; a rating LEVEL ("Average") does. A label
    # seen in more than one data row is a category value, never a subtotal.
    if label_counts.get(t, 0) != 1:
        return False
    # A real subtotal sits on an otherwise-numeric row. The lone exception kept
    # for the committed oracles (s1_3 "AVG ... monthly average, not a sample"): a
    # cell that is ONLY a bare summary keyword ("avg"/"total"/"mean") is a
    # subtotal marker even beside a prose note -- but a COMPOUND label ("Class A
    # Average", "Mean Street", "Total Recall") must earn it by sitting on
    # numbers, so a rating level or a proper name survives. See B4 report.
    return _others_num_or_na(ne) or (" " not in t)


def _is_footer(row) -> bool:
    ne = [c for c in row if _nonempty(c)]
    if not ne:
        return False
    first = ne[0].value
    if isinstance(first, str):
        t = first.strip().casefold()
        if _FOOTER_ALWAYS.match(t):                       # (a) prose lead-in
            return True
        if _FOOTER_COND.match(t):                          # (b) summary keyword
            return _others_num_or_na(ne) or _fill(row) < 0.34
    return _fill(row) < 0.34 and all(_texty(c) for c in ne)  # (c) sparse all-text


def _is_repeated_header(row, header_cf) -> bool:
    ne_idx = [c for c, cell in enumerate(row) if _nonempty(cell)]
    if len(ne_idx) < max(2, (len(row) + 1) // 2):
        return False
    for c in ne_idx:
        v = row[c].value
        if not isinstance(v, str) or v.strip().casefold() != header_cf[c]:
            return False
    return True


# --------------------------------------------------------------------------
# clean
# --------------------------------------------------------------------------
def clean(grid: Grid, header_row: int | None = None, keep_hidden: bool = True,
          restore_rows: frozenset = frozenset()) -> Table:
    rows = grid.rows
    excel = grid.excel_rows
    if not rows:
        return Table([], [], [], header_row=0, sheet=grid.sheet, source=grid.source)
    width = len(rows[0])
    idx_of = {e: i for i, e in enumerate(excel)}

    # 1. header detection (on the pre-merge-fill grid). Always run the detector
    # so a forced header_row still carries the detector's candidates/ambiguity
    # (B5): forcing the row must not un-render the app's header picker, or the
    # user's pick oscillates. A forced row only overrides which index we USE.
    det_i, candidates, ambiguous = _detect_header(grid)
    if header_row is not None and header_row in idx_of:
        h_i = idx_of[header_row]
    else:
        h_i = det_i
    h_e = excel[h_i]

    filled = _merge_filled(grid)

    # 2. two-row header (look-down, then look-up) OR units row
    two_row = False
    top_i, bottom_i, units_i = h_i, None, None
    down = h_i + 1
    if down < len(rows) and _strfrac(rows[down]) >= 0.6 and (
        _fill(rows[h_i]) < 0.6 or _hspan_ge2(grid.merged, h_e)
    ):
        two_row, top_i, bottom_i = True, h_i, down
    else:
        up = h_i - 1
        top_ne = [c for c in rows[up]] if up >= 0 else []
        top_nonempty = [c for c in top_ne if _nonempty(c)]
        if (
            up >= 0
            and len(top_nonempty) >= 2
            and all(_texty(c) for c in top_nonempty)
            and _fill(rows[up]) < 0.6
            and _hspan_ge2(grid.merged, excel[up])
        ):
            two_row, top_i, bottom_i = True, up, h_i
        elif down < len(rows) and _is_units_row(rows, down):
            units_i = down

    top_e = excel[top_i]
    if two_row:
        header_row2 = excel[bottom_i]
        units_row_out = None
        header = _name_columns(filled[top_i], filled[bottom_i], width)
        units = [None] * width
        last_hdr_e = header_row2
    elif units_i is not None:
        header_row2 = None
        units_row_out = excel[units_i]
        header = _name_columns(filled[top_i], None, width)
        units = [(_txt(c) or None) for c in rows[units_i]]
        last_hdr_e = units_row_out
    else:
        header_row2 = None
        units_row_out = None
        header = _name_columns(filled[top_i], None, width)
        units = [None] * width
        last_hdr_e = top_e

    data_start = last_hdr_e + 1
    header_cf = [n.strip().casefold() for n in header]

    log = []
    if grid.merged:
        log.append(f"filled {len(grid.merged)} merged range(s)")
    if units_row_out is not None:
        log.append(f"consumed units row at row {units_row_out}")

    # rows above the top header -> title/notes, logged (not data)
    dropped = []
    for i in range(top_i):
        e = excel[i]
        reason = "blank spacer" if _is_blank(rows[i]) else "title/notes above header"
        dropped.append((e, reason))

    # 3. data region (merge-filled)
    region = [(excel[i], filled[i]) for i in range(len(filled)) if excel[i] >= data_start]
    label_counts = _region_label_counts(region)

    # full-width merged note rows inside the data (snag 5)
    note_rows = {
        m[0] for m in grid.merged
        if m[0] == m[2] and (m[3] - m[1] + 1) >= width - 1 and m[0] >= data_start
    }

    # footer block: reverse-scan the trailing footer/blank rows
    footer = set()
    for e, row in reversed(region):
        if e in restore_rows:
            break
        if _is_blank(row):
            continue
        if _is_footer(row):
            footer.add(e)
            continue
        break

    # 4. classify each data row. A restore_rows pin exempts a row from the
    # summary/footer/repeated-header/merged-note verdicts (B4): the caller has
    # asserted it is real data.
    kept_rows, kept_er = [], []
    for e, row in region:
        restored = e in restore_rows
        if not restored and e in footer:
            dropped.append((e, "footer note"))
        elif _is_blank(row):
            dropped.append((e, "blank spacer"))
        elif not restored and e in note_rows:
            dropped.append((e, "merged note row"))
        elif not restored and _is_repeated_header(row, header_cf):
            dropped.append((e, "repeated header row"))
        elif not restored and _is_summary(row, label_counts):
            dropped.append((e, "summary/subtotal row"))
        elif not keep_hidden and e in grid.hidden_rows:
            dropped.append((e, "hidden row (excluded)"))
        else:
            kept_rows.append(row)
            kept_er.append(e)

    surviving_hidden = frozenset(e for e in kept_er if e in grid.hidden_rows)
    dropped.sort(key=lambda t: t[0])

    return Table(
        header=header,
        rows=kept_rows,
        excel_rows=kept_er,
        header_row=top_e,
        two_row=two_row,
        header_row2=header_row2,
        units_row=units_row_out,
        data_start_row=data_start,
        units=units,
        candidates=candidates,
        ambiguous=ambiguous,
        dropped_rows=dropped,
        hidden_rows=surviving_hidden,
        hidden_cols=grid.hidden_cols,
        log=log,
        sheet=grid.sheet,
        source=grid.source,
    )
