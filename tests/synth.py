"""In-memory synthetic fixture builders (PLAN §9 / §9.2 residual cases).

The generated corpus (Waves 1-2) plus these builders is the full ingest contract
(FIXTURE_CURATION_WAVE2 "Fable bottom line"). Each builder returns
``(bytes, filename)`` ready for ``statkit.grid.load`` -- an openpyxl workbook
built in memory, never committed as a file (conftest rule). The 7th residual
case (a cp1252 0x80-0x9F byte) is already covered by the committed
``gen_w2_10_1.txt`` and is intentionally not rebuilt here.

Cases (PLAN §9.2 "Registry-shape gaps -> build in synth.py"):
  * ``space_thousands``  -- "1 234" space-thousands (decimal pattern E)
  * ``percent_text``     -- percent text at n>=30
  * ``ordered_vocabs``   -- non-agree ordered vocabularies (frequency/severity/
                            low-med-high) -> ORDINAL   [wired into infer here]
  * ``long_k3``          -- k>=3 long layout (rm_anova / friedman + S20)
  * ``mcnemar_happy``    -- non-degenerate McNemar wide layout (b, c > 0)
  * ``rc_grid``          -- an R x C count grid (3x4) with a Total row + column

The last three are ready builders consumed by Chunks 6/8/9; here they only get a
"does it ingest cleanly" smoke test (test_synth.py).
"""
from __future__ import annotations

from io import BytesIO

import openpyxl


def _wb_bytes(sheet_name, rows, fmts=None):
    """rows = list[list[cell value]]; fmts = optional {(r,c) 0-based: number_format}."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_name
    for r, row in enumerate(rows, start=1):
        for c, val in enumerate(row, start=1):
            cell = ws.cell(row=r, column=c, value=val)
            if fmts and (r - 1, c - 1) in fmts:
                cell.number_format = fmts[(r - 1, c - 1)]
    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue(), f"{sheet_name}.xlsx"


# --- §9.2: "1 234" space-thousands (decimal pattern E) --------------------
def space_thousands():
    vals = ["1 234", "5 678", "9 012", "12 345", "6 789",
            "3 210", "8 765", "4 321", "10 000", "2 500"]
    rows = [["City", "Population"]] + [[f"C{i}", v] for i, v in enumerate(vals)]
    return _wb_bytes("SpaceThousands", rows)


# --- §9.2: percent text at n>=30 ------------------------------------------
def percent_text(n=30):
    vals = [f"{40 + (i % 55)}%" for i in range(n)]           # "40%".."94%", n rows
    rows = [["Subject", "Completion"]] + [[f"S{i}", v] for i, v in enumerate(vals)]
    return _wb_bytes("PercentText", rows)


# --- §9.2: non-agree ordered vocabularies -> ORDINAL ----------------------
FREQUENCY = ["Never", "Rarely", "Sometimes", "Often", "Always"]
SEVERITY = ["None", "Mild", "Moderate", "Severe"]
LOW_MED_HIGH = ["Low", "Medium", "High"]


def ordered_vocabs(reps=4):
    header = ["Frequency", "Severity", "Rating"]
    rows = [header]
    for i in range(len(FREQUENCY) * reps):        # enough rows, each level recurs
        rows.append([
            FREQUENCY[i % len(FREQUENCY)],
            SEVERITY[i % len(SEVERITY)],
            LOW_MED_HIGH[i % len(LOW_MED_HIGH)],
        ])
    return _wb_bytes("OrderedVocabs", rows)


# --- §9.2: k>=3 long layout (rm_anova / friedman + S20) -------------------
def long_k3(subjects=8):
    """One row per (subject, condition); 3 within-conditions."""
    conditions = ["Baseline", "Week4", "Week8"]
    rows = [["Subject", "Condition", "Score"]]
    for s in range(1, subjects + 1):
        for k, cond in enumerate(conditions):
            rows.append([f"P{s:02d}", cond, 50 + s + 5 * k])
    return _wb_bytes("LongK3", rows)


# --- §9.2: non-degenerate McNemar (wide, b>0 and c>0) ---------------------
def mcnemar_happy():
    """Before/After binary with both discordant cells non-empty (b=4, c=3)."""
    a = 6 * [("Yes", "Yes")]           # concordant +
    d = 5 * [("No", "No")]             # concordant -
    b = 4 * [("Yes", "No")]            # discordant b
    c = 3 * [("No", "Yes")]            # discordant c
    rows = [["Before", "After"]] + [list(p) for p in (a + b + c + d)]
    return _wb_bytes("McNemarHappy", rows)


# --- §9.2: R x C count grid (3x4) with Total row + column -----------------
def rc_grid():
    rows = [
        ["Region", "Product A", "Product B", "Product C", "Total"],
        ["North", 12, 7, 9, 28],
        ["South", 5, 14, 6, 25],
        ["East", 8, 8, 11, 27],
        ["Total", 25, 29, 26, 80],
    ]
    return _wb_bytes("CountGrid", rows)


ALL_BUILDERS = (
    space_thousands, percent_text, ordered_vocabs,
    long_k3, mcnemar_happy, rc_grid,
)
