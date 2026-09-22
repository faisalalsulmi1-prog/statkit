"""Word report — a python-docx document assembled in memory (PLAN §7, L2/L3).

``build_report(result, spec=None) -> io.BytesIO`` returns a ``.docx`` built
entirely in memory (``Document().save(BytesIO)``) — NEVER to disk (L3; the
mechanical guard is tests/test_no_ai_no_io.py). Word only, no PDF (L2). English
only (L1) — every string here is English and the numbers come straight off the
``Result``.

Fixed section order (§7), driven by the ``Result`` (which already carries
everything a report needs):

  Title · 1 Summary (the plain-English sentence) · 2 Results (an APA numbers
  panel + any coefficient / contingency / expected / post-hoc / Table-1 tables)
  · 3 Figures (charts.py -> BytesIO -> add_picture) · 4 Assumption checks ·
  5 Notes & advisories (Findings) · 6 Methods (a one-line methods statement +
  library versions + citation).

A blocked ``Result`` still produces a report: the summary sentence IS the block
reason (§6 rule 18) and no chart is embedded.

Only stdlib (io, sys) + numpy + pandas + docx (L3 allowlist). No I/O to disk.
"""
from __future__ import annotations

import io
import sys

import matplotlib
import numpy as np
import pandas as pd
import scipy
import statsmodels
from docx import Document
from docx.shared import Inches

from . import charts, fmt, sentences

# Columns whose cells are p-values -> APA form (fmt.p), never 2-dp truncation
# (a real p of .0029 would otherwise print "0"). PR(>F) is statsmodels' ANOVA
# p-column header.
_P_COLUMNS = frozenset({"p", "p_holm", "p_raw", "PR(>F)"})


# --------------------------------------------------------------------------
# small formatting helpers
# --------------------------------------------------------------------------
def _cell(v, is_p: bool = False) -> str:
    """A DataFrame cell -> display string. A p-value cell renders APA (``fmt.p``,
    e.g. ".003" not the truncated "0"); ints stay ints; other floats -> 2 dp."""
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        fv = float(v)
        if np.isnan(fv):
            return ""
        return fmt.p(fv) if is_p else fmt.num(fv, 2)
    if isinstance(v, (np.integer,)):
        return str(int(v))
    return str(v)


def _df_table(doc, df: pd.DataFrame, caption: str, index_header: str = "",
              all_p: bool = False):
    """Render a DataFrame as a Table Grid. The index becomes the first column
    UNLESS it is a plain ``RangeIndex`` (then the bare 0/1/2 column is omitted).
    p-columns (or every cell, when ``all_p`` — a whole p-matrix) render APA."""
    if df is None or df.empty:
        return
    if caption:
        doc.add_paragraph(caption, style="Caption")
    cols = [str(c) for c in df.columns]
    show_index = not isinstance(df.index, pd.RangeIndex)
    off = 1 if show_index else 0
    p_flags = [all_p or (c in _P_COLUMNS) for c in cols]
    table = doc.add_table(rows=1, cols=len(cols) + off)
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    if show_index:
        hdr[0].text = index_header
    for j, c in enumerate(cols):
        hdr[j + off].text = c
    for idx, row in df.iterrows():
        cells = table.add_row().cells
        if show_index:
            cells[0].text = _cell(idx)
        for j, c in enumerate(df.columns):
            cells[j + off].text = _cell(row[c], is_p=p_flags[j])


def _kv_table(doc, rows, caption: str = ""):
    """A two-column key/value table (the APA numbers panel)."""
    rows = [(k, v) for k, v in rows if v not in ("", None)]
    if not rows:
        return
    if caption:
        doc.add_paragraph(caption, style="Caption")
    table = doc.add_table(rows=0, cols=2)
    table.style = "Table Grid"
    for k, v in rows:
        cells = table.add_row().cells
        cells[0].text = str(k)
        cells[1].text = str(v)


def _categoricals_tables(doc, cats):
    """Render describe's categorical counts (means.py ``extra['categoricals']``),
    one table per variable — so a categorical-only describe (where
    ``r.descriptives is None``) still gets a results table. Levels/groups are
    already display strings. Two shapes:
      no-group  {header: {level: count}}          -> a level/count table;
      grouped   {header: {group: {level: count}}} -> a level × group count matrix.
    """
    for header, counts in cats.items():
        grouped = counts and all(isinstance(v, dict) for v in counts.values())
        doc.add_paragraph(f"Counts — {header}", style="Caption")
        if grouped:
            groups = list(counts.keys())
            levels = []                              # levels in first-seen order
            for g in groups:
                for lv in counts[g]:
                    if lv not in levels:
                        levels.append(lv)
            table = doc.add_table(rows=1, cols=1 + len(groups))
            table.style = "Table Grid"
            hdr = table.rows[0].cells
            hdr[0].text = header
            for j, g in enumerate(groups):
                hdr[j + 1].text = str(g)
            for lv in levels:
                cells = table.add_row().cells
                cells[0].text = str(lv)
                for j, g in enumerate(groups):
                    cells[j + 1].text = str(counts[g].get(lv, 0))
        else:
            table = doc.add_table(rows=1, cols=2)
            table.style = "Table Grid"
            hdr = table.rows[0].cells
            hdr[0].text = header
            hdr[1].text = "Count"
            for lv, cnt in counts.items():
                cells = table.add_row().cells
                cells[0].text = str(lv)
                cells[1].text = str(cnt)


# --------------------------------------------------------------------------
# the APA numbers panel (§7 section 5)
# --------------------------------------------------------------------------
def _perfect_fit(r) -> bool:
    """Perfect fit: Cohen's f² (the regression effect) is non-finite — r² has
    reached 1.0, so F is astronomically large / infinite and not meaningful. Keyed
    on the f² effect specifically (a non-finite Cramér's V / odds ratio in another
    test does NOT make its own statistic meaningless)."""
    return (r.effect is not None and r.effect[0] == "f2"
            and not np.isfinite(float(r.effect[1])))


def _stat_str(r) -> str:
    if r.statistic is None:
        return ""
    name, val = r.statistic
    # Perfect fit: F is astronomical / infinite and not meaningful — render the
    # value as the em dash (never "inf", never a 30-digit number); a note row
    # explains it. Normal finite regressions keep the .2f rendering (byte-identical).
    sval = fmt.NON_FINITE if _perfect_fit(r) else f"{float(val):.2f}"
    if len(r.df) == 2:
        return f"{name}({fmt.num(r.df[0])}, {fmt.num(r.df[1])}) = {sval}"
    if len(r.df) == 1:
        return f"{name}({fmt.num(r.df[0])}) = {sval}"
    return f"{name} = {sval}"


def _apa_rows(r):
    rows = []
    if r.statistic is not None:
        rows.append(("Test statistic", _stat_str(r)))
        if _perfect_fit(r):
            rows.append(("Note", "Perfect fit — the model fits the data exactly, "
                                 "so F is not meaningful."))
    if r.p is not None:
        rows.append(("p-value", fmt.p(r.p)))
    if r.estimate is not None:
        name, val = r.estimate
        if not np.isfinite(float(val)):
            s = "not estimable (empty cell)"
        else:
            s = f"{float(val):.2f}"
            if r.estimate_ci is not None:
                s += f", {fmt.ci(*r.estimate_ci)}"
        rows.append((name, s))
    if r.effect is not None:
        name, val = r.effect
        if not np.isfinite(float(val)):
            # A non-finite effect (perfect-fit f², empty-cell Cramér's V) renders
            # as the em dash — matching the app panel (fmt.num) — never a bare "inf".
            s = fmt.NON_FINITE
            if r.effect_label:
                s += f" ({r.effect_label})"
        else:
            s = f"{float(val):.2f}"
            if r.effect_ci is not None:
                s += f", {fmt.ci(*r.effect_ci)}"
            if r.effect_label:
                s += f" ({r.effect_label}"
                s += f", {r.effect_source})" if r.effect_source else ")"
        rows.append((f"Effect size — {name}", s))
    omega = r.extra.get("omega_sq")
    if omega is not None:
        ov = float(omega)
        rows.append(("ω²", "< 0 (reported as 0)" if ov < 0 else f"{ov:.2f}"))
    used = r.n.get("used")
    if used is not None:
        rows.append(("N analysed", str(used)))
    dropped = r.n.get("dropped")
    if dropped:
        rows.append(("N excluded (missing)", str(dropped)))
    rows.append(("Test", f"two-sided, alpha = {f'{r.alpha:.2f}'.lstrip('0')}"))
    return rows


# --------------------------------------------------------------------------
# the methods line (§7)
# --------------------------------------------------------------------------
def _methods_line(r, spec) -> str:
    total = r.n.get("total", r.n.get("used", 0))
    dropped = r.n.get("dropped", 0)
    parts = [f"Analysis used {r.test_name}"]
    if r.method:
        parts[0] += f" ({r.method})"
    parts.append("two-sided, alpha = .05.")
    if r.variant_notes:
        parts.append(" ".join(v.rstrip(".") + "." for v in r.variant_notes))
    if r.effect is not None:
        src = f" interpreted per {r.effect_source}" if r.effect_source else ""
        parts.append(f"Effect size: {r.effect[0]}{src}.")
    parts.append(f"Missing values were excluded listwise "
                 f"({dropped} of {total} rows).")
    line = " ".join(parts)
    ver = (f"Computed in Python {sys.version_info.major}.{sys.version_info.minor} "
           f"with pandas {pd.__version__}, NumPy {np.__version__}, "
           f"SciPy {scipy.__version__}, statsmodels {statsmodels.__version__} "
           f"and matplotlib {matplotlib.__version__}.")
    cite = f" Citation: {spec.citation}." if spec is not None and spec.citation else ""
    return line + " " + ver + cite


# --------------------------------------------------------------------------
# the Data section (§7 / S11) — provenance for a reproducible report
# --------------------------------------------------------------------------
# Known ``extra`` DataFrames -> (caption, whole-matrix-is-p?). Rendered after the
# results table so a reader sees the crosstab / residuals / ANOVA / p-matrices.
_EXTRA_TABLES = (
    ("observed", "Observed counts", False),
    ("adjusted_residuals",
     "Adjusted standardized residuals (|z| > 1.96 flagged)", False),
    ("anova_type3", "Type III ANOVA (Sum contrasts)", False),
    ("anova_typ2", "ANOVA of the regression (Type II)", False),
    ("p_raw", "Raw p-values", True),
    ("p_holm", "Holm-adjusted p-values", True),
    ("n", "Pairwise n", False),
)


def _role_labels(spec) -> dict:
    """role name -> student-facing label, gathered across every layout."""
    labels = {}
    if spec is not None:
        for roles in spec.contract.roles_by_layout.values():
            for role in roles:
                labels.setdefault(role.name, role.label)
    return labels


def _data_section(doc, spec, dataset, bound):
    """The "Data" section (S11): source, columns used, prep log, N accounting.
    Rendered only when ``dataset``/``bound`` are supplied; ``getattr`` keeps it
    defensive against partial stubs."""
    if dataset is None and bound is None:
        return
    doc.add_heading("Data", level=1)

    source = getattr(dataset, "source", "") or ""
    sheet = getattr(dataset, "sheet", "") or ""
    if source or sheet:
        loc = source or "(uploaded file)"
        if sheet:
            loc += f" — sheet {sheet}"
        doc.add_paragraph(f"Source: {loc}")
    n_read = getattr(dataset, "n_rows_read", 0)
    if n_read:
        doc.add_paragraph(f"Rows read (after cleaning): {n_read}")
    header_row = getattr(dataset, "header_row", 0)
    if header_row:
        doc.add_paragraph(f"Header row: {header_row}")

    # Columns used — role label -> source header(s) -> effective kind
    columns = getattr(bound, "columns", None) or {}
    if columns:
        labels = _role_labels(spec)
        kinds = getattr(bound, "kinds", {}) or {}
        table = doc.add_table(rows=1, cols=3)
        table.style = "Table Grid"
        for j, h in enumerate(("Role", "Column", "Type")):
            table.rows[0].cells[j].text = h
        for role, headers in columns.items():
            cells = table.add_row().cells
            cells[0].text = labels.get(role, role)
            cells[1].text = ", ".join(str(h) for h in headers)
            k = kinds.get(role)
            cells[2].text = getattr(k, "value", "") if k is not None else ""

    # Preparation log (clean + coerce lines)
    log = getattr(dataset, "log", ()) or ()
    if log:
        doc.add_paragraph("Preparation log:")
        for line in log:
            doc.add_paragraph(str(line), style="List Bullet")

    # N accounting
    n_total = getattr(bound, "n_total", None)
    n_used = getattr(bound, "n_used", None)
    if n_total is not None and n_used is not None:
        doc.add_paragraph(f"Rows analysed: {n_used} of {n_total}.")
    for reason, count in (getattr(bound, "dropped", {}) or {}).items():
        doc.add_paragraph(f"{count} excluded — {reason}", style="List Bullet")


def _appendices(doc, dataset, bound):
    """Appendix A (dropped rows) and Appendix B (unparsed cells) — only when
    there is something to list."""
    # Appendix A — dropped rows: bound-stage (cap 200) then clean-stage.
    a_rows = list((getattr(bound, "dropped_rows", ()) or ())[:200])
    a_rows += list(getattr(dataset, "dropped_rows", ()) or ())
    if a_rows:
        doc.add_heading("Appendix A — Dropped rows", level=1)
        table = doc.add_table(rows=1, cols=2)
        table.style = "Table Grid"
        table.rows[0].cells[0].text = "Excel row"
        table.rows[0].cells[1].text = "Reason"
        for er, reason in a_rows[:200]:
            cells = table.add_row().cells
            cells[0].text = str(er)
            cells[1].text = str(reason)

    # Appendix B — unparsed (failed) cells of the bound columns' profiles.
    used_headers = set()
    for headers in (getattr(bound, "columns", {}) or {}).values():
        used_headers.update(str(h) for h in headers)
    prof_by_name = {p.name: p for p in getattr(dataset, "profiles", ()) or ()}
    b_rows = []
    for h in used_headers:
        p = prof_by_name.get(h)
        if p is None:
            continue
        for er, raw in getattr(p, "failed_cells", ()) or ():
            b_rows.append((h, er, raw))
    if b_rows:
        doc.add_heading("Appendix B — Unparsed cells (treated as missing)", level=1)
        table = doc.add_table(rows=1, cols=3)
        table.style = "Table Grid"
        for j, h in enumerate(("Column", "Excel row", "Cell text")):
            table.rows[0].cells[j].text = h
        for col, er, raw in b_rows[:50]:
            cells = table.add_row().cells
            cells[0].text = str(col)
            cells[1].text = str(er)
            cells[2].text = str(raw)


# --------------------------------------------------------------------------
# public entry
# --------------------------------------------------------------------------
def build_report(result, spec=None, dataset=None, bound=None) -> io.BytesIO:
    """Assemble the Word report for ``result`` and return it as an in-memory
    ``io.BytesIO`` (a ``.docx``). ``spec`` (the ``TestSpec``) is optional and only
    adds the citation to the methods line. ``dataset``/``bound`` are optional and,
    when given, add a "Data" section (source, columns used, prep log, N
    accounting) plus Appendix A (dropped rows) / B (unparsed cells)."""
    r = result
    doc = Document()

    # Title
    doc.add_heading(r.test_name, level=0)

    # 1 Summary — the plain-English sentence (block reason when blocked)
    doc.add_heading("Summary", level=1)
    doc.add_paragraph(sentences.render(r))

    # Data — provenance (only when dataset/bound supplied)
    _data_section(doc, spec, dataset, bound)

    if r.status == "blocked":
        # A blocked report stops at the reason + methods; nothing was computed.
        _notes(doc, r)
        doc.add_heading("Methods", level=1)
        doc.add_paragraph(_methods_line(r, spec))
        _appendices(doc, dataset, bound)
        return _save(doc)

    # 2 Results — APA numbers panel + any structured tables
    doc.add_heading("Results", level=1)
    _kv_table(doc, _apa_rows(r))
    if r.descriptives is not None:
        _df_table(doc, r.descriptives, "Descriptive statistics (Table 1)")
    cats = r.extra.get("categoricals")
    if cats:
        _categoricals_tables(doc, cats)
    if r.table is not None:
        _df_table(doc, r.table, "Results table")
    if r.expected is not None:
        _df_table(doc, r.expected, "Expected counts", index_header="row \\ column")
    if r.posthoc is not None:
        _df_table(doc, r.posthoc, f"Post-hoc comparisons ({r.posthoc_name})")
    # Known extra DataFrames (crosstab, residuals, ANOVA tables, p-matrices)
    for key, caption, all_p in _EXTRA_TABLES:
        val = r.extra.get(key)
        if isinstance(val, pd.DataFrame):
            _df_table(doc, val, caption, all_p=all_p)

    # 3 Figures — charts rendered to memory, embedded as pictures
    figs = charts.chart(r)
    if figs:
        doc.add_heading("Figures", level=1)
        for caption, png in figs:
            doc.add_picture(io.BytesIO(png), width=Inches(6.0))
            doc.add_paragraph(caption, style="Caption")

    # 4 Assumption checks
    if r.checks:
        doc.add_heading("Assumption checks", level=1)
        table = doc.add_table(rows=1, cols=4)
        table.style = "Table Grid"
        for j, h in enumerate(("Check", "Statistic", "p", "Verdict / note")):
            table.rows[0].cells[j].text = h
        for chk in r.checks:
            cells = table.add_row().cells
            cells[0].text = chk.name
            cells[1].text = "" if chk.statistic is None else fmt.num(chk.statistic, 3)
            cells[2].text = "" if chk.p is None else fmt.p(chk.p)
            cells[3].text = (chk.note if chk.passed is None
                             else ("passed" if chk.passed else "not met")
                             + (f" — {chk.note}" if chk.note else ""))

    # 5 Notes & advisories
    _notes(doc, r)

    # 6 Methods
    doc.add_heading("Methods", level=1)
    doc.add_paragraph(_methods_line(r, spec))

    # Appendices — dropped rows / unparsed cells (only when dataset/bound given)
    _appendices(doc, dataset, bound)

    return _save(doc)


def _notes(doc, r):
    findings = [f for f in r.findings if f.text]
    if not findings:
        return
    doc.add_heading("Notes & advisories", level=1)
    for f in findings:
        para = doc.add_paragraph(style="List Bullet")
        para.add_run(f.text)
        if f.suggest_test:
            para.add_run(f"  (suggested test: {f.suggest_test})")


def _save(doc) -> io.BytesIO:
    buf = io.BytesIO()
    doc.save(buf)                     # BytesIO target -> in memory (L3-safe)
    buf.seek(0)
    return buf
