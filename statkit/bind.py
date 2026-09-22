"""Dataset + Contract + column choices -> a canonical analysis frame (PLAN §6/§2).

``bind()`` is the one place layout normalisation and listwise deletion happen, so
the runners always receive a DataFrame whose columns are named by ROLE (D14: N
accounting travels with it). The Contract drives everything, so binding and the
sanity checks cannot drift from the menu (PLAN §2).

Three normalisers (PLAN §3 / §9.1 bind list):

  * **wide -> long** -- a test whose canonical layout is ``long`` given a ``wide``
    sheet (one numeric column per group/condition): stack the group columns into
    ``outcome`` + ``group`` (group label = the source header). Blank cells in a
    wide group column are LAYOUT PADDING (unequal group sizes), not dropped
    observations -- wide independent/paired sheets are physically identical
    (PLAN D13), so a blank there cannot be told from "this group is smaller".
  * **long -> wide** -- a test whose canonical layout is ``wide`` given a ``long``
    sheet (subject / condition / outcome): pivot to one column per condition
    (``before``/``after`` for a 2-condition paired test, else ``measures__i``).
    A subject missing any condition is dropped as an incomplete subject. For this
    pivoted layout ``Bound.columns`` also carries the CONDITION LABELS naming the
    derived wide columns (``before``/``after`` -> the two condition names, or
    ``measures`` -> the ordered condition names), since the user's ``columns`` name
    only subject/condition/outcome; the runners, charts and report read them back.
  * **table -> long** -- a count grid (row labels in the first column + category
    count columns) or a category+count table, expanded to one row per counted
    unit (``table_to_long`` / ``counts_to_long``); Total row/column dropped.

Two parked contracts nailed here (STATE.md Chunk-1 follow-ons):
  * **wilcoxon one-sample** -- the registry gives wilcoxon only wide (before/after)
    and long (subject/condition/outcome) layouts, but the PLAN §3 row also allows
    a one-sample mode "vs mu0 when bound from the t_1s suggestion". With no
    single-column layout in the contract to edit, bind recognises it structurally:
    a lone ``outcome`` column (no before/after/condition) + the ``mu0`` param ->
    a single canonical ``outcome`` column; the runner (Chunk 9) does
    ``wilcoxon(x - mu0)`` when there is no ``after`` column. SNAG, see report.
  * **table_to_long** -- the count-grid normaliser above.

Listwise deletion (D14): every dropped row is counted by reason and, where a real
1-based Excel row exists, recorded in ``dropped_rows`` (capped at 200).
"""
from __future__ import annotations

import math
import re

import pandas as pd

from . import levels
from .model import Bound, Contract, Finding, Kind, Role, TestSpec

_DROP_CAP = 200
_TOTAL_RE = ("total", "totals", "grand total", "sum", "all")


class BindError(ValueError):
    """A sheet cannot be bound as the chosen test expects (e.g. a count grid with
    fractional or negative cells). Carries a student-facing message."""


def _check_count(n, column, label) -> None:
    """A count cell must be a whole, non-negative, finite number. Percentages
    (33.3) and negatives are a wrong-column / wrong-shape mistake, not data to
    silently round -- fail closed with a message that names the offending cell."""
    f = float(n)
    if not math.isfinite(f) or f < 0 or abs(f - round(f)) > 1e-9:
        raise BindError(
            f"Count cells must be whole, non-negative numbers: found {n!r} in "
            f"column '{column}' (row label {label!r}). If these are percentages, "
            "upload the raw counts.")


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _profiles_by_name(dataset) -> dict:
    return {p.name: p for p in dataset.profiles}


def _effective_kind(role: Role, profile) -> Kind:
    """First kind in role.accepts that the column can be (PLAN §2 rule a)."""
    if profile is not None:
        for k in role.accepts:
            if k in profile.kinds:
                return k
    return role.accepts[0]


def _role_by_name(contract: Contract, name: str) -> Role | None:
    for roles in contract.roles_by_layout.values():
        for r in roles:
            if r.name == name:
                return r
    return None


def _is_total(label) -> bool:
    return isinstance(label, str) and label.strip().casefold() in _TOTAL_RE


def _notna(v) -> bool:
    return not pd.isna(v)


# --------------------------------------------------------------------------
# NB2 coding choke: a role column USED numerically but STORED as text is turned
# into numbers HERE, once, gated on the role's EFFECTIVE kind -- so a rank/means
# runner's ``to_numpy(float)`` never meets a raw Likert label ("Strongly agree")
# or a stray junk cell. A CATEGORICAL-effective column keeps its labels
# (chi2/fisher/mcnemar/cochran must), and an already-numeric column is untouched.
# S-I: a BINARY-effective column in a NUMERIC single-column role (correlation/
# regression x/y) is point-biserial coded 0/1 (success=1) -- categorical roles,
# whose effective kind is never BINARY, are unaffected.
# S-N: when the column cannot be coded (a BINARY role column with != 2 levels, or
# an ORDINAL column with no known order) the choke returns the column UNCHANGED and
# a BLOCK finding, never a blanked all-NaN column -- so n_used stays honest and the
# block (not a phantom "0 usable rows") is what disables the Run.
# S-Q: a role that reads its values AS LABELS (accepts CATEGORICAL: describe / chi2)
# keeps an unordered ordinal as text rather than blocking it -- only a numeric-only
# ordinal role (a rank test) needs the order, so only it can raise ORDORDER.
# --------------------------------------------------------------------------
def _is_textual(series) -> bool:
    """A column stored as text (object/string), not a numeric/datetime dtype. An
    ordinal already numeric-coded (1/2/3, float64) is NOT textual and is left."""
    return not pd.api.types.is_numeric_dtype(series)


def _code_ordinal(series, profile, header):
    """Map an ORDINAL-on-text column to its 1-based rank in ``profile.level_order``
    (a value in no level -> NaN), returning (coded_series, Finding). The order is
    stored casefolded, so match on the casefolded display string; the note names
    the VERBATIM labels (``profile.levels``) in ordinal order.

    S-J(bind): when the column is ORDINAL but its order is UNKNOWN (no vocab, e.g.
    an override on an unrecognised scale), leave the column unchanged and BLOCK
    with a real reason -- otherwise the text reaches the rank runner's
    ``to_numpy(float)`` and raises ``could not convert string to float``."""
    order = tuple(getattr(profile, "level_order", None) or ())
    if not order:
        k = len(profile.levels)
        # S-N(bind): no order to rank against -> leave the column UNCHANGED (return
        # None, not a blanked all-NaN column) so listwise deletion keeps every row
        # and n_used stays honest. The BLOCK stops the rank runner before its
        # to_numpy(float) ever meets the raw text (a block disables the Run).
        return None, Finding(
            "block",
            f"I don't know the order of the levels in '{header}' "
            f"({', '.join(profile.levels)}). Recode them as numbers in your sheet "
            f"(1 = lowest … {k} = highest), or pick a numeric column.",
            code="ORDORDER")
    rank = {lab.casefold(): i + 1 for i, lab in enumerate(order)}
    coded = [rank.get(levels.display(v).strip().casefold()) for v in series]
    new = pd.array([float(c) if c is not None else None for c in coded],
                   dtype="float64")
    verbatim = {lab.casefold(): lab for lab in (profile.levels or ())}
    ordered = [verbatim.get(cf, cf) for cf in order]
    pairs = ", ".join(f"'{lab}'={i + 1}" for i, lab in enumerate(ordered))
    note = Finding("info", f"Ordinal labels coded 1..{len(ordered)}: {pairs}.",
                   code="ORDCODE")
    return new, note


def _code_series(series, effective_kind: Kind, profile, role=None, header=""):
    """Transform one role column by its effective kind, or leave it. Returns
    ``(new_series_or_None, Finding_or_None)``; ``None`` series means unchanged."""
    if not _is_textual(series):                  # already numeric (incl coded ordinal)
        return None, None
    # S-I: a Yes/No TEXT column used in a NUMERIC single-column role (a correlation
    # x/y, a regression predictor) is point-biserial coded 0/1 here, so the runner's
    # to_numpy(float) never meets 'No'. Skipped for categorical roles (chi2/fisher
    # row/col, prop/logistic outcome), whose effective kind is never BINARY.
    if (effective_kind is Kind.BINARY and _is_textual(series) and role is not None
            and role.accepts[0] is Kind.NUMERIC and role.max == 1):
        # S-N(bind): a yes/no (point-biserial) coding needs EXACTLY two levels. A
        # 3+-level column (an override, or a mis-typed scale) has no 0/1 mapping --
        # BLOCK naming the levels rather than silently dropping the surplus one(s)
        # to NaN (which would shrink n and pool two distinct levels onto 0).
        uniq = levels._unique(series)
        if len(uniq) != 2:
            shown = ", ".join(levels.display(v) for v in uniq)
            return None, Finding(
                "block",
                f"'{header}' has {len(uniq)} levels ({shown}); a yes/no coding "
                "needs exactly two. Pick a two-level column, or change its kind "
                "under 'Column kinds'.", code="BINLEVELS")
        success, _ = levels.pick_success(series)
        other = next(v for v in uniq if v != success)
        coded = [1.0 if (_notna(v) and v == success)
                 else 0.0 if (_notna(v) and v == other)
                 else None for v in series]
        note = Finding("info", f"'{header}' coded {levels.display(success)}=1, "
                       f"{levels.display(other)}=0 (point-biserial).", code="BINCODE")
        return pd.array(coded, dtype="float64"), note
    if effective_kind is Kind.ORDINAL:
        # S-Q(bind): a label-reading role (describe / chi2, whose accepts include
        # CATEGORICAL) can use an UNORDERED ordinal AS LABELS -- don't force an
        # ORDORDER block or code it; keep the text so it can be counted. Only a
        # numeric-only ordinal role (a rank test) falls through to _code_ordinal.
        order = tuple(getattr(profile, "level_order", None) or ())
        if not order and role is not None and Kind.CATEGORICAL in role.accepts:
            return None, None
        return _code_ordinal(series, profile, header)
    if effective_kind is Kind.NUMERIC:           # mixed numeric+junk: junk -> NaN
        return pd.to_numeric(series, errors="coerce"), None
    return None, None                            # CATEGORICAL/BINARY/... keep labels


def _code_frame(df, columns, roles, contract, profs):
    """Code every role column that needs it (ORDINAL/NUMERIC on text), returning
    ``(df, findings)``. ``df`` is copied only when a column actually changes, so a
    sheet with nothing to code is untouched. The header->profile->effective-kind
    is resolved exactly as ``bind`` resolves ``kinds``."""
    new_cols: dict = {}
    findings: list[Finding] = []
    for role_name, headers in columns.items():
        r = roles.get(role_name) or _role_by_name(contract, role_name)
        if r is None:
            continue
        for h in headers:
            prof = profs.get(h)
            coded, note = _code_series(df[h], _effective_kind(r, prof), prof, r, h)
            if coded is not None:
                new_cols[h] = coded
            if note is not None:
                findings.append(note)
    if new_cols:
        df = df.assign(**new_cols)
    return df, tuple(findings)


# --------------------------------------------------------------------------
# table normalisers (PLAN §3: expanded by table_to_long)
# --------------------------------------------------------------------------
def table_to_long(df, counts, row_label_col=None, drop_totals=True):
    """A count GRID -> long (row, col): one row per counted unit.

    ``counts`` = the column-category count headers; ``row_label_col`` = the header
    of the row-label column (default: the first column not in ``counts``). A Total
    column (already excluded from ``counts``) and a Total row (row label in the
    total vocabulary) are dropped when ``drop_totals``.
    """
    counts = list(counts)
    if row_label_col is None:
        row_label_col = next(c for c in df.columns if c not in counts)
    rows, cols = [], []
    for _, r in df.iterrows():
        label = r[row_label_col]
        if drop_totals and _is_total(label):
            continue
        for c in counts:
            n = r[c]
            if pd.isna(n):
                continue
            _check_count(n, c, label)
            k = int(round(float(n)))
            rows.extend([label] * k)
            cols.extend([c] * k)
    return pd.DataFrame({"row": rows, "col": cols})


def counts_to_long(df, category_col, count_col, drop_totals=True):
    """A category+count table -> long (category): one row per counted unit."""
    out = []
    for _, r in df.iterrows():
        label = r[category_col]
        if drop_totals and _is_total(label):
            continue
        n = r[count_col]
        if pd.isna(n):
            continue
        _check_count(n, count_col, label)
        out.extend([label] * int(round(float(n))))
    return pd.DataFrame({"category": out})


# --------------------------------------------------------------------------
# the public entry point
# --------------------------------------------------------------------------
def bind(spec: TestSpec, dataset, columns: dict, layout=None,
         params: dict | None = None) -> Bound:
    """Produce the canonical Bound for ``spec`` from ``dataset`` + role->columns."""
    params = dict(params or {})
    contract = spec.contract
    profs = _profiles_by_name(dataset)
    excel = list(dataset.excel_rows)
    df = dataset.df

    # wilcoxon one-sample: a lone outcome column + mu0, no formal layout.
    if spec.id == "wilcoxon" and set(columns) == {"outcome"}:
        return _one_sample(spec, df, columns, profs, excel, params)

    layout = layout or _infer_layout(contract, columns)
    roles = {r.name: r for r in contract.roles_by_layout[layout]}
    kinds = {}
    for role_name, headers in columns.items():
        r = roles.get(role_name) or _role_by_name(contract, role_name)
        if r is not None and headers:
            kinds[role_name] = _effective_kind(r, profs.get(headers[0]))

    # NB2: code text columns used numerically BEFORE the builders select them, so
    # the canonical frame is already numeric where a runner needs it to be.
    df, code_findings = _code_frame(df, columns, roles, contract, profs)

    canonical = contract.canonical
    derived: dict = {}
    bind_findings: tuple[Finding, ...] = ()
    if layout == canonical == "long":
        # describe summarises each variable over its OWN non-missing values (the
        # runner does .dropna() per column), and corr_matrix reports pairwise-
        # complete correlations (the runner masks per pair) -- neither may
        # cross-column listwise-delete, or a row missing one variable would drop
        # from every other variable too (PLAN snags 8.3 / S5).
        data, n_total, n_used, dropped, drows = _long_passthrough(
            df, columns, roles, excel,
            listwise=(spec.id not in ("describe", "corr_matrix")))
    elif layout == canonical == "wide":
        data, n_total, n_used, dropped, drows = _wide_passthrough(
            df, columns, roles, excel)
    elif layout == "wide" and canonical == "long":
        data, n_total, n_used, dropped, drows = _wide_to_long(df, columns, excel)
    elif layout == "long" and canonical == "wide":
        data, n_total, n_used, dropped, drows, derived, bind_findings = _long_to_wide(
            spec, df, columns, profs, excel)
    elif layout == "table":
        data, n_total, n_used, dropped, drows = _table_bind(
            spec, df, columns, excel)
    else:  # pragma: no cover - a contract we don't yet normalise
        raise ValueError(
            f"{spec.id}: no binder for layout={layout!r} canonical={canonical!r}")

    # For a pivoted (long -> wide) layout the user's columns name subject/condition
    # /outcome; `derived` carries the condition labels naming the derived wide
    # columns (before/after or measures), which the runners/charts read back.
    out_columns = {k: tuple(v) for k, v in columns.items()}
    out_columns.update(derived)
    return Bound(
        test=spec, layout=layout, columns=out_columns,
        kinds=kinds, params=params, data=data,
        n_total=n_total, n_used=n_used, dropped=dropped,
        dropped_rows=tuple(drows[:_DROP_CAP]),
        findings=code_findings + bind_findings,
    )


def _infer_layout(contract: Contract, columns: dict):
    """Pick the layout whose role names are all present in ``columns``."""
    provided = set(columns)
    best = None
    for lay, roles in contract.roles_by_layout.items():
        names = {r.name for r in roles}
        # a provided role must belong to the layout; every required role present.
        required = {r.name for r in roles if r.min >= 1}
        if provided <= names and required <= provided:
            if lay == contract.canonical:
                return lay
            best = best or lay
    return best or contract.canonical


# --------------------------------------------------------------------------
# canonical builders
# --------------------------------------------------------------------------
def _select(df, headers):
    return [df[h] for h in headers]


def _long_passthrough(df, columns, roles, excel, listwise=True):
    """Canonical long, given long: role columns renamed, row-wise listwise.

    ``listwise=False`` (describe) keeps every row -- a per-variable summary must
    not drop a row from all variables because one variable is missing (snag 8.3).
    """
    canon = {}
    for role_name, headers in columns.items():
        r = roles.get(role_name)
        if r is not None and (r.max is None or r.max > 1) and len(headers) != 1:
            for i, h in enumerate(headers):
                canon[f"{role_name}__{i}"] = df[h].to_numpy()
        elif r is not None and (r.max is None or r.max > 1):
            canon[f"{role_name}__0"] = df[headers[0]].to_numpy()
        else:
            canon[role_name] = df[headers[0]].to_numpy()
    frame = pd.DataFrame(canon)
    if not listwise:
        n = len(frame)
        return frame.reset_index(drop=True), n, n, {}, []
    return _row_listwise(frame, excel, "missing value")


def _wide_passthrough(df, columns, roles, excel):
    """Canonical wide, given wide: before/after or measures[], pairwise-complete."""
    canon = {}
    for role_name, headers in columns.items():
        r = roles.get(role_name)
        if r is not None and (r.max is None or r.max > 1):
            for i, h in enumerate(headers):
                canon[f"{role_name}__{i}"] = df[h].to_numpy()
        else:
            canon[role_name] = df[headers[0]].to_numpy()
    frame = pd.DataFrame(canon)
    return _row_listwise(frame, excel, "incomplete subject")


def _wide_to_long(df, columns, excel):
    """wide -> long: stack the group columns; blanks are layout padding."""
    headers = list(columns["groups"])
    out_vals, grp_vals, ex = [], [], []
    for h in headers:
        col = df[h]
        for i, v in enumerate(col):
            if pd.isna(v):
                continue
            out_vals.append(v)
            grp_vals.append(h)
            ex.append(excel[i] if i < len(excel) else i + 2)
    frame = pd.DataFrame({"outcome": out_vals, "group": grp_vals})
    n = len(frame)
    return frame, n, n, {}, []


def _condition_order(profile, series):
    if profile is not None and profile.levels:
        order = [lv for lv in profile.levels]
        extra = [v for v in dict.fromkeys(series.dropna().tolist()) if v not in order]
        return order + extra
    return list(dict.fromkeys(series.dropna().tolist()))


def _long_to_wide(spec, df, columns, profs, excel):
    """long -> wide: pivot subject x condition -> before/after or measures__i."""
    subj_h = columns["subject"][0]
    cond_h = columns["condition"][0]
    out_h = columns["outcome"][0]
    work = pd.DataFrame({
        "subject": df[subj_h].to_numpy(),
        "condition": df[cond_h].to_numpy(),
        "outcome": df[out_h].to_numpy(),
        "excel": excel[:len(df)],
    })
    order = _condition_order(profs.get(cond_h), work["condition"])
    wide_roles = spec.contract.roles_by_layout["wide"]
    two = {r.name for r in wide_roles} == {"before", "after"} and len(order) == 2
    names = ["before", "after"] if two else [f"measures__{i}" for i in range(len(order))]

    rows, dropped, drows = [], {}, []
    n_total = 0
    dup = 0
    for subj, g in work.groupby("subject", sort=False):
        n_total += 1
        vals = {}
        for lv in order:
            sel = g[g["condition"] == lv]["outcome"]
            if len(sel) > 1:                    # duplicate subject x condition rows:
                dup += len(sel) - 1             # first kept below, the rest ignored
            v = sel.iloc[0] if len(sel) and _notna(sel.iloc[0]) else None
            vals[lv] = v
        if any(v is None for v in vals.values()):
            dropped["incomplete subject"] = dropped.get("incomplete subject", 0) + 1
            er = int(g["excel"].iloc[0])
            drows.append((er, "incomplete subject"))
            continue
        rows.append([vals[lv] for lv in order])
    frame = pd.DataFrame(rows, columns=names)
    findings = () if not dup else (Finding(
        "flag", f"{dup} duplicate measurement(s) ignored (first value kept per "
        "subject×condition).", code="DUP"),)
    # Name the derived wide columns by their condition labels (B6): a 2-condition
    # paired test gets before/after; a k-measure test gets the ordered measures.
    # S-B(a): render the condition labels through levels.display so a numeric-coded
    # condition (Time = 1/2) reads "1"/"2", not "1.0"/"2.0", downstream.
    if two:
        derived = {"before": (levels.display(order[0]),),
                   "after": (levels.display(order[1]),)}
    else:
        derived = {"measures": tuple(levels.display(lv) for lv in order)}
    return frame, n_total, len(frame), dropped, drows, derived, findings


def _sum_counts(df, count_col, label_col):
    """Total of a count column over non-Total rows (no expansion)."""
    total = 0
    for _, r in df.iterrows():
        if _is_total(r[label_col]) or pd.isna(r[count_col]):
            continue
        _check_count(r[count_col], count_col, r[label_col])
        total += int(round(float(r[count_col])))
    return total


# S15 threshold: never materialise more rows than this; check.S15 then blocks.
_EXPAND_CAP = 1_000_000


def _table_bind(spec, df, columns, excel):
    """table -> long: count grid (row,col) or category+count (category).

    The count total is summed first; a grid over ``_EXPAND_CAP`` is NOT expanded
    (that would blow up memory) -- an empty frame is returned with n_used = total
    so check.S15 can block it.
    """
    if "category" in columns:                       # chi2_gof
        cat, cnt = columns["category"][0], columns["counts"][0]
        total = _sum_counts(df, cnt, label_col=cat)
        if total > _EXPAND_CAP:
            return pd.DataFrame({"category": []}), total, total, {}, []
        frame = counts_to_long(df, cat, cnt)
    else:                                            # chi2_ind / fisher grid
        counts = list(columns["counts"])
        row_label = next(c for c in df.columns if c not in counts)
        total = sum(_sum_counts(df, c, label_col=row_label) for c in counts)
        if total > _EXPAND_CAP:
            return pd.DataFrame({"row": [], "col": []}), total, total, {}, []
        frame = table_to_long(df, counts=counts)
    n = len(frame)
    return frame, n, n, {}, []


def _one_sample(spec, df, columns, profs, excel, params):
    """wilcoxon one-sample: a single outcome column vs mu0 (canonical: outcome)."""
    h = columns["outcome"][0]
    role = _role_by_name(spec.contract, "outcome")
    prof = profs.get(h)
    ek = _effective_kind(role, prof) if role else Kind.NUMERIC
    kinds = {"outcome": ek} if role else {}
    coded, note = _code_series(df[h], ek, prof, role, h)      # NB2: text -> numbers
    series = coded if coded is not None else df[h].to_numpy()
    frame = pd.DataFrame({"outcome": series})
    data, n_total, n_used, dropped, drows = _row_listwise(
        frame, excel, "missing value")
    params.setdefault("mu0", 0.0)
    return Bound(
        test=spec, layout="long", columns={"outcome": tuple(columns["outcome"])},
        kinds=kinds, params=params, data=data,
        n_total=n_total, n_used=n_used, dropped=dropped,
        dropped_rows=tuple(drows[:_DROP_CAP]),
        findings=(note,) if note is not None else (),
    )


# --------------------------------------------------------------------------
# Contract-driven satisfiability + greedy auto-binding.
#
# ONE definition of "which columns can fill a role" / "can this spec bind this
# sheet" / "what is the minimal contract-satisfying column assignment", shared by
# the Streamlit menu-greying + column pickers (app.py) AND the end-to-end corpus
# gate (tests/test_e2e.py) so they cannot drift (PLAN §2). None of this imports
# streamlit, so it lives in the package under the zero-streamlit gate.
# --------------------------------------------------------------------------
_HIDDEN_KINDS = (Kind.EMPTY, Kind.IGNORE)


def _levels_count(dataset, name: str) -> int:
    """Distinct usable levels of a column: the profile's when set, else the actual
    distinct count (a numeric-default column carries no levels)."""
    for p in dataset.profiles:
        if p.name == name:
            if p.n_levels:
                return p.n_levels
            break
    try:
        return int(dataset.df[name].dropna().nunique())
    except Exception:  # pragma: no cover - defensive
        return 0


# S-P(bind): a Total / % / percentage column is a SUMMARY of the counts, not a
# category count column -- offering it to the count-grid role double-counts the
# table (and a percent column is not even a count).
# NB5: match WHOLE WORDS ('_' read as a space), never substrings -- a Likert
# answer column 'Totally agree' is a real count column. 'sum' / 'all' stay
# exact-match via _TOTAL_RE.
# R1: CamelCase is split at each lower->upper seam BEFORE the word match, so a
# glued 'GrandTotal' / 'RowTotal' reads 'grand total' / 'row total' (a summary)
# while 'TotallyAgree' reads 'totally agree' (a real count column).
# R2: the two kinds of summary have different SCOPE -- a percent-like header is a
# RATE, never a count, so it is skipped for EVERY counts role (chi2_gof's single
# count column included); a Total-like header only double-counts the multi-
# column GRID (chi2_ind / fisher: ``max=None``) -- chi2_gof's single count column
# may legitimately be called 'Total'.
# R3: 'percent' / 'percentage(s)' are whole words -- 'Percentile' is a real column.
_PERCENT_RE = re.compile(r"%|\bpct\b|\bpercent(age)?s?\b")
_TOTAL_WORD_RE = re.compile(r"\b(sub)?totals?\b")


def _is_summary_header(name, role: Role) -> bool:
    words = (re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(name))     # R1: GrandTotal -> Grand Total
             .strip().casefold().replace("_", " "))
    if _PERCENT_RE.search(words) is not None:                   # R2: every counts role
        return True
    return role.max is None and (                               # S-P/NB5: the GRID only
        words in _TOTAL_RE or _TOTAL_WORD_RE.search(words) is not None)


def role_columns(role: Role, dataset) -> list[str]:
    """Columns whose kinds are compatible with ``role`` (and, for a level-bounded
    grouping role, whose distinct-level count is in range)."""
    lo, hi = role.levels
    bounded = role.levels != (1, None)
    out: list[str] = []
    for p in dataset.profiles:
        if p.kinds[0] in _HIDDEN_KINDS:
            continue
        if not any(k in p.kinds for k in role.accepts):
            continue
        if role.name == "counts" and _is_summary_header(p.name, role):
            continue
        if bounded:
            nl = _levels_count(dataset, p.name)
            if nl < lo or (hi is not None and nl > hi):
                continue
        out.append(p.name)
    return out


def role_reason(role: Role, dataset) -> str:
    """The greyed-out reason when ``role`` cannot be filled (app menu help)."""
    kinds = "/".join(k.value for k in role.accepts)
    lo, hi = role.levels
    if role.levels != (1, None):
        rng = f"{lo}–{hi}" if hi is not None else f"{lo}+"
        return f"needs a {kinds} column with {rng} groups; your sheet has none"
    if role.min > 1:
        return f"needs {role.min} {kinds} columns; your sheet has fewer"
    # S-K(b): don't lie "your sheet has none" when the sheet DOES hold numeric-
    # looking columns that were typed ID (all-distinct); name them and say how to fix.
    if Kind.NUMERIC in role.accepts and not role_columns(role, dataset):
        ids = [p.name for p in dataset.profiles
               if p.kinds == (Kind.ID,) and p.name in dataset.df.columns
               and pd.api.types.is_numeric_dtype(dataset.df[p.name])]
        if ids:
            return (f"{', '.join(repr(n) for n in ids)} were read as ID columns "
                    "(identifiers, not measurements); if they are measurements, "
                    "change their kind to Numeric under 'Column kinds'.")
    return f"needs a {kinds} column; your sheet has none"


def _feasible(required, opts) -> bool:
    """Can each required role get its MINIMUM distinct column(s)? Greedy, scarcest
    first -- this is the app's menu-greying test, unchanged (min-based)."""
    used: set = set()
    for r in sorted(required, key=lambda r: len(opts[r.name])):
        need = max(r.min, 1)
        avail = [c for c in opts[r.name] if c not in used]
        if len(avail) < need:
            return False
        used.update(avail[:need])
    return True


def _assemble(required, opts) -> dict | None:
    """The role->columns assignment a sensible user would pick: a single-column
    role gets one column; a MULTI-column role (max None or >1) gets ALL eligible
    columns up to its max (a count grid needs every count column, a matrix every
    variable). Scarcer roles first, so a greedy multi role never starves a scarcer
    single role. Returns None if some role cannot reach its minimum."""
    used: set = set()
    assign: dict = {}
    for r in sorted(required, key=lambda r: len(opts[r.name])):
        avail = [c for c in opts[r.name] if c not in used]
        if len(avail) < max(r.min, 1):        # N12: fail closed, never a partial ()
            return None
        multi = r.max is None or r.max > 1
        if multi:
            want = len(avail) if r.max is None else min(r.max, len(avail))
        else:
            want = 1
        if want < max(r.min, 1):
            return None
        take = tuple(avail[:want])
        assign[r.name] = take
        used.update(take)
    return assign


def _level_set(dataset, name: str) -> frozenset:
    for p in dataset.profiles:
        if p.name == name and p.levels:
            return frozenset(p.levels)
    try:
        return frozenset(str(v) for v in dataset.df[name].dropna().unique())
    except Exception:  # pragma: no cover - defensive
        return frozenset()


def _restrict_same_level_set(opts: dict, dataset) -> dict:
    """McNemar / Cochran (``same_level_set``) need columns that SHARE one outcome
    level set (S7). Keep only the columns of the level set shared by the most
    candidates -- a real paired/repeated variable (Yes/No across visits), not an
    unrelated binary column (Attended = TRUE/FALSE) that happens to be 2-level."""
    cands = {c for cols in opts.values() for c in cols}
    if not cands:
        return opts
    groups: dict = {}
    for c in cands:
        groups.setdefault(_level_set(dataset, c), []).append(c)
    keep = set(max(groups.values(), key=len))
    return {role: [c for c in cols if c in keep] for role, cols in opts.items()}


def auto_columns(spec: TestSpec, dataset, layout: str) -> dict | None:
    """A contract-satisfying role->columns assignment for ``layout`` (multi roles
    take all eligible columns), or None if the sheet cannot bind it. Optional roles
    (min 0) are left unbound."""
    roles = spec.contract.roles_by_layout[layout]
    opts = {r.name: role_columns(r, dataset) for r in roles}
    if spec.contract.same_level_set:
        opts = _restrict_same_level_set(opts, dataset)
    required = [r for r in roles if r.min >= 1]
    return _assemble(required, opts)


def satisfies_layout(spec: TestSpec, dataset, layout: str) -> tuple[bool, str]:
    roles = spec.contract.roles_by_layout[layout]
    opts = {r.name: role_columns(r, dataset) for r in roles}
    required = [r for r in roles if r.min >= 1]
    for r in required:
        if len(opts[r.name]) < max(r.min, 1):
            return False, role_reason(r, dataset)
    if not _feasible(required, opts):
        return False, "not enough distinct compatible columns for every role"
    return True, ""


def satisfiable_layouts(spec: TestSpec, dataset) -> list[str]:
    return [lay for lay in spec.contract.roles_by_layout
            if satisfies_layout(spec, dataset, lay)[0]]


def satisfies(spec: TestSpec, dataset) -> tuple[bool, str]:
    reasons = []
    for layout in spec.contract.roles_by_layout:
        ok, why = satisfies_layout(spec, dataset, layout)
        if ok:
            return True, ""
        reasons.append(why)
    return False, reasons[0] if reasons else "cannot bind to this sheet"


# --------------------------------------------------------------------------
# row-wise listwise deletion shared by the passthrough builders
# --------------------------------------------------------------------------
def _row_listwise(frame, excel, reason):
    n_total = len(frame)
    keep_mask = frame.notna().all(axis=1)
    dropped, drows = {}, []
    for i, keep in enumerate(keep_mask):
        if not keep:
            dropped[reason] = dropped.get(reason, 0) + 1
            er = excel[i] if i < len(excel) else i + 2
            drows.append((int(er), reason))
    kept = frame[keep_mask].reset_index(drop=True)
    return kept, n_total, len(kept), dropped, drows
