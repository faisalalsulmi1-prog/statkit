#!/usr/bin/env python3
"""Integrity gate for the StatKit fixture corpus.

Run with the generation venv (has openpyxl + xlrd):
    /Users/faisal/statkit/.venv-gen/bin/python validate_fixtures.py [DIR]

It does NOT judge whether a fixture is a *good* test case (that is Fable's job).
It checks that:
  1. every spreadsheet / text / legacy-xls fixture loads (openpyxl, csv, xlrd);
  2. the builder .py is present (its big xlsx is built on demand, never committed);
  3. each truth.json sidecar parses and its basic counts are self-consistent;
  4. each expect.json oracle parses, uses the PLAN Kind enum + S/D codes, and its
     header_row / n_rows CROSS-CHECK against a fresh re-read of the real bytes.

Tolerant of schema drift from weak models: every problem is reported, nothing
crashes the run. Optional DIR arg lets it run against a copied tree (used for the
RED-first self-test).
"""
import csv
import json
import re
import sys
from pathlib import Path

DEFAULT_GEN = Path("/Users/faisal/statkit/tests/fixtures/generated")

# PLAN 4.4 Kind enum (lower-case string values) and S/D codes (PLAN 5).
KINDS = {"numeric", "ordinal", "categorical", "binary", "id", "date", "empty", "ignore"}
CODE_RE = re.compile(r"^[SD]\d{1,2}$")
SHEET_XT = (".xlsx", ".xlsm", ".xls", ".csv", ".txt")

try:
    import openpyxl
except ImportError:
    print("openpyxl missing - run with /Users/faisal/statkit/.venv-gen/bin/python")
    sys.exit(2)
try:
    import xlrd
except ImportError:
    xlrd = None


# ---------------- loaders (return {sheet_name: [rows]} of raw values) ----------------
def load_xlsx(path):
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    out = {}
    for ws in wb.worksheets:
        out[ws.title] = [list(r) for r in ws.iter_rows(values_only=True)]
    wb.close()
    return out


def load_xls(path):
    if xlrd is None:
        raise RuntimeError("xlrd not installed")
    wb = xlrd.open_workbook(path)
    out = {}
    for sh in wb.sheets():
        out[sh.name] = [[sh.cell_value(r, c) for c in range(sh.ncols)] for r in range(sh.nrows)]
    return out


def load_delim(path):
    raw = path.read_bytes()
    bom = raw[:3] == b"\xef\xbb\xbf"
    txt = None
    for enc in ("utf-8", "cp1252"):
        try:
            txt = raw.decode("utf-8-sig") if bom else raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if txt is None:
        raise ValueError("undecodable text file")
    try:
        delim = csv.Sniffer().sniff(txt[:4096], delimiters=",;\t|").delimiter
    except Exception:
        delim = ","
    grid = [row for row in csv.reader(txt.splitlines(), delimiter=delim)]
    return {"(single)": grid}


def load_any(path):
    """Return (kind, {sheet: grid})."""
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xlsm"):
        return "xlsx", load_xlsx(path)
    if ext == ".xls":
        return "xls", load_xls(path)
    if ext == ".csv":
        return "csv", load_delim(path)
    if ext == ".txt":
        return "txt", load_delim(path)
    raise ValueError(f"unhandled extension {ext}")


def nonblank(row):
    return any(c is not None and str(c).strip() != "" for c in row)


def main():
    gen = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GEN
    if not gen.exists():
        print(f"no generated dir: {gen}")
        return 0

    sheets = sorted(p for p in gen.iterdir() if p.suffix.lower() in SHEET_XT)
    builders = sorted(gen.glob("*.py"))
    truths = sorted(gen.glob("*.truth.json"))
    expects = sorted(gen.glob("*.expect.json"))

    print("=== StatKit fixture integrity gate ===")
    print(f"dir: {gen}")
    print(f"spreadsheets: {len(sheets)}  builders: {len(builders)}  "
          f"truth: {len(truths)}  expect: {len(expects)}\n")

    problems = []

    # 1. every spreadsheet / txt / xls loads -> keep grids for cross-checks
    grids = {}
    for p in sheets:
        try:
            kind, sh = load_any(p)
            grids[p.name] = (kind, sh)
        except Exception as e:
            problems.append(f"[UNREADABLE] {p.name}: {type(e).__name__}: {e}")

    # 2. builder .py present
    builder_names = {b.name for b in builders}

    # helper: resolve an expect/truth 'file' to a loadable grid dict (or None if builder)
    def resolve(fname):
        if fname.endswith(".py"):
            return "builder", None
        if fname in grids:
            return grids[fname][0], grids[fname][1]
        return None, None

    # 3. truth.json sidecars parse + basic self-consistency
    referenced = set()
    for t in truths:
        try:
            d = json.loads(t.read_text())
        except Exception as e:
            problems.append(f"[BAD JSON] {t.name}: {e}")
            continue
        fname = d.get("file")
        if not fname:
            problems.append(f"[NO file FIELD] {t.name}")
            continue
        referenced.add(fname)
        kind, shdict = resolve(fname)
        if kind is None:
            if fname.endswith(".py"):
                pass
            else:
                problems.append(f"[MISSING FILE] {t.name} references '{fname}' not present/loadable")
                continue
        if kind == "builder":
            if fname not in builder_names:
                problems.append(f"[MISSING BUILDER] {t.name} references builder '{fname}' not present")
            continue
        # max rows over the workbook (loose bound for the truth cross-check)
        maxrows = max((len(g) for g in shdict.values()), default=0)
        for s in d.get("sheets", []) or []:
            hr = s.get("header_row")
            nd = s.get("n_data_rows")
            if isinstance(hr, int) and hr > maxrows:
                problems.append(f"[HEADER OOR] {t.name}/{s.get('name')}: header_row {hr} > file rows {maxrows}")
            if isinstance(hr, int) and isinstance(nd, int) and (hr + nd) > maxrows + 2:
                problems.append(f"[ROWS OVERFLOW] {t.name}/{s.get('name')}: header {hr}+{nd} > {maxrows} rows")
            if s.get("is_data") and not s.get("columns"):
                problems.append(f"[NO COLUMNS] {t.name}/{s.get('name')} (is_data)")

    # 4. expect.json oracles: parse, enum-check, and header_row/n_rows CROSS-CHECK on bytes
    expected_files = set()
    for e in expects:
        try:
            d = json.loads(e.read_text())
        except Exception as ex:
            problems.append(f"[BAD JSON] {e.name}: {ex}")
            continue
        for req in ("file", "fixture_id", "sheets"):
            if req not in d:
                problems.append(f"[EXPECT NO {req}] {e.name}")
        fname = d.get("file")
        if not fname:
            continue
        expected_files.add(fname)
        kind, shdict = resolve(fname)
        is_builder = fname.endswith(".py")
        if kind is None and not is_builder:
            problems.append(f"[EXPECT MISSING FILE] {e.name} references '{fname}' not present/loadable")
            continue
        if is_builder and fname not in builder_names:
            problems.append(f"[EXPECT MISSING BUILDER] {e.name} references '{fname}' not present")

        for s in d.get("sheets", []) or []:
            sname = s.get("name")
            tag = f"{e.name}/{sname}"
            # enum checks on columns
            for col in s.get("columns", []) or []:
                k = col.get("kind")
                if k not in KINDS:
                    problems.append(f"[BAD KIND] {tag}/{col.get('name')}: kind {k!r} not in Kind enum")
                for kk in col.get("kinds", []) or []:
                    if kk not in KINDS:
                        problems.append(f"[BAD KIND] {tag}/{col.get('name')}: kinds member {kk!r} not in Kind enum")
            eb = s.get("expect_block", {}) or {}
            for chk in eb.get("checks_expected", []) or []:
                code = chk.get("code") if isinstance(chk, dict) else chk
                if not (isinstance(code, str) and CODE_RE.match(code)):
                    problems.append(f"[BAD CODE] {tag}: checks_expected code {code!r} not S#/D#")
            # n_cols consistency
            cols = s.get("columns", []) or []
            if isinstance(s.get("n_cols"), int) and s["n_cols"] != len(cols):
                problems.append(f"[N_COLS] {tag}: n_cols {s['n_cols']} != {len(cols)} columns")
            if s.get("role") == "data" and not cols:
                problems.append(f"[EXPECT NO COLUMNS] {tag} (role=data)")

            # header_row / n_rows cross-check against real bytes: DATA sheets only
            # (empty/non_data/info aux sheets carry no analysable table to cross-check).
            if s.get("role") != "data" or is_builder or shdict is None:
                continue
            if sname in shdict:
                grid = shdict[sname]
            elif len(shdict) == 1:
                grid = next(iter(shdict.values()))
            else:
                problems.append(f"[EXPECT SHEET NOT FOUND] {tag}: no sheet named {sname!r} in {fname}")
                continue
            maxrow = len(grid)
            for field in ("header_row", "header_row2", "units_row", "data_start_row"):
                v = s.get(field)
                if isinstance(v, int) and not (1 <= v <= maxrow):
                    problems.append(f"[ROW OOR] {tag}: {field} {v} outside 1..{maxrow}")
            ds = s.get("data_start_row")
            nrows = s.get("n_rows")
            if not (isinstance(ds, int) and isinstance(nrows, int)):
                problems.append(f"[EXPECT MISSING n_rows/data_start] {tag}")
                continue
            span = grid[ds - 1: maxrow]
            nb = sum(1 for r in span if nonblank(r))
            dropped = s.get("dropped_rows", []) or []
            dropped_nb = 0
            for dr in dropped:
                rn = dr.get("row") if isinstance(dr, dict) else dr
                if isinstance(rn, int) and rn >= ds and rn - 1 < maxrow and nonblank(grid[rn - 1]):
                    dropped_nb += 1
            recomputed = nb - dropped_nb
            if recomputed != nrows:
                problems.append(f"[N_ROWS MISMATCH] {tag}: expect n_rows={nrows} but bytes give "
                                f"{recomputed} (nonblank {nb} in rows {ds}..{maxrow} minus {dropped_nb} dropped non-blank)")

    # 5. coverage: every fixture (spreadsheet + txt + builder) has an expect.json
    all_fixtures = {p.name for p in sheets} | builder_names
    for name in sorted(all_fixtures):
        if name not in expected_files:
            problems.append(f"[NO EXPECT] {name}: no expect.json authored")
    # orphan spreadsheets with no truth sidecar
    for name in grids:
        if name not in referenced:
            problems.append(f"[ORPHAN SHEET] {name}: no truth sidecar references it")

    print(f"loadable spreadsheets: {len(grids)}/{len(sheets)}")
    print(f"expect.json authored: {len(expected_files)}/{len(all_fixtures)} fixtures\n")
    if problems:
        print(f"--- {len(problems)} problem(s) ---")
        for pb in problems:
            print("  " + pb)
    else:
        print("no integrity problems - all fixtures loadable; every expect.json parses and cross-checks.")
    print(f"\nSUMMARY sheets={len(sheets)} builders={len(builders)} loadable={len(grids)} "
          f"truth={len(truths)} expect={len(expects)} problems={len(problems)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
