"""L3 hard gate: the shipped tool has NO AI/LLM and does NO network or disk I/O.

WHY AN ALLOWLIST, NOT A BANLIST
-------------------------------
A finite ban-list of "bad" module names passes vacuously: the day someone
imports an AI library that isn't on the list, the build stays green and a
guarantee we advertised is silently broken. (Hard-won lesson: "a finite banned
list passes vacuously" -- grepping for six bad names misses the seventh.) So the
import check here is a RULE, not a roster of villains: every top-level import in
`statkit/` and `app.py` must name a module we have explicitly sanctioned as
pure-compute-and-safe. Anything else -- openai, anthropic, cohere, torch,
transformers, a library invented next year, OR a stdlib I/O module such as
socket / urllib / http / subprocess / pickle / sqlite3 -- fails by omission.
The denominator is "every module that exists"; the allowlist is the small,
audited set we vouch for.

WHAT THE CALL-SCAN IS (a drift guard, NOT a sandbox)
----------------------------------------------------
The import allowlist above is the real disk/network guard: it closes off every
NEW capability by omission. The call-scan does a different, narrower job -- it
enumerates the I/O verbs the ALLOWED libraries themselves already expose, and
requires their file targets to be in-memory buffers:
  * the builtin ``open(...)``                      (open a file on disk)
  * ``io.open`` / ``io.FileIO`` / ``io.open_code`` (disk I/O via an allowed module)
  * pandas ``read_*`` / ``to_*`` / ``*_clipboard`` (read or serialize external data)
  * ``savefig`` / ``.save`` / ``load_workbook`` /
    ``open_workbook`` / ``Document(...)``          (matplotlib / python-docx /
        openpyxl / xlrd: a Name or BytesIO/StringIO target is in-memory and
        ALLOWED; a string, f-string, concatenation, attribute or subscript path
        writes/reads disk and FAILS)
  * ``np.savetxt`` / ``plt.imsave`` / ``urlopen`` /
    ``get_rdataset`` / ``st.connection`` ...       (network / dataset / disk verbs)
  * ``st.cache_data`` / ``st.cache_resource``      (Streamlit persistence; L3
        says uploads live only in session_state)
  * ``__import__`` / ``exec`` / ``eval`` / ``compile`` and ``sys.modules`` /
    ``sys.path``                                   (dynamic code / import machinery
        that could sidestep the static import allowlist)

This scan is a DRIFT GUARD FOR MAINTAINERS, not a sandbox against a hostile
contributor. It catches the plausible ways an honest edit re-introduces I/O
through a library we ship; it does NOT claim to be complete against someone
deliberately obfuscating (aliasing a builtin, hiding a call behind getattr,
building a verb name at runtime). The security guarantee that closes those
holes is the import allowlist -- e.g. dynamic-import evasion
(``importlib.import_module("open"+"ai")``) is stopped because ``importlib``
is not allowlisted, not because this scan enumerates it.

SCOPE
-----
Every ``*.py`` under ``statkit/`` plus the top-level ``app.py``. The ``tests/``
tree is not shipped and is not scanned (these test files themselves use ast /
pathlib, which are forbidden *inside the package* but fine here). Streamlit is
allowed by this gate because ``app.py`` hosts it; keeping Streamlit OUT of the
package is the separate job of ``test_zero_streamlit.py``.
"""
import ast
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
STREAMLIT_CONFIG = ROOT / ".streamlit" / "config.toml"
PACKAGE = ROOT / "statkit"
APP = ROOT / "app.py"

# --- the allowlist -------------------------------------------------------
# Pure-compute stdlib modules that do no network/disk I/O just by being used.
# Deliberately EXCLUDES os, pathlib, socket, ssl, urllib, http, ftplib, smtplib,
# subprocess, asyncio, selectors, pickle, marshal, shelve, dbm, sqlite3,
# tempfile, shutil, mmap, ctypes, importlib, json, zoneinfo, gzip, zipfile,
# tarfile -- adding any of those back is a deliberate L3 decision, not an edit.
STDLIB_SAFE = frozenset({
    "__future__", "abc", "dataclasses", "enum", "typing", "typing_extensions",
    "collections", "functools", "itertools", "operator", "copy", "types",
    "keyword", "math", "cmath", "statistics", "decimal", "fractions", "numbers",
    "random", "re", "string", "textwrap", "unicodedata", "datetime", "calendar",
    "io", "csv", "contextlib", "warnings", "sys", "bisect", "heapq", "array",
})
# Third-party analysis libraries this project is pinned to (see requirements.txt).
# None of these is an AI/LLM library. `streamlit` is allowed only because app.py
# is scanned here too; test_zero_streamlit.py confines it to app.py.
THIRD_PARTY_SAFE = frozenset({
    "pandas", "numpy", "scipy", "statsmodels", "matplotlib", "mpl_toolkits",
    "openpyxl", "xlrd", "docx", "patsy", "dateutil", "streamlit",
})
ALLOWED_ROOTS = STDLIB_SAFE | THIRD_PARTY_SAFE | {"statkit"}

# --- forbidden calls (complete *given* the import allowlist above) --------
PANDAS_WRITERS = frozenset({
    "to_csv", "to_excel", "to_pickle", "to_parquet", "to_feather", "to_hdf",
    "to_sql", "to_stata", "to_orc", "to_xml", "to_gbq", "to_hdf5",
})
# NOTE (2026-09-21 session fix): a name-based DISK_MUTATORS set was removed here.
# Every function it listed (os.system/os.remove/os.rename/os.replace, pathlib
# mkdir/unlink/chmod, shutil ...) is reachable ONLY by importing os / pathlib /
# shutil / subprocess -- all already forbidden by the import allowlist above. So
# the set added ZERO real protection but DID false-positive on ubiquitous
# in-memory methods that share those names: `list.remove`, `str.replace`,
# `pandas.DataFrame.replace` / `.rename`. Matching by bare `.attr(...)` cannot
# tell a filesystem call from a pure-compute one; the import allowlist is the
# real disk/network guard and keeps every such capability out of reach.
STREAMLIT_CACHE = frozenset({"cache_data", "cache_resource"})

# --- L3 hardening (batch 1H) ---------------------------------------------
# These rules do NOT replace the import allowlist (the real disk/network guard);
# they enumerate the I/O verbs the ALLOWED libraries themselves expose, so a
# maintainer who reaches for one through pandas / matplotlib / openpyxl / io is
# caught. This is a DRIFT GUARD, not a sandbox against a hostile contributor.

# Rule 1: dynamic code / import primitives that evade the static import allowlist.
DYNAMIC_EXEC = frozenset({"__import__", "exec", "eval", "compile"})
# Rule 2: io.<attr> that reaches disk (io.BytesIO/StringIO are NOT here -> allowed).
IO_DISK_ATTRS = frozenset({"open", "FileIO", "open_code"})
# Rule 2: attribute calls that read/fetch/persist regardless of their arguments.
NETWORK_DATASET_ATTRS = frozenset({
    "connection", "experimental_connection",   # st.connection(...) DB access
    "get_rdataset", "download_all",             # statsmodels / nltk remote fetch
    "urlopen",                                  # urllib network read
    "imsave", "savetxt",                        # matplotlib / numpy disk writers
})
# Rule 2: pandas readers and clipboard bridges (external data in/out).
PANDAS_READERS = frozenset({
    "read_csv", "read_table", "read_fwf", "read_excel", "read_json", "read_html",
    "read_xml", "read_parquet", "read_pickle", "read_sql", "read_sql_query",
    "read_sql_table", "read_clipboard", "to_clipboard",
})
# Rule 4: verbs whose file argument must be an in-memory buffer, not a disk path.
PATH_VERBS = frozenset({
    "open", "savefig", "save", "load_workbook", "open_workbook", "Document",
})
PATH_KWARGS = frozenset({"fname", "filename", "path", "path_or_buf", "file"})
INMEM_BUFFERS = frozenset({"BytesIO", "StringIO"})


def _iter_py_files():
    files = sorted(PACKAGE.rglob("*.py"))
    if APP.exists():
        files.append(APP)
    return files


def _root_of(dotted: str) -> str:
    return dotted.split(".", 1)[0]


def _call_name(func) -> str | None:
    """The bare callable name of a Call's func, whether ``foo(...)`` (Name) or
    ``x.foo(...)`` (Attribute). None for anything more exotic."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _path_arg(call: ast.Call):
    """The file argument of a path-taking verb: the first positional, else a
    keyword named fname/filename/path/path_or_buf/file. None => the call passes
    no path at all (``Document()``, ``open_workbook(file_contents=...)``)."""
    if call.args:
        return call.args[0]
    for kw in call.keywords:
        if kw.arg in PATH_KWARGS:
            return kw.value
    return None


def _target_is_in_memory(arg) -> bool:
    """A path target is safe ONLY if it is a local ``Name`` (a buffer variable
    like ``buf``) or a direct ``BytesIO()``/``StringIO()`` construction. A
    Constant, JoinedStr (f-string), BinOp (concatenation), Attribute, Subscript,
    or a Call to any other function may resolve to a disk path and FAILS."""
    if isinstance(arg, ast.Name):
        return True
    if isinstance(arg, ast.Call):
        return _call_name(arg.func) in INMEM_BUFFERS
    return False


def _scan_source(src: str, rel: str):
    """Return (import_violations, call_violations) for one module's source.

    Split out from ``_scan`` so each rule can be exercised on a snippet in a
    unit test (RED-first) without needing a file on disk. ``rel`` is only the
    label that appears in a violation string.
    """
    tree = ast.parse(src, filename=rel)
    imports, calls = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = _root_of(alias.name)
                if root not in ALLOWED_ROOTS:
                    imports.append(f"{rel}:{node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # relative import -> internal package, allowed
            root = _root_of(node.module or "")
            if root not in ALLOWED_ROOTS:
                imports.append(f"{rel}:{node.lineno}: from {node.module} import ...")
        elif isinstance(node, ast.Attribute):
            # Rule 3: touching the import machinery / sys.path ANYWHERE in the
            # tree (not only in a call) reaches forbidden modules or rewrites the
            # search path. `sys` is allowlisted for sys.argv/version/stderr, so
            # this narrows the ban to exactly the two dangerous attributes.
            if isinstance(node.value, ast.Name) and node.value.id == "sys" and node.attr in ("modules", "path"):
                calls.append(f"{rel}:{node.lineno}: sys.{node.attr} (import machinery / path manipulation)")
        elif isinstance(node, ast.Call):
            func = node.func
            name = _call_name(func)                       # bare verb, Name or Attribute form
            is_name = isinstance(func, ast.Name)
            is_attr = isinstance(func, ast.Attribute)
            is_io = is_attr and isinstance(func.value, ast.Name) and func.value.id == "io"
            if is_name and name == "open":
                # Builtin open() -- unconditional: even open(var) may be a disk path.
                calls.append(f"{rel}:{node.lineno}: builtin open()")
            elif is_name and name in DYNAMIC_EXEC:                         # rule 1
                calls.append(f"{rel}:{node.lineno}: {name}(...) dynamic code/import evades the allowlist")
            elif is_attr and name in PANDAS_WRITERS:
                calls.append(f"{rel}:{node.lineno}: .{name}(...) writes to disk")
            elif is_attr and name in PANDAS_READERS:                      # rule 2
                calls.append(f"{rel}:{node.lineno}: .{name}(...) reads external data")
            elif is_attr and name in STREAMLIT_CACHE:
                calls.append(f"{rel}:{node.lineno}: st.{name}(...) persists data")
            elif is_io and name in IO_DISK_ATTRS:                         # rule 2
                calls.append(f"{rel}:{node.lineno}: io.{name}(...) opens a file on disk")
            elif is_attr and name in NETWORK_DATASET_ATTRS:               # rule 2
                calls.append(f"{rel}:{node.lineno}: .{name}(...) network / dataset / disk access")
            elif name in PATH_VERBS:                                      # rule 4
                arg = _path_arg(node)
                if arg is not None and not _target_is_in_memory(arg):
                    calls.append(f"{rel}:{node.lineno}: {name}(<non-buffer path>) may touch disk")
    return imports, calls


def _scan(path: Path):
    """Thin wrapper: read a shipped file and scan its source."""
    rel = str(path.relative_to(ROOT))
    return _scan_source(path.read_text(encoding="utf-8"), rel)


def test_scanner_sees_the_shipped_code():
    """Guard against a path bug making the whole gate pass vacuously.

    (Hard-won lesson: infrastructure failure looks like a quality result -- a
    gate that silently scans zero files is green and worthless.)
    """
    files = _iter_py_files()
    names = {f.name for f in files}
    assert PACKAGE.is_dir(), f"package dir missing: {PACKAGE}"
    assert "__init__.py" in names, "statkit/__init__.py was not scanned"
    assert "app.py" in names, "app.py was not scanned"


def test_no_forbidden_imports():
    violations = []
    for path in _iter_py_files():
        imports, _ = _scan(path)
        violations.extend(imports)
    assert not violations, (
        "Non-allowlisted import(s) in shipped code (AI/LLM or network/disk "
        "capability). If one is genuinely pure-compute and safe, add its root "
        "to the allowlist in this file as a deliberate L3 decision:\n  "
        + "\n  ".join(violations)
    )


def test_no_io_or_persistence_calls():
    violations = []
    for path in _iter_py_files():
        _, calls = _scan(path)
        violations.extend(calls)
    assert not violations, (
        "Disk-write / persistence call(s) in shipped code (L3 forbids touching "
        "disk; use io.BytesIO in memory):\n  " + "\n  ".join(violations)
    )


# --- L3 hardening (batch 1H): the call-scan is a drift guard, and these are the
# concrete evasions it must catch. Each snippet reaches disk/network or evades
# the import allowlist THROUGH AN ALLOWED LIBRARY, so the import check alone
# cannot see it. (`fig.savefig(p)` -- a bare Name target -- is deliberately NOT
# here: a variable buffer is the legitimate in-memory idiom the shipped code
# uses; the f-string and string-literal forms are the real evasions.)
_KNOWN_EVASIONS = [
    'fig.savefig("out.png")',            # string path -> disk
    'fig.savefig(f"{n}.png")',           # f-string path -> disk (dodged old str-only check)
    'doc.save("x.docx")',                # python-docx string path -> disk
    'io.open("x")',                      # builtin open under an allowed module
    'io.FileIO("x")',                    # raw file object
    '__import__("os")',                  # dynamic import evades the allowlist
    'sys.modules["os"]',                 # reach a forbidden module via the loader
    'exec("import openai")',             # execute an arbitrary import string
    'pd.read_csv("https://x")',          # pandas reader = network/disk read
    'sm.datasets.get_rdataset("x")',     # statsmodels remote dataset fetch
    'openpyxl.load_workbook("x.xlsx")',  # workbook path -> disk read
    'st.connection("sql")',              # streamlit DB connection
    'eval("2+2")',                       # arbitrary expression evaluation
]


@pytest.mark.parametrize("src", _KNOWN_EVASIONS)
def test_gate_catches_known_evasions(src):
    """Every known evasion yields at least one violation from _scan_source."""
    imports, calls = _scan_source(src, "snippet.py")
    assert imports or calls, f"evasion slipped through the gate: {src!r}"


@pytest.mark.parametrize("src", [
    'fig.savefig(buf, format="png")',                       # matplotlib -> BytesIO
    'doc.save(buf)',                                        # python-docx -> BytesIO
    'openpyxl.load_workbook(BytesIO(data), read_only=True)',  # in-memory workbook
    'xlrd.open_workbook(file_contents=data)',              # bytes, no path positional
])
def test_gate_allows_shipped_idioms(src):
    """The in-memory idioms the shipped code relies on must stay clean.

    Guards the hardening against false positives that would flag charts.py,
    report.py and grid.py -- the shipped code is clean today and must remain so.
    """
    imports, calls = _scan_source(src, "snippet.py")
    assert not imports and not calls, f"false positive on shipped idiom: {src!r}\n{calls}"


def test_streamlit_config_disables_telemetry_and_caps_uploads():
    """The 'no tracking' promise is backed by a checked-in config, not prose."""
    assert STREAMLIT_CONFIG.exists(), f"missing {STREAMLIT_CONFIG}"
    cfg = tomllib.loads(STREAMLIT_CONFIG.read_text(encoding="utf-8"))
    assert cfg["browser"]["gatherUsageStats"] is False, "usage stats not disabled"
    assert cfg["server"]["maxUploadSize"] == 25, "upload size not capped at 25 MB"
