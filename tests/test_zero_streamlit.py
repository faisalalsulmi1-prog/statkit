"""Architectural gate: the `statkit/` package imports ZERO Streamlit.

The package must be a pure library, unit-testable without a Streamlit runtime;
Streamlit belongs only in the top-level app.py (PLAN §2). This AST-scans every
`*.py` under statkit/ for any import whose root module is `streamlit` -- so it
covers `import streamlit`, `import streamlit as st`, `from streamlit import x`,
and `from streamlit.foo import y`, and a NEW package module is scanned
automatically. Dynamic-import evasion is separately impossible: the L3 gate
(test_no_ai_no_io.py) forbids `importlib` inside the package.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "statkit"


def _package_files():
    return sorted(PACKAGE.rglob("*.py"))


def _streamlit_imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits = []
    rel = path.relative_to(ROOT)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] == "streamlit":
                    hits.append(f"{rel}:{node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".", 1)[0] == "streamlit":
                hits.append(f"{rel}:{node.lineno}: from {node.module} import ...")
    return hits


def test_package_files_were_scanned():
    """Fail loudly rather than pass vacuously if the package path is wrong."""
    files = _package_files()
    assert PACKAGE.is_dir(), f"package dir missing: {PACKAGE}"
    assert any(f.name == "__init__.py" for f in files), "statkit/__init__.py not found"


def test_statkit_package_has_no_streamlit_imports():
    violations = []
    for path in _package_files():
        violations.extend(_streamlit_imports(path))
    assert not violations, (
        "statkit/ must not import streamlit (it belongs only in app.py):\n  "
        + "\n  ".join(violations)
    )
