"""Deployment-readiness gate: the artefacts Streamlit Community Cloud reads are
coherent with the tested environment and cannot smuggle a package past the L3
import allowlist.

* requirements.txt (runtime) -- every line is an exact ``name==version`` pin;
  every pinned distribution is either a direct dependency (the distribution
  behind one of the THIRD_PARTY_SAFE import roots in test_no_ai_no_io.py) or a
  transitive requirement of one, computed from installed package metadata. So a
  distribution nothing we ship depends on (openai, pytest, ...) fails BY
  OMISSION -- a rule, not a ban-list -- and every pin equals the version
  installed here, so the suite certifies exactly what Cloud will install.
  (One-directional on purpose: pins must be inside the closure; the closure may
  hold platform-conditional extras such as ``watchdog`` on Linux that the macOS
  freeze omitted.)
* requirements-dev.txt -- layers on the runtime file and is where pytest lives.
* .gitignore -- virtualenvs, caches, macOS metadata and .streamlit/secrets.toml
  can never be committed; and no secrets file exists to leak.
"""
import importlib.metadata as md
from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from test_no_ai_no_io import THIRD_PARTY_SAFE

ROOT = Path(__file__).resolve().parent.parent
REQ = ROOT / "requirements.txt"
REQ_DEV = ROOT / "requirements-dev.txt"
GITIGNORE = ROOT / ".gitignore"
SECRETS = ROOT / ".streamlit" / "secrets.toml"


def _pins(text: str):
    """(canonical name, version) per non-comment line. Raises ValueError on
    anything that is not an exact ``name==version`` pin: ranges, bare names,
    ``-r``/``-e``/URLs, extras, wildcards, environment markers."""
    out = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name, sep, ver = line.partition("==")
        if (not sep or not ver
                or any(ch in name for ch in " <>=!~[;@/")
                or any(ch in ver for ch in " ,;<>=!~*")):
            raise ValueError(f"not an exact name==version pin: {raw!r}")
        out.append((canonicalize_name(name), ver))
    return out


def _direct_distributions():
    """Distributions behind the import roots the L3 gate sanctions."""
    dist_of = md.packages_distributions()
    return {canonicalize_name(d) for root in THIRD_PARTY_SAFE for d in dist_of.get(root, ())}


def _closure(seeds):
    """Transitive Requires-Dist closure of ``seeds`` with markers evaluated for
    THIS interpreter/platform (extras excluded)."""
    seen, stack = set(), list(seeds)
    while stack:
        dist = stack.pop()
        if dist in seen:
            continue
        seen.add(dist)
        try:
            reqs = md.requires(dist) or ()
        except md.PackageNotFoundError:
            continue
        for spec in reqs:
            try:
                req = Requirement(spec)
            except InvalidRequirement:
                continue
            if req.marker and not req.marker.evaluate({"extra": ""}):
                continue
            stack.append(canonicalize_name(req.name))
    return seen


def test_scanner_sees_the_real_files():
    """Fail loudly, not vacuously, if a path is wrong."""
    assert REQ.exists() and REQ_DEV.exists() and GITIGNORE.exists()
    assert len(_pins(REQ.read_text(encoding="utf-8"))) >= 10


def test_runtime_pins_are_exact():
    _pins(REQ.read_text(encoding="utf-8"))  # raises on any loose line


@pytest.mark.parametrize("line", [
    "numpy>=2",                                       # range
    "pandas",                                         # unpinned
    "-r requirements-dev.txt",                        # include
    "scipy==1.18.*",                                  # wildcard
    "streamlit[snowflake]==1.64.0",                   # extra
    'watchdog==6.0.0; platform_system != "Darwin"',   # marker
])
def test_pin_parser_rejects_loose_lines(line):
    with pytest.raises(ValueError):
        _pins(line)


def test_every_runtime_pin_is_reachable_from_a_shipped_import():
    pins = dict(_pins(REQ.read_text(encoding="utf-8")))
    direct = _direct_distributions()
    assert direct, "no direct distributions resolved -- allowlist/venv mismatch"
    allowed = _closure(direct)
    smuggled = sorted(set(pins) - allowed)
    assert not smuggled, (
        "requirements.txt pins distribution(s) that nothing we ship imports or "
        f"depends on (AI/dev package smuggled past the import allowlist?): {smuggled}")
    unpinned = sorted(direct - set(pins))
    assert not unpinned, f"direct dependency not pinned in requirements.txt: {unpinned}"


@pytest.mark.parametrize("dist", ["pytest", "openai", "torch"])
def test_closure_excludes_dev_and_ai_distributions(dist):
    """Negative control: the closure must not reach the dev tool (installed
    here) or an AI library, or the test above would pass vacuously."""
    assert canonicalize_name(dist) not in _closure(_direct_distributions())


def test_runtime_pins_match_installed_versions():
    mismatches = []
    for name, ver in _pins(REQ.read_text(encoding="utf-8")):
        try:
            installed = md.version(name)
        except md.PackageNotFoundError:
            installed = "(not installed)"
        if installed != ver:
            mismatches.append((name, ver, installed))
    assert not mismatches, f"pin != installed (suite would certify untested versions): {mismatches}"


def test_dev_requirements_layer_on_runtime_and_own_pytest():
    lines = [l.split("#", 1)[0].strip() for l in REQ_DEV.read_text(encoding="utf-8").splitlines()]
    assert "-r requirements.txt" in lines
    assert any(l.startswith("pytest==") for l in lines)


@pytest.mark.parametrize("pattern", [
    ".venv/", ".venv-gen/", "__pycache__/", ".pytest_cache/",
    ".DS_Store", ".streamlit/secrets.toml",
])
def test_gitignore_excludes(pattern):
    lines = {l.strip() for l in GITIGNORE.read_text(encoding="utf-8").splitlines()}
    assert pattern in lines, f"{pattern!r} missing from .gitignore"


def test_no_secrets_file_exists():
    """StatKit needs no secrets; none may exist to be committed by accident."""
    assert not SECRETS.exists()
