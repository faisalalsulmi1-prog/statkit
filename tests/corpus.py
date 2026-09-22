"""Corpus loaders for the committed fixture set (tests/fixtures/generated/).

Chunk 5's test_corpus.py imports these to iterate every synthetic fixture, read
its expect.json oracle, and hand its bytes to statkit.grid.load. This module
lives in tests/ and is NOT part of the shipped package, so it may use
json / pathlib / tempfile / importlib freely -- the L3 no-io gate scans only
statkit/ + app.py, never the test tree.

The 60k-row large-sheet case (fixture id "w2_2_build") has no committed data
file: it is built on demand from gen_w2_2_build.py and cached in-process.
"""
from __future__ import annotations

import functools
import importlib.util
import json
import os
import tempfile
from pathlib import Path

GENERATED = Path(__file__).resolve().parent / "fixtures" / "generated"

# Data-file extensions a committed fixture may use, in resolution order.
_DATA_EXTS = ("xlsx", "xlsm", "xls", "csv", "txt")


def fixture_ids() -> list[str]:
    """Every fixture id with a gen_<id>.expect.json sidecar, sorted.

    Excludes the dropped/ subdirectory (glob is non-recursive). Includes the
    builder id "w2_2_build".
    """
    return [
        p.name[len("gen_"):-len(".expect.json")]
        for p in sorted(GENERATED.glob("gen_*.expect.json"))
    ]


def expect(fixture_id: str) -> dict:
    """The parsed expect.json oracle (authored from the real bytes)."""
    return json.loads((GENERATED / f"gen_{fixture_id}.expect.json").read_text())


def truth(fixture_id: str) -> dict:
    """The generator's self-reported truth.json (counts unreliable -- prefer expect)."""
    return json.loads((GENERATED / f"gen_{fixture_id}.truth.json").read_text())


def fixture_path(fixture_id: str) -> Path | None:
    """The committed data file for a fixture, or None for a builder-only fixture."""
    for ext in _DATA_EXTS:
        p = GENERATED / f"gen_{fixture_id}.{ext}"
        if p.exists():
            return p
    return None


def fixture_bytes(fixture_id: str) -> tuple[bytes, str]:
    """(bytes, filename) ready to hand to statkit.grid.load.

    Reads the committed file, or builds the 60k-row workbook on demand (cached)
    for the builder fixture.
    """
    p = fixture_path(fixture_id)
    if p is not None:
        return p.read_bytes(), p.name
    if (GENERATED / f"gen_{fixture_id}.py").exists():
        return _build(fixture_id), f"gen_{fixture_id}.xlsx"
    raise FileNotFoundError(f"no data file or builder for fixture {fixture_id!r}")


def load_grids(fixture_id: str):
    """Load a fixture through the real reader (statkit.grid.load)."""
    from statkit import grid

    data, name = fixture_bytes(fixture_id)
    return grid.load(data, name)


@functools.lru_cache(maxsize=None)
def _build(fixture_id: str) -> bytes:
    """Run gen_<id>.py's build_large_sheet into a temp file; return its bytes."""
    mod = _import_builder(fixture_id)
    fd, tmp = tempfile.mkstemp(suffix=".xlsx")
    os.close(fd)
    try:
        mod.build_large_sheet(tmp)
        return Path(tmp).read_bytes()
    finally:
        os.remove(tmp)


def _import_builder(fixture_id: str):
    path = GENERATED / f"gen_{fixture_id}.py"
    spec = importlib.util.spec_from_file_location(f"gen_{fixture_id}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
