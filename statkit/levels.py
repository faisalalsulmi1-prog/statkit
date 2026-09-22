"""Shared level-matching for outcome columns (batch 1B; B3 / S12 plumbing).

One place that turns a stored outcome value into the string a student sees,
lists a column's levels, resolves a user's chosen level (a display string from a
dropdown) back to the actual stored value, and picks the "success" level for the
proportion / logistic runners. ``props.py``, ``regress.py`` and ``advise.py``
all route through here, so a numeric 0/1 outcome (stored as floats) and a
"1"/"0" dropdown choice (strings) can never drift apart -- the B3 bug, where the
chosen level was compared with ``==`` against a value of a different type and
silently missed, so the runner reported the wrong proportion or mislabelled a
failed fit as "separation".

Pure: only ``fmt``, ``pandas``/``numpy`` and ``model.Finding``. No I/O, no state.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import fmt
from .model import Finding

# Success/event tokens (moved here from props.py / advise.py). Matched against a
# level's DISPLAY string, so "1.0" is not needed -- display(1.0) is already "1".
POSITIVE = frozenset(
    {"yes", "true", "1", "positive", "success", "present", "y", "t"})


def _isna(v) -> bool:
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):        # pragma: no cover - non-scalar guard
        return False


def display(v) -> str:
    """A stored level value as the string a student sees.

    An int-valued float prints as an int ('1', not '1.0'); a **fractional** float
    prints losslessly and distinctly ('0.125', not the 2-dp-rounded '0.12'; and
    0.001 != 0.002 rather than both collapsing to '0'), so two genuinely distinct
    levels can never share a label and merge downstream (the S-D bug). ``repr`` on
    a Python float is the shortest string that round-trips, so it is exact yet free
    of binary noise like '0.30000000000000004'. A missing value is ''; a string is
    verbatim; ints, int-valued floats and non-finite values keep ``fmt.num``."""
    if _isna(v):
        return ""
    if isinstance(v, (int, float, np.number)):
        if isinstance(v, (float, np.floating)):
            x = float(v)                       # normalise np.float64 -> Python float
            if math.isfinite(x) and not x.is_integer():
                return repr(x)                 # lossless, distinct, noise-free
        return fmt.num(v)                       # int / int-valued float / non-finite
    return str(v)


def display_index(index):
    """Relabel a pandas ``Index`` (or a plain list of labels) through ``display``.

    Returns a new ``pd.Index`` when given one, otherwise a list -- so a downstream
    caller can losslessly relabel numeric-coded crosstab rows/columns or chart
    ticks without every site re-implementing the int-vs-fractional rule."""
    labels = [display(v) for v in index]
    return pd.Index(labels) if isinstance(index, pd.Index) else labels


def _unique(series) -> list:
    """First-seen distinct non-null values of a Series."""
    return list(dict.fromkeys(series.dropna().tolist()))


def display_levels(series, profile=None) -> list[str]:
    """The column's levels as display strings: the profile's levels when it
    carries them (already display labels), else the first-seen unique non-null
    values rendered by ``display``."""
    if profile is not None and getattr(profile, "levels", None):
        return list(profile.levels)
    return [display(v) for v in _unique(series)]


def resolve(chosen, series):
    """The actual stored value whose ``display`` equals ``chosen`` (a display
    string), or that equals ``chosen`` directly; ``None`` if no level matches."""
    target = str(chosen).strip()
    for v in _unique(series):
        if display(v) == target or v == chosen:
            return v
    return None


def pick_success(series, chosen=None):
    """Pick the success/event level of ``series``.

    Returns ``(value, None)`` on success or ``(None, Finding)`` when ``chosen``
    is given but present in no level. With ``chosen`` given, resolve it against
    the actual values; with ``chosen`` None, the first level whose display is a
    positive token (yes/true/1/...), else the last level in string-sorted order.
    """
    if chosen is not None:
        v = resolve(chosen, series)
        if v is None:
            lv = ", ".join(display_levels(series))
            return None, Finding(
                "block",
                f"The chosen level '{chosen}' is not present in the outcome "
                f"column after cleaning (levels: {lv}).", code="LEVEL")
        return v, None
    uniq = _unique(series)
    if not uniq:
        return None, None
    for v in uniq:
        if display(v).strip().casefold() in POSITIVE:
            return v, None
    return sorted(uniq, key=str)[-1], None
