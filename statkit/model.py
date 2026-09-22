"""StatKit domain model — the ubiquitous language (PLAN §2).

The one place the vocabulary lives, so the menu, the column pickers, the sanity
checks, the runners, the sentences and the report all speak the same terms and
cannot drift. Only stdlib + pandas typing; NO streamlit (asserted by
tests/test_zero_streamlit.py).

Glossary (PLAN §2): Grid = raw sheet cells · Table = cleaned grid · Dataset =
typed DataFrame + ColumnProfiles · Kind = what a column can be used as · Role =
a slot a test needs filled · Contract = the roles/params/layouts a test accepts
· Layout = how the sheet is shaped · Bound = a test + columns + canonical data
· Finding = a block/flag/info message · Check = one assumption-test outcome ·
Result = everything a sentence, chart or report will ever need.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Literal

import pandas as pd


class Kind(str, Enum):
    """What a column can be *used as*. A column can carry several kinds; the
    effective one is decided per role by Role.accepts order (PLAN §2 rule a)."""
    NUMERIC = "numeric"
    ORDINAL = "ordinal"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    ID = "id"
    DATE = "date"
    EMPTY = "empty"
    IGNORE = "ignore"


Layout = Literal["long", "wide", "table", "one_sample"]
Severity = Literal["block", "flag", "info"]
Goal = Literal["describe", "compare", "relate", "counts", "predict", "assumptions"]
Pairing = Literal["independent", "paired", "any"]


@dataclass(frozen=True)
class ColumnProfile:
    name: str                          # header verbatim (deduplicated with " (2)")
    kinds: tuple[Kind, ...]            # possible kinds in PREFERENCE order (§4.4);
                                       # kinds[0] is the default kind (UI + reports);
                                       # membership is what role-binding intersects.
                                       # After an override this is a single-element tuple.
    n_total: int
    n_missing: int
    n_levels: int
    levels: tuple[str, ...]            # display labels (first-seen order, or level_order)
    level_order: tuple[str, ...] | None = None   # set only when ORDINAL
    unit: str | None = None            # "kg", "%", "$"
    n_censored: int = 0                # censored / non-detect cells (treated as missing)
    notes: tuple[str, ...] = ()        # human lines: "7 text cells treated as missing"
    failed_cells: tuple[tuple[int, str], ...] = ()   # (excel_row, raw_text), first 20


@dataclass(frozen=True)
class Role:
    name: str                          # outcome|group|subject|condition|x|y|variables|row|col|
                                       # predictors|factor_a|factor_b|measures|counts|category|before|after
    accepts: tuple[Kind, ...]          # PREFERENCE ORDER; effective kind = first in accepts ∩ profile.kinds
    label: str                         # student-facing
    min: int = 1
    max: int | None = 1                # None = unbounded
    levels: tuple[int, int | None] = (1, None)   # allowed distinct levels (categorical-ish roles)
    help: str = ""


@dataclass(frozen=True)
class Param:
    name: str
    kind: Literal["float", "int", "bool", "level", "proportions", "choice"]
    default: object
    label: str
    of_role: str | None = None         # "level"/"proportions" enumerate this role's levels
    choices: tuple[str, ...] = ()
    help: str = ""


@dataclass(frozen=True)
class Contract:
    roles_by_layout: dict[Layout, tuple[Role, ...]]   # which layouts are accepted, roles per layout
    canonical: Layout                                  # what bind() normalises to
    params: tuple[Param, ...] = ()
    pairing: Pairing = "independent"
    same_level_set: bool = False                       # McNemar / Cochran: all bound columns share one level set
    min_n: int = 3


@dataclass(frozen=True)
class Finding:
    severity: Severity
    text: str
    suggest_test: str | None = None
    suggest_params: tuple[tuple[str, object], ...] = ()
    code: str = ""                     # "S2", "D3" — for tests


@dataclass(frozen=True)
class Check:                           # one assumption test
    name: str                          # "Brown-Forsythe (Levene, center=median)"
    statistic: float | None
    p: float | None
    passed: bool | None                # None = not applicable
    note: str                          # "n=3: Shapiro not run"


@dataclass
class Bound:
    test: "TestSpec"
    layout: Layout
    columns: dict[str, tuple[str, ...]]   # role -> source headers verbatim
    kinds: dict[str, Kind]                # role -> effective kind
    params: dict[str, object]
    data: pd.DataFrame                    # CANONICAL: columns named by role; multi-column roles -> f"{role}__{i}"
    n_total: int
    n_used: int
    dropped: dict[str, int]               # reason -> count
    dropped_rows: tuple[tuple[int, str], ...]   # (excel_row, reason), first 200
    findings: tuple[Finding, ...] = ()    # from check()

    @property
    def blocked(self) -> bool:
        return any(f.severity == "block" for f in self.findings)


@dataclass
class Result:
    test_id: str
    test_name: str                        # EXACT variant: "Welch's independent-samples t-test"
    status: Literal["ok", "blocked"]
    alpha: float = 0.05
    statistic: tuple[str, float] | None = None      # ("t", 2.31)
    df: tuple[float, ...] = ()                       # () | (df,) | (df1, df2)
    p: float | None = None
    estimate: tuple[str, float] | None = None       # ("mean difference (A − B)", 3.2)
    estimate_ci: tuple[float, float] | None = None
    effect: tuple[str, float] | None = None         # ("Hedges' g", 0.62)
    effect_ci: tuple[float, float] | None = None
    effect_label: str | None = None                 # "medium"
    effect_source: str = ""                         # "Cohen (1988)"
    n: dict[str, int] = field(default_factory=dict)  # {"total","used","dropped","<level>":n,"pairs","nonzero_pairs"}
    labels: dict[str, str] = field(default_factory=dict)   # role -> header verbatim; "level_hi","level_lo"
    groups: tuple[str, ...] = ()                     # level labels in reporting order
    higher: str | None = None                        # group label whose values were higher (computed by runner)
    descriptives: pd.DataFrame | None = None         # Table 1
    table: pd.DataFrame | None = None                # coefficients / contingency / ANOVA table
    expected: pd.DataFrame | None = None             # χ² expected counts
    posthoc: pd.DataFrame | None = None
    posthoc_name: str | None = None
    checks: tuple[Check, ...] = ()
    findings: tuple[Finding, ...] = ()               # advise() + runner gating
    variant_notes: tuple[str, ...] = ()              # "exact method (n≤8, no ties)", "3 zero differences dropped"
    method: str = ""                                 # library call with args, for the methods line
    arrays: dict[str, object] = field(default_factory=dict)   # numpy arrays for charts ONLY
    extra: dict[str, object] = field(default_factory=dict)    # test-specific scalars (r2, aic, converged...)


@dataclass(frozen=True)
class TestSpec:
    id: str
    name: str
    goal: Goal
    pairing: Pairing
    contract: Contract
    run: Callable[[Bound], Result]
    sentence: Callable[[Result], str]
    chart: Callable[[Result], tuple[tuple[str, bytes], ...]]   # ((caption, png_bytes), ...)
    family: str                                                # F-2G ...
    library: str                                               # "scipy.stats.ttest_ind(equal_var=False, ...)"
    citation: str
    menu_help: str
    aliases: tuple[str, ...] = ()
