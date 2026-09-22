"""StatKit — the Streamlit wizard (PLAN §8, Chunk 14).

Thin ORCHESTRATION only: every statistic, every repair, every sentence and the
Word report live in the ``statkit`` package (which imports zero Streamlit). This
file is the ONE place Streamlit is allowed (tests/test_zero_streamlit.py), it
holds all state in ``st.session_state`` (no ``st.cache_*``, no disk — the L3 gate
tests/test_no_ai_no_io.py enforces this), and it reads the uploaded file from its
in-memory buffer only.

The wizard walks: upload -> pick sheet / header / column kinds -> state the goal
(and the paired-vs-independent question) -> pick a test from a Contract-driven
menu that greys out what cannot bind -> point at role-compatible columns -> run
-> read the numbers, the plain-English sentence, the chart(s) and the advisories
-> one-click switch to a suggested robust alternative -> download the report.

Menu greying and the column pickers both read the SAME per-test ``Contract`` (via
``_role_columns`` / ``_satisfies``), so the menu, the pickers and the checks can
never drift apart (PLAN §2).
"""
from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from statkit import (
    bind, charts, check, clean, coerce, fmt, grid, infer, levels, report,
    sentences,
)
from statkit.model import Kind
from statkit.registry import REGISTRY, SPECS

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

# Friendly names for the raw layout ids (S9): the student never sees "one_sample".
LAYOUT_LABELS = {
    "long": "Long (one row per observation)",
    "wide": "Wide (one column per group / condition)",
    "table": "A table of counts",
    "one_sample": "One sample vs a reference value",
}

# Number-format presets the student can force per column (S6). "(auto)" passes
# nothing (keeps detection); the rest are the (decimal, thousands) pairs the
# detector in coerce.py produces — the only distinctions that change a reading.
CONVENTION_LABELS = {
    "(auto)": None,
    "1,234.5  (thousands ',' · decimal '.')": coerce.Convention(decimal=".", thousands=","),
    "1.234,5  (thousands '.' · decimal ',')": coerce.Convention(decimal=",", thousands="."),
    "1 234,5  (thousands space · decimal ',')": coerce.Convention(decimal=",", thousands=" "),
}

# One-sample reference values that must be entered, never silently defaulted to a
# hypothesis the student did not choose (NICE): per test id -> its required param.
REQUIRED_FLOATS = {"t_1s": ("mu0",), "prop_1": ("p0",)}


def _required_floats(spec, layout):
    """Float params the student MUST enter for this spec + active layout, never
    silently defaulted to a hypothesis they didn't choose (NICE). The one-sample
    Wilcoxon (layout ``one_sample``) needs μ₀ exactly like the one-sample t-test;
    the paired Wilcoxon (wide/long) needs none."""
    if spec.id == "wilcoxon" and layout == "one_sample":
        return ("mu0",)
    return REQUIRED_FLOATS.get(spec.id, ())

# Wizard selections cleared when a NEW file is uploaded, so choices from the last
# sheet do not leak (NICE). Prefixes cover the per-column / per-role keys.
_RESET_PREFIXES = ("kind::", "col::", "cols::", "param::", "conv::", "merge::")
_RESET_KEYS = ("test_id", "layout", "header_row", "sheet", "_ran",
               "restore_rows", "_run_sig", "_suppress_stale_reset")

GOALS = ("describe", "compare", "relate", "counts", "predict", "assumptions")
GOAL_LABELS = {
    "describe": "Describe my data",
    "compare": "Compare groups / conditions",
    "relate": "Measure a relationship",
    "counts": "Analyse counts / categories",
    "predict": "Predict an outcome",
    "assumptions": "Check an assumption",
}
PAIRED_GOALS = ("compare", "counts")

# per-column kind override labels -> Kind (the "(auto)" default keeps inference).
KIND_LABELS = {
    "Numeric": Kind.NUMERIC, "Ordinal": Kind.ORDINAL, "Categorical": Kind.CATEGORICAL,
    "Binary": Kind.BINARY, "ID": Kind.ID, "Date": Kind.DATE, "Ignore": Kind.IGNORE,
}
OVERRIDE_OPTIONS = ("(auto)", *KIND_LABELS)


# ==========================================================================
# Contract-driven satisfiability + auto-binding live in statkit.bind, so the
# menu-greying, the column pickers and the end-to-end corpus gate share ONE
# definition and cannot drift (PLAN §2). Thin aliases keep this file readable.
# ==========================================================================
_role_columns = bind.role_columns
_satisfies = bind.satisfies
_sat_layouts = bind.satisfiable_layouts


def _column_levels(dataset, name: str) -> list[str]:
    """A column's levels as the DISPLAY strings a student sees (B3): a numeric
    0/1 level reads as '0'/'1', never '0.0'/'1.0'. The chosen string is passed
    straight through as the param; the runners resolve it back themselves."""
    prof = next((p for p in dataset.profiles if p.name == name), None)
    try:
        return levels.display_levels(dataset.df[name], prof)
    except Exception:  # pragma: no cover
        return list(prof.levels) if prof and prof.levels else []


# ==========================================================================
# small session helpers
# ==========================================================================
def _guard_options(key, options):
    """Drop a stale stored selection that is no longer a valid option."""
    cur = st.session_state.get(key)
    if isinstance(cur, list):
        st.session_state[key] = [c for c in cur if c in options]
    elif cur is not None and cur not in options:
        del st.session_state[key]


def _filled_roles() -> set:
    """Role names the student has already selected a column for (col::/cols:: keys
    with a real value), so a rebind can pick a layout those columns actually fill."""
    filled = set()
    for key, val in st.session_state.items():
        if key.startswith("col::") and val and val != "(none)":
            filled.add(key[len("col::"):])
        elif key.startswith("cols::") and val:
            filled.add(key[len("cols::"):])
    return filled


def _suggestion_layout(sug):
    """A layout for the suggested test that REUSES the columns already picked, so a
    rebind never lands on a layout whose role pickers are empty (the Wilcoxon-from-
    one-sample-t dead-end: its Wide before/after pickers were empty). Keep the
    current layout when the suggested test shares it and its required roles are
    filled; else the first of the suggested test's layouts whose required roles are
    all already filled; else None (leave the layout to the app's normal default)."""
    layouts = sug.contract.roles_by_layout
    filled = _filled_roles()

    def matches(lay):
        return {r.name for r in layouts[lay] if r.min >= 1} <= filled

    cur = st.session_state.get("layout")
    if cur in layouts and matches(cur):
        return cur
    return next((lay for lay in layouts if matches(lay)), None)


def _apply_suggestion(suggest_test, suggest_params):
    """Suggestion-rebind callback: switch the test and/or its params, re-run."""
    if suggest_test:
        st.session_state["test_id"] = suggest_test
        # The menu is filtered by pairing (app §5), so a suggested test with a
        # fixed pairing (e.g. Wilcoxon = paired, offered from the one-sample
        # t-test) must carry that pairing across, or it is filtered out and the
        # click dead-ends to "Choose a test above." An "any"-pairing suggestion
        # leaves the student's current answer untouched.
        sug = REGISTRY.get(suggest_test)
        if sug is not None and sug.pairing != "any":
            st.session_state["pairing"] = sug.pairing
        # Carry a valid layout too (S-A): a multi-layout suggested test whose
        # layout box would otherwise default to a layout the student's columns
        # don't fill (Wilcoxon -> Wide) dead-ends. Only set it when a layout that
        # reuses the current columns exists, else leave the default in place.
        if sug is not None:
            lay = _suggestion_layout(sug)
            if lay is not None:
                st.session_state["layout"] = lay
    for k, v in suggest_params:
        st.session_state[f"param::{k}"] = v
    st.session_state["_ran"] = True
    # the change below is intentional -> don't let the stale-result reset undo it.
    st.session_state["_suppress_stale_reset"] = True


def _reset_on_new_upload(filename, data):
    """Clear stale wizard selections when a genuinely new file is uploaded, so
    choices from the previous sheet don't leak (NICE). Keyed on name + bytes."""
    sig = (filename, len(data), hash(data))
    if st.session_state.get("_upload_sig") == sig:
        return
    for k in list(st.session_state.keys()):
        if k.startswith(_RESET_PREFIXES) or k in _RESET_KEYS:
            del st.session_state[k]
    st.session_state["_upload_sig"] = sig


def _proposed_merges(table) -> dict:
    """Per-column level-merge proposals from the RAW cleaned cells (S7).

    Computed off the raw cells (not the post-merge profile) so applying a merge
    can't erase its own control. Only columns with a real proposal appear."""
    out = {}
    for j, name in enumerate(table.header):
        vals = [row[j].value for row in table.rows
                if isinstance(row[j].value, str) and row[j].value.strip()]
        mapping = infer.propose_merges(vals) if vals else {}
        if mapping:
            out[name] = mapping
    return out


def _name_matches(spec, query: str) -> bool:
    """A test matches a search when the query is a substring of its name or any
    alias (case-insensitive) — so a student can find a test by the name they
    know (S8)."""
    q = query.lower()
    return any(q in h.lower() for h in (spec.name, *spec.aliases))


def _stale_reset(test_id, columns, params):
    """Clear a shown result when the chosen test / columns / params change, so a
    changed selection doesn't keep displaying the previous result (NICE)."""
    sig = repr((test_id, sorted(columns.items()), sorted(params.items())))
    if st.session_state.get("_run_sig") == sig:
        return
    st.session_state["_run_sig"] = sig
    if st.session_state.pop("_suppress_stale_reset", False):
        return                 # a suggestion just set _ran on purpose; keep it
    st.session_state["_ran"] = False


# ==========================================================================
# rendering
# ==========================================================================
def _render_numbers(r):
    lines = []
    if r.statistic:
        # Perfect-fit regression (Cohen's f² non-finite): r² has reached 1.0 so the
        # reported F is astronomical / infinite and NOT meaningful — render "—" and
        # a note, matching the report + sentence (W6). Every finite result is
        # byte-identical. Keyed on the f² effect specifically, exactly as the report
        # and sentence layers do, so a non-finite Cramér's V / odds ratio elsewhere
        # does not blank its own (meaningful) statistic.
        perfect = (r.effect is not None and r.effect[0] == "f2"
                   and not math.isfinite(float(r.effect[1])))
        sval = fmt.NON_FINITE if perfect else fmt.num(r.statistic[1])
        lines.append(f"- **{r.statistic[0]}** = {sval}")
        if perfect:
            lines.append("- Perfect fit — the model fits the data exactly, so F "
                         "is not meaningful.")
    if r.df:
        lines.append("- **df** = " + ", ".join(fmt.num(d) for d in r.df))
    if r.p is not None:
        lines.append(f"- **{fmt.p(r.p)}**")
    if r.estimate:
        s = f"- **{r.estimate[0]}** = {fmt.num(r.estimate[1])}"
        if r.estimate_ci:
            s += f", {fmt.ci(*r.estimate_ci)}"
        lines.append(s)
    if r.effect:
        s = f"- **{r.effect[0]}** = {fmt.num(r.effect[1])}"
        if r.effect_label:
            s += f" ({r.effect_label})"
        lines.append(s)
    if lines:
        st.markdown("\n".join(lines))


def _render_findings(r):
    for f in r.findings:
        if f.severity == "block":
            st.error(f.text)
        elif f.severity == "flag":
            st.warning(f.text)
        else:
            st.info(f.text)
    # one-click switch to a suggested test / setting (dedup identical suggestions)
    seen, i = set(), 0
    for f in r.findings:
        if not (f.suggest_test or f.suggest_params):
            continue
        sig = (f.suggest_test, tuple(f.suggest_params))
        if sig in seen:
            continue
        seen.add(sig)
        if f.suggest_test and f.suggest_test in REGISTRY:
            label = f"Use {REGISTRY[f.suggest_test].name} instead"
        else:
            label = "Apply the suggested setting"
        st.button(label, key=f"suggest::{i}", on_click=_apply_suggestion,
                  args=(f.suggest_test, tuple(f.suggest_params)))
        i += 1


def _render_result(spec, result, dataset=None, bound=None):
    st.header("Result")
    st.markdown(f"**{result.test_name}**")
    st.markdown(sentences.render(result))

    if result.status == "blocked":     # the sentence IS the block reason
        _render_findings(result)
        return

    _render_numbers(result)

    if result.descriptives is not None:
        st.caption("Descriptive statistics (Table 1)")
        st.dataframe(result.descriptives)
    # describe's categorical counts (means.py extra['categoricals']) — one table per
    # variable, so a categorical-only describe (descriptives is None) still shows
    # data (S-C). Labels/groups are already display strings. Two shapes mirror the
    # report: no-group {level: count} -> a level/count table; grouped
    # {group: {level: count}} -> a level × group count matrix.
    for header, counts in (result.extra.get("categoricals") or {}).items():
        st.caption(f"Counts — {header}")
        grouped = counts and all(isinstance(v, dict) for v in counts.values())
        if grouped:
            mat = pd.DataFrame(counts).fillna(0).astype(int)
            mat.index.name = header
            st.dataframe(mat.reset_index())
        else:
            st.dataframe(pd.DataFrame({header: list(counts),
                                       "Count": list(counts.values())}))
    if result.table is not None:
        st.caption("Results table")
        st.dataframe(result.table)
    if result.posthoc is not None:
        st.caption(f"Post-hoc comparisons ({result.posthoc_name})")
        st.dataframe(result.posthoc)
    if result.expected is not None:
        st.caption("Expected counts")
        st.dataframe(result.expected)

    for caption, png in charts.chart(result):
        st.image(png, caption=caption)

    _render_findings(result)

    if result.checks:
        with st.expander("Assumption checks"):
            for c in result.checks:
                verdict = c.note if c.passed is None else (
                    "passed" if c.passed else "not met")
                st.write(f"{c.name}: {verdict}")

    st.download_button(
        "Download Word report",
        data=report.build_report(result, spec, dataset=dataset,
                                 bound=bound).getvalue(),
        file_name=f"statkit_{result.test_id}.docx",
        mime=DOCX_MIME,
    )


# ==========================================================================
# the wizard
# ==========================================================================
def main():
    st.set_page_config(page_title="StatKit")
    st.title("StatKit")
    st.caption("Run the test you already know by name on your messy spreadsheet.")

    up = st.file_uploader("Upload a spreadsheet",
                          type=["xlsx", "xlsm", "xls", "csv", "txt"])
    if up is None:
        st.info("Upload an .xlsx, .xls or .csv file to begin.")
        return
    data, filename = up.getvalue(), up.name
    _reset_on_new_upload(filename, data)   # NICE: drop last sheet's selections

    # S14: a bad upload gets a stated, actionable message — never a traceback.
    try:
        grids = grid.load(data, filename)
    except grid.GridError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # keep the app alive on any unreadable file
        st.error(f"Sorry, I couldn't read '{filename}': {exc}")
        return
    if not grids:
        st.error("No data found in that file.")
        return

    # 1. sheet picker (only when >1 non-empty sheet)
    if len(grids) > 1:
        titles = [g.sheet for g in grids]
        _guard_options("sheet", titles)
        chosen = st.selectbox("Which sheet?", titles, key="sheet")
        g = grids[titles.index(chosen)]
    else:
        g = grids[0]

    exclude_hidden = st.checkbox("Exclude hidden rows/columns", value=False,
                                 key="exclude_hidden")

    # 2. clean -> Table. Clean ONCE from the persisted header_row (B5): clean now
    # keeps candidates/ambiguity even for a forced row, so the picker stays
    # populated and the selection can't oscillate. restore_rows (B4) exempts the
    # chosen dropped rows from being dropped again.
    restore = frozenset(st.session_state.get("restore_rows", []))
    header_row = st.session_state.get("header_row")
    table = clean.clean(g, header_row=header_row, keep_hidden=not exclude_hidden,
                        restore_rows=restore)
    if table.ambiguous:
        st.warning("I could not tell which row is the header — please confirm it.")
        rows = [er for er, _ in table.candidates] or [table.header_row]
        _guard_options("header_row", rows)
        st.selectbox("Header row", rows,
                     format_func=lambda r: f"Row {r}", key="header_row")

    # 3. infer -> Dataset. Per-column kind overrides, number-convention overrides
    # (S6) and level-merge maps (S7) are all read from session_state BEFORE infer
    # and rendered as controls AFTER, so a choice on the last run feeds this one.
    overrides = {}
    conventions = {}
    for col in table.header:
        label = st.session_state.get(f"kind::{col}")
        if label and label != "(auto)":
            overrides[col] = KIND_LABELS[label]
        clabel = st.session_state.get(f"conv::{col}")
        if clabel and clabel != "(auto)":
            conventions[col] = CONVENTION_LABELS[clabel]

    proposals = _proposed_merges(table)                # S7 (off the raw cells)
    merges = {name: mapping for name, mapping in proposals.items()
              if st.session_state.get(f"merge::{name}")}

    dataset = infer.infer(table, overrides=overrides, merges=merges,
                          conventions=conventions)

    with st.expander("What I read from your sheet"):
        st.dataframe(dataset.df.head(50))
        for line in list(table.log) + list(dataset.log):
            st.write("- " + line)
        if table.dropped_rows:
            st.caption("Rows dropped as titles / notes / blanks / subtotals:")
            for er, why in table.dropped_rows[:50]:
                st.write(f"  - row {er}: {why}")
            # B4: let the student bring a wrongly-dropped row back. Options carry
            # the current selection too, so a restored row (now kept, so gone from
            # dropped_rows) stays a valid choice instead of un-restoring itself.
            options = sorted(set(er for er, _ in table.dropped_rows)
                             | set(st.session_state.get("restore_rows", [])))
            _guard_options("restore_rows", options)
            st.multiselect("Bring dropped rows back into the data", options,
                           key="restore_rows", format_func=lambda er: f"Row {er}")

    if proposals:
        with st.expander("Tidy up category spellings"):
            for name, mapping in proposals.items():
                preview = "; ".join(f"'{s}' → '{c}'" for s, c in mapping.items())
                st.checkbox(f"Merge in '{name}': {preview}", key=f"merge::{name}")

    with st.expander("Number format (override if a column's numbers were misread)"):
        conv_opts = list(CONVENTION_LABELS)
        for col in table.header:
            _guard_options(f"conv::{col}", conv_opts)
            st.selectbox(col, conv_opts, key=f"conv::{col}")

    with st.expander("Column kinds (override if I guessed wrong)"):
        for p in dataset.profiles:
            opts = list(OVERRIDE_OPTIONS)
            _guard_options(f"kind::{p.name}", opts)
            note = f"{p.name} — read as **{p.kinds[0].value}**"
            if p.unit:
                note += f" (unit: {p.unit})"
            if p.notes:
                note += " · " + "; ".join(p.notes)
            st.markdown(note)
            st.selectbox("kind", opts, key=f"kind::{p.name}",
                         label_visibility="collapsed")

    # 4. goal + the single paired/independent question (L5)
    st.header("What do you want to do?")
    goal = st.selectbox("Goal", GOALS, format_func=lambda x: GOAL_LABELS[x],
                        key="goal")
    pairing = None
    if goal in PAIRED_GOALS:
        pairing = st.radio(
            "Were the measurements taken on the same people/units (paired), "
            "or on different ones (independent)?",
            ("independent", "paired"), key="pairing", horizontal=True)

    # 5. Contract-driven menu: this goal's tests, greying out what cannot bind.
    # A non-empty name/alias search (S8) overrides the goal menu, so a student who
    # knows the test by name ("welch", "shapiro") finds it without the goal step.
    st.header("Pick a test")
    query = st.text_input("Search tests by name (optional)", key="test_search").strip()
    if query:
        menu = [s for s in SPECS if _name_matches(s, query)]
    else:
        menu = [s for s in SPECS if s.goal == goal]
        if pairing is not None:
            menu = [s for s in menu if s.pairing in (pairing, "any")]
    if not menu:
        st.info(f"No test matches “{query}”." if query
                else "No test matches that goal here.")
        return

    # drop a stale test choice that no longer belongs to this goal/pairing
    menu_ids = {s.id for s in menu}
    if st.session_state.get("test_id") not in menu_ids:
        st.session_state.pop("test_id", None)

    for s in menu:
        ok, why = _satisfies(s, dataset)
        st.button(s.name, key=f"pick::{s.id}", disabled=not ok,
                  help=(s.menu_help if ok else why),
                  on_click=lambda sid=s.id: st.session_state.__setitem__("test_id", sid))

    test_id = st.session_state.get("test_id")
    if not test_id:
        st.info("Choose a test above.")
        return
    spec = REGISTRY[test_id]

    # 6. column pickers (role-compatible only) + params, all Contract-driven
    st.header(spec.name)
    sat = _sat_layouts(spec, dataset)
    if not sat:
        st.warning(_satisfies(spec, dataset)[1])
        return
    if len(sat) > 1:
        _guard_options("layout", sat)
        layout = st.selectbox("Data layout", sat, key="layout",
                              format_func=lambda x: LAYOUT_LABELS.get(x, x))
    else:
        layout = sat[0]

    roles = spec.contract.roles_by_layout[layout]
    columns, ready = {}, True
    for r in roles:
        opts = _role_columns(r, dataset)
        multi = r.max is None or r.max > 1
        if multi:
            key = f"cols::{r.name}"
            _guard_options(key, opts)
            sel = st.multiselect(r.label, opts, key=key, help=r.help)
            if sel:
                columns[r.name] = tuple(sel)
            if len(sel) < max(r.min, 0):
                ready = False
        else:
            key = f"col::{r.name}"
            choices = opts if r.min >= 1 else ["(none)", *opts]
            _guard_options(key, choices)
            sel = st.selectbox(r.label, choices, key=key, help=r.help)
            if sel and sel != "(none)":
                columns[r.name] = (sel,)
            elif r.min >= 1:
                ready = False

    params = _render_params(spec, dataset, columns, layout)

    # NICE: a one-sample reference value (μ₀ / p₀) must be entered, not silently
    # defaulted to a hypothesis the student never chose. Until it is, Run is held.
    req_floats = _required_floats(spec, layout)
    need = [p.label for p in spec.contract.params
            if p.name in req_floats and params.get(p.name) is None]
    if need:
        ready = False
        st.info("Enter " + ", ".join(need) + " to run this test.")
        for p in spec.contract.params:                 # don't pass a None param on
            if p.name in req_floats:
                params.pop(p.name, None)

    # NICE: a changed test / column / param clears a stale shown result.
    _stale_reset(test_id, columns, params)

    # 7. bind + structural checks -> know blocks before offering Run
    bound, blocked = None, False
    if ready and columns:
        try:
            bound = bind.bind(spec, dataset, columns, layout=layout, params=params)
            profs = {p.name: p for p in dataset.profiles}
            # MERGE, don't overwrite: keep every structural finding check emits
            # (the runners rely on them all being present) AND preserve bind's own
            # advisories (e.g. the DUP duplicate-measurement warning) that check
            # never emits. Dedup by code so nothing doubles.
            bind_findings = bound.findings
            structural = check.check(bound, profs)
            present = {f.code for f in structural}
            bound.findings = structural + tuple(
                f for f in bind_findings if f.code not in present)
            blocks = [f for f in bound.findings if f.severity == "block"]
            for f in blocks:
                st.error(f.text)
            blocked = bool(blocks)
        except bind.BindError as exc:  # a stated wrong-shape message (S14)
            st.error(str(exc))
            bound = None
        except Exception:  # keep the wizard alive; show a student message, not a trace
            st.error("Sorry, I couldn't prepare your data for this test. Try "
                     "picking different columns, or choose a different test.")
            bound = None

    run_clicked = st.button("Run test", key="run",
                            disabled=blocked or bound is None)
    if run_clicked:
        st.session_state["_ran"] = True

    # 8. run + render (reactive after the first Run; suggestion-rebind re-enters here)
    if st.session_state.get("_ran") and bound is not None and not blocked:
        try:
            result = spec.run(bound)
        except Exception:  # a student message, never the raw Python exception text
            st.error("Sorry, this test could not be computed on your data. Try "
                     "picking different columns, or choose a different test.")
            return
        _render_result(spec, result, dataset=dataset, bound=bound)


def _render_params(spec, dataset, columns, layout):
    required = _required_floats(spec, layout)
    roles = {r.name: r for r in spec.contract.roles_by_layout[layout]}
    params = {}
    for p in spec.contract.params:
        key = f"param::{p.name}"
        if p.kind == "bool":
            params[p.name] = st.checkbox(p.label, value=bool(p.default), key=key)
        elif p.kind == "float":
            if p.name in required:           # NICE: start empty -> must be entered
                raw = st.number_input(p.label, value=None, key=key,
                                      help=p.help or "Enter a reference value.")
                params[p.name] = None if raw is None else float(raw)
            else:
                params[p.name] = float(st.number_input(
                    p.label, value=float(p.default or 0.0), key=key))
        elif p.kind == "int":
            params[p.name] = int(st.number_input(
                p.label, value=int(p.default or 0), step=1, key=key))
        elif p.kind == "choice":
            opts = list(p.choices) or [str(p.default)]
            _guard_options(key, opts)
            params[p.name] = st.selectbox(p.label, opts, key=key)
        elif p.kind == "level":
            role = roles.get(p.of_role)
            headers = columns.get(p.of_role, ())
            if role is not None and (role.max is None or role.max > 1):
                # A per-column reference role (regression predictors): one level
                # selectbox per CATEGORICAL predictor, keyed by its header. A
                # numeric predictor has no reference level, so it is skipped (F12)
                # — mirroring how the runner Treatment-codes only non-numeric
                # columns. The resulting {header: level} dict is what bind/the
                # runner already accept.
                chosen = {}
                for h in headers:
                    if pd.api.types.is_numeric_dtype(dataset.df[h]):
                        continue
                    lvls = _column_levels(dataset, h)
                    if not lvls:
                        continue
                    hk = f"{key}::{h}"
                    _guard_options(hk, lvls)
                    chosen[h] = st.selectbox(f"{p.label} — {h}", lvls, key=hk)
                if chosen:
                    params[p.name] = chosen
            else:
                lvls = _column_levels(dataset, headers[0]) if headers else []
                if lvls:
                    _guard_options(key, lvls)
                    params[p.name] = st.selectbox(p.label, lvls, key=key)
        elif p.kind == "proportions":
            txt = st.text_input(f"{p.label} (comma-separated, blank = equal)", key=key)
            vals = _parse_props(txt)
            if vals:
                params[p.name] = vals
    return params


def _parse_props(txt):
    if not txt or not txt.strip():
        return None
    try:
        return tuple(float(x) for x in txt.replace(";", ",").split(",") if x.strip())
    except ValueError:
        return None


main()

# Credit/contact footer — top level so it renders once at the page bottom on
# EVERY view, even the early-return branches inside main() (e.g. no upload yet).
st.divider()
st.caption("Made by Faisal Alsulami · contact: "
           "[faisal.alsulmi1@gmail.com](mailto:faisal.alsulmi1@gmail.com)")
