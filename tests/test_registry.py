"""RED list for statkit.registry — the closed 27-entry test registry.

This is the scope document made executable (PLAN §3, §9.1 "model/registry").
The contracts must be COMPLETE and CORRECT even though every runner is a stub:
that is the whole point of Chunk 1. So these tests check the structural
invariants the menu, pickers and checks will rely on — not the (unimplemented)
statistics.
"""
import pandas as pd
import pytest

from statkit import bind, registry
from statkit.infer import Dataset
from statkit.model import ColumnProfile, Kind  # NB: TestSpec is referenced via registry.* only,
# never imported by name here — importing a `Test*`-named class into a test
# module makes pytest try to collect it (a spurious PytestCollectionWarning).

SPECS = registry.SPECS
REGISTRY = registry.REGISTRY

KNOWN_FAMILIES = {
    "F-2G", "F-KG", "F-RANK2", "F-ASSOC", "F-ASSOC-M",
    "F-CAT", "F-REG", "F-CHK", "(table only)",
}
GOALS = {"describe", "compare", "relate", "counts", "predict", "assumptions"}
PAIRINGS = {"independent", "paired", "any"}
# "one_sample" added for wilcoxon-vs-mu0 (the t_1s -> Wilcoxon suggestion, S9).
LAYOUTS = {"long", "wide", "table", "one_sample"}

# The closed scope (PLAN §3): 24 inferential tests + 3 utilities = 27 entries.
EXPECTED_IDS = {
    "describe", "normality", "homogeneity",
    "t_1s", "t_ind", "t_paired",
    "anova_1w", "anova_2w", "rm_anova",
    "mwu", "wilcoxon", "kruskal", "friedman",
    "pearson", "spearman", "kendall", "corr_matrix",
    "chi2_ind", "chi2_gof", "fisher", "mcnemar", "cochran_q", "prop_1", "prop_2",
    "ols_simple", "ols_multi", "logistic",
}


def _role_names(spec) -> set[str]:
    names: set[str] = set()
    for roles in spec.contract.roles_by_layout.values():
        names |= {r.name for r in roles}
    return names


# --- the count and the exact closed set (§9.1: "27 entries exactly") ------
def test_exactly_27_entries():
    assert len(SPECS) == 27


def test_the_closed_set_of_ids_is_exactly_the_scope():
    assert {s.id for s in SPECS} == EXPECTED_IDS


def test_ids_are_unique():
    ids = [s.id for s in SPECS]
    assert len(ids) == len(set(ids))


def test_registry_dict_is_keyed_by_id():
    assert set(REGISTRY) == {s.id for s in SPECS}
    for spec in SPECS:
        assert REGISTRY[spec.id] is spec


# --- contract structure (§9.1: "canonical ∈ roles_by_layout") -------------
def test_canonical_layout_is_an_accepted_layout():
    for spec in SPECS:
        assert spec.contract.canonical in spec.contract.roles_by_layout, spec.id


def test_every_layout_key_is_a_valid_layout_with_at_least_one_role():
    for spec in SPECS:
        for layout, roles in spec.contract.roles_by_layout.items():
            assert layout in LAYOUTS, (spec.id, layout)
            assert len(roles) >= 1, (spec.id, layout)


def test_roles_are_well_formed():
    for spec in SPECS:
        for roles in spec.contract.roles_by_layout.values():
            for r in roles:
                assert r.accepts, (spec.id, r.name)
                assert all(isinstance(k, Kind) for k in r.accepts), (spec.id, r.name)
                assert r.min >= 0, (spec.id, r.name)
                assert r.max is None or r.max >= max(r.min, 1), (spec.id, r.name)
                lo, hi = r.levels
                assert lo >= 1 and (hi is None or hi >= lo), (spec.id, r.name)


# --- params (§9.1: "every Param of_role names a real role") ---------------
def test_param_of_role_names_a_real_role():
    for spec in SPECS:
        real = _role_names(spec)
        for p in spec.contract.params:
            if p.of_role is not None:
                assert p.of_role in real, (spec.id, p.name, p.of_role)


def test_level_and_proportion_params_declare_their_role():
    # A "level"/"proportions" param enumerates a role's levels, so it MUST say
    # which role (otherwise the picker has nothing to enumerate).
    for spec in SPECS:
        for p in spec.contract.params:
            if p.kind in ("level", "proportions"):
                assert p.of_role is not None, (spec.id, p.name)


def test_choice_params_have_choices_including_their_default():
    for spec in SPECS:
        for p in spec.contract.params:
            if p.kind == "choice":
                assert p.choices, (spec.id, p.name)
                assert p.default in p.choices, (spec.id, p.name)


def test_param_names_unique_within_a_spec():
    for spec in SPECS:
        names = [p.name for p in spec.contract.params]
        assert len(names) == len(set(names)), spec.id


# --- callables (§9.1: "every spec's run/sentence/chart callable") ---------
def test_run_sentence_chart_are_callable():
    for spec in SPECS:
        assert callable(spec.run), spec.id
        assert callable(spec.sentence), spec.id
        assert callable(spec.chart), spec.id


def test_runner_sentence_and_chart_are_wired():
    # Runners (Chunk 8), sentences (Chunk 12) and charts (Chunk 13) are all wired
    # to their real functions (see test_wiring.py / test_sentence_wiring.py /
    # test_charts.py); none may remain the loud _todo stub.
    for spec in SPECS:
        assert spec.run is not registry._todo, spec.id
        assert spec.sentence is not registry._todo, spec.id
        assert spec.chart is not registry._todo, spec.id
        assert getattr(spec.chart, "__name__", "") != "_todo", spec.id


def test_chart_of_none_result_does_not_crash():
    # spec.chart must be callable on any Result without raising; a chart-less test
    # or a blocked Result yields () rather than a NotImplementedError.
    class _Blocked:
        status = "blocked"
        test_id = "t_ind"
    assert REGISTRY["t_ind"].chart(_Blocked()) == ()
    assert REGISTRY["describe"].chart(_Blocked()) == ()


# --- families (§9.1: "every family string ∈ known set") -------------------
def test_family_strings_are_known():
    for spec in SPECS:
        assert spec.family in KNOWN_FAMILIES, (spec.id, spec.family)


# --- aliases (§9.1: "aliases unique") ------------------------------------
def test_aliases_are_globally_unique():
    seen: list[str] = []
    for spec in SPECS:
        seen.extend(spec.aliases)
    assert len(seen) == len(set(seen)), (
        "duplicate alias(es): "
        + ", ".join(sorted({a for a in seen if seen.count(a) > 1}))
    )


def test_aliases_do_not_collide_with_ids():
    ids = {s.id for s in SPECS}
    for spec in SPECS:
        assert not (set(spec.aliases) & ids), (spec.id, spec.aliases)


def test_aliases_are_lowercase():
    # The menu search lower-cases the query; aliases must match that casing.
    for spec in SPECS:
        for a in spec.aliases:
            assert a == a.lower(), (spec.id, a)


# --- goal / pairing / completeness ---------------------------------------
def test_goal_and_pairing_are_valid():
    for spec in SPECS:
        assert spec.goal in GOALS, (spec.id, spec.goal)
        assert spec.pairing in PAIRINGS, (spec.id, spec.pairing)
        assert spec.contract.pairing in PAIRINGS, (spec.id, spec.contract.pairing)


def test_descriptive_fields_are_filled():
    # "Contracts complete for all 27" — no empty name/library/citation/help.
    for spec in SPECS:
        assert spec.name.strip(), spec.id
        assert spec.library.strip(), spec.id
        assert spec.citation.strip(), spec.id
        assert spec.menu_help.strip(), spec.id


# --- pinned correctness facts from PLAN §3 -------------------------------
def test_only_mcnemar_and_cochran_use_same_level_set():
    same = {s.id for s in SPECS if s.contract.same_level_set}
    assert same == {"mcnemar", "cochran_q"}


def test_t_ind_defaults_to_welch():
    # D1: Welch is the default; Student's only via an explicit checkbox.
    ev = {p.name: p for p in REGISTRY["t_ind"].contract.params}["equal_var"]
    assert ev.kind == "bool" and ev.default is False


def test_t_ind_alias_welch_resolves_and_pearson_alias_point_biserial():
    # D19 examples: "welch" -> t_ind, "point-biserial" -> pearson.
    assert "welch" in REGISTRY["t_ind"].aliases
    assert "point-biserial" in REGISTRY["pearson"].aliases


def test_wide_only_paired_tests_normalise_to_wide():
    for tid in ("t_paired", "rm_anova", "wilcoxon", "friedman"):
        assert REGISTRY[tid].contract.canonical == "wide", tid


def test_table_layout_tests_expand_to_long():
    for tid in ("chi2_ind", "chi2_gof", "fisher"):
        c = REGISTRY[tid].contract
        assert "table" in c.roles_by_layout and c.canonical == "long", tid


# --- S9: wilcoxon one-sample layout satisfiable on a single numeric column --
def test_wilcoxon_one_sample_layout_is_satisfiable_on_a_single_numeric_column():
    # A single numeric column is a valid wilcoxon-vs-mu0 bind (the t_1s -> Wilcoxon
    # suggestion). Without the "one_sample" layout in the contract, satisfies()
    # dead-ends (wide needs 2 columns, long needs 3).
    df = pd.DataFrame({"Score": pd.array([1.0, -2.0, 3.0, 0.5, 4.0], dtype="float64")})
    prof = ColumnProfile(name="Score", kinds=(Kind.NUMERIC,), n_total=5,
                         n_missing=0, n_levels=0, levels=())
    ds = Dataset(df=df, profiles=(prof,), excel_rows=(2, 3, 4, 5, 6))
    assert bind.satisfies(REGISTRY["wilcoxon"], ds)[0] is True
    b = bind.bind(REGISTRY["wilcoxon"], ds, {"outcome": ("Score",)},
                  layout="one_sample", params={"mu0": 5})
    assert b.data.columns.tolist() == ["outcome"]
