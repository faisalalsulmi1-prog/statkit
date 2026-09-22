"""RED list for the residual synth builders (PLAN §9.2).

The number-convention / ordered-vocab cases are wired into infer NOW (assert the
inferred kind); the registry-shape cases (long k>=3, McNemar, R x C grid) get a
"ingests cleanly" smoke test -- they are consumed by Chunks 6/8/9.
"""
import synth

from statkit import clean, grid, infer
from statkit.model import Kind


def _first_table(builder):
    data, name = builder()
    grids = grid.load(data, name)
    assert grids, f"{builder.__name__} produced no grid"
    return clean.clean(grids[0])


# --- number-convention cases (wired into infer) ---------------------------
def test_space_thousands_reads_as_numeric():
    ds = infer.infer(_first_table(synth.space_thousands))
    pop = ds.profiles[1]                       # "Population"
    assert pop.name == "Population"
    assert pop.kinds[0] == Kind.NUMERIC
    assert pop.n_missing == 0 and len(pop.failed_cells) == 0
    assert ds.df["Population"].iloc[0] == 1234.0


def test_percent_text_at_n30_reads_numeric_with_percent_unit():
    ds = infer.infer(_first_table(synth.percent_text))
    comp = ds.profiles[1]                      # "Completion"
    assert comp.kinds[0] == Kind.NUMERIC
    assert comp.unit == "%"
    assert len(comp.failed_cells) == 0
    assert ds.df["Completion"].iloc[0] == 40.0     # "40%" -> 40 percent points


# --- ordered vocabularies -> ORDINAL --------------------------------------
def test_ordered_vocab_frequency_is_ordinal():
    ds = infer.infer(_first_table(synth.ordered_vocabs))
    freq, sev, rating = ds.profiles
    assert freq.kinds == (Kind.ORDINAL, Kind.CATEGORICAL)
    assert sev.kinds == (Kind.ORDINAL, Kind.CATEGORICAL)
    assert rating.kinds == (Kind.ORDINAL, Kind.CATEGORICAL)
    # level_order follows the vocabulary, not first-seen order
    assert freq.level_order == ("never", "rarely", "sometimes", "often", "always")
    assert rating.level_order == ("low", "medium", "high")
    # "none" is no longer an NA token (it is real data in a stats tool: severity
    # none / "no symptoms"), so a "None" severity is kept as the first ordinal
    # level, not dropped as missing.
    assert sev.level_order == ("none", "mild", "moderate", "severe")
    assert sev.n_levels == 4
    assert sev.n_missing == 0


# --- registry-shape builders: smoke (they ingest cleanly) -----------------
def test_long_k3_ingests_with_three_conditions():
    ds = infer.infer(_first_table(synth.long_k3))
    names = [p.name for p in ds.profiles]
    assert names == ["Subject", "Condition", "Score"]
    condition = ds.profiles[1]
    assert condition.n_levels == 3                # k>=3 within factor (S20 / rm)
    assert ds.profiles[2].kinds[0] == Kind.NUMERIC


def test_mcnemar_happy_ingests_two_binary_columns():
    ds = infer.infer(_first_table(synth.mcnemar_happy))
    before, after = ds.profiles
    assert before.name == "Before" and after.name == "After"
    assert Kind.BINARY in before.kinds and Kind.BINARY in after.kinds
    assert before.n_levels == 2 and after.n_levels == 2   # b,c > 0 -> both levels present


def test_rc_grid_ingests_as_a_wide_count_table():
    table = _first_table(synth.rc_grid)
    assert table.header[0] == "Region"
    assert "Total" in table.header                # Total column survives
    ds = infer.infer(table)
    # the three product count columns read numeric
    assert ds.profiles[1].kinds[0] == Kind.NUMERIC
    assert ds.profiles[2].kinds[0] == Kind.NUMERIC


def test_all_builders_load_without_error():
    for builder in synth.ALL_BUILDERS:
        data, name = builder()
        grids = grid.load(data, name)
        assert grids and grids[0].rows, builder.__name__
