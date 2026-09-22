"""RED-first gate for the sentence wiring (PLAN §10, Chunk 12).

Every one of the 27 ``TestSpec``s ships ``sentence`` as the ``_todo`` stub until
Chunk 12 points it at the real ``sentences.render``. This gate asserts none is
left as the stub, that the 27 wired renderers are EXACTLY the ids in
``sentences.SENTENCES``, and that dispatch through the registry actually returns
a plain-English string for a hand-built Result (no stats run).
"""
from __future__ import annotations

from statkit import registry, sentences
from statkit.model import Result
from statkit.registry import REGISTRY


def test_no_sentence_is_the_todo_stub():
    for spec in REGISTRY.values():
        assert callable(spec.sentence), spec.id
        assert spec.sentence is not registry._todo, spec.id
        assert getattr(spec.sentence, "__name__", "") != "_todo", spec.id


def test_wired_sentences_cover_every_registry_id():
    assert set(sentences.SENTENCES) == set(REGISTRY)
    for sid, spec in REGISTRY.items():
        assert spec.sentence is sentences.SENTENCES[sid], sid


def test_dispatch_through_registry_returns_a_string():
    # A minimal but populated Result per family, run through spec.sentence.
    r = Result(
        test_id="t_ind", test_name="Welch's independent-samples t-test",
        status="ok", statistic=("t", 2.0), df=(20.0,), p=0.03,
        estimate=("mean difference (A − B)", 1.0), estimate_ci=(0.1, 1.9),
        effect=("Cohen's d", 0.5), effect_label="medium",
        effect_source="Cohen (1988)",
        n={"total": 22, "used": 22, "dropped": 0}, groups=("A", "B"),
        higher="A", labels={"outcome": "Y"})
    s = REGISTRY["t_ind"].sentence(r)
    assert isinstance(s, str) and s
    assert "Welch's independent-samples t-test" in s
    assert "n = " in s and ("p = " in s or "p < " in s)
