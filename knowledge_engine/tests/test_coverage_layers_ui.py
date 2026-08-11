"""Coverage summary: WHY/HOW/MECHANIC depth layers for mastery UI."""

from __future__ import annotations

from knowledge_engine.src.node_deep_dive.concept_map_state import build_coverage_summary
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    SessionMemory,
    SubConceptRecord,
)


def _mem(*rows: SubConceptRecord, directive: str = "") -> SessionMemory:
    m = SessionMemory()
    m.sub_concepts = list(rows)
    m.last_eval_directive = directive
    return m


def test_coverage_layers_overall_thirds():
    mem = _mem(
        SubConceptRecord(
            id="sc_a",
            label="Иерархия",
            why_passed=True,
            how_passed=False,
            mechanic_passed=False,
            status="partial",
        ),
        SubConceptRecord(
            id="sc_b",
            label="Делегирование",
            why_passed=True,
            how_passed=False,
            mechanic_passed=False,
            status="partial",
        ),
        directive="PROBE_NEXT_LAYER:HOW",
    )
    cov = build_coverage_summary(mem)
    assert cov is not None
    assert cov.layers is not None
    assert cov.layers.why.status == "passed"
    assert cov.layers.how.status == "in_progress"
    assert cov.active_layer == "HOW"
    assert cov.overall_score == 33  # only WHY fully across items


def test_coverage_gloss_hint_when_why_how_without_mechanic():
    mem = _mem(
        SubConceptRecord(
            id="sc_a",
            label="Иерархия",
            why_passed=True,
            how_passed=True,
            mechanic_passed=False,
            status="verified",
        ),
        directive="PASSED_WITH_GLOSS",
    )
    cov = build_coverage_summary(mem)
    assert cov is not None
    assert cov.layers is not None
    assert cov.layers.mechanic.status == "gloss"
    assert "Gloss" in cov.gloss_hint or "механик" in cov.gloss_hint.lower()
    assert cov.overall_score == 67  # 2/3 layers
    assert cov.items[0].why_passed is True
    assert cov.items[0].how_passed is True
    assert cov.items[0].status_hint == "Не хватает механик реализации"


def test_status_hint_untouched_is_laconic():
    mem = _mem(
        SubConceptRecord(
            id="sc_x",
            label="Иерархия агентов",
            status="unchecked",
        )
    )
    cov = build_coverage_summary(mem)
    assert cov is not None
    assert cov.items[0].status_hint == "Ещё не затронута"
    assert "Иерархия" not in cov.items[0].status_hint
