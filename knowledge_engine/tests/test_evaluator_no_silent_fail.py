"""No silent failures in sub-concept evaluator pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from knowledge_engine.src.node_deep_dive.memory_schemas import (
    SessionMemory,
    SubConceptRecord,
)
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput
from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (
    apply_degraded_threshold,
    process_sub_concept_user_answer,
    run_sub_concept_gap_eval,
)


def _node(layer: str = "advanced") -> NodeDataInput:
    return NodeDataInput(
        node_id="subagent_architectures",
        title="Subagent architectures",
        layer=layer,
        category="agents",
        brief_summary="hierarchy",
        core_concepts=["hierarchy"],
        learning_goal="Understand agent hierarchy",
    )


def _mem_pending() -> SessionMemory:
    mem = SessionMemory()
    mem.sub_concepts = [
        SubConceptRecord(
            id="иерархия_агентов",
            label="Иерархия агентов",
            success_criterion="WHY+HOW",
            status="unchecked",
        )
    ]
    mem.pending_evaluation_concept_id = "иерархия_агентов"
    mem.asked_question_sub_concept_id = "иерархия_агентов"
    return mem


def test_degraded_threshold_leaves_partial_not_unchecked():
    row = SubConceptRecord(
        id="иерархия_агентов",
        label="Иерархия агентов",
        status="unchecked",
    )
    d = apply_degraded_threshold(row, layer="advanced", reason="empty_updates")
    assert row.status == "partial"
    assert d == "PROBE_NEXT_LAYER:WHY"
    assert row.why_passed is False
    assert "Автооценка" in (row.focus_hint or "")


def test_empty_updates_applies_degraded_not_silent():
    mem = _mem_pending()
    node = _node()
    fake = MagicMock()
    fake.updates = []

    with patch(
        "knowledge_engine.src.node_deep_dive.sub_concept_evaluator."
        "run_gemini_structured_with_chain",
        return_value=fake,
    ):
        d = run_sub_concept_gap_eval(
            "Длинный ответ про async event loop и пул субагентов для иерархии.",
            mem,
            node,
            "test_anchor",
            concept_id="иерархия_агентов",
        )
    row = mem.sub_concepts[0]
    assert d == "PROBE_NEXT_LAYER:WHY"
    assert row.status == "partial"
    assert "evaluator_degraded" in (row.evidence or "")


def test_id_mismatch_single_update_soft_accepted():
    from knowledge_engine.schemas.llm_contracts.tutor import SubConceptStatusUpdate

    mem = _mem_pending()
    node = _node()
    fake = MagicMock()
    fake.updates = [
        SubConceptStatusUpdate(
            id="wrong_id_slug",
            why_passed=True,
            how_passed=True,
            mechanic_passed=False,
            evidence="async hierarchy",
            focus_hint="",
        )
    ]
    with patch(
        "knowledge_engine.src.node_deep_dive.sub_concept_evaluator."
        "run_gemini_structured_with_chain",
        return_value=fake,
    ):
        d = run_sub_concept_gap_eval(
            "Достаточно длинный ответ про иерархию и async.",
            mem,
            node,
            "test_anchor",
            concept_id="иерархия_агентов",
        )
    row = mem.sub_concepts[0]
    assert d == "PASSED_WITH_GLOSS"
    assert row.status == "verified"
    assert row.why_passed is True
    assert row.how_passed is True


def test_process_sets_directive_after_degraded():
    mem = _mem_pending()
    node = _node()
    fake = MagicMock()
    fake.updates = []
    with patch(
        "knowledge_engine.src.node_deep_dive.sub_concept_evaluator."
        "run_gemini_structured_with_chain",
        return_value=fake,
    ):
        process_sub_concept_user_answer(
            "Достаточно длинный ответ пользователя про субагентов и RAM.",
            mem,
            node,
            "test_anchor",
        )
    assert mem.last_eval_directive == "PROBE_NEXT_LAYER:WHY"
    assert mem.sub_concepts[0].status == "partial"
    assert mem.pending_evaluation_concept_id == ""
    assert "частично" in (mem.last_evaluator_feedback or "").lower() or (
        mem.last_evaluator_feedback or ""
    )
