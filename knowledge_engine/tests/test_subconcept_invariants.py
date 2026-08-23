"""Strict sub-concept invariants: lecture anchor, asked binding, question drift."""

from __future__ import annotations

from knowledge_engine.src.node_deep_dive.lecture_scope import resolve_lecture_scope
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    SessionMemory,
    SubConceptRecord,
)
from knowledge_engine.src.node_deep_dive.schemas import DeepDiveLLMOutput
from knowledge_engine.src.node_deep_dive.subconcept_invariants import (
    enforce_question_sub_concept_invariant,
    resolve_active_subconcept_id,
)


def _hooks_memory() -> SessionMemory:
    mem = SessionMemory()
    mem.sub_concepts = [
        SubConceptRecord(
            id="pre_execution_hooks",
            label="Pre-execution hooks",
            success_criterion="Gateway intercept before tool exec",
            status="verified",
        ),
        SubConceptRecord(
            id="post_execution_validation",
            label="Post-execution validation",
            success_criterion="Validate tool results / hallucinations",
            status="unchecked",
        ),
    ]
    mem.next_question_concept_id = "post_execution_validation"
    mem.asked_question_sub_concept_id = ""
    mem.pending_evaluation_concept_id = ""
    mem.active_window = [
        {
            "role": "user",
            "content": "Pre-execution hooks перехватывают tool_use до исполнения",
        },
        {
            "role": "tutor",
            "content": "Разбор Pre-execution hooks и AJV. Как устроен gateway?",
        },
    ]
    mem.learning_phase = "dense_material"
    return mem


def test_lecture_scope_ignores_chat_history_when_active_subconcept_set():
    mem = _hooks_memory()
    scope, focus = resolve_lecture_scope(
        "Дай плотный материал по теме.",
        mem,
        lecture_button_pressed=True,
    )
    assert scope == "targeted_lecture"
    assert "subconcept_id=post_execution_validation" in focus
    assert "Post-execution validation" in focus
    # Must not fall back to prior user/tutor turns about pre_execution
    assert "AJV" not in focus
    assert "перехватывают tool_use" not in focus


def test_active_subconcept_prefers_asked_until_verified():
    mem = _hooks_memory()
    mem.sub_concepts[1].status = "partial"
    mem.asked_question_sub_concept_id = "post_execution_validation"
    mem.next_question_concept_id = "pre_execution_hooks"  # stale / wrong
    assert resolve_active_subconcept_id(mem) == "post_execution_validation"


def test_question_drift_force_corrects_to_active():
    mem = _hooks_memory()
    out = DeepDiveLLMOutput(
        technical_explanation="текст",
        follow_up_question="Как устроена post-execution проверка?",
        question_sub_concept_id="pre_execution_hooks",
    )
    repaired, drifted = enforce_question_sub_concept_invariant(mem, out)
    assert drifted is True
    assert repaired.question_sub_concept_id == "post_execution_validation"


def test_fully_mastered_follow_up_is_stripped():
    mem = _hooks_memory()
    mem.sub_concepts[0].why_passed = True
    mem.sub_concepts[0].how_passed = True
    mem.sub_concepts[0].mechanic_passed = True
    mem.sub_concepts[0].status = "verified"
    mem.sub_concepts[1].status = "verified"
    mem.sub_concepts[1].why_passed = True
    mem.sub_concepts[1].how_passed = True
    mem.sub_concepts[1].mechanic_passed = True
    mem.next_question_concept_id = "pre_execution_hooks"
    out = DeepDiveLLMOutput(
        technical_explanation="текст",
        follow_up_question="Как устроены pre-execution hooks?",
        question_sub_concept_id="pre_execution_hooks",
    )
    repaired, drifted = enforce_question_sub_concept_invariant(mem, out)
    assert drifted is True
    assert repaired.follow_up_question == ""
    assert repaired.question_sub_concept_id is None
    assert resolve_active_subconcept_id(mem) == ""
