"""Pydantic Layer Drill contracts + FSM schema selection."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from knowledge_engine.schemas.drill_schemas import (
    ActiveDrillStepResponse,
    LayerCompletionTutorOutput,
    StandardDrillTutorOutput,
    TechnicalConceptAudit,
    THEORY_BODY_HARD_MIN_WORDS,
)
from knowledge_engine.src.node_deep_dive.drill_orchestrator import (
    drill_response_to_llm_output,
    render_drill_markdown,
    select_drill_response_schema,
)
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    LayerDrillSession,
    SessionMemory,
    SubConceptRecord,
)
from knowledge_engine.src.node_deep_dive.star_task_fsm import start_layer_drill


def _dense_theory() -> str:
    return ("архитектура потока данных изоляция состояния аллокация ") * 60


def _ok_audit() -> TechnicalConceptAudit:
    return TechnicalConceptAudit(
        feedback_kind="EXACT",
        accuracy_grade="EXACT_AND_CORRECT",
        user_claims_analysis=["Предыдущий ответ закрыл инвариант владения."],
        detected_errors_or_misconceptions=[],
        confirmation="Ответ закрыл инвариант владения указателем.",
    )


def _active_session() -> LayerDrillSession:
    return LayerDrillSession(
        is_active=True,
        target_layer="HOW",
        target_sub_concept_ids=["pyobject", "reference_count", "type_pointer"],
        current_index=0,
        status="DRILL_IN_PROGRESS",
    )


def test_active_drill_schema_requires_question() -> None:
    theory = _dense_theory()
    with pytest.raises(ValidationError):
        ActiveDrillStepResponse(
            status_header="[Слой HOW: Проверено 0/3 подтем. Переходим к подтеме №1: «A»]",
            feedback_on_previous="Короткий разбор предыдущего ответа студента.",
            theory_body=theory,
        )
    with pytest.raises(ValidationError):
        ActiveDrillStepResponse(
            status_header="[Слой HOW: Проверено 0/3 подтем. Переходим к подтеме №1: «A»]",
            feedback_on_previous="Короткий разбор предыдущего ответа студента.",
            next_question="Как устроен refcnt?",
        )
    with pytest.raises(ValidationError):
        ActiveDrillStepResponse(
            status_header="[Слой HOW: Проверено 0/3 подтем. Переходим к подтеме №1: «A»]",
            feedback_on_previous="Короткий разбор предыдущего ответа студента.",
            theory_body=theory,
            next_question="без знака вопроса",
        )
    parsed = ActiveDrillStepResponse(
        audit=_ok_audit(),
        status_header="[Слой HOW: Проверено 0/3 подтем. Переходим к подтеме №1: «A»]",
        theory_body=theory,
        next_question="Как устроен refcnt?",
    )
    assert "архитектура" in parsed.theory_body
    assert "?" in parsed.next_question


def _header() -> str:
    return "[Слой HOW: Проверено 0/3 подтем. Переходим к подтеме №1: «A»]"


def test_theory_body_hard_floor_150_soft_target_warns() -> None:
    too_short = ("слово ") * (THEORY_BODY_HARD_MIN_WORDS - 10)
    with pytest.raises(ValidationError, match="150"):
        ActiveDrillStepResponse(
            audit=_ok_audit(),
            status_header=_header(),
            theory_body=too_short,
            next_question="Как устроен refcnt?",
        )
    mid = ("слово ") * 200
    with patch("knowledge_engine.ui.run_log.trace") as mock_trace:
        parsed = ActiveDrillStepResponse(
            audit=_ok_audit(),
            status_header=_header(),
            theory_body=mid,
            next_question="Как устроен refcnt?",
        )
    assert len(parsed.theory_body.split()) == 200
    mock_trace.assert_called()
    warn = mock_trace.call_args[0][0]
    assert "WARN drill theory_body" in warn
    assert "words=200" in warn


def test_orchestrator_selects_correct_pydantic_schema_based_on_fsm() -> None:
    mem = SessionMemory(
        layer_drill=_active_session(),
        sub_concepts=[
            SubConceptRecord(
                id="pyobject",
                label="PyObject header",
                status="verified",
                why_passed=True,
            ),
            SubConceptRecord(
                id="reference_count",
                label="Reference count",
                status="verified",
                why_passed=True,
            ),
            SubConceptRecord(
                id="type_pointer",
                label="Type pointer",
                status="verified",
                why_passed=True,
            ),
        ],
    )
    assert select_drill_response_schema(mem) is ActiveDrillStepResponse
    assert mem.layer_drill.has_more_questions() is True

    start_layer_drill(mem, "HOW")
    mem.layer_drill.current_index = 2
    mem.sub_concepts[2].how_passed = True
    mem.layer_drill.advance_or_complete()
    assert mem.layer_drill.has_more_questions() is False
    assert mem.layer_drill.status == "DRILL_COMPLETE"
    assert select_drill_response_schema(mem) is LayerCompletionTutorOutput
    assert StandardDrillTutorOutput is ActiveDrillStepResponse
    assert "next_question" not in LayerCompletionTutorOutput.model_fields
    assert "next_question" in ActiveDrillStepResponse.model_fields


def test_host_assembles_markdown_from_validated_objects() -> None:
    active = ActiveDrillStepResponse(
        audit=_ok_audit(),
        status_header="[Слой MECH: Проверено 1/4 подтем. Переходим к подтеме №2: «x»]",
        theory_body=_dense_theory(),
        next_question="Где падает refcnt до нуля?",
    )
    md = render_drill_markdown(active)
    assert md.startswith("[Слой MECH:")
    assert "**Вопрос:** Где падает refcnt до нуля?" in md
    out = drill_response_to_llm_output(active, concept_id="pyobject")
    assert out.ready_for_transition is False
    assert out.follow_up_question.startswith("**Вопрос:**")

    done = LayerCompletionTutorOutput(
        praise="Слой MECH закрыт — все четыре подтемы пройдены.",
        layer_summary="Разобрали refcnt, tp_dealloc и инварианты владения указателем.",
        transition_framing="Хотите углубиться в Advanced/Deep или перейти к следующей теме?",
    )
    done_md = render_drill_markdown(done)
    assert "Слой MECH закрыт" in done_md
    assert "Вопрос:" not in done_md
    packed = drill_response_to_llm_output(done)
    assert packed.ready_for_transition is True
    assert packed.quick_replies == []
    assert packed.follow_up_question == done.transition_framing
    assert packed.feedback_on_answer == done.praise
    assert packed.technical_explanation == done.layer_summary


def test_prompt_factory_isolates_active_drill_json_contract() -> None:
    from knowledge_engine.src.node_deep_dive.prompt_factory import (
        select_system_prompt_and_mode,
    )

    mem = SessionMemory(
        layer_drill=_active_session(),
        sub_concepts=[
            SubConceptRecord(id="pyobject", label="PyObject header", status="verified"),
        ],
    )
    system, _, _ = select_system_prompt_and_mode(
        "[mode:deep_dive_how] Разбери архитектуру.",
        default_system_prompt="DEFAULT",
        memory=mem,
    )
    assert "JSON OUTPUT (ActiveDrillStepResponse)" in system
    assert "JSON OUTPUT (DeepDiveTutorContract)" not in system
    assert "ANTI-SYCOPHANCY INVARIANTS" in system
    assert "audit" in system
    assert "DEFAULT" not in system
