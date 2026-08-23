"""Concept map: pending evaluation binding (deterministic state)."""

from __future__ import annotations

import pytest

from knowledge_engine.src.node_deep_dive.concept_map import (
    advance_next_question_after_evaluation,
    format_concept_map_for_tutor,
    list_verified_sub_concept_ids,
    resolve_evaluation_target_id,
    resolve_pending_evaluation_id,
    select_next_sub_concept,
    set_pending_evaluation_for_tutor_turn,
)
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    SessionMemory,
    SubConceptRecord,
)


def _memory_with_concepts() -> SessionMemory:
    mem = SessionMemory()
    mem.sub_concepts = [
        SubConceptRecord(
            id="определение_инструментов",
            label="Определение инструментов",
            success_criterion="Tool schema",
            status="unchecked",
        ),
        SubConceptRecord(
            id="регистрация_ресурсов",
            label="Регистрация ресурсов",
            success_criterion="URI registration",
            status="verified",
        ),
        SubConceptRecord(
            id="обработка_ошибок",
            label="Обработка ошибок",
            success_criterion="Retries and fallback",
            status="unchecked",
        ),
    ]
    return mem


def test_resolve_pending_uses_stored_pending_only():
    mem = _memory_with_concepts()
    mem.pending_evaluation_concept_id = "определение_инструментов"
    user = "Circuit breaker, retries и fallback при сбоях MCP"
    assert resolve_pending_evaluation_id(mem, user) == "определение_инструментов"
    assert resolve_evaluation_target_id(mem) == "определение_инструментов"


def test_set_pending_from_focus_sub_concept_id():
    mem = SessionMemory()
    mem.sub_concepts = [
        SubConceptRecord(
            id="react_цикл",
            label="ReAct цикл",
            success_criterion="ReAct",
            status="verified",
        ),
        SubConceptRecord(
            id="трассировка_мыслей",
            label="Трассировка мыслей",
            success_criterion="Tracing",
            status="unchecked",
        ),
    ]
    mem.pending_evaluation_concept_id = "react_цикл"
    cid = set_pending_evaluation_for_tutor_turn(mem, "трассировка_мыслей")
    assert cid == "трассировка_мыслей"
    assert mem.pending_evaluation_concept_id == "трассировка_мыслей"
    assert mem.asked_question_sub_concept_id == "трассировка_мыслей"
    assert resolve_evaluation_target_id(mem) == "трассировка_мыслей"


def test_select_next_skips_verified():
    mem = _memory_with_concepts()
    nxt = select_next_sub_concept(mem)
    assert nxt is not None
    assert nxt.id != "регистрация_ресурсов"
    assert nxt.status != "verified"


def test_advance_next_after_eval_moves_focus():
    mem = _memory_with_concepts()
    mem.sub_concepts[0].status = "verified"
    mem.asked_question_sub_concept_id = "определение_инструментов"
    cid = advance_next_question_after_evaluation(
        mem, evaluated_id="определение_инструментов"
    )
    assert cid == "обработка_ошибок"
    assert mem.next_question_concept_id == "обработка_ошибок"
    assert mem.asked_question_sub_concept_id == ""


def test_advance_next_keeps_partial_focus():
    mem = _memory_with_concepts()
    mem.sub_concepts[0].status = "partial"
    mem.sub_concepts[0].focus_hint = "Нужна механика KV invalidation"
    cid = advance_next_question_after_evaluation(
        mem, evaluated_id="определение_инструментов"
    )
    assert cid == "определение_инструментов"
    assert mem.next_question_concept_id == "определение_инструментов"
    assert mem.asked_question_sub_concept_id == "определение_инструментов"


def test_list_verified_sub_concept_ids_only_verified_status():
    mem = _memory_with_concepts()
    mem.sub_concepts[0].status = "partial"
    mem.sub_concepts[2].status = "gap"
    assert list_verified_sub_concept_ids(mem) == ["регистрация_ресурсов"]


def test_gap_eval_system_has_depth_and_topic_rules():
    from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (
        GAP_EVAL_SYSTEM,
    )

    assert "You do NOT decide mastery" in GAP_EVAL_SYSTEM
    assert "why_passed" in GAP_EVAL_SYSTEM
    assert "how_passed" in GAP_EVAL_SYSTEM
    assert "mechanic_passed" in GAP_EVAL_SYSTEM
    assert "focus_hint" in GAP_EVAL_SYSTEM
    assert "accuracy_grade" in GAP_EVAL_SYSTEM
    assert "EXACT_AND_CORRECT" in GAP_EVAL_SYSTEM
    assert "evaluation_target" in GAP_EVAL_SYSTEM
    assert "STRICT SCOPE & ABSTRACTION CEILING" in GAP_EVAL_SYSTEM
    assert "last_tutor_question" in GAP_EVAL_SYSTEM
    assert "Silence or omission" in GAP_EVAL_SYSTEM


def test_merge_layer_flags_is_cumulative_or():
    from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (
        merge_layer_flags,
    )

    why, how, mech = merge_layer_flags(True, False, False, False, True, False)
    assert (why, how, mech) == (True, True, False)
    why2, how2, mech2 = merge_layer_flags(True, True, False, False, False, True)
    assert (why2, how2, mech2) == (True, True, True)
    # Never clears prior True
    why3, how3, mech3 = merge_layer_flags(True, True, True, False, False, False)
    assert (why3, how3, mech3) == (True, True, True)


def test_passes_threshold_foundation_advanced_sota():
    from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (
        passes_threshold,
    )

    ok, d = passes_threshold("foundation", True, False, False)
    assert ok and d == "PASSED_WITH_GLOSS"
    ok, d = passes_threshold("foundation", True, True, True)
    assert ok and d == "PASSED_CLEAN"
    ok, d = passes_threshold("foundation", False, True, True)
    assert not ok and d == "PROBE_NEXT_LAYER:WHY"

    ok, d = passes_threshold("advanced", True, True, False)
    assert ok and d == "PASSED_WITH_GLOSS"
    ok, d = passes_threshold("advanced", True, False, False)
    assert not ok and d == "PROBE_NEXT_LAYER:HOW"
    ok, d = passes_threshold("advanced", True, True, True)
    assert ok and d == "PASSED_CLEAN"

    ok, d = passes_threshold("sota", True, True, False)
    assert not ok and d == "PROBE_NEXT_LAYER:MECHANIC"
    ok, d = passes_threshold("sota", True, True, True)
    assert ok and d == "PASSED_CLEAN"
    ok, d = passes_threshold("sota", False, False, False)
    assert not ok and d == "PROBE_NEXT_LAYER:WHY"


def test_apply_threshold_or_merge_and_passed_with_gloss():
    from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (
        apply_threshold_to_sub_concept,
    )

    row = SubConceptRecord(
        id="kv_cache",
        label="KV-cache",
        success_criterion="WHY+HOW",
        status="unchecked",
        why_passed=True,
        how_passed=False,
        mechanic_passed=False,
    )
    d1 = apply_threshold_to_sub_concept(
        row, layer="advanced", why=False, how=True, mechanic=False
    )
    assert row.why_passed is True
    assert row.how_passed is True
    assert row.mechanic_passed is False
    assert row.status == "verified"
    assert d1 == "PASSED_WITH_GLOSS"

    # Prior flags never cleared on a weak turn
    d2 = apply_threshold_to_sub_concept(
        row, layer="advanced", why=False, how=False, mechanic=False
    )
    assert row.why_passed is True
    assert row.how_passed is True
    assert row.status == "verified"
    assert d2 == "PASSED_WITH_GLOSS"

    row_f = SubConceptRecord(
        id="intro_why",
        label="Intro",
        success_criterion="WHY",
        status="unchecked",
    )
    d3 = apply_threshold_to_sub_concept(
        row_f, layer="foundation", why=True, how=False, mechanic=False
    )
    assert row_f.status == "verified"
    assert d3 == "PASSED_WITH_GLOSS"


def test_is_empty_answer_guard():
    from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (
        _is_empty_answer,
    )

    assert _is_empty_answer("")
    assert _is_empty_answer("ok")
    assert not _is_empty_answer("достаточно длинный ответ")


def test_tutor_prompt_explicit_ban_praise_on_partial_gap():
    from knowledge_engine.src.node_deep_dive.dialogue_prompt_en import (
        DIALOGUE_SYSTEM_INSTRUCTION_EN,
        DIALOGUE_THRESHOLD_DIRECTIVE_EN,
        FEEDBACK_TRANSPARENCY_REQUIREMENT_EN,
    )
    from knowledge_engine.src.node_deep_dive.tutor_prompt_builder import (
        GLOBAL_REGISTRY_PROMPT_RULES,
        build_dialogue_system,
    )

    dialogue_system = build_dialogue_system()
    assert "отлично" in DIALOGUE_SYSTEM_INSTRUCTION_EN
    assert "отлично" in dialogue_system
    assert "last_evaluator_feedback" in dialogue_system
    assert "HOST TRANSPARENCY" in FEEDBACK_TRANSPARENCY_REQUIREMENT_EN
    assert "HOST TRANSPARENCY" in dialogue_system
    assert "last_evaluator_focus_hint" in dialogue_system
    assert "📋" in FEEDBACK_TRANSPARENCY_REQUIREMENT_EN
    assert "Do not emit" in FEEDBACK_TRANSPARENCY_REQUIREMENT_EN or "FORBIDDEN" in FEEDBACK_TRANSPARENCY_REQUIREMENT_EN
    assert "PASSED_WITH_GLOSS" in DIALOGUE_THRESHOLD_DIRECTIVE_EN
    assert "DO NOT generate a new technical" in DIALOGUE_THRESHOLD_DIRECTIVE_EN
    assert "PASSED_WITH_GLOSS" in dialogue_system
    assert "PROBE_NEXT_LAYER" in dialogue_system
    assert "evaluator" in GLOBAL_REGISTRY_PROMPT_RULES.lower()
    assert "VERIFIED" in GLOBAL_REGISTRY_PROMPT_RULES
    assert "PARTIAL" in dialogue_system or "GAP" in dialogue_system


def test_concept_map_exposes_structured_focus_hint_and_evidence():
    mem = _memory_with_concepts()
    mem.sub_concepts[0].status = "partial"
    mem.sub_concepts[0].evidence = "Контейнеризация и seccomp"
    mem.sub_concepts[0].focus_hint = "Не раскрыта работа cgroups v2 и ulimits"
    mem.sub_concepts[0].why_passed = True
    mem.last_eval_directive = "PROBE_NEXT_LAYER:HOW"
    mem.asked_question_sub_concept_id = "определение_инструментов"
    mem.next_question_concept_id = "определение_инструментов"
    block = format_concept_map_for_tutor(mem, focus_id="определение_инструментов")
    assert "[EVALUATOR_TRANSPARENCY]" in block
    assert "last_evaluator_focus_hint: Не раскрыта работа cgroups v2 и ulimits" in block
    assert "last_evaluator_evidence: Контейнеризация и seccomp" in block
    assert "last_evaluator_status: PARTIAL" in block
    assert "last_eval_directive: PROBE_NEXT_LAYER:HOW" in block
    assert "THRESHOLD_DIRECTIVE: ask ONLY about layer HOW" in block
    assert "W1H0M0" in block


def test_partial_grade_does_not_close_layer():
    from knowledge_engine.schemas.drill_schemas import AnswerAccuracyGrade
    from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (
        apply_threshold_to_sub_concept,
        passes_threshold,
    )

    ok, d = passes_threshold(
        "advanced",
        True,
        True,
        False,
        accuracy_grade=AnswerAccuracyGrade.PARTIAL,
        detected_errors=["Пропущен инвариант владения."],
    )
    assert ok is False
    assert d == "PROBE_NEXT_LAYER:MECHANIC"

    row = SubConceptRecord(
        id="kv_cache",
        label="KV-cache",
        success_criterion="WHY+HOW",
        status="unchecked",
    )
    directive = apply_threshold_to_sub_concept(
        row,
        layer="advanced",
        why=True,
        how=True,
        mechanic=False,
        evidence="Мотивация кэша названа.",
        llm_focus_hint="Не раскрыт инвариант инвалидации.",
        accuracy_grade=AnswerAccuracyGrade.PARTIAL,
        detected_errors=["Пропущен инвариант инвалидации."],
        correct_claims=["KV-cache убирает повторный проход по ключам."],
    )
    assert row.why_passed is False
    assert row.how_passed is False
    assert row.status == "partial"
    assert row.failed_attempts == 1
    assert row.last_accuracy_grade == "PARTIAL"
    assert "инвариант" in (row.focus_hint or "")
    assert directive == "PROBE_NEXT_LAYER:WHY"


def test_misunderstanding_soft_regresses_partial_to_gap():
    from knowledge_engine.schemas.drill_schemas import AnswerAccuracyGrade
    from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (
        apply_threshold_to_sub_concept,
    )

    row = SubConceptRecord(
        id="kv_cache",
        label="KV-cache",
        status="partial",
        why_passed=False,
        failed_attempts=1,
    )
    apply_threshold_to_sub_concept(
        row,
        layer="advanced",
        why=True,
        how=False,
        mechanic=False,
        llm_focus_hint="Смешение KV-cache с page cache.",
        accuracy_grade=AnswerAccuracyGrade.MISUNDERSTANDING,
        detected_errors=["KV-cache описан как page cache ОС."],
    )
    assert row.status == "gap"
    assert row.why_passed is False
    assert row.failed_attempts == 2
    assert row.last_accuracy_grade == "MISUNDERSTANDING"


def test_partial_does_not_unverify():
    from knowledge_engine.schemas.drill_schemas import AnswerAccuracyGrade
    from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (
        apply_threshold_to_sub_concept,
    )

    row = SubConceptRecord(
        id="kv_cache",
        label="KV-cache",
        status="verified",
        why_passed=True,
        how_passed=False,
    )
    apply_threshold_to_sub_concept(
        row,
        layer="advanced",
        why=False,
        how=True,
        mechanic=False,
        accuracy_grade=AnswerAccuracyGrade.PARTIAL,
        detected_errors=["HOW без инвариантов ролей."],
        correct_claims=["Pipeline стадий назван верно."],
        llm_focus_hint="Нужны инварианты разделения ролей.",
    )
    assert row.status == "verified"
    assert row.why_passed is True
    assert row.how_passed is False
    assert row.focus_hint


def test_select_next_attracts_failed_attempts():
    mem = SessionMemory()
    mem.sub_concepts = [
        SubConceptRecord(
            id="alpha",
            label="Alpha",
            status="partial",
            failed_attempts=0,
        ),
        SubConceptRecord(
            id="beta",
            label="Beta",
            status="partial",
            failed_attempts=2,
            last_accuracy_grade="MISUNDERSTANDING",
        ),
        SubConceptRecord(
            id="gamma",
            label="Gamma",
            status="unchecked",
        ),
    ]
    nxt = select_next_sub_concept(mem)
    assert nxt is not None
    assert nxt.id == "beta"


def test_partial_status_update_requires_correct_claims() -> None:
    from pydantic import ValidationError

    from knowledge_engine.schemas.llm_contracts.tutor import SubConceptStatusUpdate

    with pytest.raises(ValidationError):
        SubConceptStatusUpdate(
            id="kv_cache",
            why_passed=True,
            how_passed=False,
            mechanic_passed=False,
            accuracy_grade="PARTIAL",
            detected_errors_or_misconceptions=["Пропущен инвариант."],
            correct_claims=[],
        )


def test_select_next_does_not_reorder_active_drill():
    from knowledge_engine.src.node_deep_dive.memory_schemas import LayerDrillSession

    mem = SessionMemory()
    mem.sub_concepts = [
        SubConceptRecord(
            id="pyobject",
            label="PyObject",
            status="verified",
            why_passed=True,
            how_passed=False,
        ),
        SubConceptRecord(
            id="refcnt",
            label="refcnt",
            status="partial",
            why_passed=True,
            how_passed=False,
            failed_attempts=5,
            last_accuracy_grade="MISUNDERSTANDING",
        ),
    ]
    mem.layer_drill = LayerDrillSession(
        is_active=True,
        target_layer="HOW",
        target_sub_concept_ids=["pyobject", "refcnt"],
        current_index=0,
        status="DRILL_IN_PROGRESS",
    )
    nxt = select_next_sub_concept(mem)
    assert nxt is not None
    assert nxt.id == "pyobject"


def test_subconcept_evidence_accumulates_across_eval_turns():
    from knowledge_engine.schemas.drill_schemas import AnswerAccuracyGrade
    from knowledge_engine.src.node_deep_dive.memory_schemas import (
        accumulate_evidence_text,
    )
    from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (
        apply_threshold_to_sub_concept,
    )

    assert "WHY" in accumulate_evidence_text("WHY закрыт.", "HOW: arena→pool")
    assert "HOW: arena→pool" in accumulate_evidence_text("WHY закрыт.", "HOW: arena→pool")
    kept = accumulate_evidence_text(
        "реальный зачёт GIL",
        "evaluator_degraded: empty_or_failed_llm_update",
    )
    assert "реальный зачёт GIL" in kept
    assert "evaluator_degraded" not in kept

    row = SubConceptRecord(id="gil_mutex", label="GIL mutex", status="unchecked")
    apply_threshold_to_sub_concept(
        row,
        layer="foundation",
        why=True,
        how=False,
        mechanic=False,
        evidence="GIL сериализует байткод.",
        llm_focus_hint="Не раскрыт eval_breaker.",
        accuracy_grade=AnswerAccuracyGrade.PARTIAL,
        detected_errors=["Нет eval_breaker."],
        correct_claims=["GIL сериализует байткод."],
    )
    first = row.evidence
    assert "GIL сериализует байткод" in first
    apply_threshold_to_sub_concept(
        row,
        layer="foundation",
        why=True,
        how=True,
        mechanic=False,
        evidence="eval_breaker проверяется между итерациями.",
        llm_focus_hint="Не раскрыт ceval switch при I/O.",
        accuracy_grade=AnswerAccuracyGrade.PARTIAL,
        detected_errors=["Нет switch при I/O."],
        correct_claims=["eval_breaker между итерациями."],
    )
    assert "GIL сериализует байткод" in (row.evidence or "")
    assert "eval_breaker" in (row.evidence or "")
