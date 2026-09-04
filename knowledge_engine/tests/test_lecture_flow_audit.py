"""Lecture mode (mode:lecture) contract: theory first, then a closing question."""

from __future__ import annotations

from knowledge_engine.services.lecture_body_format import clip_lecture_keeping_checkpoint
from knowledge_engine.services.lecture_rag_context import build_lecture_generation_payload
from knowledge_engine.src.node_deep_dive.engine import _compose_dense_chat_message
from knowledge_engine.src.node_deep_dive.lecture_prompt_en import (
    LECTURE_GAP_STEERING_RULES,
    LECTURE_MODE_STRUCTURE_RULES,
)
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    SessionMemory,
    SubConceptRecord,
)
from knowledge_engine.src.node_deep_dive.prompt_factory import (
    is_factory_control_mode,
    parse_tutor_mode_prefix,
    select_system_prompt_and_mode,
)
from knowledge_engine.src.node_deep_dive.prompt_types import InteractionPromptMode
from knowledge_engine.src.node_deep_dive.schemas import (
    DenseMaterialOutput,
    NodeDataInput,
)
from knowledge_engine.src.node_deep_dive.tutor_behavior_state import (
    build_tutor_behavior_state,
    resolve_tutor_mode,
)
from knowledge_engine.src.node_deep_dive.tutor_field_limits import SCHEMA_TUTOR_MESSAGE_MAX
from knowledge_engine.src.node_deep_dive.tutor_dialogue import (
    extract_follow_up_from_chat_text,
)
from knowledge_engine.src.node_deep_dive.tutor_prompt_builder import compose_system_prompt


def _assert_lecture_structure(prompt: str) -> None:
    text = prompt or ""
    assert "PART 1:" in text
    assert "DENSE LECTURE" in text
    assert "PART 2:" in text
    assert "CLOSING QUESTION" in text
    assert "CRITICAL NEGATIVE CONSTRAINT" in text
    assert "EXCLUSIVELY" in text
    assert "End your message with exactly ONE" not in text
    assert "NEVER end a lecture response without asking a question" not in text


def test_lecture_mode_generates_theory_then_question() -> None:
    open_q = "Почему CPython arena не отдаёт память ОС сразу после free()?"
    mem = SessionMemory()
    mem.last_tutor_follow_up_question = open_q
    mem.learning_mode = "lecture"

    body, mode = parse_tutor_mode_prefix(
        "[mode:lecture] Дай плотный материал по теме."
    )
    assert mode == "lecture"
    assert "плотный" in body.lower()
    assert not is_factory_control_mode(mode)

    default = compose_system_prompt(InteractionPromptMode.LECTURE_DENSE, context=None)
    system, factory_mode, cleaned = select_system_prompt_and_mode(
        "[mode:lecture] Дай плотный материал по теме.",
        default_system_prompt=default,
    )
    assert factory_mode == "lecture"
    assert "плотный" in cleaned.lower()
    _assert_lecture_structure(system)
    _assert_lecture_structure(LECTURE_MODE_STRUCTURE_RULES)
    assert LECTURE_MODE_STRUCTURE_RULES.strip() in system
    assert system.count("[MANDATORY RESPONSE STRUCTURE FOR MODE:LECTURE]") >= 1
    assert "BUDGET ALLOCATION" in system
    assert "CHECKPOINT ALIGNMENT" in system
    assert "[TARGET_FOCUS_AND_GAPS]" in system
    assert LECTURE_GAP_STEERING_RULES.strip() in system

    tutor_mode = resolve_tutor_mode(
        "ANSWER",
        "chat",
        "lecture",
        "dense_material",
        "[mode:lecture] Дай плотный материал по теме.",
    )
    assert tutor_mode == "lecture_dense"
    state = build_tutor_behavior_state(
        "ANSWER",
        "chat",
        "lecture",
        "dense_material",
        "[mode:lecture] Дай плотный материал по теме.",
        memory=mem,
    )
    assert state["current_mode"] == "lecture_dense"
    next_action = state["next_action"]
    assert "PART 1" in next_action
    assert "PART 2" in next_action
    assert "exclusively" in next_action.lower()
    assert "mirrored" not in next_action.lower()
    assert "Самопроверка" in next_action
    assert "arena" in next_action or open_q[:40] in next_action

    node = NodeDataInput(
        node_id="python_object_model",
        title="Object model",
        layer="foundation",
        category="python",
        brief_summary="CPython objects",
        core_concepts=["arena"],
        learning_goal="memory",
    )
    payload = build_lecture_generation_payload(
        node,
        "",
        "лекция",
        "arena 256KB [R1]",
        "",
        "",
        memory=mem,
    )
    assert "[OPEN_NODE_QUESTION]" in payload
    assert "free()" in payload
    assert "PART 2" in payload

    # Mock LLM: long PART 1 + structured checkpoint (no live API).
    theory = ("Иерархия arena → pool → block снижает malloc. [R1]\n\n") * 400
    checkpoint = (
        "Почему CPython не возвращает пустую arena ОС, пока в ней остаётся "
        "хотя бы один занятый block?"
    )
    dense = DenseMaterialOutput(
        lecture_body=theory,
        checkpoint_prompt=checkpoint,
        summary="arena/pool/block",
    )
    chat = _compose_dense_chat_message(dense)
    assert chat.endswith("?") or checkpoint in chat
    technical, follow = extract_follow_up_from_chat_text(chat)
    assert "arena" in technical.lower() or "pool" in technical.lower()
    assert follow.endswith("?")
    assert "?" in follow
    assert len(chat) <= SCHEMA_TUTOR_MESSAGE_MAX
    assert checkpoint in chat or follow.rstrip().endswith("?")


def test_clip_lecture_keeps_checkpoint_when_body_overflows() -> None:
    q = "Какой размер pool в CPython и почему он совпадает со страницей ОС?"
    body = ("x" * 11_800) + " теория аллокатора без вопроса"
    out = clip_lecture_keeping_checkpoint(body, q, limit=SCHEMA_TUTOR_MESSAGE_MAX)
    assert len(out) <= SCHEMA_TUTOR_MESSAGE_MAX
    assert q in out
    assert out.rstrip().endswith("?")
    _, follow = extract_follow_up_from_chat_text(out)
    assert follow.endswith("?")


def test_lecture_factory_does_not_force_isolated_control() -> None:
    system, mode, _ = select_system_prompt_and_mode(
        "[mode:lecture] Дай плотный материал по теме.",
        default_system_prompt="DEFAULT_DENSE",
    )
    assert mode == "lecture"
    assert not is_factory_control_mode("lecture")
    assert "DEFAULT_DENSE" in system
    assert "PART 2: MANDATORY CLOSING QUESTION" in system


def test_lecture_payload_passes_evaluator_transparency_and_anchors_checkpoint() -> None:
    mem = SessionMemory()
    mem.last_eval_directive = "PROBE_NEXT_LAYER:HOW"
    mem.last_tutor_follow_up_question = (
        "Почему GIL сериализует выполнение байткода в одном процессе?"
    )
    mem.asked_question_sub_concept_id = "gil_mutex"
    mem.next_question_concept_id = "gil_mutex"
    mem.pending_evaluation_concept_id = "gil_mutex"
    mem.sub_concepts = [
        SubConceptRecord(
            id="gil_mutex",
            label="GIL mutex",
            success_criterion="WHY+HOW",
            status="partial",
            why_passed=True,
            evidence="GIL сериализует байткод в одном процессе.",
            focus_hint="Не раскрыты ceval.c, eval_breaker и переключение при I/O.",
        )
    ]
    node = NodeDataInput(
        node_id="gil_internals",
        title="GIL internals",
        layer="foundation",
        category="python",
        brief_summary="CPython GIL",
        core_concepts=["GIL"],
        learning_goal="mutex",
    )
    payload = build_lecture_generation_payload(
        node,
        "",
        "лекция",
        "ceval eval_breaker [R1]",
        "",
        "",
        memory=mem,
    )
    assert "[EVALUATOR_TRANSPARENCY]" in payload
    assert "last_evaluator_focus_hint: Не раскрыты ceval.c" in payload
    assert "[TARGET_FOCUS_AND_GAPS]" in payload
    assert "probe_layer: HOW" in payload
    assert "last_eval_directive: PROBE_NEXT_LAYER:HOW" in payload
    assert "already-passed layer" in payload
    assert "[OPEN_NODE_QUESTION]" in payload

    state = build_tutor_behavior_state(
        "ANSWER",
        "chat",
        "lecture",
        "dense_material",
        "[mode:lecture] Дай плотный материал по теме.",
        memory=mem,
    )
    next_action = state["next_action"]
    assert "PART 1" in next_action
    assert "PART 2" in next_action
    assert "80%" in next_action
    assert "ceval.c" in next_action or "focus_hint" in next_action
    assert "HOW" in next_action
    assert "blind RE-STATE" in next_action
    assert "Самопроверка" in next_action
    assert "exclusively" in next_action.lower()
