"""Lecture-request turns must not run gap evaluation / credit scoreboards."""

from __future__ import annotations

from knowledge_engine.services.lecture_body_format import (
    append_checkpoint_to_lecture_body,
    strip_lecture_credit_scoreboard,
)
from knowledge_engine.src.node_deep_dive.lecture_scope import is_lecture_request_message
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    SessionMemory,
    SubConceptRecord,
)
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput
from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (
    process_sub_concept_user_answer,
)


def _node() -> NodeDataInput:
    return NodeDataInput(
        node_id="rules_and_commands_engine",
        title="Rules",
        layer="foundation",
        category="agents",
        brief_summary="x",
        core_concepts=["rules"],
        learning_goal="g",
    )


def test_is_lecture_request_message_detects_mode_and_stub():
    assert is_lecture_request_message("[mode:lecture] Дай плотный материал по теме.")
    assert is_lecture_request_message("Дай плотный материал по теме.")
    assert not is_lecture_request_message(
        "Приоритизация инструкций делается через граф весов."
    )


def test_process_sub_concept_skips_lecture_request(monkeypatch):
    mem = SessionMemory()
    mem.pending_evaluation_concept_id = "приоритизация_инструкций"
    mem.sub_concepts = [
        SubConceptRecord(
            id="приоритизация_инструкций",
            label="Приоритизация",
            status="unchecked",
        )
    ]
    called = {"n": 0}

    def _boom(*_a, **_k):
        called["n"] += 1
        raise AssertionError("gap eval must not run for lecture request")

    monkeypatch.setattr(
        "knowledge_engine.src.node_deep_dive.sub_concept_evaluator.run_sub_concept_gap_eval",
        _boom,
    )
    process_sub_concept_user_answer(
        "[mode:lecture] Дай плотный материал по теме.",
        mem,
        _node(),
        "anchor",
    )
    assert called["n"] == 0
    assert mem.pending_evaluation_concept_id == "приоритизация_инструкций"
    assert mem.sub_concepts[0].status == "unchecked"


def test_strip_lecture_credit_scoreboard():
    raw = (
        "---\n"
        "**📋 Что уже зачтено:** кэширование.\n"
        "**🎯 Чего не хватило для полного зачёта:** механизм арбитража.\n"
        "---\n\n"
        "В условиях мультиагентных систем…\n"
    )
    cleaned = strip_lecture_credit_scoreboard(raw)
    assert "зачтено" not in cleaned.lower()
    assert "мультиагентных" in cleaned


def test_append_checkpoint_to_lecture_body():
    body = "Лекция про арбитраж."
    q = "Как разрешить конфликт политик?"
    assert append_checkpoint_to_lecture_body(body, q).endswith(q)
    assert append_checkpoint_to_lecture_body(f"{body}\n\n{q}", q).count(q) == 1
