"""Lecture-request turns must not run gap evaluation / credit scoreboards."""

from __future__ import annotations

from knowledge_engine.services.lecture_body_format import (
    append_checkpoint_to_lecture_body,
    strip_lecture_credit_scoreboard,
    strip_trailing_checkpoint_from_lecture_body,
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
    assert mem.evaluator_skipped is True


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
    out = append_checkpoint_to_lecture_body(body, q)
    assert out.endswith(q)
    assert "**Самопроверка:**" in out
    assert append_checkpoint_to_lecture_body(f"{body}\n\n{q}", q).count(q) == 1


def test_append_checkpoint_dedupes_plain_paragraph_before_marker_block():
    q = (
        "Почему в многопоточной программе на CPython потоки делят общую память кучи, "
        "и почему GIL работает как мьютекс выполнения байт-кода?"
    )
    theory = ("GIL сериализует выполнение байт-кода. " * 40).strip()
    duplicated = f"{theory}\n\n{q}\n\n**Самопроверка:** {q}"
    out = append_checkpoint_to_lecture_body(duplicated, q)
    assert out.count(q) == 1
    assert out.endswith(q)
    assert theory in out
    assert out.index("**Самопроверка:**") > out.index(theory[:40])


def test_strip_trailing_checkpoint_removes_tail_question_only():
    q = "Как GIL защищает кучу CPython?"
    body = f"Теория про GIL и PyObject.\n\n{q}"
    stripped = strip_trailing_checkpoint_from_lecture_body(body, q)
    assert q not in stripped
    assert "PyObject" in stripped


def test_strip_trailing_checkpoint_ignores_punctuation_and_yo():
    q = "Почему GIL сериализует байт-код, а не бизнес-операции?"
    tail = "Почему GIL сериализует байткод а не бизнес операции"
    theory = "Глобальная блокировка интерпретатора защищает кучу."
    stripped = strip_trailing_checkpoint_from_lecture_body(f"{theory}\n\n{tail}", q)
    assert "байткод" not in stripped
    assert "блокировка" in stripped
    yo_q = "Зачем нужен счётчик ссылок?"
    yo_tail = "Зачем нужен счетчик ссылок"
    stripped_yo = strip_trailing_checkpoint_from_lecture_body(
        f"{theory}\n\n{yo_tail}", yo_q
    )
    assert "счетчик" not in stripped_yo
    assert "блокировка" in stripped_yo


def test_strip_trailing_checkpoint_cuts_quiz_headers_regardless_of_question():
    theory = "GIL сериализует выполнение байт-кода в одном процессе."
    q = "Как устроен ceval?"
    for header in (
        "**Самопроверка:** другой текст вопроса?",
        "Самопроверка: ещё один вариант.",
        "Вопрос: совершенно другая формулировка?",
    ):
        stripped = strip_trailing_checkpoint_from_lecture_body(
            f"{theory}\n\n{header}",
            q,
        )
        assert "другой" not in stripped
        assert "ещё один" not in stripped
        assert "совершенно" not in stripped
        assert "сериализует" in stripped


def test_gil_internals_trace_duplicate_is_stripped_once():
    """Real gil_internals JSON: last lecture_body paragraph == checkpoint with markdown ticks."""
    from knowledge_engine.src.node_deep_dive.engine import _compose_dense_chat_message
    from knowledge_engine.src.node_deep_dive.schemas import DenseMaterialOutput

    q = (
        "Почему отсутствие единственной глобальной блокировки интерпретатора "
        "привело бы к разрушению внутренних указателей и структуры бакетов "
        "в динамических коллекциях (например, словарях) даже при условии "
        "атомарности операций с отдельным счетчиком ссылок ob_refcnt?"
    )
    body_q = (
        "Почему отсутствие единственной глобальной блокировки интерпретатора "
        "привело бы к разрушению внутренних указателей и структуры бакетов "
        "в динамических коллекциях (например, словарях) даже при условии "
        "атомарности операций с отдельным счетчиком ссылок `ob_refcnt`?"
    )
    theory = "GIL функционирует как взаимно исключающий мьютекс на уровне интерпретатора."
    stripped = strip_trailing_checkpoint_from_lecture_body(f"{theory}\n\n{body_q}", q)
    assert "разрушению" not in stripped
    assert "мьютекс" in stripped

    dense = DenseMaterialOutput(
        lecture_body=f"{theory}\n\n{body_q}",
        checkpoint_prompt=q,
        summary="GIL",
    )
    chat = _compose_dense_chat_message(dense)
    assert chat.count("разрушению") == 1
    assert chat.count("**Самопроверка:**") == 1
    assert chat.endswith("?")


def test_lecture_prompts_forbid_checkpoint_inside_lecture_body():
    from knowledge_engine.schemas.llm_contracts.tutor import (
        STRUCTURED_LECTURE_FIELD_RULES,
        StructuredLectureResponse,
    )
    from knowledge_engine.src.node_deep_dive.lecture_prompt_en import (
        LECTURE_MODE_STRUCTURE_RULES,
    )
    from knowledge_engine.src.node_deep_dive.tutor_behavior_state import (
        resolve_tutor_mode,
        build_tutor_behavior_state,
    )

    assert "CRITICAL NEGATIVE CONSTRAINT" in LECTURE_MODE_STRUCTURE_RULES
    assert "EXCLUSIVELY" in LECTURE_MODE_STRUCTURE_RULES
    assert "End your message with exactly ONE" not in LECTURE_MODE_STRUCTURE_RULES
    assert "CRITICAL NEGATIVE CONSTRAINT" in STRUCTURED_LECTURE_FIELD_RULES
    lecture_desc = StructuredLectureResponse.model_fields["lecture_body"].description or ""
    assert "checkpoint_prompt" in lecture_desc
    assert "FORBIDDEN" in lecture_desc

    mode = resolve_tutor_mode(
        "ANSWER", "chat", "lecture", "dense_material", "[mode:lecture] лекция"
    )
    assert mode == "lecture_dense"
    state = build_tutor_behavior_state(
        "ANSWER", "chat", "lecture", "dense_material", "[mode:lecture] лекция"
    )
    assert "mirrored" not in state["next_action"].lower()
    assert "exclusively in checkpoint_prompt" in state["next_action"]
