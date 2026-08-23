"""Regression: dense lecture question must bind pending (no silent credit loss)."""

from __future__ import annotations

from knowledge_engine.src.node_deep_dive.graph.nodes.commit_turn import commit_turn_node
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    SessionMemory,
    SubConceptRecord,
)
from knowledge_engine.src.node_deep_dive.schemas import (
    DeepDiveLLMOutput,
    NodeDataInput,
    NodeDeepDiveRequest,
)
from knowledge_engine.src.node_deep_dive.tutor_dialogue import (
    deep_dive_llm_output_from_chat_text,
    extract_follow_up_from_chat_text,
)
from knowledge_engine.src.node_deep_dive.tutor_field_limits import (
    SCHEMA_FOLLOW_UP_QUESTION_MAX,
)
from knowledge_engine.services.lecture_body_format import append_checkpoint_to_lecture_body


def _mem() -> SessionMemory:
    mem = SessionMemory()
    mem.sub_concepts = [
        SubConceptRecord(
            id="детерминированные_защитные_барьеры",
            label="Детерминированные защитные барьеры",
            status="unchecked",
        ),
        SubConceptRecord(
            id="мониторинг_стека_вызовов",
            label="Мониторинг стека вызовов",
            status="unchecked",
        ),
    ]
    mem.next_question_concept_id = "детерминированные_защитные_барьеры"
    return mem


def _req() -> NodeDeepDiveRequest:
    return NodeDeepDiveRequest(
        curriculum_id="agentic_systems_architecture",
        node_data=NodeDataInput(
            node_id="governed_agent_pipelines",
            title="Governed agent pipelines",
            layer="sota",
            category="agents",
            brief_summary="x",
            core_concepts=["guardrails"],
            learning_goal="g",
        ),
        user_action="chat",
        user_message="Дай плотный материал по теме.",
    )


def test_extract_follow_up_self_check_marker():
    body = "Лекция про барьеры.\n\n**Самопроверка:** Где срабатывает hook?"
    tech, follow = extract_follow_up_from_chat_text(body)
    assert "Лекция" in tech
    assert "Где срабатывает hook?" in follow
    assert "**Самопроверка:**" not in follow


def test_extract_follow_up_trailing_question_without_marker():
    """Incident shape: lecture ends with a ? paragraph, no Самопроверка marker."""
    body = (
        "Детерминированные барьеры выполняются вне зоны LLM.\n\n"
        "Если агент в цикле обращается к инструменту с невалидными аргументами, "
        "на каком этапе конвейера должен срабатывать детерминированный барьер "
        "и почему валидация средствами Python эффективнее системного промпта?"
    )
    tech, follow = extract_follow_up_from_chat_text(body)
    assert "вне зоны LLM" in tech
    assert follow.endswith("?")
    assert "на каком этапе конвейера" in follow
    out = deep_dive_llm_output_from_chat_text(body)
    assert (out.follow_up_question or "").endswith("?")
    assert "вне зоны LLM" in (out.technical_explanation or "")


def test_append_checkpoint_uses_self_check_marker():
    out = append_checkpoint_to_lecture_body("Body.", "Как устроен middleware?")
    assert "**Самопроверка:**" in out
    tech, follow = extract_follow_up_from_chat_text(out)
    assert tech == "Body."
    assert "middleware" in follow


def test_append_checkpoint_normalizes_plain_tail_question():
    q = "Почему GIL сериализует байт-код, а не бизнес-операции?"
    body = f"Глобальная блокировка интерпретатора защищает кучу. {q}"
    out = append_checkpoint_to_lecture_body(body, q)
    assert "**Самопроверка:**" in out
    tech, follow = extract_follow_up_from_chat_text(out)
    assert "Глобальная блокировка" in tech
    assert follow.endswith("?")
    assert len(follow) < 400
    parsed = deep_dive_llm_output_from_chat_text(out)
    assert len(parsed.follow_up_question or "") <= SCHEMA_FOLLOW_UP_QUESTION_MAX
    assert "Глобальная блокировка" in (parsed.technical_explanation or "")


def test_extract_follow_up_long_single_paragraph():
    """Regression: one glued paragraph must not become the entire follow_up field."""
    q = (
        "Почему в многопоточной программе на CPython потоки делят общую память кучи, "
        "и почему GIL работает как мьютекс выполнения байт-кода, "
        "а не как инструмент ожидания завершения бизнес-операций?"
    )
    body = ("Глобальная блокировка интерпретатора (GIL) — мьютекс CPython. " * 80) + q
    tech, follow = extract_follow_up_from_chat_text(body)
    assert follow.endswith("?")
    assert "GIL" in follow
    assert len(follow) < 600
    assert "мьютекс CPython" in tech
    out = deep_dive_llm_output_from_chat_text(body)
    assert len(out.follow_up_question or "") <= SCHEMA_FOLLOW_UP_QUESTION_MAX
    assert len(out.technical_explanation or "") > len(out.follow_up_question or "")


def test_dense_llm_output_prefers_structured_checkpoint():
    from knowledge_engine.src.node_deep_dive.engine import _compose_dense_chat_message
    from knowledge_engine.src.node_deep_dive.schemas import DenseMaterialOutput

    q = (
        "Почему в многопоточной программе на CPython потоки делят общую память кучи, "
        "и почему GIL работает как мьютекс выполнения байт-кода?"
    )
    theory = ("GIL сериализует выполнение байт-кода в CPython. " * 120) + q
    dense = DenseMaterialOutput(lecture_body=theory, checkpoint_prompt=q)
    tutor = _compose_dense_chat_message(dense)
    out = deep_dive_llm_output_from_chat_text(
        tutor,
        follow_up_question=q,
        technical_explanation=theory[: -len(q)].strip(),
    )
    assert out.follow_up_question == q
    assert len(out.follow_up_question) <= 2000
    assert len(out.technical_explanation or "") > len(out.follow_up_question)


def test_commit_turn_binds_pending_from_trailing_question():
    mem = _mem()
    assert not (mem.pending_evaluation_concept_id or "").strip()
    lecture = (
        "Барьеры — middleware на границе tool/LLM.\n\n"
        "На каком этапе конвейера должен срабатывать детерминированный барьер "
        "при росте невалидных аргументов?"
    )
    llm_out = deep_dive_llm_output_from_chat_text(
        lecture,
        question_sub_concept_id="детерминированные_защитные_барьеры",
    )
    assert (llm_out.follow_up_question or "").endswith("?")

    state = {
        "request": _req(),
        "memory": mem,
        "anchor": "test-anchor",
        "tutor_message": lecture,
        "llm_out": llm_out,
        "focus_sub_concept_id": "детерминированные_защитные_барьеры",
    }
    out = commit_turn_node(state)
    mem2 = out["memory"]
    assert mem2.pending_evaluation_concept_id == "детерминированные_защитные_барьеры"
    assert mem2.asked_question_sub_concept_id == "детерминированные_защитные_барьеры"


def test_commit_turn_binds_pending_when_llm_out_missed_follow_up():
    """llm_out without follow_up, but tutor text has trailing ? → still bind."""
    mem = _mem()
    lecture = (
        "Изоляция сбоев через hooks.\n\n"
        "Как предотвратить race conditions при параллельном исполнении инструментов?"
    )
    llm_out = DeepDiveLLMOutput(
        technical_explanation=lecture,
        follow_up_question="",
        question_sub_concept_id="",
    )
    state = {
        "request": _req(),
        "memory": mem,
        "anchor": "test-anchor",
        "tutor_message": lecture,
        "llm_out": llm_out,
        "focus_sub_concept_id": "детерминированные_защитные_барьеры",
    }
    out = commit_turn_node(state)
    assert out["memory"].pending_evaluation_concept_id == (
        "детерминированные_защитные_барьеры"
    )
    assert (out["llm_out"].follow_up_question or "").endswith("?")
