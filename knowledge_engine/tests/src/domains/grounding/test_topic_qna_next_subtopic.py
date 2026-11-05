"""Topic Q&A: «Следующая подтема (без зачёта)».

Кнопка шлёт ``[action:next_subtopic]``: текущая подтема получает
``deferred=True`` (статус НЕ меняется — зачёта нет), открытый вопрос
снимается, следующей выбирается неотложенная незачтённая подтема;
к отложенным возвращаемся, когда другие кончатся."""

from __future__ import annotations

from knowledge_engine.src.domains.grounding.concept_map import (
    is_quick_reply_control_message,
)
from knowledge_engine.src.domains.grounding.concept_map_state import (
    _status_hint_for_sub_concept,
    defer_current_sub_concept,
    select_next_sub_concept,
)
from knowledge_engine.src.domains.grounding.control_intent import (
    _classify_explicit_and_exact,
)
from knowledge_engine.src.domains.grounding.intent_definitions import (
    CHIP_NEXT_SUBTOPIC,
)
from knowledge_engine.src.domains.grounding.memory_schemas import (
    SessionMemory,
    SubConceptRecord,
)
from knowledge_engine.src.domains.grounding.prompt_factory import (
    display_user_after_mode_prefix,
)

_BUTTON = f"[action:next_subtopic] {CHIP_NEXT_SUBTOPIC}"


def _mem(*rows: tuple[str, str]) -> SessionMemory:
    mem = SessionMemory()
    mem.sub_concepts = [
        SubConceptRecord(id=cid, label=f"Подтема {cid}", status=status)
        for cid, status in rows
    ]
    return mem


def test_button_is_explicit_control_chip_and_not_scored() -> None:
    assert _classify_explicit_and_exact(_BUTTON) == "next_subtopic"
    assert _classify_explicit_and_exact(CHIP_NEXT_SUBTOPIC) == "next_subtopic"
    assert is_quick_reply_control_message(_BUTTON, SessionMemory())


def test_defer_keeps_status_clears_pending_and_moves_on() -> None:
    mem = _mem(("vec_db", "partial"), ("sharding", "unchecked"))
    mem.pending_evaluation_concept_id = "vec_db"
    mem.pending_evaluation_interaction_axis = "topic_qna"
    mem.asked_question_sub_concept_id = "vec_db"

    assert defer_current_sub_concept(mem) == "vec_db"

    row = mem.sub_concepts[0]
    assert row.deferred is True
    assert row.status == "partial"  # без зачёта
    assert mem.pending_evaluation_concept_id == ""
    assert mem.pending_evaluation_interaction_axis == ""
    assert select_next_sub_concept(mem).id == "sharding"


def test_deferred_go_last_and_come_back_when_nothing_else_left() -> None:
    # gap обычно идёт первым, но отложенный gap — после неотложенного partial.
    mem = _mem(("aa", "gap"), ("bb", "partial"), ("cc", "verified"))
    mem.sub_concepts[0].deferred = True
    assert select_next_sub_concept(mem).id == "bb"

    mem.sub_concepts[1].deferred = True
    assert select_next_sub_concept(mem).id == "aa"  # остались только отложенные


def test_verified_current_is_not_deferred() -> None:
    mem = _mem(("done", "verified"), ("next", "unchecked"))
    mem.next_question_concept_id = "done"
    assert defer_current_sub_concept(mem) == ""
    assert mem.sub_concepts[0].deferred is False


def test_status_hint_and_history_text() -> None:
    row = SubConceptRecord(id="vec_db", label="Векторные БД", status="partial")
    row.deferred = True
    assert "без зачёта" in _status_hint_for_sub_concept(row)
    assert display_user_after_mode_prefix(_BUTTON)[0] == CHIP_NEXT_SUBTOPIC
