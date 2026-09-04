"""Streaming compose for DeepDiveTutorContract fields."""

from __future__ import annotations

from knowledge_engine.services.gemini_json_stream import (
    DRILL_ACTIVE_STREAM_FIELDS,
    TUTOR_DIALOGUE_STREAM_FIELDS,
    TutorDialogueFieldsStreamFilter,
)


def test_tutor_dialogue_stream_filter_composes_in_order():
    chunks: list[str] = []
    filt = TutorDialogueFieldsStreamFilter(chunks.append)
    raw = (
        '{"confirmation":"Hi","technical_explanation":"Body","'
        'follow_up_question":"Next?"}'
    )
    filt.feed(raw)
    filt.flush()
    assert "".join(chunks) == "Hi\n\nBody\n\nNext?"
    assert TUTOR_DIALOGUE_STREAM_FIELDS[0] == "confirmation"


def test_drill_stream_waits_for_status_header_before_audit_confirmation():
    """Gemini fills nested audit.confirmation before status_header.

    Append-only SSE must not glue confirmation+header, then reprint confirmation.
    """
    chunks: list[str] = []
    filt = TutorDialogueFieldsStreamFilter(
        chunks.append, fields=DRILL_ACTIVE_STREAM_FIELDS
    )
    confirmation = (
        "Отличный ответ. Все ключевые аспекты управления памятью "
        "для статических и динамических типов разобраны корректно."
    )
    header = (
        "[Слой MECH: Проверено 2/2 подтем. Переходим к финальному анализу: "
        "«динамическая типизация»]"
    )
    filt.feed(
        '{"audit":{"feedback_kind":"EXACT","confirmation":"' + confirmation + '"'
    )
    assert "".join(chunks) == ""
    filt.feed(',"correction_breakdown":""},"status_header":"' + header[:20])
    assert "".join(chunks) == header[:20]
    assert confirmation not in "".join(chunks)
    filt.feed(header[20:] + '"')
    streamed = "".join(chunks)
    assert streamed == f"{header}\n\n{confirmation}"
    assert streamed.count(confirmation) == 1
    filt.feed(',"theory_body":"Теория слоя.","next_question":"Что такое refcnt?"}')
    filt.flush()
    final = "".join(chunks)
    assert final.count(confirmation) == 1
    assert final.startswith(header)
    assert "**Вопрос:** Что такое refcnt?" in final
    assert confirmation + "[" not in final.replace("\n", "")
