"""Streaming compose for DeepDiveTutorContract fields."""

from __future__ import annotations

from knowledge_engine.src.adapters.llm_providers.gemini_json_stream import (
    DRILL_ACTIVE_STREAM_FIELDS,
    TOPIC_QNA_EXPLAIN_STREAM_FIELDS,
    TOPIC_QNA_TUTOR_STREAM_FIELDS,
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


def test_topic_qna_tutor_stream_filter_matches_field_by_field_dispatch():
    """Regression: Topic Q&A dialogue turns streamed as a raw, growing JSON

    blob instead of clean text — TopicQnaTutorContract/TopicQnaExplainContract
    (tutor.py) had no matching schema_name branch in
    ChatSessionManager.send_chat_message_stream, so the field-extraction
    filter used everywhere else (regex-based partial JSON field decode,
    same mechanism as TUTOR_DIALOGUE_STREAM_FIELDS) never engaged for them.
    Field sets mirror the non-topic_qna ones minus follow_up_question,
    which does not exist on either schema."""
    chunks: list[str] = []
    filt = TutorDialogueFieldsStreamFilter(
        chunks.append, fields=TOPIC_QNA_TUTOR_STREAM_FIELDS
    )
    raw = '{"confirmation":"Верно.","technical_explanation":"Разбор темы."}'
    filt.feed(raw)
    filt.flush()
    assert "".join(chunks) == "Верно.\n\nРазбор темы."

    chunks2: list[str] = []
    filt2 = TutorDialogueFieldsStreamFilter(
        chunks2.append, fields=TOPIC_QNA_EXPLAIN_STREAM_FIELDS
    )
    filt2.feed('{"technical_explanation":"Ответ на вопрос пользователя."}')
    filt2.flush()
    assert "".join(chunks2) == "Ответ на вопрос пользователя."


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
    filt.feed('{"audit":{"feedback_kind":"EXACT","confirmation":"' + confirmation + '"')
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
