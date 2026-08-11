"""Streaming compose for DeepDiveTutorContract fields."""

from __future__ import annotations

from knowledge_engine.services.gemini_json_stream import (
    TUTOR_DIALOGUE_STREAM_FIELDS,
    TutorDialogueFieldsStreamFilter,
)


def test_tutor_dialogue_stream_filter_composes_in_order():
    chunks: list[str] = []
    filt = TutorDialogueFieldsStreamFilter(chunks.append)
    raw = (
        '{"feedback_on_answer":"Hi","technical_explanation":"Body","'
        'follow_up_question":"Next?"}'
    )
    filt.feed(raw)
    filt.flush()
    assert chunks[-1] == "Hi\n\nBody\n\nNext?"
    assert TUTOR_DIALOGUE_STREAM_FIELDS[0] == "feedback_on_answer"
