"""Windowed History (Zero-Overhead Summarization) + Sub-thread Isolation.

No real network/LLM calls: ``_build_history_content`` takes its
``types_module`` (normally ``google.genai.types``) as a plain parameter, so
tests pass a lightweight fake with the same ``Content``/``Part`` shape
instead of importing the real SDK — same "no real external calls" tier as
``tests/benchmarks/test_concept_affinity_rag.py``.
"""

from __future__ import annotations

from knowledge_engine.src.config import settings as ke_settings
from knowledge_engine.src.domains.grounding.chat_session_manager import (
    ChatSessionManager,
    _extract_bullet_summary,
)
from knowledge_engine.src.domains.grounding.memory_schemas import (
    SubConceptRecord,
    build_sub_concept_status_lookup,
)
from knowledge_engine.src.domains.grounding.tutor import (
    DeepDiveTutorContract,
    StructuredLectureResponse,
    TopicQnaLectureResponse,
    TopicQnaTutorContract,
)
from knowledge_engine.src.domains.grounding.tutor_field_limits import (
    SCHEMA_BULLET_ITEM_MAX_CHARS,
    SCHEMA_BULLET_SUMMARY_MAX_ITEMS,
)


class _FakePart:
    def __init__(self, text: str) -> None:
        self.text = text

    @staticmethod
    def from_text(text: str) -> "_FakePart":
        return _FakePart(text)


class _FakeContent:
    def __init__(self, role: str, parts: list[_FakePart]) -> None:
        self.role = role
        self.parts = parts


class _FakeTypesModule:
    Content = _FakeContent
    Part = _FakePart


def _texts(history: list[_FakeContent]) -> list[tuple[str, str]]:
    return [(c.role, c.parts[0].text) for c in history]


def _mgr() -> ChatSessionManager:
    return ChatSessionManager("test-scope")


# --- 1. message_bullet_summary assembly in Pydantic contracts -----------------


def test_message_bullet_summary_present_on_all_five_contracts():
    for cls in (
        StructuredLectureResponse,
        TopicQnaLectureResponse,
        DeepDiveTutorContract,
        TopicQnaTutorContract,
    ):
        assert "message_bullet_summary" in cls.model_fields, cls.__name__


def test_message_bullet_summary_clips_items_and_length():
    long_item = "x" * (SCHEMA_BULLET_ITEM_MAX_CHARS + 50)
    obj = StructuredLectureResponse.model_construct(
        lecture_body="body",
        next_recommended_subtopics=["a", "b", "c"],
        message_bullet_summary=[long_item] * (SCHEMA_BULLET_SUMMARY_MAX_ITEMS + 3),
    )
    # model_construct skips validators — re-validate explicitly via round-trip.
    validated = StructuredLectureResponse.model_validate(obj.model_dump())
    assert len(validated.message_bullet_summary) == SCHEMA_BULLET_SUMMARY_MAX_ITEMS
    assert all(
        len(item) <= SCHEMA_BULLET_ITEM_MAX_CHARS
        for item in validated.message_bullet_summary
    )


def test_extract_bullet_summary_pulls_field_from_raw_json_text():
    payload = DeepDiveTutorContract(
        audit={
            "feedback_kind": "EXACT",
            "accuracy_grade": "EXACT_AND_CORRECT",
            "user_claims_analysis": ["user got it right"],
            "detected_errors_or_misconceptions": [],
            "confirmation": "Correct.",
        },
        message_bullet_summary=["claim one [S1]", "claim two [R2]"],
    )
    text = payload.model_dump_json()
    bullets = _extract_bullet_summary(text, DeepDiveTutorContract)
    assert bullets == ["claim one [S1]", "claim two [R2]"]


def test_extract_bullet_summary_returns_none_on_bad_json_or_missing_schema():
    assert _extract_bullet_summary("not json", DeepDiveTutorContract) is None
    assert _extract_bullet_summary('{"a": 1}', None) is None


# --- 2 & 3. get_or_create_live_chat windowing: raw N-1, bullets/fallback N-x --


def _sample_turns() -> list[dict]:
    return [
        {"role": "user", "content": "Q1", "sub_concept_id": "concept_a"},
        {
            "role": "model",
            "content": "FULL RAW ANSWER ONE",
            "sub_concept_id": "concept_a",
            "bullet_summary": ["claim 1a", "claim 1b"],
        },
        {
            "role": "user",
            "content": "Q2 no bullets available",
            "sub_concept_id": "concept_a",
        },
        {
            "role": "model",
            "content": "FULL RAW ANSWER TWO (no bullet_summary stored)",
            "sub_concept_id": "concept_a",
        },
        {"role": "user", "content": "Q3 latest", "sub_concept_id": "concept_a"},
        {
            "role": "model",
            "content": "FULL RAW ANSWER THREE (latest)",
            "sub_concept_id": "concept_a",
        },
    ]


def test_windowed_history_substitutes_old_model_turns_with_bullets(monkeypatch):
    monkeypatch.setattr(ke_settings, "DIALOG_WINDOWED_HISTORY_ENABLED", True)
    mgr = _mgr()
    history = mgr._build_history_content(_FakeTypesModule, _sample_turns())
    texts = _texts(history)
    # Oldest model turn (has bullet_summary) -> substituted with bullets.
    assert texts[1] == ("model", "- claim 1a\n- claim 1b")
    # Latest (last) pair stays fully raw regardless of the flag.
    assert texts[-2] == ("user", "Q3 latest")
    assert texts[-1] == ("model", "FULL RAW ANSWER THREE (latest)")


def test_windowed_history_fallback_truncates_when_bullet_summary_missing(
    monkeypatch,
):
    monkeypatch.setattr(ke_settings, "DIALOG_WINDOWED_HISTORY_ENABLED", True)
    mgr = _mgr()
    history = mgr._build_history_content(_FakeTypesModule, _sample_turns())
    texts = _texts(history)
    # Second-oldest model turn has NO bullet_summary -> truncation fallback,
    # not silently dropped and not left as arbitrarily-long raw text.
    assert texts[3] == ("model", "FULL RAW ANSWER TWO (no bullet_summary stored)"[:400])


def test_windowed_history_substitutes_old_user_turns_with_status_lookup(monkeypatch):
    monkeypatch.setattr(ke_settings, "DIALOG_WINDOWED_HISTORY_ENABLED", True)
    mgr = _mgr()
    sc = SubConceptRecord(
        id="concept_a",
        label="Concept A",
        status="verified",
        last_accuracy_grade="EXACT_AND_CORRECT",
    )
    lookup = build_sub_concept_status_lookup([sc])
    history = mgr._build_history_content(
        _FakeTypesModule, _sample_turns(), sub_concept_status_lookup=lookup
    )
    texts = _texts(history)
    assert texts[0] == (
        "user",
        "Ответ по [Concept A]: verified, grade=EXACT_AND_CORRECT",
    )


def test_subthread_isolation_collapses_foreign_sub_concept_turns(monkeypatch):
    monkeypatch.setattr(ke_settings, "DIALOG_SUBTHREAD_ISOLATION_ENABLED", True)
    mgr = _mgr()
    turns = [
        {
            "role": "model",
            "content": "FULL ANSWER ABOUT B",
            "sub_concept_id": "concept_b",
            "bullet_summary": ["b claim"],
        },
        {"role": "user", "content": "Q latest", "sub_concept_id": "concept_a"},
        {
            "role": "model",
            "content": "FULL ANSWER LATEST",
            "sub_concept_id": "concept_a",
        },
    ]
    history = mgr._build_history_content(
        _FakeTypesModule, turns, current_sub_concept_id="concept_a"
    )
    texts = _texts(history)
    # Foreign sub_concept (b) turn collapses to its bullet summary even
    # though isolation alone (without windowed history) would otherwise
    # leave non-last turns raw.
    assert texts[0] == ("model", "- b claim")
    assert texts[-1] == ("model", "FULL ANSWER LATEST")


# --- 4. Flags off => identical to pre-existing behavior -----------------------


def test_both_flags_off_keeps_all_turns_raw(monkeypatch):
    # Explicit, not relied-on-default: this must hold even if the real
    # environment (.env) has these flags turned on for production use.
    monkeypatch.setattr(ke_settings, "DIALOG_WINDOWED_HISTORY_ENABLED", False)
    monkeypatch.setattr(ke_settings, "DIALOG_SUBTHREAD_ISOLATION_ENABLED", False)
    mgr = _mgr()
    turns = _sample_turns()
    history = mgr._build_history_content(_FakeTypesModule, turns)
    texts = _texts(history)
    expected = [
        ("model" if t["role"] in ("model", "tutor") else "user", t["content"])
        for t in turns
    ]
    assert texts == expected


# --- 5. Тезисы at the TOP of the displayed tutor message ---------------------


def test_deep_dive_llm_output_carries_message_bullet_summary_from_raw_contract():
    """DeepDiveLLMOutput.model_validate(raw.model_dump()) (engine.py) must not
    silently drop message_bullet_summary — it previously had no such field,
    so the raw contract's thesis claims never survived the host-normalized
    conversion and were never displayed anywhere."""
    from knowledge_engine.src.domains.grounding.schemas import DeepDiveLLMOutput

    raw = DeepDiveTutorContract(
        audit={
            "feedback_kind": "EXACT",
            "accuracy_grade": "EXACT_AND_CORRECT",
            "user_claims_analysis": ["ok"],
            "detected_errors_or_misconceptions": [],
            "confirmation": "Correct.",
        },
        message_bullet_summary=["claim one [S1]", "claim two [R2]"],
    )
    out = DeepDiveLLMOutput.model_validate(raw.model_dump())
    assert out.message_bullet_summary == ["claim one [S1]", "claim two [R2]"]


def test_compose_tutor_dialogue_orders_evaluator_message_thesis_question():
    """Порядок: оценщик (плашка+audit feedback) → сообщение
    (technical_explanation) → Тезисы → вопрос (follow_up_question). Тезисы
    идут ПОСЛЕ основного текста (там их реально пишет Gemini в JSON — после
    technical_explanation, перед follow_up_question) и после того места, где
    уже останавливается стриминг — так финальная подмена текста только
    ДОПИСЫВАЕТ хвост, а не переставляет уже прочитанное сверху."""
    from knowledge_engine.src.domains.grounding.schemas import DeepDiveLLMOutput
    from knowledge_engine.src.domains.grounding.tutor_dialogue import (
        compose_tutor_dialogue_from_output,
    )

    out = DeepDiveLLMOutput(
        feedback_on_answer="Уточните механизм X.",
        technical_explanation="Разбор темы.",
        follow_up_question="Почему так?",
        message_bullet_summary=["claim one [S1]", "claim two [R2]"],
    )
    composed = compose_tutor_dialogue_from_output(out)
    assert composed.startswith("Уточните механизм X.")
    assert composed.index("Уточните механизм X.") < composed.index("Разбор темы.")
    assert composed.index("Разбор темы.") < composed.index("Тезисы")
    assert composed.index("Тезисы") < composed.index("Почему так?")


def test_compose_tutor_dialogue_omits_thesis_block_when_empty():
    from knowledge_engine.src.domains.grounding.tutor_dialogue import (
        compose_tutor_dialogue_message,
    )

    composed = compose_tutor_dialogue_message(
        feedback_on_answer="fb", message_bullet_summary=[]
    )
    assert "Тезисы" not in composed
    assert composed == "fb"


def test_record_turn_tags_sub_concept_id_and_bullet_summary():
    mgr = _mgr()
    mgr.create_new_session("gemini-3.5-flash-lite", "lab")
    mgr.record_turn(
        "lab",
        "user answer",
        "tutor reply",
        bullet_summary=["claim x"],
        sub_concept_id="concept_z",
    )
    stored = mgr.get("lab")
    assert stored.api_turns[0]["sub_concept_id"] == "concept_z"
    assert stored.api_turns[1]["sub_concept_id"] == "concept_z"
    assert stored.api_turns[1]["bullet_summary"] == ["claim x"]
