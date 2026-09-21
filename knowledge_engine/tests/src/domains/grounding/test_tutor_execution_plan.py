"""CoT Execution Plan (ENABLE_TUTOR_EXECUTION_PLAN, default off) — a hidden
`execution_plan` field (short Mermaid causal graph) generated FIRST, before
technical_explanation, so a lightweight model plans the causal chain before
writing prose. Must never reach the client, streamed or not.
"""

from __future__ import annotations

import inspect
import json

from knowledge_engine.src.adapters.llm_providers.gemini_json_stream import (
    TUTOR_DIALOGUE_STREAM_FIELDS,
    TutorDialogueFieldsStreamFilter,
    structured_stream_text_field,
)
from knowledge_engine.src.domains.grounding.engine import _resolve_tutor_response_schema
from knowledge_engine.src.domains.grounding.schemas import DeepDiveLLMOutput
from knowledge_engine.src.domains.grounding.tutor import (
    DeepDiveTutorContract,
    DeepDiveTutorWithPlanContract,
    StructuredLectureResponse,
    StructuredLectureResponseWithPlanContract,
    structured_lecture_to_dense,
)


def _sample_audit(feedback_kind: str = "EXACT") -> dict:
    if feedback_kind == "EXACT":
        return {
            "feedback_kind": "EXACT",
            "accuracy_grade": "EXACT_AND_CORRECT",
            "user_claims_analysis": ["ok"],
            "detected_errors_or_misconceptions": [],
            "confirmation": "Correct.",
        }
    return {
        "feedback_kind": "NEEDS_CORRECTION",
        "accuracy_grade": "PARTIAL",
        "user_claims_analysis": ["ok"],
        "detected_errors_or_misconceptions": ["missing X"],
        "correction_breakdown": "Missing X.",
    }


# --- 1. Contract switching (Step 2.3 / engine.py::_resolve_tutor_response_schema) ---


def test_resolver_returns_plain_contract_when_flag_off(monkeypatch):
    monkeypatch.setattr(
        "knowledge_engine.src.config.settings.ENABLE_TUTOR_EXECUTION_PLAN", False
    )
    schema = _resolve_tutor_response_schema(
        star_guard=False,
        evaluator_skipped=False,
        factory_mode="",
        drill_schema=None,
        interaction_axis="lecture_self_check",
        last_eval_directive="",
    )
    assert schema is DeepDiveTutorContract


def test_resolver_returns_plan_contract_when_flag_on(monkeypatch):
    monkeypatch.setattr(
        "knowledge_engine.src.config.settings.ENABLE_TUTOR_EXECUTION_PLAN", True
    )
    schema = _resolve_tutor_response_schema(
        star_guard=False,
        evaluator_skipped=False,
        factory_mode="",
        drill_schema=None,
        interaction_axis="lecture_self_check",
        last_eval_directive="",
    )
    assert schema is DeepDiveTutorWithPlanContract


def test_flag_never_affects_other_schema_branches(monkeypatch):
    """drill_schema / star_guard / topic_qna paths are untouched by the flag
    — the task only asked for DeepDiveTutorContract to gain a plan variant."""
    monkeypatch.setattr(
        "knowledge_engine.src.config.settings.ENABLE_TUTOR_EXECUTION_PLAN", True
    )

    class _FakeDrillSchema:
        pass

    schema = _resolve_tutor_response_schema(
        star_guard=False,
        evaluator_skipped=False,
        factory_mode="",
        drill_schema=_FakeDrillSchema,
        interaction_axis="lecture_self_check",
        last_eval_directive="",
    )
    assert schema is _FakeDrillSchema

    from knowledge_engine.src.domains.grounding.tutor import DeepDiveDeepAnalysisContract

    schema = _resolve_tutor_response_schema(
        star_guard=True,
        evaluator_skipped=False,
        factory_mode="",
        drill_schema=None,
        interaction_axis="lecture_self_check",
        last_eval_directive="",
    )
    assert schema is DeepDiveDeepAnalysisContract


# --- 2. Schema shape: execution_plan genuinely first, no field drift ------------


def test_execution_plan_is_the_first_declared_field():
    assert next(iter(DeepDiveTutorWithPlanContract.model_fields)) == "execution_plan"


def test_field_set_matches_deep_dive_tutor_contract():
    """Guards the DRY tradeoff: DeepDiveTutorWithPlanContract re-declares
    every DeepDiveTutorContract field by hand (Pydantic always appends
    subclass-added fields last, so a plain subclass cannot put execution_plan
    first — verified empirically). If someone edits one contract and forgets
    the other, this must fail."""
    assert set(DeepDiveTutorWithPlanContract.model_fields) == set(
        DeepDiveTutorContract.model_fields
    ) | {"execution_plan"}


def test_execution_plan_is_required_and_bounded():
    field = DeepDiveTutorWithPlanContract.model_fields["execution_plan"]
    assert field.is_required()
    payload = {
        "execution_plan": "x" * 100_000,
        "audit": _sample_audit(),
        "technical_explanation": "text",
        "follow_up_question": "q?",
    }
    try:
        DeepDiveTutorWithPlanContract.model_validate(payload)
        raised = False
    except Exception:
        raised = True
    assert raised, "execution_plan must have an enforced max_length"


# --- 3. Validation on a realistic JSON response ---------------------------------


def test_contract_validates_a_realistic_plan_response():
    payload = {
        "execution_plan": (
            "graph TD\nA[AppendOnly] --> B[PartsAccumulation] --> C{Can it "
            "merge instantly? No, because...} --> D[MergeTree] -. "
            "BANNED_IN_TEXT .-> E[SHADOW: Read IO Impact]"
        ),
        "audit": _sample_audit(),
        "technical_explanation": "Каждая вставка создаёт независимый парт [S1].",
        "message_bullet_summary": ["Парты независимы [S1]."],
        "follow_up_question": "Что произойдёт при росте числа партов?",
        "question_sub_concept_id": "parts",
    }
    obj = DeepDiveTutorWithPlanContract.model_validate(payload)
    assert obj.execution_plan.startswith("graph TD")
    assert obj.feedback_on_answer == "Correct."


def test_execution_plan_is_dropped_when_converting_to_host_llm_output():
    """DeepDiveLLMOutput has no execution_plan field — Pydantic v2 default
    (extra='ignore') silently drops it, which is exactly the desired
    never-reaches-the-client behavior."""
    payload = {
        "execution_plan": "graph TD\nA --> B -. BANNED_IN_TEXT .-> C[SHADOW: X]",
        "audit": _sample_audit(),
        "technical_explanation": "text",
        "follow_up_question": "q?",
    }
    raw = DeepDiveTutorWithPlanContract.model_validate(payload)
    out = DeepDiveLLMOutput.model_validate(raw.model_dump())
    assert not hasattr(out, "execution_plan")
    assert "execution_plan" not in out.model_dump()


# --- 4. Streaming filter must never leak execution_plan -------------------------


def test_execution_plan_not_in_stream_field_allowlist():
    assert "execution_plan" not in TUTOR_DIALOGUE_STREAM_FIELDS


def test_structured_stream_text_field_opts_out_for_plan_contract():
    assert structured_stream_text_field(DeepDiveTutorWithPlanContract) is None


def test_chat_session_manager_routes_plan_contract_through_dialogue_filter():
    """Regression: without this, send_chat_message_stream's schema_name
    dispatch would not recognize 'DeepDiveTutorWithPlanContract' (a new,
    unrelated class name — not a DeepDiveTutorContract subclass), field_filter
    would stay None, and raw growing JSON (including execution_plan) would
    be streamed to the client unfiltered via the bare stream_callback
    fallback. Source-level check since exercising the real method needs a
    live google.genai chat session."""
    from knowledge_engine.src.domains.grounding.chat_session_manager import (
        ChatSessionManager,
    )

    src = inspect.getsource(ChatSessionManager.send_chat_message_stream)
    assert "DeepDiveTutorWithPlanContract" in src


def test_dialogue_stream_filter_never_emits_execution_plan_content():
    emitted: list[str] = []
    filt = TutorDialogueFieldsStreamFilter(emitted.append)
    raw = {
        "execution_plan": "graph TD\nA --> B -. BANNED_IN_TEXT .-> C[SHADOW: Bottleneck]",
        "audit": _sample_audit(),
        "technical_explanation": "Каждая вставка создаёт независимый парт.",
        "follow_up_question": "Что произойдёт при росте числа партов?",
    }
    buffer = json.dumps(raw, ensure_ascii=False)
    # Feed byte-by-byte to exercise the same incremental path as real streaming.
    for i in range(0, len(buffer), 7):
        filt.feed(buffer[i : i + 7])
    filt.flush()
    full = "".join(emitted)
    assert "graph TD" not in full
    assert "SHADOW" not in full
    assert "BANNED_IN_TEXT" not in full
    assert "Каждая вставка" in full
    assert "Что произойдёт" in full


# --- 5. Lecture variant (node_content_generator.py::generate_dense_material) ----


def test_lecture_execution_plan_is_first_and_field_set_matches():
    fields = list(StructuredLectureResponseWithPlanContract.model_fields)
    assert fields[0] == "execution_plan"
    assert set(fields) - {"execution_plan"} == set(
        StructuredLectureResponse.model_fields
    )


def test_node_content_generator_selects_lecture_plan_contract_behind_flag(monkeypatch):
    import knowledge_engine.src.domains.grounding.node_content_generator as ncg

    monkeypatch.setattr(
        "knowledge_engine.src.config.settings.ENABLE_TUTOR_EXECUTION_PLAN", True
    )
    src = inspect.getsource(ncg.generate_dense_material)
    assert "StructuredLectureResponseWithPlanContract" in src
    assert "ENABLE_TUTOR_EXECUTION_PLAN" in src


def test_lecture_plan_contract_converts_to_dense_material_dropping_plan():
    payload = {
        "execution_plan": "graph TD\nA --> B -. BANNED_IN_TEXT .-> C[SHADOW: Read IO]",
        "lecture_body": "Каждый парт независим [S1].",
        "next_recommended_subtopics": ["a", "b", "c"],
        "message_bullet_summary": ["Парты независимы [S1]."],
        "follow_up_question": "Что произойдёт при накоплении партов?",
    }
    raw = StructuredLectureResponseWithPlanContract.model_validate(payload)
    dense = structured_lecture_to_dense(raw)
    assert not hasattr(dense, "execution_plan")
    assert "execution_plan" not in dense.model_dump()
    assert dense.lecture_body == "Каждый парт независим [S1]."
    assert dense.follow_up_question == "Что произойдёт при накоплении партов?"


def test_lecture_stream_field_stays_lecture_body_for_plan_contract():
    assert (
        structured_stream_text_field(StructuredLectureResponseWithPlanContract)
        == "lecture_body"
    )
