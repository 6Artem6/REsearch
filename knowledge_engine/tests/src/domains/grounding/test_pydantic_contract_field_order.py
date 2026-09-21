"""All tutor-message contracts follow one 4-tier field order:

1. Plan (CoT) — ``execution_plan``, when the *WithPlanContract variant is
   used. Deliberately generated first even though it is never shown to the
   learner, so a lightweight model plans the causal chain before writing
   prose — moving it after the message tier would defeat the whole feature.
2. Message (User-Facing Text) — audit/lecture_body/technical_explanation,
   follow_up_question, message_bullet_summary, references/used_sources —
   in the order the reply is actually read.
3. Additional materials & context — node_status, summary,
   referenced_diagram_id, code_snippets, introduced_terms, bridge_to_next.
4. System & RAG fields — new_gap_to_record, extracted_concepts,
   verified_sub_concept_ids, question_sub_concept_id, diagrams_referenced,
   next_recommended_subtopics, ready_for_transition, suggested_next_step,
   quick_replies (Host-owned / LightRAG / knowledge-graph bookkeeping that
   never becomes learner-authored content).

Within each tier, relative declaration order is preserved by design, not
enforced field-by-field here — this file only guards the coarse invariant
that no tier-4 (system) field is ever declared before a tier-2/3 field.
"""

from __future__ import annotations

from knowledge_engine.src.domains.grounding.tutor import (
    DeepDiveExplainContract,
    DeepDiveTutorContract,
    DeepDiveTutorWithPlanContract,
    StructuredLectureResponse,
    StructuredLectureResponseWithPlanContract,
    TopicQnaExplainContract,
    TopicQnaLectureResponse,
    TopicQnaTutorContract,
)

_DIALOGUE_SYSTEM_FIELDS = {
    "new_gap_to_record",
    "verified_sub_concept_ids",
    "question_sub_concept_id",
    "ready_for_transition",
    "suggested_next_step",
    "quick_replies",
}

_LECTURE_SYSTEM_FIELDS = {
    "diagrams_referenced",
    "next_recommended_subtopics",
    "extracted_concepts",
}

_EXPLAIN_SYSTEM_FIELDS = {
    "verified_sub_concept_ids",
    "question_sub_concept_id",
    "ready_for_transition",
    "suggested_next_step",
    "quick_replies",
}


def _assert_system_fields_last(
    fields: list[str], system_fields: set[str], *, skip_first: bool = False
) -> None:
    body = fields[1:] if skip_first else fields
    seen_system = False
    for name in body:
        is_system = name in system_fields
        if is_system:
            seen_system = True
        elif seen_system:
            raise AssertionError(
                f"non-system field {name!r} declared after a system field in {fields}"
            )


def test_deep_dive_tutor_contract_system_fields_last():
    _assert_system_fields_last(
        list(DeepDiveTutorContract.model_fields), _DIALOGUE_SYSTEM_FIELDS
    )


def test_deep_dive_tutor_with_plan_contract_execution_plan_first_then_grouped():
    fields = list(DeepDiveTutorWithPlanContract.model_fields)
    assert fields[0] == "execution_plan"
    _assert_system_fields_last(fields, _DIALOGUE_SYSTEM_FIELDS, skip_first=True)


def test_topic_qna_tutor_contract_system_fields_last():
    _assert_system_fields_last(
        list(TopicQnaTutorContract.model_fields), _DIALOGUE_SYSTEM_FIELDS
    )


def test_structured_lecture_response_system_fields_last():
    _assert_system_fields_last(
        list(StructuredLectureResponse.model_fields), _LECTURE_SYSTEM_FIELDS
    )


def test_structured_lecture_response_with_plan_contract_execution_plan_first():
    fields = list(StructuredLectureResponseWithPlanContract.model_fields)
    assert fields[0] == "execution_plan"
    _assert_system_fields_last(fields, _LECTURE_SYSTEM_FIELDS, skip_first=True)


def test_topic_qna_lecture_response_system_fields_last():
    _assert_system_fields_last(
        list(TopicQnaLectureResponse.model_fields), _LECTURE_SYSTEM_FIELDS
    )


def test_deep_dive_explain_contract_system_fields_last():
    _assert_system_fields_last(
        list(DeepDiveExplainContract.model_fields), _EXPLAIN_SYSTEM_FIELDS
    )


def test_topic_qna_explain_contract_system_fields_last():
    _assert_system_fields_last(
        list(TopicQnaExplainContract.model_fields), _EXPLAIN_SYSTEM_FIELDS
    )


def test_explain_field_set_matches_base():
    from knowledge_engine.src.domains.grounding.tutor import _ExplainFieldsBase

    assert set(DeepDiveExplainContract.model_fields) == set(
        _ExplainFieldsBase.model_fields
    ) | {"follow_up_question", "question_sub_concept_id"}


def test_message_tier_fields_precede_context_tier_in_dialogue():
    """audit/technical_explanation/follow_up_question/message_bullet_summary/

    references must all come before node_status/summary/referenced_diagram_id/
    introduced_terms (tier 2 before tier 3)."""
    fields = list(DeepDiveTutorContract.model_fields)
    message_tier = {
        "audit",
        "technical_explanation",
        "follow_up_question",
        "message_bullet_summary",
        "references",
    }
    context_tier = {"node_status", "summary", "referenced_diagram_id", "introduced_terms"}
    last_message_idx = max(fields.index(f) for f in message_tier if f in fields)
    first_context_idx = min(fields.index(f) for f in context_tier if f in fields)
    assert last_message_idx < first_context_idx, fields


def test_message_tier_fields_precede_context_tier_in_lecture():
    fields = list(StructuredLectureResponse.model_fields)
    message_tier = {
        "lecture_body",
        "follow_up_question",
        "message_bullet_summary",
        "used_sources",
    }
    context_tier = {
        "summary",
        "referenced_diagram_id",
        "code_snippets",
        "introduced_terms",
        "bridge_to_next",
    }
    last_message_idx = max(fields.index(f) for f in message_tier if f in fields)
    first_context_idx = min(fields.index(f) for f in context_tier if f in fields)
    assert last_message_idx < first_context_idx, fields
