"""Prompt compositor: mode isolation and size guardrails."""

from __future__ import annotations

from knowledge_engine.src.node_deep_dive.prompt_types import (
    DIALOGUE_SYSTEM_PROMPT_MAX_LEN,
    DIALOGUE_SYSTEM_PROMPT_MIN_LEN,
    InteractionPromptMode,
    PromptComposeContext,
)
from knowledge_engine.src.node_deep_dive.tutor_prompt_builder import (
    build_dense_system,
    build_dialogue_system,
    build_lecture_chat_system,
    compose_system_prompt,
)


def test_dialogue_system_prompt_is_english_core():
    text = build_dialogue_system()
    assert "DeepDiveTutorContract" in text
    assert "You are an expert technical Tutor" in text
    assert "fluent Russian" in text
    assert "technical_explanation" in text
    assert "peer-диалог" not in text
    text = build_dialogue_system()
    assert "FIELD-BY-FIELD GENERATION RULES" not in text
    assert "Generate technical lectures matching JSON Schema contract" not in text
    assert "lecture_body (ОБЯЗАТЕЛЬНО)" not in text
    assert "lecture_body MUST include extracted_concepts" not in text


def test_lecture_dense_rules_not_triplicated():
    text = build_dense_system()
    assert text.count("RAG GROUNDING, CITATION & CODE IN JSON") == 1
    assert text.count("STRICT GROUNDEDNESS & CITATION INTEGRITY") == 1
    assert text.count("FIELD-BY-FIELD GENERATION RULES") == 1
    assert "[R1]" in text


def test_lecture_chat_uses_tutor_contract_not_structured_lecture():
    text = build_lecture_chat_system()
    assert "DeepDiveTutorContract" in text
    assert "FIELD-BY-FIELD GENERATION RULES" not in text


def test_dialogue_prompt_length_budget(caplog):
    import logging

    caplog.set_level(
        logging.INFO, logger="knowledge_engine.src.node_deep_dive.tutor_prompt_builder"
    )
    text = compose_system_prompt(InteractionPromptMode.DIALOGUE_FEEDBACK)
    n = len(text)
    assert (
        DIALOGUE_SYSTEM_PROMPT_MIN_LEN <= n <= DIALOGUE_SYSTEM_PROMPT_MAX_LEN
    ), f"unexpected dialogue system len={n}"
    assert any("tutor_prompt dialogue system" in r.message for r in caplog.records)


def test_recency_moved_out_of_cached_system_into_dynamic_context():
    """Recency stays out of system_instruction (cache-stable); lives on ctx.recency_tail."""
    ctx = PromptComposeContext()
    text = compose_system_prompt(InteractionPromptMode.DIALOGUE_FEEDBACK, context=ctx)
    assert "=== CRITICAL_RULES_RECENCY" not in text
    assert ctx.recency_tail.count("=== CRITICAL_RULES_RECENCY") == 1


def test_dialogue_depth_and_evaluation_rules():
    text = build_dialogue_system()
    assert "DEPTH_AND_EVALUATION_RULES" in text
    assert "CONTEXT-BOUNDED QUESTION FACTORY" in text
    assert "Senior Systems Architect" in text
    assert "follow_up_question" in text
    assert "TOPIC COMPLETION INSTRUCTIONS" in text
    assert "Host Pathway Flag" in text
    assert "base_complete" in text
    assert "optional_fork" in text
    assert "overlay_offer" in text
    assert "EVALUATOR_CRITIQUE_JSON" in text
    assert "Host-owned (Python)" in text
    assert "USER CHOICE HANDLING" not in text
    assert "Базовая теория закрыта/усвоена" in text  # only as FORBIDDEN
    assert "DEEP DIVE & MECH CONTENT RULES" in text
    assert "Host next_action" in text or "chip processing" in text.lower()
    # Must not teach LLM to invent chip menus / compute flags.
    assert 'quick_replies=[' not in text
    assert "WHEN USER REQUESTS" not in text


def test_lecture_chat_topic_completion_cta():
    text = build_lecture_chat_system()
    assert "TOPIC COMPLETION" in text
    assert "Host Pathway Flag" in text or "pathway" in text.lower()
    assert "USER CHOICE HANDLING" not in text
