"""Prompt compositor: mode isolation and size guardrails."""

from __future__ import annotations

from knowledge_engine.src.node_deep_dive.prompt_types import (
    DIALOGUE_SYSTEM_PROMPT_MAX_LEN,
    DIALOGUE_SYSTEM_PROMPT_MIN_LEN,
    InteractionPromptMode,
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


def test_recency_appears_once_in_dialogue():
    text = build_dialogue_system()
    assert text.count("=== CRITICAL_RULES_RECENCY") == 1


def test_dialogue_depth_and_evaluation_rules():
    text = build_dialogue_system()
    assert "DEPTH_AND_EVALUATION_RULES" in text
    assert "Senior Systems Architect" in text
    assert "follow_up_question" in text
    assert "TOPIC COMPLETION RULE" in text
    assert "ready_for_transition=true" in text
    assert "MUST NOT be empty" in text
    assert "Do NOT invent or name next curriculum nodes" in text or (
        "Do NOT invent or name the next curriculum node" in text
    )
    assert (
        "OPTIONAL LAYERS" in text
        or "OPTIONAL_LAYER" in text
        or "опциональный слой" in text
    )
    assert "sota" in text.lower()
    assert "Хочу Gloss" in text
    assert "DEEP DIVE & MECH EXTRACTION" in text
    assert "DO NOT CLOSE THE NODE AUTOMATICALLY" in text


def test_lecture_chat_topic_completion_cta():
    text = build_lecture_chat_system()
    assert "TOPIC COMPLETION" in text
    assert "do NOT invent next node" in text.lower() or "GRAPH LIMIT" in text
    assert "sota" in text.lower() or "SotA" in text
