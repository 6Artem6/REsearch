"""Prompt Factory: [mode:…] selector for Quick Reply chips."""

from __future__ import annotations

from knowledge_engine.src.node_deep_dive.concept_map import (
    classify_gloss_fork_choice,
    is_quick_reply_control_message,
    orchestrate_tutor_llm_output,
)
from knowledge_engine.src.node_deep_dive.deep_dive_how_prompt import (
    DEEP_DIVE_HOW_PROMPT,
)
from knowledge_engine.src.node_deep_dive.deep_dive_mech_prompt import (
    DEEP_DIVE_MECH_PROMPT,
)
from knowledge_engine.src.node_deep_dive.gloss_summary_prompt import (
    GLOSS_SUMMARY_PROMPT,
)
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    SessionMemory,
    SubConceptRecord,
)
from knowledge_engine.src.node_deep_dive.prompt_factory import (
    display_user_after_mode_prefix,
    factory_mode_to_gloss_choice,
    is_factory_control_mode,
    parse_tutor_mode_prefix,
    select_system_prompt_and_mode,
)
from knowledge_engine.src.node_deep_dive.schemas import DeepDiveLLMOutput


def test_parse_mode_prefixes() -> None:
    body, mode = parse_tutor_mode_prefix(
        "[mode:deep_dive_mech] Разбери механики и код темы."
    )
    assert mode == "deep_dive_mech"
    assert body.startswith("Разбери механики")

    body, mode = parse_tutor_mode_prefix(
        "[mode:deep_dive_how] Разбери архитектуру темы."
    )
    assert mode == "deep_dive_how"
    assert "архитектур" in body.lower()

    body, mode = parse_tutor_mode_prefix(
        "[mode:gloss] Сформируй сжатую выжимку (Glossary) по оставшимся слоям."
    )
    assert mode == "gloss"
    assert "Glossary" in body or "выжимк" in body.lower()

    body, mode = parse_tutor_mode_prefix("обычный ответ без префикса")
    assert mode == "default"
    assert body == "обычный ответ без префикса"


def test_select_isolated_prompts() -> None:
    system, mode, cleaned = select_system_prompt_and_mode(
        "[mode:deep_dive_mech] Разбери механики и код темы.",
        default_system_prompt="DEFAULT",
    )
    assert mode == "deep_dive_mech"
    assert system != "DEFAULT"
    assert DEEP_DIVE_MECH_PROMPT.strip() in system
    assert cleaned.startswith("Разбери")

    system, mode, _ = select_system_prompt_and_mode(
        "[mode:deep_dive_how] x",
        default_system_prompt="DEFAULT",
    )
    assert mode == "deep_dive_how"
    assert DEEP_DIVE_HOW_PROMPT.strip() in system

    system, mode, _ = select_system_prompt_and_mode(
        "[mode:gloss] x",
        default_system_prompt="DEFAULT",
    )
    assert mode == "gloss"
    assert GLOSS_SUMMARY_PROMPT.strip() in system

    system, mode, cleaned = select_system_prompt_and_mode(
        "plain",
        default_system_prompt="DEFAULT",
    )
    assert mode == "default"
    assert system == "DEFAULT"
    assert cleaned == "plain"


def test_factory_mode_maps_to_gloss_choice() -> None:
    assert factory_mode_to_gloss_choice("deep_dive_mech") == "mech"
    assert factory_mode_to_gloss_choice("deep_dive_how") == "how"
    assert factory_mode_to_gloss_choice("gloss") == "gloss"
    assert is_factory_control_mode("deep_dive_mech")
    assert is_factory_control_mode("deep_analysis")
    assert not is_factory_control_mode("default")
    assert not is_factory_control_mode("lecture")


def test_parse_and_select_deep_analysis() -> None:
    from knowledge_engine.src.node_deep_dive.deep_analysis_prompt import (
        DEEP_ANALYSIS_PROMPT,
    )

    body, mode = parse_tutor_mode_prefix("[mode:deep_analysis] challenge")
    assert mode == "deep_analysis"
    assert body == "challenge"
    system, mode, cleaned = select_system_prompt_and_mode(
        "[mode:deep_analysis] challenge",
        default_system_prompt="DEFAULT",
    )
    assert mode == "deep_analysis"
    assert DEEP_ANALYSIS_PROMPT.strip() in system
    assert cleaned == "challenge"


def test_classify_mode_prefixed_chips() -> None:
    assert (
        classify_gloss_fork_choice("[mode:deep_dive_mech] Разбери механики и код темы.")
        == "mech"
    )
    assert (
        classify_gloss_fork_choice("[mode:deep_dive_how] Разбери архитектуру темы.")
        == "how"
    )
    assert (
        classify_gloss_fork_choice(
            "[mode:gloss] Сформируй сжатую выжимку (Glossary) по оставшимся слоям."
        )
        == "gloss"
    )
    assert is_quick_reply_control_message(
        "[mode:deep_dive_mech] Разбери механики и код темы."
    )


def test_orchestrate_mode_mech_holds_transition() -> None:
    mem = SessionMemory(
        pending_evaluation_concept_id="agg",
        sub_concepts=[
            SubConceptRecord(
                id="agg",
                label="agg",
                status="verified",
                why_passed=True,
                how_passed=True,
                mechanic_passed=False,
            )
        ],
    )
    out = DeepDiveLLMOutput(
        ready_for_transition=False,
        follow_up_question="Что если score=0?",
        question_sub_concept_id="agg",
    )
    packed = orchestrate_tutor_llm_output(
        mem,
        out,
        user_message="[mode:deep_dive_mech] Разбери механики и код темы.",
        node_layer="advanced",
    )
    assert packed.ready_for_transition is False


def test_overlay_chip_payload_keeps_simple_body() -> None:
    body, mode = parse_tutor_mode_prefix(
        "[mode:deep_design] Архитектурный дизайн"
    )
    assert mode == "deep_design"
    assert body == "Архитектурный дизайн"
    body, mode = parse_tutor_mode_prefix(
        "[mode:advanced_analysis] Анализ уязвимостей"
    )
    assert mode == "advanced_analysis"
    assert body == "Анализ уязвимостей"


def test_tag_only_overlay_display_fallback() -> None:
    body, mode = display_user_after_mode_prefix("[mode:deep_design]")
    assert mode == "deep_design"
    assert body == "Архитектурный дизайн"
    body, mode = display_user_after_mode_prefix("[mode:advanced_analysis]")
    assert mode == "advanced_analysis"
    assert body == "Анализ уязвимостей"
    raw_body, raw_mode = parse_tutor_mode_prefix("[mode:deep_design]")
    assert raw_mode == "deep_design"
    assert raw_body == ""
