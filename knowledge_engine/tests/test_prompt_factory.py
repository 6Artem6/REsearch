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
    body, mode = parse_tutor_mode_prefix("[mode:deep_design] Архитектурный дизайн")
    assert mode == "deep_design"
    assert body == "Архитектурный дизайн"
    body, mode = parse_tutor_mode_prefix("[mode:advanced_analysis] Анализ уязвимостей")
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


# ---------------------------------------------------------------------------
# blitz / socratic / self_check / next_module — Intent Routing & Evaluator
# Bypass refactor: each tag gets its own isolated system prompt, is forced
# off the dense-lecture route, and (when the evaluator was skipped this turn)
# the JSON contract is swapped to the no-audit explain tail.
# ---------------------------------------------------------------------------


def test_parse_and_select_new_control_mode_tags() -> None:
    from knowledge_engine.src.node_deep_dive.blitz_mode_prompt import (
        BLITZ_MODE_PROMPT,
    )
    from knowledge_engine.src.node_deep_dive.next_module_prompt import (
        NEXT_MODULE_PROMPT,
    )
    from knowledge_engine.src.node_deep_dive.self_check_mode_prompt import (
        SELF_CHECK_MODE_PROMPT,
    )
    from knowledge_engine.src.node_deep_dive.socratic_mode_prompt import (
        SOCRATIC_MODE_PROMPT,
    )

    cases = (
        ("[mode:blitz] x", "blitz", BLITZ_MODE_PROMPT),
        ("[mode:socratic] x", "socratic", SOCRATIC_MODE_PROMPT),
        ("[mode:self_check] x", "self_check", SELF_CHECK_MODE_PROMPT),
        ("[mode:next_module] x", "next_module", NEXT_MODULE_PROMPT),
    )
    for raw, expected_mode, expected_prompt in cases:
        body, mode = parse_tutor_mode_prefix(raw)
        assert mode == expected_mode
        assert body == "x"
        system, mode2, _cleaned = select_system_prompt_and_mode(
            raw, default_system_prompt="DEFAULT"
        )
        assert mode2 == expected_mode
        assert expected_prompt.strip() in system
        assert is_factory_control_mode(expected_mode)


def test_new_control_modes_never_force_dense_lecture() -> None:
    for mode in ("blitz", "socratic", "self_check", "next_module"):
        assert is_factory_control_mode(mode)


def test_free_text_blitz_request_resolves_isolated_prompt() -> None:
    """Step 2 (vector match, no [mode:] tag) must also select the isolated
    prompt — not just fall back to the default tutor system prompt."""
    from knowledge_engine.src.node_deep_dive.blitz_mode_prompt import (
        BLITZ_MODE_PROMPT,
    )
    from knowledge_engine.src.node_deep_dive.intent_definitions import (
        INTENT_REFERENCE_PHRASES,
    )
    from knowledge_engine.src.node_deep_dive.vector_intent_router import (
        VectorIntentRouter,
        set_vector_intent_router_for_tests,
    )
    from knowledge_engine.tests.intent_embed_probe import lexical_probe_embed

    router = VectorIntentRouter(
        threshold=0.82,
        embed_fn=lexical_probe_embed,
        persist=False,
        auto_sync=True,
        enabled=True,
    )
    set_vector_intent_router_for_tests(router)
    try:
        phrase = INTENT_REFERENCE_PHRASES["blitz"][0]
        system, mode, _cleaned = select_system_prompt_and_mode(
            phrase, default_system_prompt="DEFAULT"
        )
        assert mode == "blitz"
        assert BLITZ_MODE_PROMPT.strip() in system
    finally:
        set_vector_intent_router_for_tests(None)


def test_evaluator_skipped_swaps_new_mode_prompt_to_explain_tail() -> None:
    """When the host already skipped the gap evaluator this turn (control
    chip / tag), the isolated blitz prompt must still end in the no-audit
    explain contract — never the accusatory audit/confirmation contract."""
    from types import SimpleNamespace

    from knowledge_engine.src.node_deep_dive.prompt_factory import (
        _EXPLAIN_JSON_TAIL,
        _JSON_CONTRACT_TAIL,
    )

    memory = SimpleNamespace(evaluator_skipped=True, layer_drill=None)
    system, mode, _cleaned = select_system_prompt_and_mode(
        "[mode:blitz] x",
        default_system_prompt="DEFAULT",
        memory=memory,
    )
    assert mode == "blitz"
    assert _EXPLAIN_JSON_TAIL.strip() in system
    assert _JSON_CONTRACT_TAIL.strip() not in system
