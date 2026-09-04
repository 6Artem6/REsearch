"""step_analysis intent is now deterministic (VectorIntentRouter), not LLM-guessed.

Covers the refactor from prompt.txt: intent classification is removed from the
Gemini step_analysis call and resolved via classify_control_chip upstream.
"""

from __future__ import annotations

from knowledge_engine.src.node_deep_dive.control_intent import classify_control_chip
from knowledge_engine.src.node_deep_dive.intent_definitions import (
    EVALUATOR_SKIP_INTENTS,
)
from knowledge_engine.src.node_deep_dive.step_pipeline import (
    resolve_user_intent_from_chip,
)
from knowledge_engine.src.node_deep_dive.vector_intent_router import (
    VectorIntentRouter,
    set_vector_intent_router_for_tests,
)
from knowledge_engine.tests.intent_embed_probe import lexical_probe_embed

_WAL_ANSWER = (
    "Если мутации стейта постоянны, то можно пойти по архитектуре снапшота, "
    "где фиксируется одно начальное состояние, а все изменения - это набор "
    "последовательных операций над ним как в WAL. Это позволит сохранить "
    "состояние промежуточное достаточное количество раз с возможностью "
    "отката к нужному, совершив обратные операции или от начального "
    "проделав повторно операции."
)


def test_lecture_finalize_shift_focus_map_to_legacy_intents():
    """Only these three chips are translated for engine.py/tutor_behavior_state.py
    string comparisons; every other chip or no-match stays ANSWER."""
    assert resolve_user_intent_from_chip("lecture") == "INTENT_EXPLAIN"
    assert resolve_user_intent_from_chip("finalize") == "INTENT_FINALIZE"
    assert resolve_user_intent_from_chip("shift_focus") == "INTENT_SHIFT_FOCUS"
    assert resolve_user_intent_from_chip("gloss") == "ANSWER"
    assert resolve_user_intent_from_chip("") == "ANSWER"
    assert resolve_user_intent_from_chip(None) == "ANSWER"  # type: ignore[arg-type]


def test_answer_message_resolves_without_any_control_chip():
    """A substantive free-text answer never matches the vector catalog — intent
    resolution needs 0 LLM calls and 0 catalog hits to land on ANSWER."""
    router = VectorIntentRouter(
        threshold=0.82,
        embed_fn=lexical_probe_embed,
        persist=False,
        auto_sync=True,
        enabled=True,
    )
    set_vector_intent_router_for_tests(router)
    try:
        chip = classify_control_chip(_WAL_ANSWER)
        assert chip == ""
        assert resolve_user_intent_from_chip(chip) == "ANSWER"
    finally:
        set_vector_intent_router_for_tests(None)


def test_finalize_phrase_is_deterministic_and_blocks_evaluator():
    """Regression: before this refactor INTENT_FINALIZE never skipped the
    sub-concept evaluator (intent wasn't consulted there at all). The new
    'finalize' catalog chip fixes that via EVALUATOR_SKIP_INTENTS."""
    router = VectorIntentRouter(
        threshold=0.82,
        embed_fn=lexical_probe_embed,
        persist=False,
        auto_sync=True,
        enabled=True,
    )
    set_vector_intent_router_for_tests(router)
    try:
        chip = classify_control_chip("Давай закроем тему")
        assert chip == "finalize"
        assert resolve_user_intent_from_chip(chip) == "INTENT_FINALIZE"
        assert chip in EVALUATOR_SKIP_INTENTS
    finally:
        set_vector_intent_router_for_tests(None)


def test_shift_focus_phrase_does_not_block_evaluator():
    """shift_focus preserves pre-refactor behavior: the user still answered
    something — it must keep being graded, only the tutor's tone changes."""
    router = VectorIntentRouter(
        threshold=0.82,
        embed_fn=lexical_probe_embed,
        persist=False,
        auto_sync=True,
        enabled=True,
    )
    set_vector_intent_router_for_tests(router)
    try:
        chip = classify_control_chip("Давай зайдём под другим углом")
        assert chip == "shift_focus"
        assert resolve_user_intent_from_chip(chip) == "INTENT_SHIFT_FOCUS"
        assert chip not in EVALUATOR_SKIP_INTENTS
    finally:
        set_vector_intent_router_for_tests(None)
