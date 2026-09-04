"""Guardrails: free-text answers must never be mistaken for control chips."""

from __future__ import annotations

import pytest

from knowledge_engine.src.node_deep_dive.concept_map import (
    classify_gloss_fork_choice,
    is_quick_reply_control_message,
)
from knowledge_engine.src.node_deep_dive.control_intent import (
    classify_control_chip,
    is_control_chip_message,
    is_short_begin_message,
    is_short_lecture_request,
    is_short_skip_node_message,
)
from knowledge_engine.src.node_deep_dive.engine import _is_explicit_lecture_request
from knowledge_engine.src.node_deep_dive.init_context import (
    is_begin_user_message,
    user_declines_node_equivalence,
)
from knowledge_engine.src.node_deep_dive.lecture_scope import is_lecture_request_message
from knowledge_engine.src.node_deep_dive.step_pipeline import heuristic_step_analysis
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

_MECH_ANSWER = (
    "Механизм отката опирается на неизменяемые структуры данных и графовые "
    "снапшоты, где каждый узел фиксирует дельта-состояние перед вызовом "
    "внешнего инструмента, чтобы можно было безопасно откатиться. "
    "Подробно разберем принцип схемы checkpointing."
)


@pytest.fixture(autouse=True)
def _vector_router_probe(tmp_path):
    router = VectorIntentRouter(
        threshold=0.82,
        embed_fn=lexical_probe_embed,
        enabled=True,
        persist=True,
        db_path=tmp_path / "intent_lance",
        embed_model="probe-embed",
        auto_sync=True,
    )
    set_vector_intent_router_for_tests(router)
    yield router
    set_vector_intent_router_for_tests(None)


def _active_router() -> VectorIntentRouter:
    from knowledge_engine.src.node_deep_dive.vector_intent_router import (
        get_vector_intent_router,
    )

    return get_vector_intent_router()


def test_wal_architecture_answer_not_how_chip():
    intent, score = _active_router().classify(_WAL_ANSWER)
    assert intent == ""
    assert score < 0.82
    assert classify_gloss_fork_choice(_WAL_ANSWER) == ""
    assert not is_quick_reply_control_message(_WAL_ANSWER)
    assert not is_control_chip_message(_WAL_ANSWER)
    assert classify_control_chip(_WAL_ANSWER) == ""


def test_mechanism_word_in_long_answer_not_mech_chip():
    assert not is_control_chip_message(_MECH_ANSWER)
    assert classify_gloss_fork_choice(_MECH_ANSWER) == ""


def test_short_formula_answer_not_mech_chip():
    """Regression: substring «по формуле» must not skip evaluator via mech chip."""
    msg = "по формуле key = hash(payload)"
    intent, score = _active_router().classify(msg)
    assert intent == ""
    assert score < 0.82
    assert classify_gloss_fork_choice(msg) == ""
    assert not is_control_chip_message(msg)


def test_exact_chips_still_work():
    assert classify_gloss_fork_choice("Дожать HOW") == "how"
    assert classify_gloss_fork_choice("Дожать MECH") == "mech"
    assert classify_gloss_fork_choice("Хочу Gloss") == "gloss"
    assert classify_gloss_fork_choice("Идем дальше") == "next"
    assert is_control_chip_message("Дожать MECH")


def test_vector_matches_short_chip_paraphrase():
    intent, score = _active_router().classify("дожать how")
    assert intent == "how"
    assert score >= 0.82
    assert classify_control_chip("хочу mech") == "mech"
    assert classify_control_chip("дай краткий glossary по слоям") == "gloss"


def test_explicit_mode_tags_still_work():
    assert (
        classify_gloss_fork_choice("[mode:deep_dive_how] Разбери архитектуру темы.")
        == "how"
    )
    assert (
        classify_gloss_fork_choice(
            "[mode:deep_dive_mech] Разбери механики и код темы."
        )
        == "mech"
    )
    assert (
        classify_control_chip("[mode:deep_analysis] Задачка со звёздочкой")
        == "deep_analysis"
    )
    assert is_short_lecture_request("[mode:lecture] Дай плотный материал по теме.")
    assert is_lecture_request_message("[mode:lecture] Дай плотный материал по теме.")


def test_long_answer_with_podrobno_not_lecture_or_explain():
    msg = (
        "Я подробно описал, как работает middleware валидации аргументов "
        "инструмента: сначала Pydantic schema, затем reject с structured error, "
        "и только после этого ответ может попасть обратно в контекст LLM."
    )
    assert not is_lecture_request_message(msg)
    assert not _is_explicit_lecture_request(msg)
    out = heuristic_step_analysis(msg, learning_phase="dense_material")
    assert out.intent == "ANSWER"


def test_short_lecture_stub_still_lecture():
    assert is_lecture_request_message("Дай плотный материал по теме.")
    assert _is_explicit_lecture_request("Дай плотный материал по теме.")


def test_begin_and_skip_via_vector_or_exact():
    assert is_begin_user_message("начать")
    assert is_begin_user_message("[begin]")
    assert is_short_begin_message("начать")
    assert not is_begin_user_message(
        "Давай разберем архитектуру WAL подробнее, потому что "
        "начать откат без checkpoint опасно для идемпотентности платежей."
    )
    assert user_declines_node_equivalence("уже знаю")
    assert is_short_skip_node_message("пропустить")
    assert not user_declines_node_equivalence(
        "Я уже знаю OpenTelemetry на практике, но хочу уточнить, "
        "как пробрасывать correlation id через tool spans в этом пайплайне."
    )


def test_mode_selection_exact_chips_skip_llm_and_evaluator():
    """Короткий чип fast-track перехватывается до Lite/Evaluator."""
    from knowledge_engine.src.node_deep_dive.control_intent import (
        apply_mode_selection_intent,
        is_control_chip_message,
        mark_awaiting_mode_selection,
    )
    from knowledge_engine.src.node_deep_dive.intent_definitions import (
        MODE_SELECTION_SLOT,
    )
    from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory
    from knowledge_engine.src.node_deep_dive.step_pipeline import (
        should_run_step_analysis_llm,
    )

    mem = SessionMemory()
    mark_awaiting_mode_selection(mem)
    assert mem.pending_control_slot == MODE_SELECTION_SLOT
    assert not (mem.pending_evaluation_concept_id or "").strip()

    for label, intent in (
        ("практика", "practice"),
        ("проверка", "check"),
        ("пропустить", "skip"),
    ):
        assert classify_control_chip(label, memory=mem) == intent
        assert is_control_chip_message(label, memory=mem)
        assert should_run_step_analysis_llm(label, mem, "chat") is False

    assert apply_mode_selection_intent(mem, "practice") is True
    assert mem.pending_control_slot == ""
    assert mem.learning_mode == "socratic_point"


def test_mode_selection_slot_vector_paraphrase():
    """Парафраз слота идёт в BGE с приоритетом practice/check/skip."""
    from knowledge_engine.src.node_deep_dive.control_intent import (
        mark_awaiting_mode_selection,
    )
    from knowledge_engine.src.node_deep_dive.memory_schemas import SessionMemory

    mem = SessionMemory()
    mark_awaiting_mode_selection(mem)
    assert classify_control_chip("хочу практику", memory=mem) == "practice"
    assert classify_control_chip("сделаем проверку", memory=mem) == "check"


def test_praktika_inside_long_answer_is_not_practice_chip():
    msg = (
        "Я уже знаю OpenTelemetry на практике, но хочу уточнить, "
        "как пробрасывать correlation id через tool spans в этом пайплайне."
    )
    assert classify_control_chip(msg) == ""
    assert not is_control_chip_message(msg)


def test_lancedb_sync_skips_ollama_on_second_boot(tmp_path):
    """Cold start embeds once; restart with same DB embeds 0 reference phrases."""
    db = tmp_path / "intent_persist"
    calls = {"n": 0}

    def counting_embed(text: str):
        calls["n"] += 1
        return lexical_probe_embed(text)

    r1 = VectorIntentRouter(
        threshold=0.82,
        embed_fn=counting_embed,
        persist=True,
        db_path=db,
        embed_model="probe-embed",
        auto_sync=False,
    )
    stats1 = r1.sync_and_validate_intents()
    assert stats1["embedded"] > 0
    first_embeds = calls["n"]
    assert first_embeds == stats1["embedded"]
    assert stats1["loaded_from_db"] == stats1["expected"]

    # Simulate process restart: new router, same LanceDB path
    calls["n"] = 0
    r2 = VectorIntentRouter(
        threshold=0.82,
        embed_fn=counting_embed,
        persist=True,
        db_path=db,
        embed_model="probe-embed",
        auto_sync=False,
    )
    stats2 = r2.sync_and_validate_intents()
    assert stats2["embedded"] == 0
    assert calls["n"] == 0
    assert stats2["catalog_valid"] is True
    assert stats2["loaded_from_db"] == stats1["expected"]
    # Classify still works (only query embed, not reference embeds)
    before = calls["n"]
    intent, score = r2.classify("Дожать HOW")
    assert intent == "how"
    assert score >= 0.82
    assert calls["n"] == before + 1  # one query embed only


def test_long_engineering_answer_dilutes_without_tech_wordlist():
    """Semantic dilution — not a domain dictionary — keeps long answers off chips."""
    msg = (
        "Я подробно описал, как работает middleware валидации аргументов "
        "инструмента: сначала Pydantic schema, затем reject с structured error, "
        "и только после этого ответ может попасть обратно в контекст LLM."
    )
    intent, score = _active_router().classify(msg)
    assert intent == ""
    assert score < 0.82


def test_explicit_mode_prefix_is_not_diluted_by_length():
    padding = " ".join(["constraint"] * 40)
    msg = f"[mode:advanced_analysis] {padding}"
    intent, score = _active_router().classify(msg)
    assert intent == "advanced_analysis"
    assert score >= 0.82


def test_overlay_probe_order_prefers_specific_kind():
    intent, score = _active_router().classify("анализ уязвимостей")
    assert intent == "advanced_analysis"
    assert score >= 0.82
    intent, score = _active_router().classify("архитектурный дизайн")
    assert intent == "deep_design"
    assert score >= 0.82
    intent, score = _active_router().classify("задачка со звёздочкой")
    assert intent == "deep_analysis"
    assert score >= 0.82
