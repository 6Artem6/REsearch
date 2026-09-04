"""Quick-reply control chips: skip Evaluator; Gloss credits; Дожать holds transition."""

from __future__ import annotations

from knowledge_engine.src.node_deep_dive.concept_map import (
    credit_open_optional_layers,
    is_quick_reply_control_message,
    open_optional_layers,
    orchestrate_tutor_llm_output,
)
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    SessionMemory,
    SubConceptRecord,
)
from knowledge_engine.src.node_deep_dive.schemas import DeepDiveLLMOutput
from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (
    process_sub_concept_user_answer,
)


def _mem_optional_mech() -> SessionMemory:
    return SessionMemory(
        last_eval_directive="PASSED_WITH_GLOSS",
        pending_evaluation_concept_id="agg",
        asked_question_sub_concept_id="agg",
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


def test_quick_reply_control_detected() -> None:
    assert is_quick_reply_control_message("Дожать MECH")
    assert is_quick_reply_control_message("Хочу Gloss")
    assert is_quick_reply_control_message("Идем дальше")
    assert is_quick_reply_control_message(
        "[mode:deep_dive_how] Разбери архитектуру темы."
    )
    assert is_quick_reply_control_message(
        "[mode:gloss] Сформируй сжатую выжимку (Glossary) по оставшимся слоям."
    )
    assert not is_quick_reply_control_message(
        "Если confidence_score=0.0 у обоих, взвешивание сломается так:…"
    )


def test_substantive_answer_with_architecture_not_dozhat_how() -> None:
    """Regression: WAL / rollback answer contained «архитектуре» → false HOW chip."""
    msg = (
        "Если мутации стейта постоянны, то можно пойти по архитектуре снапшота, "
        "где фиксируется одно начальное состояние, а все изменения - это набор "
        "последовательных операций над ним как в WAL. Это позволит сохранить "
        "состояние промежуточное достаточное количество раз с возможностью "
        "отката к нужному, совершив обратные операции или от начального "
        "проделав повторно операции."
    )
    assert not is_quick_reply_control_message(msg)
    from knowledge_engine.src.node_deep_dive.concept_map import (
        classify_gloss_fork_choice,
    )

    assert classify_gloss_fork_choice(msg) == ""


def test_substantive_answer_with_mechanism_not_dozhat_mech() -> None:
    msg = (
        "Механизм отката опирается на неизменяемые структуры данных и графовые "
        "снапшоты, где каждый узел фиксирует дельта-состояние перед вызовом "
        "внешнего инструмента, чтобы можно было безопасно откатиться."
    )
    assert not is_quick_reply_control_message(msg)


def test_process_skips_eval_on_dozhat(monkeypatch) -> None:
    called = {"n": 0}

    def boom(*_a, **_k):
        called["n"] += 1
        raise AssertionError("gap eval must not run for chip")

    monkeypatch.setattr(
        "knowledge_engine.src.node_deep_dive.sub_concept_evaluator.run_sub_concept_gap_eval",
        boom,
    )
    from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput

    mem = _mem_optional_mech()
    node = NodeDataInput(
        node_id="n1",
        title="Subagent Architectures",
        layer="advanced",
        learning_goal="goal text here",
        core_concepts=["c1"],
    )
    process_sub_concept_user_answer("Дожать MECH", mem, node, "anchor")
    assert called["n"] == 0
    assert mem.sub_concepts[0].mechanic_passed is False


def test_process_skips_eval_on_rephrase_request(monkeypatch) -> None:
    """Free-text «переформулируй вопрос» must not be graded as a wrong/missed
    answer — pending_evaluation_concept_id and mastery flags stay untouched
    so the tutor can restate the same still-open question next turn."""
    from knowledge_engine.src.node_deep_dive.vector_intent_router import (
        VectorIntentRouter,
        set_vector_intent_router_for_tests,
    )
    from knowledge_engine.tests.intent_embed_probe import lexical_probe_embed

    called = {"n": 0}

    def boom(*_a, **_k):
        called["n"] += 1
        raise AssertionError("gap eval must not run for a rephrase request")

    monkeypatch.setattr(
        "knowledge_engine.src.node_deep_dive.sub_concept_evaluator.run_sub_concept_gap_eval",
        boom,
    )
    from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput

    router = VectorIntentRouter(
        threshold=0.82,
        embed_fn=lexical_probe_embed,
        persist=False,
        auto_sync=True,
        enabled=True,
    )
    set_vector_intent_router_for_tests(router)
    try:
        mem = _mem_optional_mech()
        node = NodeDataInput(
            node_id="n1",
            title="Subagent Architectures",
            layer="advanced",
            learning_goal="goal text here",
            core_concepts=["c1"],
        )
        process_sub_concept_user_answer(
            "Не понял вопрос, объясни иначе", mem, node, "anchor"
        )
    finally:
        set_vector_intent_router_for_tests(None)
    assert called["n"] == 0
    assert mem.evaluator_skipped is True
    assert mem.pending_evaluation_concept_id == "agg"
    assert mem.sub_concepts[0].mechanic_passed is False


def test_orchestrate_dozhat_does_not_force_ready() -> None:
    mem = _mem_optional_mech()
    out = DeepDiveLLMOutput(
        ready_for_transition=False,
        follow_up_question="Что если score=0?",
        question_sub_concept_id="agg",
        technical_explanation="```python\nclass X: ...\n```",
    )
    packed = orchestrate_tutor_llm_output(
        mem, out, user_message="Дожать MECH", node_layer="advanced"
    )
    assert packed.ready_for_transition is False
    assert mem.pending_evaluation_concept_id == "agg"


def test_gloss_credits_optional_mech() -> None:
    mem = _mem_optional_mech()
    assert open_optional_layers(mem, "advanced") == ["MECHANIC"]
    credited = credit_open_optional_layers(mem, "advanced")
    assert "MECHANIC" in credited
    assert mem.sub_concepts[0].mechanic_passed is True
    out = DeepDiveLLMOutput(
        ready_for_transition=True,
        follow_up_question="Выбери следующую ноду в UI.",
        technical_explanation="Краткий gloss по весам…",
    )
    packed = orchestrate_tutor_llm_output(
        mem, out, user_message="Хочу Gloss", node_layer="advanced"
    )
    assert packed.ready_for_transition is True
    assert not (mem.pending_evaluation_concept_id or "").strip()


def test_orchestrate_ignores_llm_orchestration_fields() -> None:
    """Host FSM owns chips/transition — LLM-invented labels and flags are dropped."""
    mem = _mem_optional_mech()
    out = DeepDiveLLMOutput(
        ready_for_transition=True,
        suggested_next_step="deep_dive_optional",
        follow_up_question="Выбери шаг.",
        quick_replies=["Дожать MECHANIC", "invented chip"],
    )
    packed = orchestrate_tutor_llm_output(
        mem, out, user_message="", node_layer="advanced"
    )
    assert packed.ready_for_transition is True  # coverage complete, host FSM
    assert packed.quick_replies == []
    assert packed.suggested_next_step == "deep_dive_optional"

    held = orchestrate_tutor_llm_output(
        mem,
        DeepDiveLLMOutput(
            ready_for_transition=True,
            follow_up_question="Что если score=0?",
            quick_replies=["Идем дальше"],
        ),
        user_message="Дожать MECH",
        node_layer="advanced",
    )
    assert held.ready_for_transition is False
    assert held.quick_replies == []
    assert held.suggested_next_step is None
