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
