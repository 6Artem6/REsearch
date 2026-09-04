"""Контракт замкнутого контекста: Evaluator и Question Factory делят SSOT."""

from __future__ import annotations

from knowledge_engine.schemas.llm_contracts.evaluator_critique import (
    EvaluatorCritiqueContract,
)
from knowledge_engine.schemas.llm_contracts.tutor import (
    IntroAssessmentContract,
    SubConceptStatusUpdate,
)
from knowledge_engine.src.node_deep_dive.context_bounded_eval import (
    CONTEXT_BOUNDED_EVAL_RULES,
    CONTEXT_BOUNDED_QUESTION_RULES,
)
from knowledge_engine.src.node_deep_dive.deep_analysis_eval_prompt import (
    ADVANCED_ANALYSIS_EVAL_SYSTEM,
    DEEP_DESIGN_EVAL_SYSTEM,
)
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    LayerDrillSession,
    SessionMemory,
    SubConceptRecord,
)
from knowledge_engine.src.node_deep_dive.prompt_factory import (
    build_how_drill_prompt,
    build_mech_drill_prompt,
    build_why_drill_prompt,
)
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput
from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (
    GAP_EVAL_SYSTEM,
    _gap_eval_payload,
)
from knowledge_engine.src.node_deep_dive.tutor_prompt_builder import (
    build_dialogue_system,
    build_intro_system,
    build_lecture_chat_system,
)


def test_ssot_blocks_are_english_and_meta_domain() -> None:
    eval_block = CONTEXT_BOUNDED_EVAL_RULES
    q_block = CONTEXT_BOUNDED_QUESTION_RULES
    assert "STRICT SCOPE & ABSTRACTION CEILING" in eval_block
    assert "CONTEXT-BOUNDED QUESTION FACTORY" in q_block
    assert "NO HIDDEN RUBRICS" in q_block
    assert "Silence or omission" in eval_block
    assert "FORBIDDEN" in eval_block
    assert "Do NOT" in q_block
    for block in (eval_block, q_block):
        assert "Запрещено" not in block
        for banned in (
            "futex",
            "cache-line",
            "Cache-Line",
            "uarch",
            "OS-kernel",
            "microarchitecture",
            "CPU",
            "kernel",
        ):
            assert banned not in block


def test_gap_eval_system_embeds_context_bound_ssot() -> None:
    assert CONTEXT_BOUNDED_EVAL_RULES.strip() in GAP_EVAL_SYSTEM
    assert "Missing MECHANIC is NOT a PARTIAL reason" in GAP_EVAL_SYSTEM


def test_overlay_eval_embeds_context_bound_ssot() -> None:
    assert CONTEXT_BOUNDED_EVAL_RULES.strip() in ADVANCED_ANALYSIS_EVAL_SYSTEM
    assert CONTEXT_BOUNDED_EVAL_RULES.strip() in DEEP_DESIGN_EVAL_SYSTEM
    assert "do not invent requirements absent from" in ADVANCED_ANALYSIS_EVAL_SYSTEM


def test_question_factory_embeds_context_bound_ssot() -> None:
    intro = build_intro_system()
    dialogue = build_dialogue_system()
    lecture = build_lecture_chat_system()
    for text in (intro, dialogue, lecture):
        assert CONTEXT_BOUNDED_QUESTION_RULES.strip() in text
    assert "Stay at the asked layer" in intro


def test_drill_prompts_embed_question_ssot() -> None:
    mem = SessionMemory(
        sub_concepts=[
            SubConceptRecord(id="gil", label="GIL", success_criterion="WHY"),
        ]
    )
    session = LayerDrillSession(
        is_active=True,
        target_layer="WHY",
        target_sub_concept_ids=["gil"],
        current_index=0,
        status="DRILL_IN_PROGRESS",
    )
    why = build_why_drill_prompt(session, memory=mem)
    how = build_how_drill_prompt(session, memory=mem)
    mech = build_mech_drill_prompt(session, memory=mem)
    for text in (why, how, mech):
        assert CONTEXT_BOUNDED_QUESTION_RULES.strip() in text


def test_gap_eval_payload_repeats_context_bound() -> None:
    mem = SessionMemory(
        node_goal="GIL internals",
        last_tutor_follow_up_question=(
            "Зачем потокам CPython нужен глобальный замок, если счётчики "
            "ссылок и так атомарны на бумаге?"
        ),
        sub_concepts=[
            SubConceptRecord(
                id="gil",
                label="GIL",
                success_criterion="Практическое понимание: GIL",
            )
        ],
    )
    node = NodeDataInput(
        node_id="gil_internals",
        title="GIL internals",
        layer="advanced",
        learning_goal="GIL internals",
        core_concepts=["GIL"],
    )
    payload = _gap_eval_payload(mem, node, "GIL сериализует байткод.", mem.sub_concepts[0])
    assert "### context_bound" in payload
    assert "strictly bounded by last_tutor_question" in payload
    assert "Зачем потокам CPython" in payload


def test_contracts_forbid_hidden_deeper_layer_rubric() -> None:
    acc = SubConceptStatusUpdate.model_fields["accuracy_grade"].description or ""
    hint = SubConceptStatusUpdate.model_fields["focus_hint"].description or ""
    intro = IntroAssessmentContract.model_fields["tutor_message"].description or ""
    edges = EvaluatorCritiqueContract.model_fields["unaccounted_edge_cases"].description or ""
    assert "explicitly requested scope" in acc
    assert "Unasked deeper layers MUST NOT reduce this grade" in acc
    assert "unasked" in hint.lower() or "NEVER" in hint
    assert "MUST appear in this text" in intro or "MUST appear" in intro
    assert "last_tutor_question" in edges
    for text in (acc, hint, intro, edges):
        for banned in ("OS", "CPU", "futex", "kernel", "uarch"):
            assert banned not in text
