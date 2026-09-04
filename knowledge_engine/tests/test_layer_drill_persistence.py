"""Layer Drill Session: Host FSM holds HOW/MECH/overlay until every queued row passes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from knowledge_engine.schemas.drill_schemas import LayerCompletionTutorOutput
from knowledge_engine.src.node_deep_dive.concept_map import (
    host_ready_for_transition,
    orchestrate_tutor_llm_output,
    set_pending_evaluation_for_tutor_turn,
)
from knowledge_engine.src.node_deep_dive.concept_map_state import build_coverage_summary
from knowledge_engine.src.node_deep_dive.drill_orchestrator import (
    drill_response_to_llm_output,
    select_drill_response_schema,
)
from knowledge_engine.src.node_deep_dive.learning_loop import build_mastery_dashboard
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    SessionMemory,
    SubConceptRecord,
)
from knowledge_engine.src.node_deep_dive.prompt_factory import select_system_prompt_and_mode
from knowledge_engine.src.node_deep_dive.schemas import DeepDiveLLMOutput, NodeDataInput
from knowledge_engine.src.node_deep_dive.star_task_fsm import (
    format_layer_drill_invariants,
    layer_drill_is_active,
    layer_drill_progress,
    start_layer_drill,
)
from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (
    process_sub_concept_user_answer,
)
from knowledge_engine.src.node_deep_dive.tutor_behavior_state import (
    build_tutor_behavior_state,
)


def _node() -> NodeDataInput:
    return NodeDataInput(
        node_id="python_object_model",
        title="Python object model",
        layer="foundation",
        core_concepts=["pyobject", "refcount", "typeptr"],
        learning_goal="Object header, refcount, type pointer",
    )


def _how_gap_memory() -> SessionMemory:
    return SessionMemory(
        last_eval_directive="PASSED_WITH_GLOSS",
        learning_phase="pathway_decision",
        topic_mastery_score=100,
        sub_concepts=[
            SubConceptRecord(
                id="pyobject",
                label="PyObject header",
                status="verified",
                why_passed=True,
                how_passed=False,
                mechanic_passed=False,
            ),
            SubConceptRecord(
                id="reference_count",
                label="Reference count",
                status="verified",
                why_passed=True,
                how_passed=False,
                mechanic_passed=False,
            ),
            SubConceptRecord(
                id="type_pointer",
                label="Type pointer",
                status="verified",
                why_passed=True,
                how_passed=False,
                mechanic_passed=False,
            ),
        ],
    )


def _how_pass_eval(cid: str) -> MagicMock:
    fake = MagicMock()
    fake.updates = [
        MagicMock(
            id=cid,
            why_passed=True,
            how_passed=True,
            mechanic_passed=False,
            evidence="architecture credited",
            focus_hint="",
            status="VERIFIED",
        )
    ]
    return fake


def _credit_how_answer(mem: SessionMemory, cid: str, node: NodeDataInput) -> None:
    assert set_pending_evaluation_for_tutor_turn(mem, cid) == cid
    with patch(
        "knowledge_engine.src.node_deep_dive.sub_concept_evaluator.run_gemini_structured_with_chain",
        return_value=_how_pass_eval(cid),
    ), patch(
        "knowledge_engine.src.node_deep_dive.concept_map.is_quick_reply_control_message",
        return_value=False,
    ), patch(
        "knowledge_engine.src.node_deep_dive.lecture_scope.is_lecture_request_message",
        return_value=False,
    ):
        process_sub_concept_user_answer(
            "Разбор архитектуры: заголовок, refcnt, ob_type.",
            mem,
            node,
            "anchor",
        )


def test_full_layer_drill_walkthrough() -> None:
    mem = _how_gap_memory()
    node = _node()
    current = start_layer_drill(mem, "HOW")
    assert current == "pyobject"
    assert mem.layer_drill.is_active is True
    assert mem.layer_drill.status == "DRILL_IN_PROGRESS"
    assert mem.layer_drill.target_layer == "HOW"
    assert mem.layer_drill.target_sub_concept_ids == [
        "pyobject",
        "reference_count",
        "type_pointer",
    ]
    snap0 = layer_drill_progress(mem)
    assert snap0["status"] == "DRILL_ACTIVE"
    assert snap0["progress"] == "0/3"

    system, mode, _ = select_system_prompt_and_mode(
        "[mode:deep_dive_how] Разбери архитектуру темы.",
        default_system_prompt="DEFAULT",
        memory=mem,
    )
    assert mode == "deep_dive_how"
    assert "DRILL_ACTIVE" in system
    assert "checked 0/3" in system or "0/3" in system
    assert "DO NOT declare the node or layer complete" in system

    _credit_how_answer(mem, "pyobject", node)
    assert mem.sub_concepts[0].how_passed is True
    assert mem.sub_concepts[1].how_passed is False
    assert mem.sub_concepts[2].how_passed is False
    assert layer_drill_is_active(mem) is True
    snap1 = layer_drill_progress(mem)
    assert snap1["progress"] == "1/3"
    assert snap1["current_sub_concept_id"] == "reference_count"
    packed1 = orchestrate_tutor_llm_output(
        mem,
        DeepDiveLLMOutput(
            ready_for_transition=True,
            follow_up_question="Как устроен refcnt?",
            technical_explanation="Базовая теория ноды полностью закрыта.",
        ),
        user_message="Разбор архитектуры: заголовок, refcnt, ob_type.",
        node_layer="foundation",
    )
    assert packed1.ready_for_transition is False
    assert host_ready_for_transition(
        mem,
        user_message="Разбор архитектуры: заголовок, refcnt, ob_type.",
        node_layer="foundation",
    ) is False
    inv1 = format_layer_drill_invariants(mem)
    assert "Progress:" in inv1
    assert "1/3" in inv1
    assert "Reference count" in inv1
    state1 = build_tutor_behavior_state(
        "ANSWER",
        "chat",
        "chat",
        "pathway_decision",
        "Разбор архитектуры: заголовок, refcnt, ob_type.",
        memory=mem,
        node_layer="foundation",
    )
    assert state1["layer_drill_session"]["progress"] == "1/3"
    assert "DRILL_ACTIVE" in state1["next_action"]
    assert "pathway=base_complete" not in state1["next_action"]

    _credit_how_answer(mem, "reference_count", node)
    assert mem.sub_concepts[1].how_passed is True
    snap2 = layer_drill_progress(mem)
    assert snap2["progress"] == "2/3"
    assert snap2["current_sub_concept_id"] == "type_pointer"
    packed2 = orchestrate_tutor_llm_output(
        mem,
        DeepDiveLLMOutput(
            ready_for_transition=True,
            follow_up_question="Где ob_type?",
            technical_explanation="Всё успешно закрыто.",
        ),
        user_message="refcount в заголовке PyObject.",
        node_layer="foundation",
    )
    assert packed2.ready_for_transition is False
    assert "2/3" in format_layer_drill_invariants(mem)

    _credit_how_answer(mem, "type_pointer", node)
    assert all(sc.how_passed for sc in mem.sub_concepts)
    assert layer_drill_is_active(mem) is False
    assert mem.layer_drill.status == "DRILL_COMPLETE"
    assert mem.is_layer_just_completed is True
    assert select_drill_response_schema(mem) is LayerCompletionTutorOutput
    assert "next_question" not in LayerCompletionTutorOutput.model_fields
    done = LayerCompletionTutorOutput(
        praise="HOW закрыт по всем трём подтемам.",
        layer_summary="Разобрали заголовок PyObject, refcnt и ob_type.",
        transition_framing="Хотите углубиться в MECH/Advanced/Deep или идти дальше?",
    )
    native = drill_response_to_llm_output(done, memory=mem)
    assert native.follow_up_question == done.transition_framing
    assert native.quick_replies == []
    packed3 = orchestrate_tutor_llm_output(
        mem,
        native,
        user_message="tp_dealloc живёт в типе.",
        node_layer="foundation",
    )
    assert packed3.ready_for_transition is True
    assert packed3.suggested_next_step == "deep_dive_optional"
    assert packed3.follow_up_question == done.transition_framing
    assert packed3.feedback_on_answer == done.praise
    assert packed3.technical_explanation == done.layer_summary
    from knowledge_engine.src.node_deep_dive.concept_map import (
        gloss_fork_quick_replies,
        open_optional_layers,
    )
    from knowledge_engine.src.node_deep_dive.intent_definitions import (
        CHIP_GLOSS,
        CHIP_MECH,
        CHIP_OVERLAY_NEXT,
    )

    chips = gloss_fork_quick_replies(open_optional_layers(mem, "foundation"))
    assert CHIP_GLOSS in chips
    assert CHIP_MECH in chips
    assert CHIP_OVERLAY_NEXT in chips
    cov = build_coverage_summary(mem)
    assert cov is not None
    assert all(item.how_passed for item in cov.items)
    assert cov.layers is not None
    assert cov.layers.how.status == "passed"
    assert cov.layers.how.score == 1.0
    dash = build_mastery_dashboard(mem, "deep_understanding")
    assert dash.coverage_summary is not None
    assert dash.coverage_summary.layers.how.score == 1.0


def _eval_update(cid: str, *, why: bool, how: bool, mechanic: bool) -> MagicMock:
    fake = MagicMock()
    fake.updates = [
        MagicMock(
            id=cid,
            why_passed=why,
            how_passed=how,
            mechanic_passed=mechanic,
            evidence="turn extract",
            focus_hint="",
            status=None,
        )
    ]
    return fake


def _score_answer(
    mem: SessionMemory,
    cid: str,
    node: NodeDataInput,
    *,
    why: bool,
    how: bool,
    mechanic: bool,
    text: str = "Ответ по архитектуре: header, refcnt, ob_type.",
) -> None:
    assert set_pending_evaluation_for_tutor_turn(mem, cid) == cid
    with patch(
        "knowledge_engine.src.node_deep_dive.sub_concept_evaluator.run_gemini_structured_with_chain",
        return_value=_eval_update(cid, why=why, how=how, mechanic=mechanic),
    ), patch(
        "knowledge_engine.src.node_deep_dive.concept_map.is_quick_reply_control_message",
        return_value=False,
    ), patch(
        "knowledge_engine.src.node_deep_dive.lecture_scope.is_lecture_request_message",
        return_value=False,
    ):
        process_sub_concept_user_answer(text, mem, node, "anchor")


def test_how_drill_does_not_credit_when_extractor_omits_how() -> None:
    """Extractor booleans are the only credit signal — Host does not auto-set HOW."""
    mem = _how_gap_memory()
    node = _node()
    start_layer_drill(mem, "HOW")
    _score_answer(
        mem,
        "pyobject",
        node,
        why=True,
        how=False,
        mechanic=False,
    )
    row = mem.sub_concepts[0]
    assert row.how_passed is False
    assert mem.last_eval_directive == "PROBE_NEXT_LAYER:HOW"
    cov = build_coverage_summary(mem)
    assert cov is not None
    how_item = next(it for it in cov.items if it.id == "pyobject")
    assert how_item.how_passed is False


def test_how_drill_does_not_credit_total_miss() -> None:
    mem = _how_gap_memory()
    node = _node()
    start_layer_drill(mem, "HOW")
    _score_answer(
        mem,
        "pyobject",
        node,
        why=False,
        how=False,
        mechanic=False,
        text="не знаю как это устроено внутри",
    )
    assert mem.sub_concepts[0].how_passed is False
    assert mem.last_eval_directive == "PROBE_NEXT_LAYER:HOW"
    cov = build_coverage_summary(mem)
    assert cov is not None
    assert next(it for it in cov.items if it.id == "pyobject").how_passed is False


def test_how_drill_clears_leftover_overlay_eval_kind() -> None:
    mem = _how_gap_memory()
    mem.pending_eval_kind = "deep_design"
    mem.star_task_status = "in_progress"
    start_layer_drill(mem, "HOW")
    assert mem.pending_eval_kind == ""
    assert mem.star_task_status == "not_started"


def test_prompt_factory_skips_drill_invariants_when_inactive() -> None:
    mem = _how_gap_memory()
    system, mode, _ = select_system_prompt_and_mode(
        "[mode:deep_dive_how] x",
        default_system_prompt="DEFAULT",
        memory=mem,
    )
    assert mode == "deep_dive_how"
    assert "DRILL_ACTIVE" not in system


def test_evaluator_latch_selects_layer_completion_schema() -> None:
    from knowledge_engine.src.node_deep_dive.drill_orchestrator import (
        is_layer_just_completed,
    )

    mem = _how_gap_memory()
    node = _node()
    start_layer_drill(mem, "HOW")
    assert mem.is_layer_just_completed is False
    assert select_drill_response_schema(mem) is not LayerCompletionTutorOutput
    _credit_how_answer(mem, "pyobject", node)
    assert mem.is_layer_just_completed is False
    _credit_how_answer(mem, "reference_count", node)
    assert mem.is_layer_just_completed is False
    _credit_how_answer(mem, "type_pointer", node)
    assert is_layer_just_completed(mem) is True
    assert select_drill_response_schema(mem) is LayerCompletionTutorOutput
    start_layer_drill(mem, "MECH")
    assert mem.is_layer_just_completed is False


def test_completed_drill_prompt_forbids_next_question() -> None:
    mem = _how_gap_memory()
    start_layer_drill(mem, "HOW")
    mem.layer_drill.current_index = 2
    for sc in mem.sub_concepts:
        sc.how_passed = True
    mem.layer_drill.advance_or_complete()
    mem.is_layer_just_completed = True
    assert mem.layer_drill.status == "DRILL_COMPLETE"
    system, _, _ = select_system_prompt_and_mode(
        "tp_dealloc живёт в типе.",
        default_system_prompt="DEFAULT",
        memory=mem,
    )
    assert "LAYER COMPLETION" in system
    assert "LayerCompletionTutorOutput" in system
    assert "There is NO next_question" in system
    assert "JSON OUTPUT (ActiveDrillStepResponse)" not in system
