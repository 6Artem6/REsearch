"""Phase 4: split asterisk-question overlay into ADVANCED (L4) vs DEEP (L5/L6)."""

from __future__ import annotations

from unittest.mock import patch

from knowledge_engine.schemas.llm_contracts.evaluator_critique import (
    EvaluatedIdea,
    EvaluatorCritiqueContract,
    IdeaStatus,
)
from knowledge_engine.src.node_deep_dive.advanced_analysis_prompt import (
    ADVANCED_ANALYSIS_PROMPT,
)
from knowledge_engine.src.node_deep_dive.concept_map import classify_gloss_fork_choice
from knowledge_engine.src.node_deep_dive.concept_map_state import (
    build_coverage_summary,
    has_overlay_award,
    list_overlay_mastery_records,
    register_deep_mastery,
)
from knowledge_engine.src.node_deep_dive.control_intent import classify_control_chip
from knowledge_engine.src.node_deep_dive.deep_analysis_eval_prompt import (
    ADVANCED_ANALYSIS_EVAL_SYSTEM,
    DEEP_DESIGN_EVAL_SYSTEM,
)
from knowledge_engine.src.node_deep_dive.deep_design_prompt import DEEP_DESIGN_PROMPT
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    SessionMemory,
    SubConceptRecord,
)
from knowledge_engine.src.node_deep_dive.prompt_factory import (
    is_factory_control_mode,
    parse_tutor_mode_prefix,
    requires_deep_analysis_guard,
    select_system_prompt_and_mode,
)
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput
from knowledge_engine.src.node_deep_dive.star_task_fsm import (
    CHIP_ADVANCED_ANALYSIS,
    CHIP_DEEP_DESIGN,
    CHIP_OVERLAY_NEXT,
    mark_star_task_in_progress,
    overlay_offer_quick_replies,
    overlay_type_for_kind,
)
from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (
    run_sub_concept_gap_eval,
)
from knowledge_engine.src.node_deep_dive.tiered_memory import sync_topic_mastery_score


def _node() -> NodeDataInput:
    return NodeDataInput(
        node_id="n1",
        title="Aggregation",
        layer="advanced",
        core_concepts=["aggregation"],
        learning_goal="Understand aggregation",
    )


def _core_verified_memory(*, pending_kind: str) -> SessionMemory:
    return SessionMemory(
        pending_evaluation_concept_id="agg",
        pending_eval_kind=pending_kind,  # type: ignore[arg-type]
        asked_question_sub_concept_id="agg",
        topic_mastery_score=100,
        sub_concepts=[
            SubConceptRecord(
                id="agg",
                label="Aggregation",
                status="verified",
                why_passed=True,
                how_passed=True,
                mechanic_passed=True,
            )
        ],
    )


def _critique(*, layer: str, passed: bool = True) -> EvaluatorCritiqueContract:
    return EvaluatorCritiqueContract(
        target_layer=layer,  # type: ignore[arg-type]
        passes_threshold=passed,
        bloom_level_matched=passed,
        analyzed_ideas=[
            EvaluatedIdea(
                idea_concept="timeout bounds hang under fan-out",
                status=IdeaStatus.STRONG,
                technical_note="Sound L4/L6 reasoning for the overlay task.",
            )
        ],
        unaccounted_edge_cases=[],
        verdict_reason="Constraints respected; overlay depth matched.",
    )


def test_factory_modes_advanced_and_deep_design() -> None:
    body, mode = parse_tutor_mode_prefix("[mode:advanced_analysis] challenge")
    assert mode == "advanced_analysis"
    assert body == "challenge"
    system, mode, _ = select_system_prompt_and_mode(
        "[mode:advanced_analysis] challenge",
        default_system_prompt="DEFAULT",
    )
    assert mode == "advanced_analysis"
    assert ADVANCED_ANALYSIS_PROMPT.strip() in system
    assert "ADVANCED_ASTERISK" in ADVANCED_ANALYSIS_PROMPT
    assert is_factory_control_mode("advanced_analysis")
    assert requires_deep_analysis_guard("advanced_analysis")

    body, mode = parse_tutor_mode_prefix("[mode:deep_design] challenge")
    assert mode == "deep_design"
    system, mode, _ = select_system_prompt_and_mode(
        "[mode:deep_design] challenge",
        default_system_prompt="DEFAULT",
    )
    assert mode == "deep_design"
    assert DEEP_DESIGN_PROMPT.strip() in system
    assert "DEEP_ASTERISK" in DEEP_DESIGN_PROMPT
    assert is_factory_control_mode("deep_design")
    assert requires_deep_analysis_guard("deep_design")


def test_overlay_offer_chips_route_to_modes() -> None:
    chips = overlay_offer_quick_replies()
    assert chips == [CHIP_DEEP_DESIGN, CHIP_OVERLAY_NEXT]
    assert CHIP_OVERLAY_NEXT in chips
    weak_chips = overlay_offer_quick_replies(weakness_tags=["race_conditions"])
    assert weak_chips == [CHIP_ADVANCED_ANALYSIS, CHIP_OVERLAY_NEXT]
    assert classify_control_chip(CHIP_ADVANCED_ANALYSIS) == "advanced_analysis"
    assert classify_control_chip(CHIP_DEEP_DESIGN) == "deep_design"
    assert classify_gloss_fork_choice(CHIP_ADVANCED_ANALYSIS) == "advanced_analysis"
    assert classify_gloss_fork_choice(CHIP_DEEP_DESIGN) == "deep_design"
    assert (
        classify_control_chip(
            "[mode:advanced_analysis] Анализ уязвимостей (задачка со звёздочкой)"
        )
        == "advanced_analysis"
    )
    assert (
        classify_control_chip(
            "[mode:deep_design] Архитектурный дизайн (сложная звёздочка)"
        )
        == "deep_design"
    )


def test_advanced_analysis_pass_awards_advanced_asterisk() -> None:
    mem = _core_verified_memory(pending_kind="advanced_analysis")
    mark_star_task_in_progress(
        mem, concept_id="agg", overlay_kind="advanced_analysis"
    )
    assert mem.pending_eval_kind == "advanced_analysis"
    score_before = sync_topic_mastery_score(mem)
    assert score_before == 100
    why0 = mem.sub_concepts[0].why_passed
    how0 = mem.sub_concepts[0].how_passed
    mech0 = mem.sub_concepts[0].mechanic_passed

    with patch(
        "knowledge_engine.src.node_deep_dive.sub_concept_evaluator.run_gemini_structured_with_chain",
        return_value=_critique(layer="DEEP", passed=True),
    ) as mock_llm:
        d = run_sub_concept_gap_eval(
            "I enumerate race windows and P99 blow-ups under fan-out timeout.",
            mem,
            _node(),
            "anchor",
            concept_id="agg",
        )
        system_arg = mock_llm.call_args[0][1]
        payload_arg = mock_llm.call_args[0][2]
        assert ADVANCED_ANALYSIS_EVAL_SYSTEM.strip() in system_arg
        assert "advanced_analysis" in payload_arg

    assert d == "DEEP_MASTERY_EARNED"
    assert overlay_type_for_kind("advanced_analysis") == "ADVANCED_ASTERISK"
    assert has_overlay_award(mem, "agg", "ADVANCED_ASTERISK")
    assert not has_overlay_award(mem, "agg", "DEEP_ASTERISK")
    recs = list_overlay_mastery_records(mem)
    assert recs[0].overlay_type == "ADVANCED_ASTERISK"
    row = mem.sub_concepts[0]
    assert row.why_passed is why0
    assert row.how_passed is how0
    assert row.mechanic_passed is mech0
    assert row.status == "verified"
    assert sync_topic_mastery_score(mem) == 100
    cov = build_coverage_summary(mem)
    assert cov is not None
    assert cov.overall_score == 100
    assert cov.overlay_awards[0].overlay_type == "ADVANCED_ASTERISK"


def test_deep_design_pass_awards_deep_asterisk() -> None:
    mem = _core_verified_memory(pending_kind="deep_design")
    mark_star_task_in_progress(mem, concept_id="agg", overlay_kind="deep_design")
    assert mem.pending_eval_kind == "deep_design"
    assert sync_topic_mastery_score(mem) == 100
    why0 = mem.sub_concepts[0].why_passed
    how0 = mem.sub_concepts[0].how_passed
    mech0 = mem.sub_concepts[0].mechanic_passed

    with patch(
        "knowledge_engine.src.node_deep_dive.sub_concept_evaluator.run_gemini_structured_with_chain",
        return_value=_critique(layer="ADVANCED", passed=True),
    ) as mock_llm:
        d = run_sub_concept_gap_eval(
            "I would shard by tenant and drop full-material RAM; trade-off is fan-out.",
            mem,
            _node(),
            "anchor",
            concept_id="agg",
        )
        system_arg = mock_llm.call_args[0][1]
        payload_arg = mock_llm.call_args[0][2]
        assert DEEP_DESIGN_EVAL_SYSTEM.strip() in system_arg
        assert "deep_design" in payload_arg

    assert d == "DEEP_MASTERY_EARNED"
    assert overlay_type_for_kind("deep_design") == "DEEP_ASTERISK"
    assert has_overlay_award(mem, "agg", "DEEP_ASTERISK")
    assert not has_overlay_award(mem, "agg", "ADVANCED_ASTERISK")
    row = mem.sub_concepts[0]
    assert row.why_passed is why0
    assert row.how_passed is how0
    assert row.mechanic_passed is mech0
    assert sync_topic_mastery_score(mem) == 100


def test_overlay_awards_do_not_change_core_topic_mastery() -> None:
    mem = _core_verified_memory(pending_kind="gap")
    mem.pending_eval_kind = ""
    mem.pending_evaluation_concept_id = ""
    assert sync_topic_mastery_score(mem) == 100
    assert register_deep_mastery(mem, "agg", overlay_type="ADVANCED_ASTERISK")
    assert register_deep_mastery(mem, "agg", overlay_type="DEEP_ASTERISK")
    assert sync_topic_mastery_score(mem) == 100
    row = mem.sub_concepts[0]
    assert row.why_passed is True
    assert row.how_passed is True
    assert row.mechanic_passed is True
    assert row.status == "verified"
    assert has_overlay_award(mem, "agg", "ADVANCED_ASTERISK")
    assert has_overlay_award(mem, "agg", "DEEP_ASTERISK")
    cov = build_coverage_summary(mem)
    assert cov is not None
    assert cov.overall_score == 100
    assert cov.deep_mastery_ids == ["agg"]
    assert {r.overlay_type for r in cov.overlay_awards} == {
        "ADVANCED_ASTERISK",
        "DEEP_ASTERISK",
    }


def test_legacy_deep_analysis_still_maps_to_deep_asterisk() -> None:
    mem = _core_verified_memory(pending_kind="deep_analysis")
    mark_star_task_in_progress(mem, concept_id="agg")
    assert mem.pending_eval_kind == "deep_analysis"
    with patch(
        "knowledge_engine.src.node_deep_dive.sub_concept_evaluator.run_gemini_structured_with_chain",
        return_value=_critique(layer="DEEP", passed=True),
    ):
        d = run_sub_concept_gap_eval(
            "I refuse full-material RAM given the stated memory class.",
            mem,
            _node(),
            "anchor",
            concept_id="agg",
        )
    assert d == "DEEP_MASTERY_EARNED"
    assert has_overlay_award(mem, "agg", "DEEP_ASTERISK")
    assert sync_topic_mastery_score(mem) == 100


def _two_core_overlay_memory(*, pending_kind: str) -> SessionMemory:
    rows = []
    for cid, label in (("agg", "Aggregation"), ("fanout", "Fan-out")):
        rows.append(
            SubConceptRecord(
                id=cid,
                label=label,
                status="verified",
                why_passed=True,
                how_passed=True,
                mechanic_passed=True,
            )
        )
    return SessionMemory(
        pending_evaluation_concept_id="agg",
        pending_eval_kind=pending_kind,  # type: ignore[arg-type]
        asked_question_sub_concept_id="agg",
        star_task_status="in_progress",
        topic_mastery_score=100,
        last_eval_directive="DEEP_MASTERY_EARNED",
        sub_concepts=rows,
    )


def test_overlay_push_binds_pending_on_verified() -> None:
    from knowledge_engine.src.node_deep_dive.concept_map import (
        set_pending_evaluation_for_tutor_turn,
        start_overlay_push,
    )

    mem = SessionMemory(
        topic_mastery_score=100,
        sub_concepts=[
            SubConceptRecord(
                id="agg",
                label="Aggregation",
                status="verified",
                why_passed=True,
                how_passed=True,
                mechanic_passed=True,
            )
        ],
    )
    assert set_pending_evaluation_for_tutor_turn(mem, "agg") == ""
    cid = start_overlay_push(mem, "deep_design")
    assert cid == "agg"
    assert mem.star_task_status == "in_progress"
    assert set_pending_evaluation_for_tutor_turn(mem, "agg") == "agg"


def test_overlay_pass_continues_to_next_core_without_award() -> None:
    from knowledge_engine.src.node_deep_dive.concept_map import (
        host_ready_for_transition,
        orchestrate_tutor_llm_output,
    )
    from knowledge_engine.src.node_deep_dive.schemas import DeepDiveLLMOutput
    from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (
        process_sub_concept_user_answer,
    )
    from knowledge_engine.src.node_deep_dive.tutor_behavior_state import (
        _next_action_for_mode,
    )

    mem = _two_core_overlay_memory(pending_kind="deep_design")
    mark_star_task_in_progress(
        mem, concept_id="agg", overlay_kind="deep_design"
    )
    with patch(
        "knowledge_engine.src.node_deep_dive.sub_concept_evaluator.run_gemini_structured_with_chain",
        return_value=_critique(layer="DEEP", passed=True),
    ):
        process_sub_concept_user_answer(
            "Shard by tenant; drop full-material RAM; trade-off is fan-out.",
            mem,
            _node(),
            "anchor",
        )
    assert has_overlay_award(mem, "agg", "DEEP_ASTERISK")
    assert not has_overlay_award(mem, "fanout", "DEEP_ASTERISK")
    assert mem.star_task_status == "in_progress"
    assert mem.pending_eval_kind == "deep_design"
    assert mem.next_question_concept_id == "fanout"
    packed = orchestrate_tutor_llm_output(
        mem,
        DeepDiveLLMOutput(
            ready_for_transition=True,
            follow_up_question="Как режете fan-out?",
            technical_explanation="Переходим к fan-out.",
        ),
        user_message="Shard by tenant; drop full-material RAM.",
        node_layer="advanced",
    )
    assert packed.ready_for_transition is False
    assert host_ready_for_transition(
        mem,
        user_message="Shard by tenant; drop full-material RAM.",
        node_layer="advanced",
    ) is False
    text = _next_action_for_mode(
        "dialogue_feedback",
        "ANSWER",
        "chat",
        "pathway_decision",
        memory=mem,
        user_message="Shard by tenant; drop full-material RAM.",
        node_layer="advanced",
    )
    assert "continue overlay" in text
    assert "fanout" in text
    assert "pathway=base_complete" not in text


def test_overlay_pass_on_last_core_resolves() -> None:
    from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (
        process_sub_concept_user_answer,
    )

    mem = _two_core_overlay_memory(pending_kind="advanced_analysis")
    register_deep_mastery(mem, "fanout", overlay_type="ADVANCED_ASTERISK")
    mark_star_task_in_progress(
        mem, concept_id="agg", overlay_kind="advanced_analysis"
    )
    with patch(
        "knowledge_engine.src.node_deep_dive.sub_concept_evaluator.run_gemini_structured_with_chain",
        return_value=_critique(layer="ADVANCED", passed=True),
    ):
        process_sub_concept_user_answer(
            "I enumerate race windows and P99 blow-ups under fan-out timeout.",
            mem,
            _node(),
            "anchor",
        )
    assert has_overlay_award(mem, "agg", "ADVANCED_ASTERISK")
    assert has_overlay_award(mem, "fanout", "ADVANCED_ASTERISK")
    assert mem.star_task_status == "resolved"
    assert not (mem.pending_eval_kind or "").strip()
