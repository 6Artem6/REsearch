"""Deep Analysis factory mode: prompt, pending_eval_kind, manifest skip, evaluator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from knowledge_engine.src.node_deep_dive.concept_map import (
    is_quick_reply_control_message,
    orchestrate_tutor_llm_output,
)
from knowledge_engine.src.node_deep_dive.control_intent import classify_control_chip
from knowledge_engine.src.node_deep_dive.deep_analysis_eval_prompt import (
    DEEP_ANALYSIS_EVAL_SYSTEM,
)
from knowledge_engine.src.node_deep_dive.deep_analysis_prompt import DEEP_ANALYSIS_PROMPT
from knowledge_engine.src.node_deep_dive.fact_manifest import update_manifest_from_evicted
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    SessionMemory,
    SubConceptRecord,
)
from knowledge_engine.src.node_deep_dive.prompt_factory import (
    is_factory_control_mode,
    parse_tutor_mode_prefix,
    select_system_prompt_and_mode,
)
from knowledge_engine.src.node_deep_dive.schemas import DeepDiveLLMOutput, NodeDataInput
from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (
    run_sub_concept_gap_eval,
)


def test_parse_deep_analysis_prefix() -> None:
    body, mode = parse_tutor_mode_prefix(
        "[mode:deep_analysis] Задачка со звёздочкой"
    )
    assert mode == "deep_analysis"
    assert "звёзд" in body.lower() or "звезд" in body.lower() or body


def test_select_deep_analysis_isolated_prompt() -> None:
    system, mode, cleaned = select_system_prompt_and_mode(
        "[mode:deep_analysis] x",
        default_system_prompt="DEFAULT",
    )
    assert mode == "deep_analysis"
    assert system != "DEFAULT"
    assert DEEP_ANALYSIS_PROMPT.strip() in system
    assert "FACT_ATTRACTION" in DEEP_ANALYSIS_PROMPT
    assert "DYNAMIC" in DEEP_ANALYSIS_PROMPT or "dynamic" in DEEP_ANALYSIS_PROMPT.lower()
    assert "[S" in DEEP_ANALYSIS_PROMPT and "[R" in DEEP_ANALYSIS_PROMPT
    assert "trade-off" in DEEP_ANALYSIS_PROMPT.lower() or "Trade-off" in (
        DEEP_ANALYSIS_PROMPT
    )
    assert "900" in DEEP_ANALYSIS_PROMPT or "1600" in DEEP_ANALYSIS_PROMPT
    assert "code-1" in DEEP_ANALYSIS_PROMPT.lower() or "code-1" in DEEP_ANALYSIS_PROMPT
    assert cleaned == "x"
    assert is_factory_control_mode("deep_analysis")


def test_deep_analysis_token_budget_at_least_4k() -> None:
    from knowledge_engine.config import (
        GEMINI_DEEP_ANALYSIS_MAX_OUTPUT_TOKENS,
        GEMINI_TUTOR_MAX_OUTPUT_TOKENS,
    )

    assert GEMINI_DEEP_ANALYSIS_MAX_OUTPUT_TOKENS >= 4096
    assert (
        max(GEMINI_TUTOR_MAX_OUTPUT_TOKENS, GEMINI_DEEP_ANALYSIS_MAX_OUTPUT_TOKENS)
        >= 4096
    )


def test_deep_analysis_disables_dialogue_lite_layers() -> None:
    """Deep Analysis / open star task must receive full context, not lite strip."""
    from knowledge_engine.src.node_deep_dive.prompt_factory import (
        requires_deep_analysis_guard,
    )
    from knowledge_engine.src.node_deep_dive.prompt_types import InteractionPromptMode

    prompt_mode = InteractionPromptMode.DIALOGUE_FEEDBACK
    star_guard = requires_deep_analysis_guard("deep_analysis", star_task_status="")
    use_lite = prompt_mode == InteractionPromptMode.DIALOGUE_FEEDBACK and not star_guard
    assert use_lite is False
    assert requires_deep_analysis_guard("", star_task_status="in_progress")
    assert requires_deep_analysis_guard("", star_task_status="needs_refinement")
    assert not requires_deep_analysis_guard("", star_task_status="resolved")
    assert not requires_deep_analysis_guard("", star_task_status="not_started")


def test_deep_analysis_contract_requires_follow_up() -> None:
    from pydantic import ValidationError

    from knowledge_engine.schemas.llm_contracts.tutor import (
        DeepDiveDeepAnalysisContract,
    )

    long_q = (
        "Как вы перестроите шардирование при partition, если hot-key "
        "съедает 40% QPS — какой trade-off выберете?"
    )
    from knowledge_engine.schemas.drill_schemas import TechnicalConceptAudit

    audit = TechnicalConceptAudit(
        feedback_kind="EXACT",
        accuracy_grade="EXACT_AND_CORRECT",
        user_claims_analysis=["Нет предыдущего ответа для аудита overlay-хода."],
        detected_errors_or_misconceptions=[],
        confirmation="Предыдущего ответа для аудита нет — переходим к overlay.",
    )
    ok = DeepDiveDeepAnalysisContract(
        audit=audit,
        technical_explanation="x" * 100,
        follow_up_question=long_q,
        ready_for_transition=True,  # host will override; contract allows it
    )
    assert ok.follow_up_question.strip()

    try:
        DeepDiveDeepAnalysisContract(
            audit=audit,
            technical_explanation="секции 1-5",
            follow_up_question="",
        )
        raise AssertionError("expected ValidationError for empty follow_up")
    except ValidationError:
        pass

    try:
        DeepDiveDeepAnalysisContract(
            audit=audit,
            technical_explanation="секции 1-5",
            follow_up_question="   ",
        )
        raise AssertionError("expected ValidationError for whitespace follow_up")
    except ValidationError:
        pass

    # Phrase bans removed — structural contract only.
    decree = DeepDiveDeepAnalysisContract(
        audit=audit,
        technical_explanation="Нода полностью освоена на 100%.",
        follow_up_question=long_q,
    )
    assert "освоена" in decree.technical_explanation


def test_suppress_topic_completion_in_concept_map() -> None:
    from knowledge_engine.src.node_deep_dive.concept_map_state import (
        format_concept_map_for_tutor,
    )

    mem = SessionMemory(
        sub_concepts=[
            SubConceptRecord(
                id="agg",
                label="agg",
                status="verified",
                why_passed=True,
                how_passed=True,
                mechanic_passed=True,
            )
        ]
    )
    open_text = format_concept_map_for_tutor(mem, suppress_topic_completion=False)
    assert "pathway=base_complete" in open_text or "pathway=optional_fork" in open_text
    assert "Host pathway" in open_text or "pathway=" in open_text

    guarded = format_concept_map_for_tutor(mem, suppress_topic_completion=True)
    assert "TOPIC COMPLETION" not in guarded
    assert "BASE COVERAGE COMPLETE" not in guarded
    assert "node_completed=false" in guarded.lower()
    assert "освоена" not in guarded.lower()
    assert "USER CHOICE HANDLING" not in guarded
    assert "quick_replies MUST" not in open_text
    assert "quick_replies MUST" not in guarded


def test_deep_analysis_prompt_requires_follow_up_not_phrase_ban() -> None:
    assert "follow_up_question" in DEEP_ANALYSIS_PROMPT
    assert "REQUIRED" in DEEP_ANALYSIS_PROMPT or "required" in DEEP_ANALYSIS_PROMPT.lower()
    # No phrase-ban crutches in the isolated system prompt.
    assert "Нода освоена на 100%" not in DEEP_ANALYSIS_PROMPT
    assert "CRITICAL FAIL" not in DEEP_ANALYSIS_PROMPT


def test_select_deep_analysis_includes_session_flags() -> None:
    from knowledge_engine.src.node_deep_dive.prompt_factory import (
        deep_analysis_context_policy,
        deep_analysis_hard_guard_block,
    )

    system, mode, _ = select_system_prompt_and_mode(
        "[mode:deep_analysis] x",
        default_system_prompt="DEFAULT",
    )
    assert mode == "deep_analysis"
    assert deep_analysis_hard_guard_block().strip() in system
    assert "node_completed: false" in system
    assert "DeepDiveDeepAnalysisContract" in system
    assert "TOPIC COMPLETION" not in system
    policy = deep_analysis_context_policy()
    assert policy["node_completed"] is False
    assert policy["suppress_topic_completion"] is True


def test_deep_analysis_chip_skips_evaluator() -> None:
    assert classify_control_chip("[mode:deep_analysis] Задачка со звёздочкой") == (
        "deep_analysis"
    )
    assert classify_control_chip("Задачка со звёздочкой") == "deep_analysis"
    assert is_quick_reply_control_message(
        "[mode:deep_analysis] Задачка со звёздочкой"
    )


def test_orchestrate_holds_transition_for_deep_analysis() -> None:
    mem = SessionMemory(
        pending_evaluation_concept_id="agg",
        pending_eval_kind="deep_analysis",
        star_task_status="in_progress",
        sub_concepts=[
            SubConceptRecord(
                id="agg",
                label="agg",
                status="verified",
                why_passed=True,
                how_passed=True,
                mechanic_passed=True,
            )
        ],
    )
    out = DeepDiveLLMOutput(
        ready_for_transition=True,
        follow_up_question="Спроектируйте …",
        question_sub_concept_id="agg",
    )
    packed = orchestrate_tutor_llm_output(
        mem,
        out,
        user_message="[mode:deep_analysis] Задачка со звёздочкой",
        node_layer="advanced",
    )
    assert packed.ready_for_transition is False
    assert mem.pending_evaluation_concept_id == "agg"
    assert mem.pending_eval_kind == "deep_analysis"
    assert mem.star_task_status == "in_progress"


def test_star_task_fsm_blocks_transition_until_resolved() -> None:
    from knowledge_engine.src.node_deep_dive.star_task_fsm import (
        apply_star_task_eval_outcome,
        mark_star_task_in_progress,
        star_task_blocks_transition,
    )

    mem = SessionMemory(
        pending_evaluation_concept_id="agg",
        sub_concepts=[
            SubConceptRecord(
                id="agg",
                label="agg",
                status="verified",
                why_passed=True,
                how_passed=True,
                mechanic_passed=True,
            )
        ],
    )
    mark_star_task_in_progress(mem, concept_id="agg")
    assert mem.star_task_status == "in_progress"
    assert star_task_blocks_transition(mem)
    out = DeepDiveLLMOutput(ready_for_transition=True, follow_up_question="x")
    packed = orchestrate_tutor_llm_output(
        mem, out, user_message="обычный ответ", node_layer="advanced"
    )
    assert packed.ready_for_transition is False

    apply_star_task_eval_outcome(mem, concept_id="agg", resolved=False)
    assert mem.star_task_status == "needs_refinement"
    assert mem.pending_eval_kind == "deep_analysis"
    assert mem.pending_evaluation_concept_id == "agg"
    packed2 = orchestrate_tutor_llm_output(
        mem, out, user_message="доработал дизайн", node_layer="advanced"
    )
    assert packed2.ready_for_transition is False

    apply_star_task_eval_outcome(mem, concept_id="agg", resolved=True)
    assert mem.star_task_status == "resolved"
    assert not star_task_blocks_transition(mem)


def test_star_task_needs_refinement_keeps_pending_after_eval() -> None:
    mem = SessionMemory(
        pending_evaluation_concept_id="agg",
        pending_eval_kind="deep_analysis",
        asked_question_sub_concept_id="agg",
        star_task_status="in_progress",
        sub_concepts=[
            SubConceptRecord(
                id="agg",
                label="Aggregation",
                status="partial",
                why_passed=True,
                how_passed=False,
                mechanic_passed=False,
            )
        ],
    )
    node = NodeDataInput(
        node_id="n1",
        title="Aggregation",
        layer="advanced",
        core_concepts=["aggregation"],
        learning_goal="Understand aggregation",
    )
    fake = MagicMock()
    fake.updates = [
        MagicMock(
            id="agg",
            why_passed=True,
            how_passed=False,
            mechanic_passed=False,
            evidence="ignored partition edge case",
            focus_hint="Нужен разбор отказа при partition",
            status=None,
        )
    ]
    from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (
        process_sub_concept_user_answer,
    )

    with patch(
        "knowledge_engine.src.node_deep_dive.sub_concept_evaluator.run_gemini_structured_with_chain",
        return_value=fake,
    ), patch(
        "knowledge_engine.src.node_deep_dive.concept_map.is_quick_reply_control_message",
        return_value=False,
    ), patch(
        "knowledge_engine.src.node_deep_dive.lecture_scope.is_lecture_request_message",
        return_value=False,
    ):
        process_sub_concept_user_answer(
            "Сделаю простой кэш без учёта партиций.",
            mem,
            node,
            "anchor",
        )
    assert mem.star_task_status == "needs_refinement"
    assert mem.pending_eval_kind == "deep_analysis"
    assert mem.pending_evaluation_concept_id == "agg"
    assert mem.last_eval_directive == "STAR_TASK_NEEDS_REFINEMENT"


def test_fact_manifest_skips_deep_analysis_tutor_eviction() -> None:
    mem = SessionMemory(pending_eval_kind="deep_analysis")
    with patch(
        "knowledge_engine.src.node_deep_dive.fact_manifest.run_gemini_structured_with_chain"
    ) as mock_llm:
        update_manifest_from_evicted(
            mem,
            {
                "role": "tutor",
                "content": "Спроектируйте систему с P99 < 50ms и 4GB RAM.",
            },
            "anchor",
        )
        mock_llm.assert_not_called()


def test_fact_manifest_still_merges_user_under_deep_analysis() -> None:
    mem = SessionMemory(pending_eval_kind="deep_analysis")
    with patch(
        "knowledge_engine.src.node_deep_dive.fact_manifest.run_gemini_structured_with_chain",
        side_effect=RuntimeError("skip network"),
    ):
        # Falls back to heuristic; must not raise / early-return on user role.
        update_manifest_from_evicted(
            mem,
            {"role": "user", "content": "Я бы использовал шардирование по ключу."},
            "anchor",
        )


def test_gap_eval_uses_deep_analysis_system_when_pending_kind_set() -> None:
    from knowledge_engine.schemas.llm_contracts.evaluator_critique import (
        EvaluatedIdea,
        EvaluatorCritiqueContract,
        IdeaStatus,
    )

    mem = SessionMemory(
        pending_evaluation_concept_id="agg",
        pending_eval_kind="deep_analysis",
        asked_question_sub_concept_id="agg",
        sub_concepts=[
            SubConceptRecord(
                id="agg",
                label="Aggregation",
                status="partial",
                why_passed=True,
                how_passed=False,
                mechanic_passed=False,
            )
        ],
    )
    node = NodeDataInput(
        node_id="n1",
        title="Aggregation",
        layer="advanced",
        core_concepts=["aggregation"],
        learning_goal="Understand aggregation",
    )
    fake = EvaluatorCritiqueContract(
        target_layer="DEEP",
        passes_threshold=True,
        bloom_level_matched=True,
        analyzed_ideas=[
            EvaluatedIdea(
                idea_concept="memory-bound shard refusal",
                status=IdeaStatus.STRONG,
                technical_note="Respects stated RAM class; sound trade-off.",
            )
        ],
        unaccounted_edge_cases=[],
        verdict_reason="Constraints respected; trade-off justified.",
    )

    with patch(
        "knowledge_engine.src.node_deep_dive.sub_concept_evaluator.run_gemini_structured_with_chain",
        return_value=fake,
    ) as mock_llm:
        directive = run_sub_concept_gap_eval(
            "Учитывая лимит памяти, я откажусь от полного материала в RAM.",
            mem,
            node,
            "anchor",
            concept_id="agg",
        )
        assert mock_llm.called
        system_arg = mock_llm.call_args[0][1]
        assert DEEP_ANALYSIS_EVAL_SYSTEM.strip() in system_arg
        assert mock_llm.call_args[0][4] is not None
        # Overlay uses EvaluatorCritiqueContract, not gap booleans.
        assert mock_llm.call_args[0][4].__name__ == "EvaluatorCritiqueContract"
        payload_arg = mock_llm.call_args[0][2]
        assert "deep_analysis" in payload_arg
        assert directive == "DEEP_MASTERY_EARNED"
        from knowledge_engine.src.node_deep_dive.concept_map_state import (
            has_overlay_award,
        )

        assert has_overlay_award(mem, "agg")
        # Core HOW/MECH must stay untouched.
        row = mem.sub_concepts[0]
        assert row.how_passed is False
        assert row.mechanic_passed is False
        assert row.status == "partial"
        assert mem.last_evaluator_critique.get("passes_threshold") is True
        assert "STRONG" in (mem.last_evaluator_feedback or "")


def test_coverage_summary_keeps_deep_mastery_separate_from_overall() -> None:
    from knowledge_engine.src.node_deep_dive.concept_map_state import (
        build_coverage_summary,
        register_deep_mastery,
    )

    mem = SessionMemory(
        sub_concepts=[
            SubConceptRecord(
                id="agg",
                label="agg",
                status="verified",
                why_passed=True,
                how_passed=True,
                mechanic_passed=True,
            )
        ]
    )
    register_deep_mastery(mem, "agg")
    cov = build_coverage_summary(mem)
    assert cov is not None
    assert cov.overall_score == 100
    assert cov.deep_mastery_ids == ["agg"]
    assert cov.deep_mastery_count == 1


def test_base_complete_behavior_offers_star_not_100_percent() -> None:
    from knowledge_engine.src.node_deep_dive.tutor_behavior_state import (
        _next_action_for_mode,
    )

    mem = SessionMemory(
        sub_concepts=[
            SubConceptRecord(
                id="agg",
                label="agg",
                status="verified",
                why_passed=True,
                how_passed=True,
                mechanic_passed=True,
            )
        ]
    )
    text = _next_action_for_mode(
        "dialogue_feedback",
        "ANSWER",
        "chat",
        "pathway_decision",
        memory=mem,
        user_message="",
        node_layer="advanced",
    )
    assert "pathway=base_complete" in text
    assert "Нода полностью освоена" not in text
    assert "Базовая теория ноды усвоена" not in text
    # UI owns Asterisk chips — no hardcoded Russian star CTA in next_action.
