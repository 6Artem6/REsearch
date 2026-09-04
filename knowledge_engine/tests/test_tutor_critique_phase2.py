"""Phase 2: Tutor consumes EvaluatorCritiqueContract JSON directly."""

from __future__ import annotations

from knowledge_engine.schemas.llm_contracts.evaluator_critique import (
    EvaluatedIdea,
    EvaluatorCritiqueContract,
    IdeaStatus,
)
from knowledge_engine.src.node_deep_dive.concept_map_state import (
    format_concept_map_for_tutor,
)
from knowledge_engine.src.node_deep_dive.deep_analysis_prompt import DEEP_ANALYSIS_PROMPT
from knowledge_engine.src.node_deep_dive.eval_result_adapter import (
    format_evaluator_critique_for_tutor,
)
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    SessionMemory,
    SubConceptRecord,
)
from knowledge_engine.src.node_deep_dive.prompt_factory import (
    select_system_prompt_and_mode,
)
from knowledge_engine.src.node_deep_dive.tutor_critique_prompt import (
    TUTOR_CRITIQUE_REVIEW_RULES,
)


def _sample_critique() -> EvaluatorCritiqueContract:
    return EvaluatorCritiqueContract(
        target_layer="DEEP",
        passes_threshold=False,
        bloom_level_matched=True,
        analyzed_ideas=[
            EvaluatedIdea(
                idea_concept="circuit_breaker_on_fanout",
                status=IdeaStatus.STRONG,
                technical_note="Correctly isolates failing workers from the gather path.",
            ),
            EvaluatedIdea(
                idea_concept="single_shared_queue",
                status=IdeaStatus.RISK,
                technical_note="Shared queue becomes a P99 hotspot under burst load.",
            ),
            EvaluatedIdea(
                idea_concept="retry_forever",
                status=IdeaStatus.WEAK,
                technical_note="Unbounded retries amplify thundering-herd load.",
            ),
        ],
        unaccounted_edge_cases=[
            "partial CancelledError during asyncio.gather",
            "poison-pill message poisoning the shared queue",
        ],
        verdict_reason="Design misses cancellation and poison-pill edges.",
    )


def test_format_evaluator_critique_includes_ideas_and_edges() -> None:
    block = format_evaluator_critique_for_tutor(_sample_critique())
    assert "[EVALUATOR_CRITIQUE_JSON]" in block
    assert "circuit_breaker_on_fanout" in block
    assert "STRONG" in block
    assert "RISK" in block
    assert "WEAK" in block
    assert "partial CancelledError" in block
    assert "poison-pill" in block
    assert "passes_threshold" in block


def test_concept_map_injects_critique_json_over_legacy_feedback() -> None:
    critique = _sample_critique()
    mem = SessionMemory(
        last_evaluator_critique=critique.model_dump(mode="json"),
        last_evaluator_feedback="legacy feedback should be secondary",
        last_eval_directive="STAR_TASK_NEEDS_REFINEMENT",
        pending_eval_kind="deep_design",
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
    block = format_concept_map_for_tutor(mem, node_layer="advanced")
    assert "[EVALUATOR_CRITIQUE_JSON]" in block
    assert "circuit_breaker_on_fanout" in block
    assert "unaccounted_edge_cases" in block or "partial CancelledError" in block
    # Prefer structured critique; legacy string may still appear elsewhere but
    # the JSON block is the Phase-2 contract surface.
    assert "analyzed_ideas" in block


def test_deep_analysis_system_includes_critique_review_rules() -> None:
    system, mode, _ = select_system_prompt_and_mode(
        "[mode:deep_analysis] design task",
        default_system_prompt="",
    )
    assert mode == "deep_analysis"
    assert "EVALUATOR_CRITIQUE_JSON" in system or "POINTWISE REVIEW" in system
    assert "STRONG" in system and "RISK" in system and "WEAK" in system
    assert "unaccounted_edge_cases" in system or "EDGE CASES" in system
    assert "Базовая теория" in system  # present only as FORBIDDEN cliché
    assert TUTOR_CRITIQUE_REVIEW_RULES.strip()[:40] in system
    assert "POINTWISE REVIEW" in DEEP_ANALYSIS_PROMPT or "POINTWISE" in system


def test_empty_critique_formats_to_empty() -> None:
    assert format_evaluator_critique_for_tutor(None) == ""
    assert format_evaluator_critique_for_tutor({}) == ""
