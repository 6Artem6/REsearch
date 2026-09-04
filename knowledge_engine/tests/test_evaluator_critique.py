"""Phase 1: EvaluatorCritiqueContract + legacy adapter + overlay isolation."""

from __future__ import annotations

from unittest.mock import patch

from knowledge_engine.schemas.llm_contracts.evaluator_critique import (
    EvaluatedIdea,
    EvaluatorCritiqueContract,
    IdeaStatus,
)
from knowledge_engine.src.node_deep_dive.eval_result_adapter import (
    critique_to_feedback_text,
    critique_to_legacy_gap_contract,
    normalize_overlay_target_layer,
)
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    SessionMemory,
    SubConceptRecord,
)
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput
from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (
    run_sub_concept_gap_eval,
)


def _sample_critique(*, passed: bool = True) -> EvaluatorCritiqueContract:
    return EvaluatorCritiqueContract(
        target_layer="DEEP",
        passes_threshold=passed,
        bloom_level_matched=passed,
        analyzed_ideas=[
            EvaluatedIdea(
                idea_concept="async gather with per-worker timeout",
                status=IdeaStatus.STRONG,
                technical_note="Bounds hang risk under fan-out.",
            ),
            EvaluatedIdea(
                idea_concept="retry entire graph on any failure",
                status=IdeaStatus.RISK,
                technical_note="Amplifies load; may break idempotency.",
            ),
            EvaluatedIdea(
                idea_concept="store full chat history in each subagent",
                status=IdeaStatus.WEAK,
                technical_note="Irrelevant to the stated memory constraint.",
            ),
        ],
        unaccounted_edge_cases=[
            "partial result aggregation when one worker times out",
        ],
        verdict_reason=(
            "Pass" if passed else "Missed timeout partial-aggregation edge"
        ),
    )


def test_critique_contract_validates() -> None:
    c = _sample_critique()
    assert c.target_layer == "DEEP"
    assert c.analyzed_ideas[0].status == IdeaStatus.STRONG
    dumped = c.model_dump(mode="json")
    again = EvaluatorCritiqueContract.model_validate(dumped)
    assert again.passes_threshold is True


def test_adapter_groups_ideas_and_keeps_core_booleans_false() -> None:
    c = _sample_critique()
    text = critique_to_feedback_text(c)
    assert "[EVALUATOR_CRITIQUE]" in text
    assert "=== STRONG ===" in text
    assert "=== RISK ===" in text
    assert "=== WEAK ===" in text
    assert "UNACCOUNTED_EDGE_CASES" in text
    legacy = critique_to_legacy_gap_contract(c, concept_id="agg")
    assert len(legacy.updates) == 1
    u = legacy.updates[0]
    assert u.how_passed is False
    assert u.mechanic_passed is False
    assert u.why_passed is False


def test_normalize_overlay_target_layer() -> None:
    assert normalize_overlay_target_layer("deep_analysis") == "DEEP"
    assert normalize_overlay_target_layer("deep_design") == "DEEP"
    assert normalize_overlay_target_layer("advanced_analysis") == "ADVANCED"
    assert normalize_overlay_target_layer("ADVANCED") == "ADVANCED"
    assert normalize_overlay_target_layer("") == "DEEP"


def test_overlay_eval_does_not_set_how_or_mech() -> None:
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
    with patch(
        "knowledge_engine.src.node_deep_dive.sub_concept_evaluator.run_gemini_structured_with_chain",
        return_value=_sample_critique(passed=True),
    ):
        d = run_sub_concept_gap_eval(
            "I add per-worker timeouts and cancel stragglers.",
            mem,
            node,
            "anchor",
            concept_id="agg",
        )
    assert d == "DEEP_MASTERY_EARNED"
    row = mem.sub_concepts[0]
    assert row.how_passed is False
    assert row.mechanic_passed is False
    assert row.status == "partial"
    from knowledge_engine.src.node_deep_dive.concept_map_state import (
        has_overlay_award,
    )

    assert has_overlay_award(mem, "agg")


def test_overlay_eval_fail_refinement_preserves_core() -> None:
    mem = SessionMemory(
        pending_evaluation_concept_id="agg",
        pending_eval_kind="deep_analysis",
        asked_question_sub_concept_id="agg",
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
    node = NodeDataInput(
        node_id="n1",
        title="Aggregation",
        layer="sota",
        core_concepts=["aggregation"],
        learning_goal="Understand aggregation",
    )
    with patch(
        "knowledge_engine.src.node_deep_dive.sub_concept_evaluator.run_gemini_structured_with_chain",
        return_value=_sample_critique(passed=False),
    ):
        d = run_sub_concept_gap_eval(
            "Just use a bigger model.",
            mem,
            node,
            "anchor",
            concept_id="agg",
        )
    assert d == "STAR_TASK_NEEDS_REFINEMENT"
    row = mem.sub_concepts[0]
    assert row.how_passed is True
    assert row.mechanic_passed is True
    assert row.status == "verified"
    assert "agg" not in (mem.deep_mastery_concepts or [])
    assert "UNACCOUNTED" in (mem.last_evaluator_feedback or "")
