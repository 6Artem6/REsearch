"""Phase 5: adaptive overlay chips + cross-node weakness ledger."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from knowledge_engine.context_drift_manager import (
    ContextDriftManager,
    SessionWeaknessLedger,
    mix_prior_weaknesses_into_eval_system,
    parse_curriculum_id_from_anchor,
    set_weakness_ledger_store_dir,
    tags_from_focus_and_critique,
)
from knowledge_engine.schemas.llm_contracts.evaluator_critique import (
    EvaluatedIdea,
    EvaluatorCritiqueContract,
    IdeaStatus,
)
from knowledge_engine.src.node_deep_dive.deep_analysis_eval_prompt import (
    ADVANCED_ANALYSIS_EVAL_SYSTEM,
)
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    SessionMemory,
    SubConceptRecord,
)
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput
from knowledge_engine.src.node_deep_dive.star_task_fsm import (
    CHIP_ADVANCED_ANALYSIS,
    CHIP_DEEP_DESIGN,
    CHIP_OVERLAY_NEXT,
    overlay_offer_quick_replies,
)
from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (
    run_sub_concept_gap_eval,
)
from knowledge_engine.src.node_deep_dive.tutor_behavior_state import (
    overlay_offer_host_chips,
)


@pytest.fixture(autouse=True)
def _isolate_ledger(tmp_path):
    set_weakness_ledger_store_dir(tmp_path)
    yield
    set_weakness_ledger_store_dir(None)


def _node(node_id: str = "node_b") -> NodeDataInput:
    return NodeDataInput(
        node_id=node_id,
        title="Aggregation",
        layer="advanced",
        core_concepts=["aggregation"],
        learning_goal="Understand aggregation",
    )


def _verified_mem(*, pending_kind: str = "advanced_analysis") -> SessionMemory:
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


def _pass_critique(*, cleared: list[str] | None = None) -> EvaluatorCritiqueContract:
    return EvaluatorCritiqueContract(
        target_layer="ADVANCED",
        passes_threshold=True,
        bloom_level_matched=True,
        analyzed_ideas=[
            EvaluatedIdea(
                idea_concept="per-worker timeout bounds hang",
                status=IdeaStatus.STRONG,
                technical_note="Closes the race window under fan-out.",
            )
        ],
        unaccounted_edge_cases=[],
        verdict_reason="L4 analysis closes prior race/P99 tags.",
        cleared_weakness_tags=list(cleared or []),
    )


def test_parse_curriculum_id_from_anchor() -> None:
    assert (
        parse_curriculum_id_from_anchor("node_deep_dive:cur_alpha:node_1")
        == "cur_alpha"
    )
    assert parse_curriculum_id_from_anchor("anchor") == ""


def test_ledger_record_and_clear_roundtrip() -> None:
    led = SessionWeaknessLedger(curriculum_id="cur_x")
    added = led.record_weaknesses(
        ["Race conditions", "P99 latency"],
        node_id="node_a",
        title="Fan-out gather",
        topic_mastery_score=100,
    )
    assert "race_conditions" in added
    assert "p99_latency" in added
    assert led.has_open_weaknesses() is True
    ctx = led.build_cross_node_prompt_context(exclude_node_id="node_b")
    assert "PRIOR WEAKNESSES" in ctx
    assert "race_conditions" in ctx
    assert "node_a" in ctx
    assert "ADVANCED_ASTERISK" in ctx
    removed = led.clear_weaknesses(
        ["race_conditions"],
        node_id="node_b",
        overlay_type="ADVANCED_ASTERISK",
    )
    assert removed == ["race_conditions"]
    assert led.open_weakness_tags() == ["p99_latency"]
    led.clear_weaknesses(None, node_id="node_b", overlay_type="DEEP_ASTERISK")
    assert led.open_weakness_tags() == []
    assert led.has_open_weaknesses() is False


def test_cross_node_transit_between_two_nodes() -> None:
    cid = "cur_phase5"
    mgr_a = ContextDriftManager(cid, persist=True)
    mgr_a.record_weaknesses(
        ["race_conditions", "p99_latency"],
        node_id="node_a",
        title="Gather timeouts",
        topic_mastery_score=100,
    )

    mgr_b = ContextDriftManager(cid, persist=True)
    assert mgr_b.open_weakness_tags() == ["race_conditions", "p99_latency"]
    ctx = mgr_b.build_cross_node_prompt_context(exclude_node_id="node_b")
    assert "node_a" in ctx
    assert "race_conditions" in ctx
    assert "Gather timeouts" in ctx
    assert "PRIOR WEAKNESSES" in ctx

    mixed = mix_prior_weaknesses_into_eval_system(
        ADVANCED_ANALYSIS_EVAL_SYSTEM,
        curriculum_id=cid,
        exclude_node_id="node_b",
        persist=True,
    )
    assert "PRIOR WEAKNESSES" in mixed
    assert "race_conditions" in mixed
    assert "cleared_weakness_tags" in ADVANCED_ANALYSIS_EVAL_SYSTEM

    mgr_b.clear_weaknesses(
        ["race_conditions", "p99_latency"],
        node_id="node_b",
        overlay_type="ADVANCED_ASTERISK",
    )
    mgr_c = ContextDriftManager(cid, persist=True)
    assert mgr_c.open_weakness_tags() == []
    ctx_after = mgr_c.build_cross_node_prompt_context(exclude_node_id="node_c")
    open_line = ctx_after.split("Open weakness_tags:")[-1].split("\n")[0]
    assert "race_conditions" not in open_line
    assert "ADVANCED_ASTERISK" in ctx_after


def test_adaptive_chips_depend_on_weakness_history() -> None:
    cid = "cur_chips"
    clean = overlay_offer_quick_replies()
    assert clean[0] == CHIP_DEEP_DESIGN
    assert CHIP_OVERLAY_NEXT in clean
    assert CHIP_ADVANCED_ANALYSIS not in clean

    host_clean = overlay_offer_host_chips(None, curriculum_id=cid, persist=True)
    assert host_clean == [CHIP_DEEP_DESIGN, CHIP_OVERLAY_NEXT]

    ContextDriftManager(cid, persist=True).record_weaknesses(
        ["how_gap"],
        node_id="node_a",
        title="How gap",
    )
    host_weak = overlay_offer_host_chips(None, curriculum_id=cid, persist=True)
    assert host_weak == [CHIP_ADVANCED_ANALYSIS, CHIP_OVERLAY_NEXT]
    assert CHIP_DEEP_DESIGN not in host_weak

    ContextDriftManager(cid, persist=True).clear_weaknesses(
        None,
        node_id="node_b",
        overlay_type="ADVANCED_ASTERISK",
    )
    host_after = overlay_offer_host_chips(None, curriculum_id=cid, persist=True)
    assert host_after == [CHIP_DEEP_DESIGN, CHIP_OVERLAY_NEXT]


def test_overlay_eval_mixes_prior_weaknesses_and_clears_tags() -> None:
    cid = "cur_eval"
    ContextDriftManager(cid, persist=True).record_weaknesses(
        ["race_conditions"],
        node_id="node_a",
        title="Prior node",
    )
    mem = _verified_mem()
    critique = _pass_critique(cleared=["race_conditions"])
    with patch(
        "knowledge_engine.src.node_deep_dive.sub_concept_evaluator.run_gemini_structured_with_chain",
        return_value=critique,
    ) as mock_llm:
        d = run_sub_concept_gap_eval(
            "I bound each worker with a timeout and cancel stragglers.",
            mem,
            _node("node_b"),
            f"node_deep_dive:{cid}:node_b",
            concept_id="agg",
        )
        system_arg = mock_llm.call_args[0][1]
        assert "PRIOR WEAKNESSES" in system_arg
        assert "race_conditions" in system_arg
        assert mock_llm.call_args[0][4].__name__ == "EvaluatorCritiqueContract"

    assert d == "DEEP_MASTERY_EARNED"
    assert critique.cleared_weakness_tags == ["race_conditions"]
    assert ContextDriftManager(cid, persist=True).open_weakness_tags() == []
    row = mem.sub_concepts[0]
    assert row.why_passed is True
    assert row.how_passed is True
    assert row.mechanic_passed is True


def test_tags_from_focus_and_critique_helper() -> None:
    tags = tags_from_focus_and_critique(
        focus_hint="Need race windows under gather",
        unaccounted_edge_cases=["P99 latency blow-up"],
        directive="PROBE_NEXT_LAYER:HOW",
    )
    assert "how_gap" in tags
    assert any("race" in t for t in tags)
    assert any("p99" in t for t in tags)
