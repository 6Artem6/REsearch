"""Phase 7: core freeze invariants, resilience, and cross-node E2E journey."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from knowledge_engine.context_drift_manager import (
    ContextDriftManager,
    set_weakness_ledger_store_dir,
)
from knowledge_engine.schemas.llm_contracts.evaluator_critique import (
    EvaluatedIdea,
    EvaluatorCritiqueContract,
    IdeaStatus,
)
from knowledge_engine.src.node_deep_dive.control_intent import classify_control_chip
from knowledge_engine.src.node_deep_dive.host_parallel import run_host_prep_sync
from knowledge_engine.src.node_deep_dive.intent_definitions import (
    CHIP_ADVANCED_ANALYSIS,
    CHIP_DEEP_DESIGN,
    CHIP_HOW,
    INTENT_NAMES,
    INTENT_RULES,
    validate_intent_catalog,
)
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    SessionMemory,
    SubConceptRecord,
)
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput
from knowledge_engine.src.node_deep_dive.star_task_fsm import (
    CHIP_OVERLAY_NEXT,
    get_star_task_status,
    overlay_type_for_kind,
    set_star_task_status,
)
from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (
    process_sub_concept_user_answer,
    run_sub_concept_gap_eval,
)
from knowledge_engine.src.node_deep_dive.tutor_behavior_state import (
    overlay_offer_host_chips,
)
from knowledge_engine.src.node_deep_dive.vector_intent_router import VectorIntentRouter
from knowledge_engine.src.resilience_manager import (
    MAX_FSM_HOPS_PER_TURN,
    classify_intent_from_rules,
    core_ready_for_overlay,
    is_llm_resilience_error,
    is_tutor_contract_validation_error,
    note_asterisk_fsm_hop,
    reset_asterisk_fsm_hops,
)
from knowledge_engine.src.telemetry_auditor import (
    HostTurnTelemetry,
    clear_host_telemetry_for_tests,
    recent_host_telemetry,
)

_KE_ROOT = Path(__file__).resolve().parents[1]
_SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".runs",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "htmlcov",
    "dist",
    "tokenizers_cache",
}
_SCAN_SUFFIXES = {
    ".py",
    ".js",
    ".mjs",
    ".cjs",
    ".ts",
    ".tsx",
    ".md",
    ".json",
    ".css",
    ".html",
    ".txt",
    ".yml",
    ".yaml",
    ".toml",
    ".mdc",
}


@pytest.fixture(autouse=True)
def _isolate_ledger(tmp_path):
    set_weakness_ledger_store_dir(tmp_path)
    yield
    set_weakness_ledger_store_dir(None)


def _glyph_needles() -> tuple[str, str]:
    """Build forbidden needles without putting the glyph in this file."""
    return chr(0x2605), chr(92) + "u2605"


def _iter_ke_text_files():
    for path in _KE_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in _SCAN_SUFFIXES:
            continue
        yield path


def _verified_mem(*, pending_kind: str = "advanced_analysis") -> SessionMemory:
    return SessionMemory(
        pending_evaluation_concept_id="agg",
        pending_eval_kind=pending_kind,  # type: ignore[arg-type]
        asked_question_sub_concept_id="agg",
        topic_mastery_score=100,
        star_task_status="in_progress",
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


def _node(node_id: str) -> NodeDataInput:
    return NodeDataInput(
        node_id=node_id,
        title="Aggregation",
        layer="advanced",
        core_concepts=["aggregation"],
        learning_goal="Understand aggregation",
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


def test_core_architecture_invariants() -> None:
    glyph, escape = _glyph_needles()
    hits: list[str] = []
    for path in _iter_ke_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        rel = str(path.relative_to(_KE_ROOT))
        if glyph in text or escape in text:
            hits.append(rel)
    assert hits == [], f"forbidden asterisk glyph in: {hits}"

    stats = validate_intent_catalog()
    assert stats["ok"] is True
    assert INTENT_RULES, "INTENT_RULES is the single source of truth"
    assert stats["intents"] == len(INTENT_RULES)
    names = list(INTENT_NAMES)
    assert names.index("advanced_analysis") < names.index("deep_analysis")
    assert names.index("deep_design") < names.index("deep_analysis")
    assert stats["overlay_order"] == [
        "advanced_analysis",
        "deep_design",
        "deep_analysis",
    ]

    incomplete = SessionMemory(
        topic_mastery_score=80,
        sub_concepts=[
            SubConceptRecord(id="core_a", label="Core A", status="partial")
        ],
    )
    assert core_ready_for_overlay(incomplete) is False
    assert overlay_offer_host_chips(incomplete, curriculum_id="inv_gate") == []

    complete = SessionMemory(
        topic_mastery_score=100,
        sub_concepts=[
            SubConceptRecord(
                id="core_a",
                label="Core A",
                status="verified",
                why_passed=True,
                how_passed=True,
                mechanic_passed=True,
            )
        ],
    )
    assert core_ready_for_overlay(complete) is True
    chips = overlay_offer_host_chips(complete, curriculum_id="inv_gate")
    assert CHIP_DEEP_DESIGN in chips
    assert CHIP_OVERLAY_NEXT in chips
    assert CHIP_ADVANCED_ANALYSIS not in chips


def test_full_student_journey_cross_node() -> None:
    cid = "cur_phase7_e2e"
    mem_a = SessionMemory(
        topic_mastery_score=40,
        sub_concepts=[
            SubConceptRecord(id="fanout", label="Fan-out", status="partial")
        ],
    )
    ContextDriftManager(cid, persist=True).record_weaknesses(
        ["race_conditions"],
        node_id="node_a",
        title="Fan-out gather",
        topic_mastery_score=40,
    )
    assert overlay_offer_host_chips(mem_a, curriculum_id=cid, persist=True) == []
    assert ContextDriftManager(cid, persist=True).open_weakness_tags() == [
        "race_conditions"
    ]

    mem_a.topic_mastery_score = 100
    mem_a.sub_concepts[0].status = "verified"
    mem_a.sub_concepts[0].why_passed = True
    mem_a.sub_concepts[0].how_passed = True
    mem_a.sub_concepts[0].mechanic_passed = True
    # Node A closed core still carries the ledger tag into node B.
    assert ContextDriftManager(cid, persist=True).open_weakness_tags() == [
        "race_conditions"
    ]

    mem_b = _verified_mem()
    chips = overlay_offer_host_chips(mem_b, curriculum_id=cid, persist=True)
    assert chips == [CHIP_ADVANCED_ANALYSIS, CHIP_OVERLAY_NEXT]
    assert classify_control_chip(CHIP_ADVANCED_ANALYSIS) == "advanced_analysis"
    assert overlay_type_for_kind("advanced_analysis") == "ADVANCED_ASTERISK"

    critique = _pass_critique(cleared=["race_conditions"])
    with patch(
        "knowledge_engine.src.node_deep_dive.sub_concept_evaluator.run_gemini_structured_with_chain",
        return_value=critique,
    ):
        d = run_sub_concept_gap_eval(
            "I bound each worker with a timeout and cancel stragglers.",
            mem_b,
            _node("node_b"),
            f"node_deep_dive:{cid}:node_b",
            concept_id="agg",
        )
    assert d == "DEEP_MASTERY_EARNED"
    assert critique.cleared_weakness_tags == ["race_conditions"]
    assert ContextDriftManager(cid, persist=True).open_weakness_tags() == []
    row = mem_b.sub_concepts[0]
    assert row.why_passed is True
    assert row.how_passed is True
    assert row.mechanic_passed is True


def test_host_telemetry_json_fields() -> None:
    clear_host_telemetry_for_tests()
    prep = run_host_prep_sync(
        CHIP_HOW,
        curriculum_id="cur_tel",
        session_id="sess_tel",
        node_id="node_tel",
        active_overlay="",
    )
    assert prep.chip == "how"
    assert prep.intent_source == "exact"
    rows = recent_host_telemetry()
    assert rows, "gather_host_prep must emit HostTurnTelemetry"
    row = rows[-1]
    assert isinstance(row, HostTurnTelemetry)
    dumped = row.model_dump()
    assert dumped["session_id"] == "sess_tel"
    assert dumped["node_id"] == "node_tel"
    assert dumped["intent_detected"] == "how"
    assert dumped["intent_source"] == "exact"
    assert "weakness_tags" in dumped
    assert dumped["latency_host_ms"] >= 0.0
    payload = row.model_dump_json()
    assert '"intent_source":"exact"' in payload or '"intent_source": "exact"' in payload


def test_vector_router_degrades_to_intent_rules() -> None:
    router = VectorIntentRouter(
        embed_fn=lambda _t: [1.0, 0.0],
        persist=False,
        auto_sync=False,
        enabled=True,
        timeout_sec=0.0,
    )
    router._ready = True
    router._labels = []
    router._matrix = np.zeros((0, 1), dtype=np.float64)
    intent, score = router.classify(CHIP_ADVANCED_ANALYSIS)
    assert intent == "advanced_analysis"
    assert score == 1.0
    assert router._degraded is True
    assert classify_intent_from_rules(CHIP_ADVANCED_ANALYSIS) == "advanced_analysis"


def test_vector_router_timeout_falls_back_to_rules() -> None:
    def hang(_text: str) -> list[float]:
        import time

        time.sleep(1.0)
        return [1.0, 0.0]

    router = VectorIntentRouter(
        embed_fn=hang,
        persist=False,
        auto_sync=False,
        enabled=True,
        timeout_sec=0.05,
    )
    router._ready = True
    router._labels = ["gloss"]
    router._matrix = np.ones((1, 2), dtype=np.float64)
    intent, _score = router.classify(CHIP_HOW)
    assert intent == "how"
    assert router._degraded is True


def test_evaluator_llm_fault_preserves_fsm() -> None:
    class QuotaErr(Exception):
        status_code = 429

        def __str__(self) -> str:
            return "429 RESOURCE_EXHAUSTED quota"

    mem = _verified_mem()
    assert get_star_task_status(mem) == "in_progress"
    with patch(
        "knowledge_engine.src.node_deep_dive.sub_concept_evaluator.run_gemini_structured_with_chain",
        side_effect=QuotaErr(),
    ):
        process_sub_concept_user_answer(
            "Timeouts on each worker cancel the hung fan-out.",
            mem,
            _node("node_b"),
            "node_deep_dive:cur_res:node_b",
        )
    assert mem.last_eval_directive == "STAR_TASK_NEEDS_REFINEMENT"
    assert get_star_task_status(mem) == "needs_refinement"
    assert mem.pending_eval_kind == "advanced_analysis"
    assert mem.sub_concepts[0].status == "verified"
    assert mem.sub_concepts[0].why_passed is True


def test_asterisk_fsm_hop_cap() -> None:
    mem = SessionMemory()
    reset_asterisk_fsm_hops(mem)
    cycle = [
        "in_progress",
        "needs_refinement",
        "resolved",
        "not_started",
        "in_progress",
        "needs_refinement",
    ]
    for status in cycle:
        set_star_task_status(mem, status)  # type: ignore[arg-type]
    assert mem.asterisk_fsm_hops == MAX_FSM_HOPS_PER_TURN
    assert get_star_task_status(mem) == "in_progress"
    assert note_asterisk_fsm_hop(mem) is False


def test_is_llm_resilience_error_codes() -> None:
    assert is_llm_resilience_error(TimeoutError("gemini timeout"))
    err429 = Exception("429 rate limit")
    assert is_llm_resilience_error(err429)
    err5 = Exception("status=500 unavailable")
    assert is_llm_resilience_error(err5)
    assert not is_llm_resilience_error(ValueError("bad json schema"))
    assert is_tutor_contract_validation_error(
        RuntimeError(
            "Gemini JSON не прошёл валидацию (node_deep_dive / chat): "
            "1 validation error for ActiveDrillStepResponse\n"
            "theory_body\n  Value error, theory_body must contain at least 150 words"
        )
    )
    assert not is_tutor_contract_validation_error(ValueError("bad json schema"))
