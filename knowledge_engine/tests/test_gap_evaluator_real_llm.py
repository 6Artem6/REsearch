"""Integration tests: gap evaluator via real Gemini Lite API."""

from __future__ import annotations

import os
import uuid

import pytest

from knowledge_engine.config import GEMINI_LITE_MODEL
from knowledge_engine.schemas.llm_contracts.tutor import SubConceptGapEvalContract
from knowledge_engine.services.gemini_stateless import run_gemini_structured_with_chain
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    SessionMemory,
    SubConceptRecord,
)
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput
from knowledge_engine.src.node_deep_dive.sub_concept_evaluator import (
    GAP_EVAL_SYSTEM,
    _gap_eval_payload,
    apply_threshold_to_sub_concept,
)

_HAS_GEMINI = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

pytestmark = pytest.mark.skipif(
    not _HAS_GEMINI,
    reason="GEMINI_API_KEY or GOOGLE_API_KEY required for real LLM gap eval tests",
)


def _wal_target_memory() -> tuple[SessionMemory, SubConceptRecord, NodeDataInput]:
    mem = SessionMemory()
    row = SubConceptRecord(
        id="wal_и_консистентность",
        label="WAL и консистентность",
        success_criterion=(
            "Объяснить write-ahead log: порядок записи, fsync, recovery после crash, "
            "trade-offs latency vs durability."
        ),
        status="unchecked",
    )
    mem.sub_concepts = [row]
    mem.node_goal = "Персистентность и консистентность в production DB"
    node = NodeDataInput(
        node_id="test_wal",
        title="WAL и durability",
        layer="foundation",
        learning_goal=mem.node_goal,
        core_concepts=["WAL"],
    )
    return mem, row, node


def _run_gap_eval_contract(
    mem: SessionMemory,
    node: NodeDataInput,
    target: SubConceptRecord,
    user_message: str,
    last_tutor_question: str,
) -> SubConceptGapEvalContract:
    mem.last_tutor_follow_up_question = last_tutor_question
    payload = _gap_eval_payload(mem, node, user_message, target)
    anchor = f"test_gap_eval_{uuid.uuid4().hex[:12]}"
    return run_gemini_structured_with_chain(
        GEMINI_LITE_MODEL,
        GAP_EVAL_SYSTEM,
        payload,
        anchor,
        SubConceptGapEvalContract,
        "test / sub_concept_gap",
        rpm_pause=False,
        chat_manager=None,
        max_output_tokens=1024,
    )


def _status_from_contract(
    mem: SessionMemory,
    target: SubConceptRecord,
    raw: SubConceptGapEvalContract,
    *,
    layer: str = "foundation",
) -> str:
    _ = mem
    updates = [u for u in (raw.updates or []) if str(u.id or "").strip() == target.id]
    if not updates:
        return target.status
    u0 = updates[0]
    apply_threshold_to_sub_concept(
        target,
        layer=layer,
        why=bool(u0.why_passed),
        how=bool(u0.how_passed),
        mechanic=bool(u0.mechanic_passed),
        evidence=u0.evidence or "",
        llm_focus_hint=u0.focus_hint or "",
    )
    return target.status


@pytest.mark.integration
def test_gap_eval_surface_list_partial():
    mem, target, node = _wal_target_memory()
    tutor_q = (
        "Как WAL обеспечивает durability при crash? Опиши механику записи и recovery."
    )
    user = "Используй WAL, Mutex, Sharding."
    raw = _run_gap_eval_contract(mem, node, target, user, tutor_q)
    status = _status_from_contract(mem, target, raw, layer=node.layer)
    assert status == "partial"
    assert status != "verified"
    hint = (target.focus_hint or "").strip()
    if not hint and raw.updates:
        hint = str((raw.updates[0].focus_hint or "")).strip()
    assert hint
    assert raw.updates and len(raw.updates) == 1


@pytest.mark.integration
def test_gap_eval_deep_answer_verified():
    mem, target, node = _wal_target_memory()
    tutor_q = "Как WAL обеспечивает durability при crash?"
    user = (
        "Сначала изменения пишутся в WAL на диск (часто group commit / fsync policy), "
        "потом в data pages. При crash recovery читает WAL с последнего checkpoint, "
        "redo незавершённые записи — так не теряем committed transactions. "
        "Trade-off: fsync на каждый commit даёт latency, batch/fsync interval снижает I/O "
        "но увеличивает window потери при power loss если sync=false."
    )
    raw = _run_gap_eval_contract(mem, node, target, user, tutor_q)
    status = _status_from_contract(mem, target, raw, layer=node.layer)
    assert status == "verified"
    assert target.why_passed is True


@pytest.mark.integration
def test_gap_eval_refusal_partial():
    mem, target, node = _wal_target_memory()
    tutor_q = "Объясни WAL и recovery после crash."
    user = "Не знаю, давай дальше."
    raw = _run_gap_eval_contract(mem, node, target, user, tutor_q)
    status = _status_from_contract(mem, target, raw, layer=node.layer)
    # Threshold Engine uses PARTIAL (not GAP) when required layers are missing.
    assert status == "partial"
    assert not target.why_passed


@pytest.mark.integration
def test_gap_eval_off_topic_partial():
    mem, target, node = _wal_target_memory()
    tutor_q = "Как WAL обеспечивает durability при crash?"
    user = (
        "ReAct цикл: модель думает, вызывает tool, получает observation и снова думает. "
        "Это не про WAL, но про agent loop."
    )
    raw = _run_gap_eval_contract(mem, node, target, user, tutor_q)
    status = _status_from_contract(mem, target, raw, layer=node.layer)
    assert status == "partial"
    assert (target.focus_hint or "").strip() or (raw.updates or [{}])[0].focus_hint
