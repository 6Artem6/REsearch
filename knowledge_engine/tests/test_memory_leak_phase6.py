"""Phase 6: tracemalloc session plateau, LanceDB pool, ledger pruning."""

from __future__ import annotations

import gc
import os
import tracemalloc
from datetime import datetime, timedelta, timezone

import pytest

from knowledge_engine.context_drift_manager import (
    ClosedWeaknessTag,
    ContextDriftManager,
    SessionWeaknessLedger,
    set_weakness_ledger_store_dir,
)
from knowledge_engine.db.lancedb_pool import (
    get_lancedb_connection,
    lancedb_pool_size,
    reset_lancedb_pool_for_tests,
)
from knowledge_engine.src.node_deep_dive.graph.nodes.commit_turn import commit_turn_node
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    SessionMemory,
    SubConceptRecord,
)
from knowledge_engine.src.node_deep_dive.schemas import (
    DeepDiveLLMOutput,
    NodeDataInput,
    NodeDeepDiveRequest,
)
from knowledge_engine.src.node_deep_dive.star_task_fsm import star_task_blocks_transition
from knowledge_engine.src.node_deep_dive.tiered_memory import (
    ACTIVE_WINDOW_MAX,
    append_to_active_window,
    pop_evicted_message,
)
from knowledge_engine.src.node_deep_dive.vector_intent_router import (
    VectorIntentRouter,
    set_vector_intent_router_for_tests,
)
from knowledge_engine.tests.intent_embed_probe import lexical_probe_embed


@pytest.fixture
def _ledger_dir(tmp_path):
    set_weakness_ledger_store_dir(tmp_path)
    yield tmp_path
    set_weakness_ledger_store_dir(None)


def _fd_count() -> int:
    try:
        return len(os.listdir("/dev/fd"))
    except OSError:
        return 0


def _session_state() -> dict:
    mem = SessionMemory(
        sub_concepts=[
            SubConceptRecord(id="wal_core", label="WAL core", status="partial"),
        ]
    )
    req = NodeDeepDiveRequest(
        curriculum_id="phase6_load",
        node_data=NodeDataInput(
            node_id="wal_node",
            title="Write-ahead log",
            layer="advanced",
            category="storage",
            brief_summary="Host-only load fixture for memory tracing.",
            core_concepts=["wal"],
            learning_goal="Keep the window bounded",
        ),
        user_action="chat",
        user_message="snapshot then replay",
    )
    return {
        "request": req,
        "memory": mem,
        "anchor": "node_deep_dive:phase6_load:wal_node",
        "focus_sub_concept_id": "wal_core",
    }


def _drive_turns(state: dict, n: int, ledger: SessionWeaknessLedger, router) -> dict:
    """Simulate n learner turns: bounded window + ledger prune + vector classify."""
    memory = state["memory"]
    for i in range(n):
        append_to_active_window(memory, "user", f"turn {i}: checkpoint then replay")
        while len(memory.active_window) > ACTIVE_WINDOW_MAX:
            pop_evicted_message(memory)
        append_to_active_window(
            memory, "tutor", f"Replay is prefix {i}. What is truncated on a crash?"
        )
        while len(memory.active_window) > ACTIVE_WINDOW_MAX:
            pop_evicted_message(memory)
        if i % 11 == 0:
            router.classify("дожать how")
        if i % 17 == 0:
            ledger.record_weaknesses(
                [f"tag_{i % 9}"],
                node_id="wal_node",
                title="WAL",
            )
        if i % 29 == 0 and ledger.open_weakness_tags():
            ledger.clear_weaknesses(node_id="wal_node", overlay_type="ADVANCED_ASTERISK")
        if i % 40 == 0:
            ledger.prune_closed_tags(max_age_hours=1, max_keep=12)
        star_task_blocks_transition(memory)
    state["memory"] = memory
    return state


def test_session_memory_plateaus_over_500_turns(_ledger_dir):
    router = VectorIntentRouter(
        threshold=0.82,
        embed_fn=lexical_probe_embed,
        persist=True,
        db_path=_ledger_dir / "intent_lance",
        embed_model="probe-embed",
        auto_sync=True,
        enabled=True,
    )
    set_vector_intent_router_for_tests(router)
    try:
        state = _session_state()
        ledger = SessionWeaknessLedger(curriculum_id="phase6_load")
        llm_out = DeepDiveLLMOutput(
            technical_explanation="Replay is a prefix of the log.",
            follow_up_question="What is truncated on a crash?",
            question_sub_concept_id="wal_core",
        )
        commit_turn_node(
            {
                **state,
                "tutor_message": (
                    llm_out.technical_explanation + "\n\n" + llm_out.follow_up_question
                ),
                "llm_out": llm_out,
            }
        )
        gc.collect()
        tracemalloc.start()
        state = _drive_turns(state, 80, ledger, router)
        gc.collect()
        warm, _peak_warm = tracemalloc.get_traced_memory()
        state = _drive_turns(state, 500, ledger, router)
        gc.collect()
        after, _peak_after = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert len(state["memory"].active_window) <= 8
        extra = after - warm
        # After warmup the window/ledger are capped — extra 500 turns must not
        # look like a linear leak of the warmup slope.
        per_warm = max(warm / 80.0, 1.0)
        per_extra = extra / 500.0
        assert extra < 4 * 1024 * 1024, f"growth after warmup={extra} bytes"
        assert per_extra < per_warm * 0.45, (
            f"per-turn after warmup {per_extra:.0f} vs warmup {per_warm:.0f}"
        )
    finally:
        set_vector_intent_router_for_tests(None)
        tracemalloc.stop()


def test_lancedb_connection_pool_reuses_one_handle(tmp_path):
    reset_lancedb_pool_for_tests()
    db_path = tmp_path / "pooled_lance"
    before_fd = _fd_count()
    routers = []
    for _ in range(12):
        routers.append(
            VectorIntentRouter(
                threshold=0.82,
                embed_fn=lexical_probe_embed,
                persist=True,
                db_path=db_path,
                embed_model="probe-embed",
                auto_sync=True,
                enabled=True,
            )
        )
    assert lancedb_pool_size() == 1
    handles = {id(r._connect_db()) for r in routers}
    assert len(handles) == 1
    assert id(get_lancedb_connection(db_path)) == next(iter(handles))
    after_fd = _fd_count()
    # Shared pool: extra routers must not open a fd per instance.
    if before_fd and after_fd:
        assert after_fd - before_fd < 8
    reset_lancedb_pool_for_tests()


def test_closed_weakness_tags_prune_and_decay(_ledger_dir):
    led = SessionWeaknessLedger(curriculum_id="phase6_load")
    led.record_weaknesses(["race_conditions", "p99_latency"], node_id="n1")
    led.clear_weaknesses(["race_conditions"], node_id="n1")
    assert "race_conditions" not in led.open_weakness_tags()
    assert any(c.tag == "race_conditions" for c in led.closed_weaknesses)

    old = (datetime.now(timezone.utc) - timedelta(hours=200)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    led.closed_weaknesses.append(
        ClosedWeaknessTag(tag="stale_edge", closed_at=old, node_id="n0")
    )
    led.node_summaries[0].weakness_tags.append("stale_edge")
    stats = led.prune_closed_tags(max_age_hours=168, max_keep=24)
    assert stats["dropped_closed"] >= 1
    assert all(c.tag != "stale_edge" for c in led.closed_weaknesses)
    assert "stale_edge" not in led.node_summaries[0].weakness_tags

    mgr = ContextDriftManager("phase6_load", persist=True)
    mgr.record_weaknesses(["overflow_tag"], node_id="n2")
    mgr.clear_weaknesses(node_id="n2")
    pruned = mgr.prune_closed_tags(max_age_hours=1, max_keep=2)
    assert "dropped_closed" in pruned
