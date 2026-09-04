"""Phase 6 host-layer SLAs (no LLM / network): exact, vector, commit+FSM+prompt."""

from __future__ import annotations

import asyncio
import time

import pytest

from knowledge_engine.src.node_deep_dive.control_intent import classify_control_chip
from knowledge_engine.src.node_deep_dive.graph.nodes.commit_turn import commit_turn_node
from knowledge_engine.src.node_deep_dive.host_parallel import (
    gather_host_prep,
    run_host_prep_sync,
)
from knowledge_engine.src.node_deep_dive.intent_definitions import (
    CHIP_HOW,
    CHIP_MECH,
)
from knowledge_engine.src.node_deep_dive.memory_schemas import (
    SessionMemory,
    SubConceptRecord,
)
from knowledge_engine.src.node_deep_dive.prompt_factory import select_system_prompt_and_mode
from knowledge_engine.src.node_deep_dive.schemas import (
    DeepDiveLLMOutput,
    NodeDataInput,
    NodeDeepDiveRequest,
)
from knowledge_engine.src.node_deep_dive.star_task_fsm import (
    overlay_offer_quick_replies,
    star_task_blocks_transition,
)
from knowledge_engine.src.node_deep_dive.vector_intent_router import (
    VectorIntentRouter,
    set_vector_intent_router_for_tests,
)
from knowledge_engine.tests.intent_embed_probe import lexical_probe_embed

_EXACT_SLA_S = 0.001
_VECTOR_SLA_S = 0.008
_HOST_CYCLE_P99_S = 0.015


def _p99(samples: list[float]) -> float:
    xs = sorted(samples)
    if not xs:
        return 0.0
    idx = min(len(xs) - 1, max(0, int(round(0.99 * (len(xs) - 1)))))
    return xs[idx]


@pytest.fixture
def probe_router(tmp_path):
    router = VectorIntentRouter(
        threshold=0.82,
        embed_fn=lexical_probe_embed,
        persist=True,
        db_path=tmp_path / "intent_lance",
        embed_model="probe-embed",
        auto_sync=True,
        enabled=True,
    )
    set_vector_intent_router_for_tests(router)
    yield router
    set_vector_intent_router_for_tests(None)


def _host_state() -> dict:
    mem = SessionMemory(
        sub_concepts=[
            SubConceptRecord(id="agg", label="Aggregation", status="partial"),
        ]
    )
    req = NodeDeepDiveRequest(
        curriculum_id="phase6_bench",
        node_data=NodeDataInput(
            node_id="agg_node",
            title="Aggregation",
            layer="advanced",
            category="systems",
            brief_summary="Host-cycle benchmark fixture without LLM calls.",
            core_concepts=["aggregation"],
            learning_goal="Measure host P99",
        ),
        user_action="chat",
        user_message=CHIP_HOW,
    )
    return {
        "request": req,
        "memory": mem,
        "anchor": "node_deep_dive:phase6_bench:agg_node",
        "focus_sub_concept_id": "agg",
    }


def test_exact_match_router_under_1ms(probe_router):
    classify_control_chip(CHIP_HOW)  # warmup imports
    samples: list[float] = []
    for _ in range(400):
        t0 = time.perf_counter()
        chip = classify_control_chip(CHIP_HOW)
        samples.append(time.perf_counter() - t0)
        assert chip == "how"
    assert _p99(samples) < _EXACT_SLA_S


def test_vector_match_under_8ms(probe_router):
    probe_router.classify("дожать how")  # warmup matrix + probe
    samples: list[float] = []
    for _ in range(200):
        t0 = time.perf_counter()
        intent, score = probe_router.classify("дожать how")
        samples.append(time.perf_counter() - t0)
        assert intent == "how"
        assert score >= 0.82
    assert _p99(samples) < _VECTOR_SLA_S


def test_gather_host_prep_runs_routing_and_prompt_concurrently(probe_router):
    prep = asyncio.run(
        gather_host_prep(
            CHIP_MECH,
            curriculum_id="phase6_bench",
            default_system_prompt="default system",
        )
    )
    assert prep.chip == "mech"
    assert prep.factory_mode in ("default", "deep_dive_mech")
    sync = run_host_prep_sync(CHIP_HOW, default_system_prompt="default system")
    assert sync.chip == "how"


def test_commit_fsm_prompt_p99_under_15ms(probe_router):
    state = _host_state()
    llm_out = DeepDiveLLMOutput(
        technical_explanation="Fan-in waits for all branches.",
        follow_up_question="Where does backpressure sit?",
        question_sub_concept_id="agg",
    )
    # Warmup: prompt assembly + one commit.
    select_system_prompt_and_mode(CHIP_HOW, default_system_prompt="sys")
    commit_turn_node(
        {
            **state,
            "tutor_message": "Fan-in waits.\n\nWhere does backpressure sit?",
            "llm_out": llm_out,
        }
    )

    samples: list[float] = []
    for i in range(120):
        mem = SessionMemory(
            sub_concepts=[
                SubConceptRecord(id="agg", label="Aggregation", status="partial"),
            ]
        )
        req = state["request"].model_copy(
            update={"user_message": f"{CHIP_HOW} #{i}"}
        )
        t0 = time.perf_counter()
        prep = run_host_prep_sync(
            CHIP_HOW,
            curriculum_id="phase6_bench",
            default_system_prompt="sys",
        )
        out = commit_turn_node(
            {
                "request": req,
                "memory": mem,
                "anchor": state["anchor"],
                "focus_sub_concept_id": "agg",
                "tutor_message": "Fan-in waits.\n\nWhere does backpressure sit?",
                "llm_out": llm_out.model_copy(),
            }
        )
        blocked = star_task_blocks_transition(out["memory"])
        overlay_offer_quick_replies(weakness_tags=[])
        samples.append(time.perf_counter() - t0)
        assert prep.chip == "how"
        assert blocked is False
    assert _p99(samples) < _HOST_CYCLE_P99_S, f"P99={_p99(samples)*1000:.2f}ms"
