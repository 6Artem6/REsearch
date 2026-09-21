"""Regression: graph nodes must not KeyError on a missing state["anchor"].

Real production failure (perf_debug.log): sub_concept_eval_node crashed with
``KeyError: 'anchor'`` reading ``state["anchor"]`` directly. "anchor" is a
pure, deterministic derivation of (curriculum_id, node_id) — see
node_session_reset.py::node_deep_dive_anchor (same format as
engine.py::_anchor) — so every node that needs it can safely recompute it
instead of trusting a channel that a resumed/stale LangGraph checkpoint may
not carry.
"""

from __future__ import annotations

from knowledge_engine.src.domains.grounding.memory_schemas import SessionMemory
from knowledge_engine.src.domains.grounding.node_session_reset import (
    node_deep_dive_anchor,
)
from knowledge_engine.src.domains.grounding.schemas import (
    NodeDataInput,
    NodeDeepDiveRequest,
)


def _req() -> NodeDeepDiveRequest:
    return NodeDeepDiveRequest(
        curriculum_id="clickhouse_mergetree_f32db532a801",
        node_data=NodeDataInput(
            node_id="clickhouse",
            title="ClickHouse MergeTree",
            layer="foundation",
            category="databases",
            brief_summary="x",
            core_concepts=["mergetree"],
            learning_goal="g",
        ),
        user_action="chat",
        user_message="Почему колоночное хранение эффективно для OLAP?",
    )


def test_sub_concept_eval_node_recovers_anchor_when_missing_from_state(monkeypatch):
    req = _req()
    mem = SessionMemory()
    mem.pending_evaluation_concept_id = "sc1"
    mem.asked_question_sub_concept_id = "sc1"

    import knowledge_engine.src.domains.grounding.graph.nodes.sub_concept_eval as node

    captured: dict[str, str] = {}

    def fake_process(user_message, memory, node_data, anchor):
        captured["anchor"] = anchor

    monkeypatch.setattr(node, "process_sub_concept_user_answer", fake_process)
    monkeypatch.setattr(node, "stored_pending_evaluation_id", lambda m: "sc1")

    state = {"request": req, "memory": mem}  # deliberately no "anchor" key
    out = node.sub_concept_eval_node(state, None)

    assert captured["anchor"] == node_deep_dive_anchor(
        "clickhouse_mergetree_f32db532a801", "clickhouse"
    )
    assert out["memory"] is mem


def test_commit_turn_node_recovers_anchor_when_missing_from_state():
    from knowledge_engine.src.domains.grounding.graph.nodes.commit_turn import (
        commit_turn_node,
    )
    from knowledge_engine.src.domains.grounding.tutor_dialogue import (
        deep_dive_llm_output_from_chat_text,
    )

    req = _req()
    mem = SessionMemory()
    llm_out = deep_dive_llm_output_from_chat_text("Ответ тьютора без вопроса.")
    state = {
        "request": req,
        "memory": mem,
        "tutor_message": "Ответ тьютора без вопроса.",
        "llm_out": llm_out,
    }  # deliberately no "anchor" key
    out = commit_turn_node(state)
    assert out["memory"] is mem


def test_step_analysis_node_recovers_anchor_when_missing_from_state(monkeypatch):
    import knowledge_engine.src.domains.grounding.graph.nodes.step_analysis as node

    req = _req()
    mem = SessionMemory()

    class _Analysis:
        concept_updates: list = []
        critical_gap = None

    monkeypatch.setattr(node, "should_run_step_analysis_llm", lambda *a, **k: False)
    monkeypatch.setattr(
        node, "heuristic_step_analysis", lambda *a, **k: _Analysis()
    )

    state = {"request": req, "memory": mem}  # deliberately no "anchor" key
    out = node.step_analysis_node(state, None)
    assert out["memory"] is mem
