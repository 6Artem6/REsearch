"""Ingest node: validate request, load session into state."""

from __future__ import annotations

from knowledge_engine.src.node_deep_dive.graph.state import TutorGraphState
from knowledge_engine.src.node_deep_dive.session_store import get_session

_RAG_PLACEHOLDER = "(не запрашивался — продолжение сессии)"


def ingest_node(state: TutorGraphState) -> TutorGraphState:
    """Load session, memory, anchor, content from ``request`` (chat/verify)."""
    from knowledge_engine.src.node_deep_dive.engine import _anchor, _ensure_memory

    req = state["request"]
    action = (req.user_action or "").strip().lower()
    node = req.node_data
    raw_user = (req.user_message or "").strip()
    if action in ("chat", "verify") and not raw_user:
        raise ValueError("user_message обязателен для chat и verify")

    session = get_session(req.curriculum_id, node.node_id)
    anchor = _anchor(req.curriculum_id, node.node_id)

    if action == "init":
        return {
            **state,
            "anchor": anchor,
            "content": session.content,
            "memory": session.memory,
            "rag_facts_count": state.get("rag_facts_count") or 0,
            "rag_fact_labels": list(state.get("rag_fact_labels") or []),
        }

    memory = _ensure_memory(session, node, _RAG_PLACEHOLDER)
    if memory.intro_question_pending and raw_user:
        memory.intro_question_pending = False

    from knowledge_engine.src.node_deep_dive.content_assets import (
        hydrate_content_diagrams_from_articles,
    )

    content = hydrate_content_diagrams_from_articles(
        session.content,
        node,
        req.curriculum_id,
    )

    return {
        **state,
        "memory": memory,
        "anchor": anchor,
        "content": content,
        "rag_facts_count": state.get("rag_facts_count") or 0,
        "rag_fact_labels": list(state.get("rag_fact_labels") or []),
    }
