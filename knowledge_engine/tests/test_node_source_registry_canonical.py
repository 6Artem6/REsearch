"""Session registry: one canonical URL per mapped src_* (no duplicate registry rows)."""

from __future__ import annotations

from unittest.mock import patch

from knowledge_engine.services.node_source_registry import (
    build_session_source_registry,
    fresh_mapped_source_ids_for_node,
)
from knowledge_engine.src.processors.source_anchors import retarget_source_anchor_tags


def test_build_session_registry_one_row_per_mapped_id() -> None:
    graph = {
        "curriculum_sources_registry": [
            {"source_id": "src_a", "url": "https://habr.com/stale", "title": "stale"},
            {"source_id": "src_a", "url": "https://habr.com/canonical", "title": "ok"},
            {"source_id": "src_b", "url": "https://arxiv.org/abs/1234", "title": "b"},
        ],
        "nodes": [],
    }
    with patch(
        "knowledge_engine.services.skill_tree_store.get_curriculum_graph",
        return_value=graph,
    ):
        reg = build_session_source_registry("cid", ["src_a", "src_b"])
    assert len(reg) == 2
    assert reg[0]["id"] == "S1"
    assert reg[0]["course_source_id"] == "src_a"
    assert reg[0]["url"] == "https://habr.com/canonical"
    assert reg[1]["id"] == "S2"
    assert reg[1]["course_source_id"] == "src_b"


def test_fresh_mapped_source_ids_prefers_nonempty_fallback() -> None:
    """req.node_data snapshot wins when it already has ids — no silent
    override by the persisted graph (see
    test_finalize_falls_back_to_request_node_without_dense_lecture)."""
    graph = {"nodes": [{"node_id": "n1", "mapped_source_ids": ["src_from_graph"]}]}
    with patch(
        "knowledge_engine.services.skill_tree_store.get_curriculum_graph",
        return_value=graph,
    ):
        out = fresh_mapped_source_ids_for_node("cid", "n1", ["src_9"])
    assert out == ["src_9"]


def test_fresh_mapped_source_ids_reads_graph_when_fallback_empty() -> None:
    """Bug scenario: a search (init lazy grounding / lecture external search /
    any future scenario) persisted new sources to the graph, but the node
    object captured earlier this turn still has mapped_source_ids=[] and
    nothing downstream refreshed it — the graph read is the safety net."""
    graph = {"nodes": [{"node_id": "n1", "mapped_source_ids": ["src_1", "src_2"]}]}
    with patch(
        "knowledge_engine.services.skill_tree_store.get_curriculum_graph",
        return_value=graph,
    ):
        out = fresh_mapped_source_ids_for_node("cid", "n1", [])
    assert out == ["src_1", "src_2"]


def test_fresh_mapped_source_ids_falls_back_when_graph_missing() -> None:
    with patch(
        "knowledge_engine.services.skill_tree_store.get_curriculum_graph",
        return_value=None,
    ):
        out = fresh_mapped_source_ids_for_node("cid", "n1", ["src_fallback"])
    assert out == ["src_fallback"]


def test_retarget_source_anchor_tags_by_url() -> None:
    old = [
        {"id": "S1", "url": "https://example.org/a"},
        {"id": "S8", "url": "https://example.org/canonical"},
    ]
    new = [
        {"id": "S1", "url": "https://example.org/canonical"},
    ]
    text = "claim [S8] and stale [S1]"
    out = retarget_source_anchor_tags(text, old, new)
    assert "[S1]" in out
    assert "[S8]" not in out
    assert "stale" not in out or "[S1]" in out
