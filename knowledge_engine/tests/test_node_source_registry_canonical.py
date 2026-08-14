"""Session registry: one canonical URL per mapped src_* (no duplicate registry rows)."""

from __future__ import annotations

from unittest.mock import patch

from knowledge_engine.services.node_source_registry import build_session_source_registry
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
