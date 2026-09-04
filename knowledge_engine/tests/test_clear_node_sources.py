"""clear_node_sources.apply_clear — regression: local LanceDB knowledge_atoms
must be cleared alongside rag_chunks/summaries for the node's resolved URLs
(pre-migration rows were left behind with no cleanup path, see
VectorStore.delete_knowledge_atoms_for_urls)."""

from __future__ import annotations

from knowledge_engine.scripts import clear_node_sources as cns


def _graph():
    return {
        "nodes": [
            {
                "node_id": "n1",
                "mapped_source_ids": [],
                "resource_urls": ["https://example.com/a"],
            }
        ],
        "curriculum_sources_registry": [],
    }


def test_apply_clear_deletes_knowledge_atoms_alongside_rag_chunks_and_summaries(
    monkeypatch,
) -> None:
    graph = _graph()
    monkeypatch.setattr(cns, "get_curriculum_graph", lambda cid: graph)
    monkeypatch.setattr(cns, "get_curriculum_meta", lambda cid: {})
    monkeypatch.setattr(cns, "save_curriculum_record", lambda *a, **kw: None)
    monkeypatch.setattr(
        cns, "reset_node_deep_dive_persistence", lambda cid, nid: True
    )
    monkeypatch.setattr(cns, "get_blocked_domains", lambda: set())

    calls: dict[str, list[str]] = {}

    class _FakeStore:
        def delete_rag_chunks_for_urls(self, urls):
            calls["rag_chunks"] = list(urls)
            return 2

        def delete_summaries_for_urls(self, urls):
            calls["summaries"] = list(urls)
            return 1

        def delete_knowledge_atoms_for_urls(self, urls):
            calls["knowledge_atoms"] = list(urls)
            return 5

    monkeypatch.setattr(cns, "VectorStore", lambda: _FakeStore())

    report = cns.apply_clear(
        "curr", "n1", clear_blocklist=False, scrub_registry=False
    )

    assert report["lance_knowledge_atoms_removed"] == 5
    assert calls["knowledge_atoms"] == ["https://example.com/a"]
    assert calls["rag_chunks"] == calls["summaries"] == calls["knowledge_atoms"]
