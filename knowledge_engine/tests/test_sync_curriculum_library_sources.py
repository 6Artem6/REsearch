from __future__ import annotations

from knowledge_engine.scripts import sync_curriculum_library_sources as sync


def test_plan_prunes_only_orphaned_registry_sources(
    monkeypatch,
) -> None:
    graph = {
        "nodes": [
            {
                "node_id": "mapped",
                "mapped_source_ids": ["src_mapped"],
                "resource_urls": [],
            },
            {
                "node_id": "direct_url",
                "mapped_source_ids": [],
                "resource_urls": ["https://example.com/direct"],
            },
        ],
        "curriculum_sources_registry": [
            {
                "source_id": "src_mapped",
                "title": "Mapped",
                "url": "https://example.com/mapped",
            },
            {
                "source_id": "src_direct",
                "title": "Direct",
                "url": "https://example.com/direct",
            },
            {
                "source_id": "src_shared",
                "title": "Shared orphan",
                "url": "https://example.com/shared",
            },
            {
                "source_id": "src_delete",
                "title": "Delete orphan",
                "url": "https://example.com/delete",
            },
        ],
    }
    other_graph = {
        "nodes": [],
        "curriculum_sources_registry": [
            {"source_id": "other", "url": "https://example.com/shared"}
        ],
    }

    monkeypatch.setattr(
        sync,
        "get_curriculum_graph",
        lambda curriculum_id: (
            graph
            if curriculum_id == "current"
            else other_graph if curriculum_id == "other" else None
        ),
    )
    monkeypatch.setattr(
        sync,
        "list_curriculum_summaries",
        lambda: [{"curriculum_id": "current"}, {"curriculum_id": "other"}],
    )
    monkeypatch.setattr(sync, "_load_all", lambda: {})

    plan = sync.build_plan("current")

    assert plan["registry_before"] == 4
    assert plan["registry_after"] == 2
    assert {row["source_id"] for row in plan["orphaned_registry_entries"]} == {
        "src_shared",
        "src_delete",
    }
    assert plan["orphaned_urls_delete_from_lancedb"] == ["https://example.com/delete"]
    assert plan["orphaned_urls_kept_for_other_curricula"] == [
        "https://example.com/shared"
    ]


def test_apply_sync_deletes_orphaned_knowledge_atoms_alongside_rag_chunks(
    monkeypatch,
) -> None:
    """Regression: knowledge_atoms writes moved to Qdrant-only, but
    pre-migration local LanceDB rows had no cleanup path — apply_sync must
    purge them for orphaned URLs exactly like rag_chunks/summaries."""
    graph = {
        "nodes": [],
        "curriculum_sources_registry": [
            {
                "source_id": "src_delete",
                "title": "Delete orphan",
                "url": "https://example.com/delete",
            }
        ],
    }
    monkeypatch.setattr(
        sync,
        "get_curriculum_graph",
        lambda curriculum_id: graph if curriculum_id == "current" else None,
    )
    monkeypatch.setattr(
        sync, "list_curriculum_summaries", lambda: [{"curriculum_id": "current"}]
    )
    monkeypatch.setattr(sync, "_load_all", lambda: {})
    monkeypatch.setattr(sync, "get_curriculum_meta", lambda curriculum_id: {})
    monkeypatch.setattr(sync, "save_curriculum_record", lambda *a, **kw: None)

    calls: dict[str, list[str]] = {}

    class _FakeStore:
        def delete_rag_chunks_for_urls(self, urls):
            calls["rag_chunks"] = list(urls)
            return 1

        def delete_summaries_for_urls(self, urls):
            calls["summaries"] = list(urls)
            return 1

        def delete_knowledge_atoms_for_urls(self, urls):
            calls["knowledge_atoms"] = list(urls)
            return 1

    monkeypatch.setattr(sync, "VectorStore", lambda: _FakeStore())
    monkeypatch.setattr(
        "knowledge_engine.services.article_diagram_store.delete_diagrams_for_urls",
        lambda urls: 0,
    )
    monkeypatch.setattr(
        "knowledge_engine.services.article_ingestion.figure_registry_service."
        "delete_figure_registry_for_urls",
        lambda urls: 0,
    )
    def _fake_delete_qdrant_urls(urls: list[str]) -> int:
        calls["qdrant"] = list(urls)
        return 1

    monkeypatch.setattr(
        "knowledge_engine.scripts.cleanup_cloud_resources.delete_qdrant_urls",
        _fake_delete_qdrant_urls,
    )

    report = sync.apply_sync("current")

    assert report["lance_knowledge_atoms_removed"] == 1
    assert calls["knowledge_atoms"] == ["https://example.com/delete"]
    assert calls["rag_chunks"] == calls["knowledge_atoms"] == calls["summaries"]
    assert calls["qdrant"] == calls["knowledge_atoms"]
    assert report["qdrant_urls_removed"] == 1
