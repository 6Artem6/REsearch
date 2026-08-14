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
