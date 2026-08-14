"""Lecture RAG source scope and LanceDB prefilter helpers."""

from __future__ import annotations

from knowledge_engine.services.lecture_rag_source_scope import (
    build_lecture_rag_source_scope,
)
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput


def test_where_doc_ids_in_sql():
    clause = VectorStore._where_doc_ids_in(["abc123", "def456"])
    assert "IN (" in clause
    assert "'abc123'" in clause


def test_scope_primary_doc_ids_from_urls():
    node = NodeDataInput(
        node_id="n1",
        title="Test node",
        layer="foundation",
        category="c",
        brief_summary="b",
        core_concepts=["c1"],
    )
    scope = build_lecture_rag_source_scope(
        "cur-1",
        node,
        ["https://example.com/a", "https://example.com/b"],
    )
    assert len(scope.primary_doc_ids) == 2
    assert VectorStore.doc_id_for_url("https://example.com/a") in scope.primary_doc_ids


def test_search_with_empty_allowlist_returns_nothing():
    store = VectorStore()
    out = store.search_rag_chunk_rows(
        "test query",
        limit=5,
        allowed_doc_ids=[],
        prefilter=True,
    )
    assert out == []


def test_append_fine_chunks_uses_primary_doc_ids_only(monkeypatch):
    from knowledge_engine.services import lecture_rag_context as lrc
    from knowledge_engine.services.lecture_rag_source_scope import LectureRagSourceScope

    calls: list[list[str] | None] = []

    class FakeStore:
        def search_rag_chunk_rows(self, *args, **kwargs):
            calls.append(kwargs.get("allowed_doc_ids"))
            return []

        def count_rag_chunks_in_scope(self, allowed):
            return 0

    monkeypatch.setattr(lrc, "VectorStore", lambda: FakeStore())
    scope = LectureRagSourceScope(
        node_id="n1",
        curriculum_id="c1",
        primary_urls=("https://a.com/x", "https://b.com/y"),
        library_urls=(),
        primary_doc_ids=(
            VectorStore.doc_id_for_url("https://a.com/x"),
            VectorStore.doc_id_for_url("https://b.com/y"),
        ),
        library_doc_ids=(),
    )
    lrc._append_fine_rag_chunk_candidates(
        "q",
        [],
        set(),
        scope=scope,
        node_id="n1",
    )
    assert len(calls) >= 1
    assert calls[0] == list(scope.primary_doc_ids)
