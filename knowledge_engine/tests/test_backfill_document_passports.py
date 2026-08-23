"""Unit tests for document passport backfill helpers (no Gemma)."""

from __future__ import annotations

from knowledge_engine.db.rag_chunks_schema import map_window_chunk_id
from knowledge_engine.schemas import DocumentSummary
from knowledge_engine.schemas.extraction import KnowledgeAtom, ScopeType
from knowledge_engine.scripts.backfill_document_passports import (
    _prefer_map_rows,
    remap_atom_source_chunk_ids,
)
from knowledge_engine.services.vector_store import VectorStore


def test_prefer_map_rows() -> None:
    rows = [
        {"chunk_id": "d_chunk_1", "chunk_text": "a"},
        {"chunk_id": "d_map_1", "chunk_text": "b"},
        {"chunk_id": "d_map_2", "chunk_text": "c"},
    ]
    preferred = _prefer_map_rows(rows)
    assert [r["chunk_id"] for r in preferred] == ["d_map_1", "d_map_2"]
    assert _prefer_map_rows([{"chunk_id": "d_chunk_1"}])[0]["chunk_id"] == "d_chunk_1"


def test_remap_atom_source_chunk_ids() -> None:
    doc_id = "abcdef0123456789abcdef01"
    synth0 = map_window_chunk_id(doc_id, 0)
    synth1 = map_window_chunk_id(doc_id, 1)
    atom = KnowledgeAtom(
        scope=ScopeType.PRINCIPLE,
        statement="Isolation reduces blast radius across agents",
        source_chunk_ids=[synth0, synth1],
    )
    remapped = remap_atom_source_chunk_ids(
        [atom],
        doc_id=doc_id,
        window_index_to_chunk_id={0: "real_chunk_a", 1: "real_chunk_b"},
    )
    assert remapped[0].source_chunk_ids == ["real_chunk_a", "real_chunk_b"]


def test_passport_is_filled() -> None:
    assert not VectorStore.passport_is_filled(None)
    assert not VectorStore.passport_is_filled(
        DocumentSummary(title="t", url="https://x.test", key_takeaways=[])
    )
    assert not VectorStore.passport_is_filled(
        DocumentSummary(title="t", url="https://x.test", key_takeaways=["short"])
    )
    assert VectorStore.passport_is_filled(
        DocumentSummary(
            title="t",
            url="https://x.test",
            key_takeaways=[
                "[SCOPE: PRINCIPLE] Isolation reduces blast radius across agents"
            ],
        )
    )
    assert VectorStore.passport_is_filled(
        DocumentSummary(
            title="t",
            url="https://x.test",
            executive_summary="Reduce synthesis passport prose about isolation.",
            key_takeaways=[],
        )
    )


def test_curriculum_scope_helpers(monkeypatch) -> None:
    from knowledge_engine.scripts import backfill_document_passports as mod

    monkeypatch.setattr(
        mod,
        "get_curriculum_graph",
        lambda cid: {"curriculum_sources_registry": [{"url": "https://example.com/a"}]},
    )
    monkeypatch.setattr(
        mod,
        "collect_curriculum_library_urls",
        lambda cid: ["https://example.com/a", "https://example.com/b"],
    )
    doc_ids, urls = mod.curriculum_doc_id_scope("agentic_systems_architecture")
    assert len(doc_ids) == 2
    assert "https://example.com/a" in urls
    assert "https://example.com/b" in urls
