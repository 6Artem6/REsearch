"""Mapped node sources: primary scope + mandatory pinned_rag chunks."""

from __future__ import annotations

from knowledge_engine.services.lecture_context_rerank import LectureContextCandidate
from knowledge_engine.services.lecture_rag_context import (
    _PINNED_RAG_LABEL,
    _append_mandatory_mapped_rag_chunks,
    _finalize_lecture_citation_candidates,
    _merge_mandatory_rag_after_rerank,
)
from knowledge_engine.services.lecture_rag_source_scope import (
    build_lecture_rag_source_scope,
    collect_mapped_source_urls,
    mapped_doc_ids_for_node,
    normalize_lecture_source_url,
)
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput


def _node(**kwargs) -> NodeDataInput:
    base = {
        "node_id": "subagent_architectures",
        "title": "Subagent architectures",
        "layer": "foundation",
        "category": "agents",
        "brief_summary": "Patterns",
        "core_concepts": ["orchestration"],
    }
    base.update(kwargs)
    return NodeDataInput(**base)


def test_mapped_urls_in_primary_scope_not_library(monkeypatch):
    graph = {
        "curriculum_sources_registry": [
            {
                "source_id": "src_5",
                "url": "https://example.com/article-mapped",
                "title": "Mapped paper",
            },
            {
                "source_id": "src_99",
                "url": "https://example.com/other-course",
                "title": "Other",
            },
        ],
        "route_sources": [],
        "nodes": [
            {
                "node_id": "subagent_architectures",
                "mapped_source_ids": ["src_5"],
            }
        ],
    }
    monkeypatch.setattr(
        "knowledge_engine.services.lecture_rag_source_scope.get_curriculum_graph",
        lambda _cid: graph,
    )
    node = _node(mapped_source_ids=["src_5"])
    urls = collect_mapped_source_urls("agentic_systems_architecture", node)
    assert urls == ["https://example.com/article-mapped"]
    scope = build_lecture_rag_source_scope(
        "agentic_systems_architecture",
        node,
        ["https://example.com/route-only"],
    )
    assert "https://example.com/article-mapped" in scope.primary_urls
    mapped_norm = normalize_lecture_source_url("https://example.com/article-mapped")
    assert all(
        normalize_lecture_source_url(u) != mapped_norm for u in scope.library_urls
    )
    did = VectorStore.doc_id_for_url("https://example.com/article-mapped")
    assert did in scope.primary_doc_ids
    assert did in mapped_doc_ids_for_node("agentic_systems_architecture", node)


def test_mandatory_pinned_rag_survives_rerank_and_gets_r_index():
    mandatory = [
        LectureContextCandidate(
            label=_PINNED_RAG_LABEL,
            formatted="",
            plain="Mapped article chunk text long enough for lecture body",
            url_key="https://example.com/article-mapped",
            source_id="docmapped",
            source_title="Mapped paper",
            doc_id="docmapped",
            chunk_index=1,
            chunks_in_doc=3,
            retrieval_score=0.91,
        ),
    ]
    mmr_only = [
        LectureContextCandidate(
            label="hybrid_semantic",
            formatted="### other",
            plain="Different unrelated chunk content for semantic hit",
            url_key="https://example.com/other",
            source_title="Other doc",
            doc_id="docother",
            chunk_index=1,
        ),
    ]
    merged = _merge_mandatory_rag_after_rerank(mandatory, mmr_only)
    finalized = _finalize_lecture_citation_candidates(merged)
    assert len(finalized) == 2
    assert finalized[0].formatted.startswith("[R1]")
    assert "Mapped article chunk" in finalized[0].formatted
    assert finalized[1].formatted.startswith("[R2]")


def test_append_mandatory_mapped_rag_chunks(monkeypatch):
    from knowledge_engine.db.rag_chunks_schema import (
        COL_CHUNK_ID,
        COL_CHUNK_INDEX,
        COL_CHUNK_TEXT,
        COL_CHUNK_VECTOR,
        COL_CHUNKS_IN_DOC,
        COL_DOC_ID,
        COL_TITLE,
        COL_URL,
    )

    doc_id = VectorStore.doc_id_for_url("https://example.com/mapped")
    vec = [0.1, 0.9, 0.0]

    class FakeStore:
        _embeddings = type(
            "E",
            (),
            {"embed_query": lambda self, q: [1.0, 0.0, 0.0]},
        )()

        def fetch_rag_chunks_by_doc_id(self, did: str):
            if did != doc_id:
                return []
            return [
                {
                    COL_CHUNK_ID: f"{doc_id}_chunk_1",
                    COL_DOC_ID: doc_id,
                    COL_URL: "https://example.com/mapped",
                    COL_TITLE: "Mapped",
                    COL_CHUNK_TEXT: "Chunk one text long enough for retrieval",
                    COL_CHUNK_INDEX: 1,
                    COL_CHUNKS_IN_DOC: 2,
                    COL_CHUNK_VECTOR: vec,
                },
                {
                    COL_CHUNK_ID: f"{doc_id}_chunk_2",
                    COL_DOC_ID: doc_id,
                    COL_URL: "https://example.com/mapped",
                    COL_TITLE: "Mapped",
                    COL_CHUNK_TEXT: "Chunk two also long enough for retrieval",
                    COL_CHUNK_INDEX: 2,
                    COL_CHUNKS_IN_DOC: 2,
                    # RU: положительный cos к qv=[1,0,0] — ортогональный
                    # [0,1,0] (cos=0.0) намеренно проверяется отдельным
                    # тестом ниже (zero-cosine exclusion), не здесь.
                    COL_CHUNK_VECTOR: [0.8, 0.2, 0.0],
                },
            ]

    monkeypatch.setattr(
        "knowledge_engine.services.lecture_rag_context.VectorStore",
        lambda: FakeStore(),
    )
    candidates: list[LectureContextCandidate] = []
    seen: set[str] = set()
    n = _append_mandatory_mapped_rag_chunks(
        "subagent query",
        candidates,
        seen,
        mapped_doc_ids=[doc_id],
        node_id="n1",
        max_per_doc=2,
    )
    assert n == 2
    assert all(c.label == _PINNED_RAG_LABEL for c in candidates)
    plains = {c.plain for c in candidates}
    assert any(p.startswith("Chunk one") for p in plains)
    assert any(p.startswith("Chunk two") for p in plains)


def test_append_mandatory_mapped_rag_chunks_excludes_zero_cosine(monkeypatch):
    """Чанк, ортогональный запросу (cos=0.0), не должен попадать в
    mandatory_rag даже если max_per_doc есть место — иначе в RAG-инспектор
    уходят бессмысленные чанки (см. баг: 3 identичных R1/R2/R3, cos=0.000)."""
    from knowledge_engine.db.rag_chunks_schema import (
        COL_CHUNK_ID,
        COL_CHUNK_INDEX,
        COL_CHUNK_TEXT,
        COL_CHUNK_VECTOR,
        COL_CHUNKS_IN_DOC,
        COL_DOC_ID,
        COL_TITLE,
        COL_URL,
    )

    doc_id = VectorStore.doc_id_for_url("https://example.com/zero-cos")

    class FakeStore:
        _embeddings = type(
            "E",
            (),
            {"embed_query": lambda self, q: [1.0, 0.0, 0.0]},
        )()

        def fetch_rag_chunks_by_doc_id(self, did: str):
            if did != doc_id:
                return []
            return [
                {
                    COL_CHUNK_ID: f"{doc_id}_relevant",
                    COL_DOC_ID: doc_id,
                    COL_URL: "https://example.com/zero-cos",
                    COL_TITLE: "Doc",
                    COL_CHUNK_TEXT: "Relevant chunk text long enough for retrieval",
                    COL_CHUNK_INDEX: 1,
                    COL_CHUNKS_IN_DOC: 2,
                    COL_CHUNK_VECTOR: [1.0, 0.0, 0.0],
                },
                {
                    COL_CHUNK_ID: f"{doc_id}_orthogonal",
                    COL_DOC_ID: doc_id,
                    COL_URL: "https://example.com/zero-cos",
                    COL_TITLE: "Doc",
                    COL_CHUNK_TEXT: "Orthogonal chunk text long enough for retrieval",
                    COL_CHUNK_INDEX: 2,
                    COL_CHUNKS_IN_DOC: 2,
                    COL_CHUNK_VECTOR: [0.0, 1.0, 0.0],
                },
                {
                    COL_CHUNK_ID: f"{doc_id}_no_vector",
                    COL_DOC_ID: doc_id,
                    COL_URL: "https://example.com/zero-cos",
                    COL_TITLE: "Doc",
                    COL_CHUNK_TEXT: "No vector chunk text long enough for retrieval",
                    COL_CHUNK_INDEX: 3,
                    COL_CHUNKS_IN_DOC: 2,
                    COL_CHUNK_VECTOR: None,
                },
            ]

    monkeypatch.setattr(
        "knowledge_engine.services.lecture_rag_context.VectorStore",
        lambda: FakeStore(),
    )
    candidates: list[LectureContextCandidate] = []
    seen: set[str] = set()
    n = _append_mandatory_mapped_rag_chunks(
        "subagent query",
        candidates,
        seen,
        mapped_doc_ids=[doc_id],
        node_id="n1",
        max_per_doc=3,
    )
    assert n == 1
    assert candidates[0].plain.startswith("Relevant chunk")
