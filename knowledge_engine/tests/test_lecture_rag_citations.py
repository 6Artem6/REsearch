"""Citation registry for lecture RAG chunks."""

from __future__ import annotations

from knowledge_engine.services.lecture_context_rerank import LectureContextCandidate
from knowledge_engine.services.lecture_rag_context import (
    _finalize_lecture_citation_candidates,
    build_rag_chunk_citation_registry,
)


def test_finalize_renumbers_and_formats():
    raw = [
        LectureContextCandidate(
            label="hybrid_semantic",
            formatted="### old header\nbody",
            plain="body text long enough for lecture chunk",
            url_key="https://example.com/a",
            source_title="Article A",
        ),
        LectureContextCandidate(
            label="route_doc",
            formatted="### other",
            plain="second chunk with different content here",
            url_key="https://example.com/b",
            source_title="Article B",
        ),
    ]
    out = _finalize_lecture_citation_candidates(raw)
    assert len(out) == 2
    assert out[0].formatted.startswith("[R1]")
    assert out[1].formatted.startswith("[R2]")
    reg = build_rag_chunk_citation_registry(out)
    assert "[R1]" in reg and "[R2]" in reg
    assert "example.com/a" in reg
    assert "example.com/b" in reg
