"""Index DocumentSummary into LanceDB rag_chunks; Exa highlights anti-bot fallback."""

from __future__ import annotations

from knowledge_engine.schemas import DocumentSummary
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.src.curriculum.schemas import CurriculumSearchHit
from knowledge_engine.ui.run_log import trace


def ingest_document_summary(
    summary: DocumentSummary,
    *,
    body_text: str | None = None,
    store: VectorStore | None = None,
) -> int:
    """
    Build doc_summary_text + doc_meta_vector, split body into sliding windows,
    persist one LanceDB row per chunk.
    """
    vs = store or VectorStore()
    n = vs.upsert_rag_chunks_from_summary(summary, body_text=body_text)
    if n:
        trace(
            f"RAG_CHUNKS ingest ✓ | chunks={n} | "
            f"{(summary.url or summary.title or '')[:72]}"
        )
    return n


def ingest_exa_highlights_fallback(
    hit: CurriculumSearchHit,
    *,
    body_text: str | None = None,
    store: VectorStore | None = None,
) -> int:
    parts = [str(x).strip() for x in (hit.key_extracts or []) if str(x).strip()]
    text = (body_text or "").strip() or "\n\n".join(parts)
    if not text and (hit.snippet or "").strip():
        text = (hit.snippet or "").strip()
    if len(text) < 40:
        return 0
    vs = store or VectorStore()
    n = vs.upsert_rag_exa_highlights_fallback(
        hit.url,
        hit.title or hit.url,
        text,
    )
    if n:
        trace(f"RAG_CHUNKS exa_highlights_fallback ✓ | chunks={n} | {hit.url[:72]}")
    return n
