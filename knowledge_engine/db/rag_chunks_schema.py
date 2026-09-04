"""LanceDB parent–child RAG chunk table (fine chunks + doc meta vectors)."""

from __future__ import annotations

RAG_CHUNKS_TABLE = "rag_chunks"

# Column names (LanceDB row keys)
COL_CHUNK_ID = "chunk_id"
COL_DOC_ID = "doc_id"
COL_URL = "url"
COL_TITLE = "title"
COL_CHUNK_TEXT = "chunk_text"
COL_CHUNK_VECTOR = "chunk_vector"
COL_DOC_SUMMARY_TEXT = "doc_summary_text"
COL_DOC_META_VECTOR = "doc_meta_vector"
COL_CHUNK_INDEX = "chunk_index"
COL_CHUNKS_IN_DOC = "chunks_in_doc"
COL_SOURCE_TYPE = "source_type"
COL_DETAIL_INSTRUCTION = "detail_instruction"
# OpenAlex / vendor trust weight in [0, 1]; missing → treat as 1.0 at query time
COL_TRUST_SCORE = "trust_score"
# Dense MAP window summary (optional; filled when writing academic/blog MAP rows)
COL_WINDOW_SUMMARY = "window_summary"
# Integrity: skip/rebuild when EMBED_MODEL changes (nomic vs bge-m3).
COL_EMBED_MODEL = "embed_model"


def map_window_chunk_id(doc_id: str, window_index: int) -> str:
    """LanceDB chunk_id for MAP window ``window_index`` (0-based → 1-based suffix)."""
    did = (doc_id or "").strip() or "doc"
    return f"{did}_map_{int(window_index) + 1}"


EXA_HIGHLIGHTS_FALLBACK_SOURCE_TYPE = "exa_highlights_fallback"
EXA_HIGHLIGHTS_FALLBACK_DETAIL_INSTRUCTION = (
    "Концентрированные факты из Exa highlights (прямой доступ заблокирован). "
    "Повышенное внимание к терминологии."
)
