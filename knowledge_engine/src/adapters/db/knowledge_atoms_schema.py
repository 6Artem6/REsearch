"""LanceDB collection for Knowledge Triangulation atoms (fact-level RAG)."""

from __future__ import annotations

KNOWLEDGE_ATOMS_TABLE = "knowledge_atoms"

COL_ID = "id"
COL_DOC_ID = "doc_id"
COL_URL = "url"
COL_STATEMENT = "statement"
COL_SCOPE = "scope"
COL_SOURCE_CHUNK_IDS = "source_chunk_ids"
COL_CONTEXT_QUOTE = "context_quote"
COL_VECTOR = "vector"
COL_EMBED_MODEL = "embed_model"
# Two-Stage Fact Relevance & Anchor Filtering (add-only) — Optional, missing
# on rows written before this field existed; treat as unscored, not 0.0.
COL_CORE_RELEVANCE_SCORE = "core_relevance_score"
