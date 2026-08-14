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
