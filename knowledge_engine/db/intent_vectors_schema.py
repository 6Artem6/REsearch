"""LanceDB table for cached VectorIntentRouter reference phrase embeddings."""

from __future__ import annotations

INTENT_VECTORS_TABLE = "intent_vectors"

COL_ID = "id"
COL_INTENT = "intent"
COL_PHRASE = "phrase"
COL_VECTOR = "vector"
# Integrity: rebuild when EMBED_MODEL changes (must be BAAI/bge-m3).
COL_EMBED_MODEL = "embed_model"
