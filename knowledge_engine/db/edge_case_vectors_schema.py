"""LanceDB table for deep_analysis edge-case lexicon embeddings."""

from __future__ import annotations

EDGE_CASE_VECTORS_TABLE = "edge_case_vectors"

COL_ID = "id"
COL_LABEL = "label"  # edge_case | trade_off | bottleneck
COL_PHRASE = "phrase"
COL_VECTOR = "vector"
COL_EMBED_MODEL = "embed_model"
