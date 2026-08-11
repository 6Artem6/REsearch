"""Doc-gate filters low doc_meta relevance before CA selection."""

from __future__ import annotations

import numpy as np

from knowledge_engine.services.chunk_cross_attention_mmr import (
    ChunkCandidate,
    select_diverse_chunks_with_cross_attention,
)


def _unit(v: tuple[float, ...]) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(arr))
    return arr / n if n > 0 else arr


def test_doc_gate_excludes_orthogonal_doc_meta():
    topic = _unit((1.0, 0.0))
    aligned = ChunkCandidate(
        text="Aligned chunk with enough characters for pool.",
        source_id="a",
        source_title="A",
        chunk_vector=_unit((1.0, 0.0)),
        doc_meta_vector=_unit((1.0, 0.0)),
        meta={"doc_id": "doc-a"},
    )
    gated_out = ChunkCandidate(
        text="Orthogonal chunk with enough characters here.",
        source_id="b",
        source_title="B",
        chunk_vector=_unit((0.0, 1.0)),
        doc_meta_vector=_unit((0.0, 1.0)),
        meta={"doc_id": "doc-b"},
    )
    picked = select_diverse_chunks_with_cross_attention(
        topic,
        [aligned, gated_out],
        top_k=2,
        alpha=1.0,
        beta=0.0,
        gamma=0.0,
        max_chunks_per_source=2,
        doc_gate_threshold=0.4,
    )
    assert len(picked) == 1
    assert picked[0].source_id == "doc-a"
