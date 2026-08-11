"""Unit tests for chunk cross-attention + MMR selection."""

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


def test_select_spreads_sources_and_penalizes_duplicates():
    topic = _unit((1.0, 0.0, 0.0))
    a_chunk = _unit((1.0, 0.0, 0.0))
    b_chunk = _unit((0.9, 0.1, 0.0))
    c_chunk = _unit((0.88, 0.48, 0.0))

    candidates = [
        ChunkCandidate(
            text="Alpha chunk about topic alignment.",
            source_id="src-a",
            source_title="Article A",
            chunk_vector=a_chunk,
            doc_meta_vector=a_chunk,
            meta={"origin_i": "0", "doc_id": "doc-a"},
        ),
        ChunkCandidate(
            text="Almost same as alpha — redundant body text.",
            source_id="src-a",
            source_title="Article A",
            chunk_vector=b_chunk,
            doc_meta_vector=b_chunk,
            meta={"origin_i": "1", "doc_id": "doc-a"},
        ),
        ChunkCandidate(
            text="Orthogonal concept from another source entirely.",
            source_id="src-b",
            source_title="Article B",
            chunk_vector=c_chunk,
            doc_meta_vector=c_chunk,
            meta={"origin_i": "2", "doc_id": "doc-b"},
        ),
    ]

    picked = select_diverse_chunks_with_cross_attention(
        topic,
        candidates,
        top_k=2,
        alpha=1.0,
        beta=0.0,
        gamma=0.8,
        max_chunks_per_source=1,
        anchor_threshold=1.01,
        semantic_dedup_threshold=0.99,
    )
    assert len(picked) == 2
    assert picked[0].formatted.startswith("[R1]")
    assert picked[1].formatted.startswith("[R2]")
    source_ids = {p.doc_id or p.source_id for p in picked}
    assert source_ids == {"doc-a", "doc-b"}


def test_max_chunks_per_source_respected():
    topic = _unit((1.0, 0.0))
    candidates = []
    for i in range(4):
        v = _unit((1.0 - i * 0.12, i * 0.12 + 0.01))
        candidates.append(
            ChunkCandidate(
                text=f"Chunk {i} from same article with enough chars.",
                source_id="one",
                source_title="Only",
                chunk_vector=v,
                doc_meta_vector=v,
                meta={"origin_i": str(i), "doc_id": "doc-one"},
            )
        )
    picked = select_diverse_chunks_with_cross_attention(
        topic,
        candidates,
        top_k=4,
        gamma=0.0,
        max_chunks_per_source=2,
        anchor_threshold=1.01,
        knee_drop_ratio=0.12,
        semantic_dedup_threshold=1.0,
    )
    assert len(picked) == 2
    assert all((p.doc_id or p.source_id) == "doc-one" for p in picked)
