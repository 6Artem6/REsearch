"""LIVE, no-mock verification of the reason the core_relevance_score
backfill (knowledge_engine/scripts/db/backfill_core_relevance_score.py) was
run: prove that the hybrid/pedagogical rerank machinery in
dialog_atoms_rag.py now actually consumes REAL production
core_relevance_score values, instead of every atom silently hitting the
None -> 0.5 neutral fallback (dialog_atoms_rag.py's own convention — see
_hybrid_rerank_pairs_async docstring) as it did for 100% of the 703 rows
before the backfill.

Hits the real Postgres ke-postgres container (no mocks) and the real local
Cross-Encoder (BGE-reranker-v2-m3, no cloud calls). No skip guard: the same
DB dependency test_postgres_vector_repository.py already exercises
unconditionally in this project's dev environment.
"""

from __future__ import annotations

import pytest

from knowledge_engine.src.domains.grounding.dialog_atoms_rag import (
    _hybrid_rerank_pairs_async,
    _rows_to_atoms,
)
from knowledge_engine.src.shared.vector_store import VectorStore

_SAMPLE_DOC_ID = "1434195885cb72ada79ceb83"  # GROUPING SETS / Хабр, 8 real atoms


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_all_real_atoms_are_scored_after_backfill() -> None:
    """The backfill's own success criterion, re-checked from the consumer
    side (dialog_atoms_rag._rows_to_atoms), not just the raw DB count."""
    store = VectorStore()
    doc_ids = await store.knowledge_atom_doc_ids()
    assert doc_ids, "no knowledge_atoms in Postgres — is ke-postgres up?"

    total = 0
    scored = 0
    for doc_id in doc_ids:
        rows = await store.fetch_knowledge_atoms_by_doc_id(doc_id)
        atoms = _rows_to_atoms(rows)
        total += len(atoms)
        scored += sum(1 for a in atoms if a.core_relevance_score is not None)

    assert total > 0
    assert scored == total, (
        f"{total - scored}/{total} real atoms still have core_relevance_score=None "
        "— backfill did not fully complete"
    )


@pytest.mark.anyio
async def test_real_core_relevance_score_has_genuine_variance() -> None:
    """Guards against a degenerate backfill that stamped the same constant
    on everything (which would make the hybrid formula pointless)."""
    store = VectorStore()
    rows = await store.fetch_knowledge_atoms_by_doc_id(_SAMPLE_DOC_ID)
    atoms = _rows_to_atoms(rows)
    assert len(atoms) >= 3
    scores = {round(a.core_relevance_score, 3) for a in atoms}
    assert len(scores) > 1, f"expected real variance, got a single value: {scores}"


@pytest.mark.anyio
async def test_hybrid_rerank_on_real_atoms_is_driven_by_real_core_score() -> None:
    """The actual point of the whole exercise: with query relevance held
    equal, ranking must be decided by the REAL core_relevance_score, not a
    universal 0.5 fallback — i.e. the production reranker
    (_hybrid_rerank_pairs_async) genuinely differentiates real facts now."""
    store = VectorStore()
    rows = await store.fetch_knowledge_atoms_by_doc_id(_SAMPLE_DOC_ID)
    pairs = []
    for row in rows:
        atoms = _rows_to_atoms([row])
        if atoms:
            pairs.append((row, atoms[0]))
    assert len(pairs) >= 3
    assert all(atom.core_relevance_score is not None for _, atom in pairs)

    # Real Cross-Encoder call, but the query is deliberately generic/topic-
    # agnostic so every atom's query_score comes back close to equal —
    # isolating core_relevance_score as the actual ranking driver, same
    # methodology as the earlier pedagogical A/B benchmarks.
    ranked = await _hybrid_rerank_pairs_async(
        "секционирование и GROUPING SETS в PostgreSQL", pairs, alpha=0.3
    )
    ranked_atoms = [atom for _, atom in ranked]

    print("\n[LIVE] Hybrid rerank on real backfilled data, alpha=0.3:")
    for _, atom in ranked[:5]:
        print(f"  core={atom.core_relevance_score:.3f} | {atom.statement[:80]}")

    # With alpha weighted toward core_relevance_score (0.3 query / 0.7 core),
    # the top-ranked real atom must be at or near the highest core score —
    # not an arbitrary DB-fetch-order artifact (which is what happened for
    # every atom pre-backfill, when core was always the 0.5 constant).
    assert (
        ranked_atoms[0].core_relevance_score
        >= sorted((a.core_relevance_score for _, a in pairs), reverse=True)[1]
    )
