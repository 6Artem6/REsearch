"""Smoke-test: real RAG retrieval + Pedagogical Reranker against the newly
migrated 5-category taxonomy (CONCEPT/MECHANIC/PRACTICE/EDGE_CASE/
ANTI_PATTERN), on the real 703 Postgres atoms.

Reuses the exact retrieval pattern already built and validated in
``compare_real_dialog_rag.py`` (real VectorStore ANN search -> real
Cross-Encoder query score -> real ``apply_pedagogical_weights``), rather
than reinventing it — same Cross-Encoder concurrency lock, same
``core_relevance_score`` hybrid-formula fallback semantics.

Two probe queries (depth_level="expert", which is the tier that most
strongly favors EDGE_CASE/ANTI_PATTERN in ``_PEDAGOGY_WEIGHTS`` — the
category the migration just added):
  1. Anti-pattern probe: recursive CTE / temp table pitfalls in PostgreSQL.
  2. Edge-case probe: memory/lock behavior under extreme load
     (obmalloc / BRIN indexes).

For each, prints the top-10 ranked atoms as
``atom_id | scope | score | statement`` and flags whether the expected
scope actually appears in the ranked list.

Usage: python knowledge_engine/scripts/test_rag_5_scopes.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from knowledge_engine.src.domains.grounding.pedagogical_reranker import (
    apply_pedagogical_weights,
)
from knowledge_engine.src.domains.grounding.schemas import RetrievalPedagogicalContext
from knowledge_engine.src.rag_gateway.cross_encoder import score_relevance_pairs
from knowledge_engine.src.shared.extraction import KnowledgeAtom, coerce_scope_type
from knowledge_engine.src.shared.vector_store import VectorStore

_FETCH_POOL = 40
_TOP_N = 10
_ALPHA = 0.7

# Same real-world Cross-Encoder concurrency guard already validated in
# backfill_core_relevance_score.py / compare_real_dialog_rag.py — the shared
# singleton's model.predict() call is not internally thread-safe.
_CE_LOCK = asyncio.Lock()


def _row_to_atom(row: dict) -> KnowledgeAtom:
    return KnowledgeAtom(
        scope=coerce_scope_type(row.get("scope")),
        statement=str(row.get("statement") or "").strip()[:2000],
        source_chunk_ids=[],
        core_relevance_score=row.get("core_relevance_score"),
        id=(str(row.get("id")).strip() or None) if row.get("id") else None,
    )


async def run_probe(
    store: VectorStore, label: str, query: str, expected_scope: str
) -> None:
    rows = await store.search_knowledge_atoms(query, limit=_FETCH_POOL)
    atoms = [_row_to_atom(row) for row in rows]
    statements = [a.statement for a in atoms]

    if not atoms:
        print(f"\n[{label}] query={query!r} -> 0 candidates retrieved.")
        return

    async with _CE_LOCK:
        query_scores = await asyncio.to_thread(score_relevance_pairs, query, statements)

    context = RetrievalPedagogicalContext(depth_level="expert")
    ranked = apply_pedagogical_weights(
        list(zip(atoms, query_scores)), context, alpha=_ALPHA
    )

    print(f"\n{'=' * 90}\n[{label}]\nQuery: {query!r}\n{'=' * 90}")
    print(f"{'atom_id':<40} | {'scope':<13} | {'score':>7} | statement")
    print("-" * 90)
    found = False
    for atom, final_score in ranked[:_TOP_N]:
        scope_val = atom.scope.value
        if scope_val == expected_scope:
            found = True
        atom_id = (atom.id or "?")[:38]
        print(
            f"{atom_id:<40} | {scope_val:<13} | {final_score:>7.3f} | {atom.statement[:90]}"
        )

    verdict = "PASS" if found else "FAIL"
    print(
        f"\n[{verdict}] expected scope={expected_scope!r} "
        f"{'found' if found else 'NOT FOUND'} in top-{_TOP_N}."
    )


async def _amain() -> None:
    store = VectorStore()
    await run_probe(
        store,
        label="Probe 1: Anti-Pattern (recursive CTE / temp tables)",
        query=(
            "Какие подводные камни и ошибки бывают при использовании "
            "рекурсивных CTE или временных таблиц в PostgreSQL?"
        ),
        expected_scope="ANTI_PATTERN",
    )
    await run_probe(
        store,
        label="Probe 2: Edge Case (memory/locks under extreme load)",
        query=(
            "Как ведет себя память или блокировки при экстремальной нагрузке "
            "на obmalloc / brin indexes?"
        ),
        expected_scope="EDGE_CASE",
    )


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
