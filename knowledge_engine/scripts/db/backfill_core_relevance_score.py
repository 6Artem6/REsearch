"""Backfill ``core_relevance_score`` on existing ``knowledge_atoms`` rows.

Two-Stage Fact Relevance & Anchor Filtering (see
``anchor_relevance.py``) only stamps ``core_relevance_score`` on atoms that
pass through ``_apply_static_anchor_filter`` during a fresh MAP -> REDUCE
ingest, gated by ``BLOG_SPATIAL_ANCHOR_FILTER_ENABLED`` (default off). Every
atom already in the database predates that code path entirely — flipping
the flag on does NOT retroactively score existing rows, only new ingests
going forward. This script closes that gap for existing data: it re-derives
a Synthetic Anchor (Title + Lead) per document from its ``rag_chunks``, and
re-scores that document's existing atoms in place (statement/scope/
provenance unchanged — only ``core_relevance_score`` gets a value).

Async batch: one Cross-Encoder call per document (all its atoms scored in a
single batched ``score_relevance_pairs`` call, not one call per atom — see
``AtomAnchorFilterService.filter_atoms_by_anchor``), documents processed
concurrently up to ``--concurrency`` at once.

Idempotent: skips documents whose atoms are already fully scored, unless
``--force``. Uses ``--threshold 0.0`` by default — a backfill should STAMP
scores, not silently cut existing atoms out of retrieval; pass a higher
``--threshold`` only if you explicitly want this run to also drop atoms.

Examples:
  # Dry-run: list documents that would be scored, no Cross-Encoder / writes
  ./.venv/bin/python knowledge_engine/scripts/db/backfill_core_relevance_score.py --dry-run

  # Score the first 5 documents (smoke test)
  ./.venv/bin/python knowledge_engine/scripts/db/backfill_core_relevance_score.py --limit 5

  # Full backfill, 8-way concurrency
  ./.venv/bin/python knowledge_engine/scripts/db/backfill_core_relevance_score.py --concurrency 8

  # Single document
  ./.venv/bin/python knowledge_engine/scripts/db/backfill_core_relevance_score.py --doc-id 793387d17120df4d3c2aec97
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("KE_TRACE_STDOUT", "1")

from knowledge_engine.src.adapters.db.knowledge_atoms_schema import (
    COL_CONTEXT_QUOTE,
    COL_CORE_RELEVANCE_SCORE,
    COL_SCOPE,
    COL_SOURCE_CHUNK_IDS,
    COL_STATEMENT,
    COL_URL,
)
from knowledge_engine.src.adapters.db.rag_chunks_schema import COL_CHUNK_TEXT, COL_TITLE
from knowledge_engine.src.core.run_log import trace
from knowledge_engine.src.domains.ingestion.article_ingestion.anchor_relevance import (
    AtomAnchorFilterService,
    get_synthetic_anchor,
)
from knowledge_engine.src.shared.extraction import KnowledgeAtom, coerce_scope_type
from knowledge_engine.src.shared.vector_store import VectorStore

logger = logging.getLogger("backfill_core_relevance_score")

_LEAD_CHARS = 500

# CRITICAL: rag_gateway/cross_encoder.py's singleton CrossEncoder is loaded
# under a threading.Lock, but concurrent model.predict() CALLS on the same
# loaded instance are NOT serialized there — two docs scoring at once (this
# script's --concurrency > 1) hung indefinitely in a live run (sentence-
# transformers' internal DataLoader appears not to tolerate concurrent
# invocation from two threads of the same process; observed: a stuck
# "Batches: 0%" progress bar + a leaked loky semaphore, 49 min of spinning
# CPU with zero progress). Serialize the CE step here — DB I/O around it
# stays concurrent under --concurrency, only score_relevance_pairs calls are
# queued. This is a real latent risk in the shared CE singleton itself, not
# something this backfill script can or should fix on its own — flagged
# separately for anyone flipping DIALOG_ATOMS_HYBRID_RERANK_ENABLED /
# DIALOG_ATOMS_PEDAGOGICAL_BOOST_ENABLED in production.
_CE_LOCK = asyncio.Lock()


@dataclass
class DocResult:
    doc_id: str
    status: str  # "scored" | "skipped_already_scored" | "skipped_no_atoms" |
    # "skipped_no_chunks" | "skipped_no_anchor" | "dry_run" | "failed"
    n_atoms: int = 0
    reason: str = ""


def _rows_to_atoms(rows: list[dict[str, Any]]) -> list[KnowledgeAtom]:
    atoms: list[KnowledgeAtom] = []
    for row in rows:
        stmt = str(row.get(COL_STATEMENT) or "").strip()
        if len(stmt) < 8:
            continue
        raw_chunks = row.get(COL_SOURCE_CHUNK_IDS)
        chunk_ids = (
            [str(x).strip() for x in raw_chunks if str(x).strip()]
            if isinstance(raw_chunks, (list, tuple, set))
            else []
        )
        atoms.append(
            KnowledgeAtom(
                scope=coerce_scope_type(row.get(COL_SCOPE)),
                statement=stmt,
                context_quote=(row.get(COL_CONTEXT_QUOTE) or "").strip() or None,
                source_chunk_ids=chunk_ids,
            )
        )
    return atoms


def _derive_title_and_lead(chunk_rows: list[dict[str, Any]]) -> tuple[str, str]:
    if not chunk_rows:
        return "", ""
    title = str(chunk_rows[0].get(COL_TITLE) or "").strip()
    body = str(chunk_rows[0].get(COL_CHUNK_TEXT) or "").strip()
    prefix = f"Article: {title}"
    if body.startswith(prefix):
        body = body[len(prefix) :].lstrip("\n")
    return title, body[:_LEAD_CHARS]


async def process_one_doc(
    store: VectorStore,
    doc_id: str,
    *,
    threshold: float,
    force: bool,
    dry_run: bool,
    semaphore: asyncio.Semaphore,
) -> DocResult:
    async with semaphore:
        rows = await store.fetch_knowledge_atoms_by_doc_id(doc_id)
        if not rows:
            return DocResult(doc_id, "skipped_no_atoms")

        if not force and all(r.get(COL_CORE_RELEVANCE_SCORE) is not None for r in rows):
            return DocResult(doc_id, "skipped_already_scored", n_atoms=len(rows))

        if dry_run:
            return DocResult(doc_id, "dry_run", n_atoms=len(rows))

        try:
            chunk_rows = await store.fetch_rag_chunks_by_doc_id(doc_id)
            title, lead = _derive_title_and_lead(chunk_rows)
            anchor = get_synthetic_anchor(title, lead)
            if not anchor:
                return DocResult(
                    doc_id,
                    "skipped_no_anchor",
                    n_atoms=len(rows),
                    reason="empty title+lead",
                )

            atoms = _rows_to_atoms(rows)
            if not atoms:
                return DocResult(doc_id, "skipped_no_atoms")

            service = AtomAnchorFilterService()
            async with _CE_LOCK:
                scored = await service.filter_atoms_by_anchor(atoms, anchor, threshold)

            url = str(rows[0].get(COL_URL) or "").strip()
            n = await store.upsert_knowledge_atoms(url, scored, doc_id=doc_id)
            trace(
                f"BACKFILL core_relevance_score ✓ | doc_id={doc_id[:12]}… "
                f"atoms={n} anchor={anchor[:60]!r}"
            )
            return DocResult(doc_id, "scored", n_atoms=n)
        except (
            Exception
        ) as exc:  # noqa: BLE001 — one failed doc must not abort the batch
            logger.exception("backfill ✗ doc_id=%s | %s", doc_id, exc)
            return DocResult(doc_id, "failed", reason=str(exc))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Backfill core_relevance_score on existing knowledge_atoms rows "
            "(Synthetic Anchor from rag_chunks, async batch across documents)."
        )
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List documents that would be scored only (no Cross-Encoder / writes)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N documents (0 = all)",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max documents scored concurrently (default: 4)",
    )
    p.add_argument("--doc-id", type=str, default="", help="Restrict to a single doc_id")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help=(
            "Score cutoff passed to filter_atoms_by_anchor (default: 0.0 — "
            "stamp every atom, drop none). Raise only if you also want this "
            "run to cut low-relevance atoms out of retrieval."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-score documents even if all their atoms already have a score",
    )
    return p.parse_args(argv)


async def _amain(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    store = VectorStore()
    doc_id_filter = (args.doc_id or "").strip()
    if doc_id_filter:
        doc_ids = [doc_id_filter]
    else:
        doc_ids = sorted(await store.knowledge_atom_doc_ids())
    if args.limit and args.limit > 0:
        doc_ids = doc_ids[: int(args.limit)]

    print(
        f"documents={len(doc_ids)} dry_run={args.dry_run} concurrency={args.concurrency}"
    )
    if not doc_ids:
        print("nothing to backfill")
        return 0

    semaphore = asyncio.Semaphore(max(1, int(args.concurrency)))
    results = await asyncio.gather(
        *[
            process_one_doc(
                store,
                did,
                threshold=float(args.threshold),
                force=bool(args.force),
                dry_run=bool(args.dry_run),
                semaphore=semaphore,
            )
            for did in doc_ids
        ]
    )

    by_status: dict[str, int] = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1
        if r.status in ("scored", "dry_run"):
            print(f"  - {r.doc_id} atoms={r.n_atoms} status={r.status}")
        elif r.status == "failed":
            print(f"  - {r.doc_id} status=failed reason={r.reason}")

    print("summary:", ", ".join(f"{k}={v}" for k, v in sorted(by_status.items())))
    return 0 if by_status.get("failed", 0) == 0 else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
