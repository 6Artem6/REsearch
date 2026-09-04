"""Backfill document passports, knowledge atoms, and window_summary in LanceDB.

Identifies ``rag_chunks`` documents that lack ``knowledge_atoms`` and/or a usable
passport in ``document_summaries``, then runs MAP → REDUCE (two_phase) over the
stored chunk texts and persists results.

Examples:
  # Curriculum-scoped dry-run
  ./.venv/bin/python knowledge_engine/scripts/backfill_document_passports.py \\
    --curriculum agentic_systems_architecture --dry-run

  # Smoke first 2 docs of that curriculum through Gemma
  ./.venv/bin/python knowledge_engine/scripts/backfill_document_passports.py \\
    --curriculum agentic_systems_architecture --limit 2

  # Single doc
  ./.venv/bin/python knowledge_engine/scripts/backfill_document_passports.py \\
    --doc-id abcdef0123456789abcdef01
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("KE_TRACE_STDOUT", "1")
# Prefer two-phase REDUCE for atom provenance merges.
os.environ.setdefault("REDUCE_STRATEGY", "two_phase")

from knowledge_engine.config import LANCE_DB_PATH, gemma_cloud_api_key_available
from knowledge_engine.db.rag_chunks_schema import (
    COL_CHUNK_ID,
    COL_CHUNK_TEXT,
    COL_TITLE,
    COL_URL,
    map_window_chunk_id,
)
from knowledge_engine.schemas.extraction import KnowledgeAtom, merge_source_chunk_ids
from knowledge_engine.services.article_ingestion.blog_spatial_pipeline import (
    _document_summary_from_final,
)
from knowledge_engine.services.article_ingestion.blog_spatial_summarizer import (
    MapReduceArticleJob,
    map_reduce_jobs_pooled_async,
)
from knowledge_engine.services.article_ingestion.paragraph_token_splitter import (
    TokenWindowChunk,
)
from knowledge_engine.services.lecture_rag_source_scope import (
    collect_curriculum_library_urls,
    normalize_lecture_source_url,
)
from knowledge_engine.services.skill_tree_store import get_curriculum_graph
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.ui.run_log import trace

logger = logging.getLogger("backfill_document_passports")


@dataclass
class BlindDocument:
    doc_id: str
    url: str
    title: str
    chunk_count: int
    missing_atoms: bool
    missing_passport: bool
    missing_window_summaries: int = 0
    reasons: list[str] = field(default_factory=list)


def curriculum_source_urls(curriculum_id: str) -> list[str]:
    """All http URLs from curriculum registry + route_sources."""
    cid = (curriculum_id or "").strip()
    if not cid:
        return []
    if not get_curriculum_graph(cid):
        raise SystemExit(f"curriculum not found: {cid!r}")
    return collect_curriculum_library_urls(cid)


def curriculum_doc_id_scope(curriculum_id: str) -> tuple[set[str], set[str]]:
    """Return (doc_ids, normalized_urls) for curriculum sources."""
    urls = curriculum_source_urls(curriculum_id)
    norm_urls = {normalize_lecture_source_url(u) for u in urls}
    doc_ids = {VectorStore.doc_id_for_url(u) for u in urls if u.startswith("http")}
    return doc_ids, norm_urls


def _prefer_map_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer ``*_map_*`` windows when present; else all ordered chunks."""
    map_rows = [r for r in rows if "_map_" in str(r.get(COL_CHUNK_ID) or "")]
    return map_rows if map_rows else list(rows)


def remap_atom_source_chunk_ids(
    atoms: list[KnowledgeAtom],
    *,
    doc_id: str,
    window_index_to_chunk_id: dict[int, str],
) -> list[KnowledgeAtom]:
    """
    MAP attaches synthetic ``{doc_id}_map_{i}`` ids; remap onto real LanceDB chunk_ids.
    """
    synth_to_real = {
        map_window_chunk_id(doc_id, idx): cid
        for idx, cid in window_index_to_chunk_id.items()
        if (cid or "").strip()
    }
    # Also allow identity when rows already use map_* ids.
    for cid in window_index_to_chunk_id.values():
        if cid:
            synth_to_real.setdefault(cid, cid)

    out: list[KnowledgeAtom] = []
    for atom in atoms:
        remapped: list[str] = []
        for raw in atom.source_chunk_ids or []:
            key = str(raw or "").strip()
            remapped.append(synth_to_real.get(key, key))
        ids = merge_source_chunk_ids(remapped)
        if ids != list(atom.source_chunk_ids or []):
            out.append(atom.model_copy(update={"source_chunk_ids": ids}))
        else:
            out.append(atom)
    return out


async def discover_blind_documents(
    store: VectorStore,
    *,
    doc_id_filter: str | None = None,
    allowed_doc_ids: set[str] | None = None,
    allowed_urls: set[str] | None = None,
    include_missing_window_summary_only: bool = False,
) -> list[BlindDocument]:
    """
    Documents in ``rag_chunks`` missing knowledge_atoms and/or passport
    (``document_summaries`` takeaways).

    When ``allowed_doc_ids`` / ``allowed_urls`` are set (curriculum scope),
    only matching LanceDB docs are considered.
    """
    atom_docs = await store.knowledge_atom_doc_ids()
    metas = store.list_rag_documents()
    want = (doc_id_filter or "").strip()
    out: list[BlindDocument] = []
    for meta in metas:
        did = str(meta.get("doc_id") or "").strip()
        if not did:
            continue
        if want and did != want:
            continue
        url = str(meta.get("url") or "").strip()
        url_key = normalize_lecture_source_url(url) if url else ""
        if allowed_doc_ids is not None or allowed_urls is not None:
            in_docs = bool(allowed_doc_ids) and did in allowed_doc_ids
            in_urls = bool(allowed_urls) and url_key in allowed_urls
            if not (in_docs or in_urls):
                continue
        title = str(meta.get("title") or "").strip() or url or did
        missing_atoms = did not in atom_docs
        passport = (
            store.fetch_latest_summary_for_url(url) if url.startswith("http") else None
        )
        missing_passport = not store.passport_is_filled(passport)
        missing_ws = int(meta.get("missing_window_summary_count") or 0)

        reasons: list[str] = []
        if missing_atoms:
            reasons.append("no_knowledge_atoms")
        if missing_passport:
            reasons.append("missing_passport")

        needs = missing_atoms or missing_passport
        if not needs and include_missing_window_summary_only and missing_ws > 0:
            reasons.append(f"missing_window_summary={missing_ws}")
            needs = True
        elif needs and missing_ws > 0:
            reasons.append(f"missing_window_summary={missing_ws}")

        if not needs:
            continue
        out.append(
            BlindDocument(
                doc_id=did,
                url=url,
                title=title,
                chunk_count=int(meta.get("chunk_count") or 0),
                missing_atoms=missing_atoms,
                missing_passport=missing_passport,
                missing_window_summaries=missing_ws,
                reasons=reasons,
            )
        )
    return out


async def backfill_one_document(
    store: VectorStore,
    blind: BlindDocument,
    *,
    force_gemma_cloud: bool,
) -> dict[str, Any]:
    """MAP → REDUCE → upsert atoms / window_summary / passport for one doc."""
    rows = store.fetch_rag_chunks_by_doc_id(blind.doc_id)
    rows = _prefer_map_rows(rows)
    usable: list[dict[str, Any]] = []
    for row in rows:
        text = str(row.get(COL_CHUNK_TEXT) or "").strip()
        if text:
            usable.append(row)
    if not usable:
        raise RuntimeError("no non-empty chunk_text rows")

    url = blind.url or str(usable[0].get(COL_URL) or "").strip()
    if not url.startswith("http"):
        raise RuntimeError(f"doc has no http url (got {url!r})")
    title = blind.title or str(usable[0].get(COL_TITLE) or "").strip() or url

    windows: list[TokenWindowChunk] = []
    index_to_chunk_id: dict[int, str] = {}
    for i, row in enumerate(usable):
        cid = str(row.get(COL_CHUNK_ID) or "").strip()
        if not cid:
            cid = map_window_chunk_id(blind.doc_id, i)
        index_to_chunk_id[i] = cid
        windows.append(
            TokenWindowChunk(
                window_index=i,
                body=str(row.get(COL_CHUNK_TEXT) or "").strip(),
            )
        )

    job = MapReduceArticleJob(
        job_id=url,
        title=title[:300],
        url=url,
        windows=windows,
        all_figure_ids=[],
        figure_registry=None,
    )
    pooled = await map_reduce_jobs_pooled_async(
        [job],
        force_gemma_cloud=force_gemma_cloud,
    )
    outcome = pooled.get(job.job_id)
    final = outcome.final if outcome else None
    if final is None:
        raise RuntimeError("map-reduce returned no final passport")

    atoms = remap_atom_source_chunk_ids(
        list(final.knowledge_atoms or []),
        doc_id=blind.doc_id,
        window_index_to_chunk_id=index_to_chunk_id,
    )
    final.knowledge_atoms = atoms

    summary = _document_summary_from_final(
        final,
        title=title,
        url=url,
        registry=None,
    )
    # Skip sliding-window re-ingest — preserve existing rag_chunks bodies.
    await store.save_summary(summary, skip_rag_ingest=True)

    n_atoms = await store.upsert_knowledge_atoms(
        url,
        atoms,
        doc_id=blind.doc_id,
    )

    summaries_by_chunk: dict[str, str] = {}
    map_results = list(outcome.map_results) if outcome else []
    for i, cid in index_to_chunk_id.items():
        m = map_results[i] if i < len(map_results) else None
        if m is None:
            continue
        ws = (m.window_summary or "").strip()
        if ws:
            summaries_by_chunk[cid] = ws
    n_ws = await store.update_rag_window_summaries(blind.doc_id, summaries_by_chunk)

    return {
        "doc_id": blind.doc_id,
        "url": url,
        "atoms": n_atoms,
        "window_summaries": n_ws,
        "takeaways": len(summary.key_takeaways or []),
        "windows": len(windows),
    }


def _progress_iter(items: list[BlindDocument], *, disable: bool):
    try:
        from tqdm import tqdm
    except ImportError:
        return items
    return tqdm(items, desc="backfill passports", unit="doc", disable=disable)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Backfill LanceDB document passports, knowledge_atoms, "
            "and rag_chunks.window_summary via MAP→REDUCE (two_phase)."
        )
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List blind documents only (no Gemma / no writes)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N blind documents (0 = all)",
    )
    p.add_argument(
        "--curriculum",
        type=str,
        default="",
        help=(
            "Restrict to sources of this curriculum_id "
            "(e.g. agentic_systems_architecture)"
        ),
    )
    p.add_argument(
        "--doc-id",
        type=str,
        default="",
        help="Restrict to a single doc_id",
    )
    p.add_argument(
        "--include-window-summary-only",
        action="store_true",
        help=(
            "Also include docs that only miss window_summary "
            "(even if atoms+passport exist)"
        ),
    )
    p.add_argument(
        "--force-gemma-cloud",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force Gemma cloud for MAP/REDUCE (default: true; --no-force-gemma-cloud for Ollama)",
    )
    return p.parse_args(argv)


async def _amain(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    force_gemma = bool(args.force_gemma_cloud)
    if force_gemma and not gemma_cloud_api_key_available() and not args.dry_run:
        print(
            "ERROR: GEMINI_API_KEY required for --force-gemma-cloud "
            "(or pass --no-force-gemma-cloud).",
            file=sys.stderr,
        )
        return 2

    curriculum_id = (args.curriculum or "").strip()
    allowed_doc_ids: set[str] | None = None
    allowed_urls: set[str] | None = None
    registry_n = 0
    if curriculum_id:
        allowed_doc_ids, allowed_urls = curriculum_doc_id_scope(curriculum_id)
        registry_n = len(allowed_urls or [])
        trace(
            f"BACKFILL scope curriculum={curriculum_id} "
            f"registry_urls={registry_n} doc_ids={len(allowed_doc_ids or [])}"
        )

    store = VectorStore()
    trace(f"BACKFILL passports ▶ | lance={LANCE_DB_PATH}")
    blinds = await discover_blind_documents(
        store,
        doc_id_filter=(args.doc_id or "").strip() or None,
        allowed_doc_ids=allowed_doc_ids,
        allowed_urls=allowed_urls,
        include_missing_window_summary_only=bool(args.include_window_summary_only),
    )
    if args.limit and args.limit > 0:
        blinds = blinds[: int(args.limit)]

    scope = f" curriculum={curriculum_id}" if curriculum_id else ""
    print(
        f"blind_documents={len(blinds)}{scope} "
        f"registry_urls={registry_n or '-'} lance={LANCE_DB_PATH}"
    )
    for b in blinds:
        print(
            f"  - {b.doc_id} chunks={b.chunk_count} "
            f"reasons={','.join(b.reasons)} | {(b.url or '')[:70]}"
        )

    if args.dry_run:
        print("dry-run: no MAP/REDUCE executed")
        return 0
    if not blinds:
        print("nothing to backfill")
        return 0

    ok = 0
    failed = 0
    for blind in _progress_iter(blinds, disable=False):
        try:
            result = await backfill_one_document(
                store,
                blind,
                force_gemma_cloud=force_gemma,
            )
            ok += 1
            logger.info(
                "backfill ✓ doc_id=%s atoms=%s window_summaries=%s takeaways=%s",
                result["doc_id"],
                result["atoms"],
                result["window_summaries"],
                result["takeaways"],
            )
            trace(
                f"BACKFILL ✓ | {blind.doc_id[:12]}… "
                f"atoms={result['atoms']} ws={result['window_summaries']}"
            )
        except Exception as exc:
            failed += 1
            logger.exception("backfill ✗ doc_id=%s | %s", blind.doc_id, exc)
            trace(f"BACKFILL ✗ | doc_id={blind.doc_id} | {exc}")

    print(f"done ok={ok} failed={failed} total={len(blinds)}")
    return 0 if failed == 0 else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
