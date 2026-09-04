"""CLI: targeted cleanup of resources actually linked to a node_id/curriculum_id.

Scope note (verified against the real schemas before writing this): Qdrant point
payloads (rag_chunks / document_summaries / knowledge_atoms) carry NO node_id/
curriculum_id field — never did, in LanceDB or Qdrant (checked full git history
of both schema files). There is no way to plumb those fields into the ingest
call chain today (source_material_pipeline.py never has a node_id/curriculum_id
in scope when it writes a chunk/atom/summary — only a search-hit source_id and
the URL). Adding unused payload fields and filtering on them would just move
the "always 0 matches" symptom one layer deeper while looking wired up.

Instead, Qdrant cleanup resolves node_id/curriculum_id → source URLs via the
EXISTING curriculum→source link (skill_tree_store.get_curriculum_graph, then
clear_node_sources._collect_node_source_urls — registry-resolved
mapped_source_ids PLUS node.resource_urls/node.source_ref, since a node's
mapped_source_ids can reference registry entries that no longer exist, in
which case its real URLs only live on resource_urls/source_ref; registry-only
resolution silently found 0 matches for such nodes — confirmed live against
agentic_systems_architecture/subagent_architectures), then deletes by `url`
(document_summaries) / `doc_id` (rag_chunks, knowledge_atoms — doc_id =
VectorStore.doc_id_for_url(url)). Requires --curriculum-id (a bare --node-id
can't be resolved to a source graph without knowing which curriculum's graph
to look in).

Gemini CacheMetadata (cloud_cache_manager.py — display_name is a hash digest /
session_id, not node/curriculum tagged) still has no such linkage anywhere, so
that section still honestly reports "not applicable" rather than faking a match.

The other real, safely-scoped resource is the Redis node-grounding lock key from
node_grounding_lock.py: `ke:lock:node_ground:{curriculum_id}:{node_id}`.

Usage:
    PYTHONPATH=. ./.venv/bin/python knowledge_engine/scripts/cleanup_cloud_resources.py \
        --node-id N3 --curriculum-id C1 [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from knowledge_engine.services.node_grounding_lock import _lock_key
from knowledge_engine.services.redis_client import get_redis, redis_enabled

_GLOB_SPECIAL = ("*", "?", "[", "]")


def _escape_glob(value: str) -> str:
    """Escape Redis SCAN MATCH glob metacharacters in a user-supplied id."""
    out = value
    for ch in _GLOB_SPECIAL:
        out = out.replace(ch, f"\\{ch}")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-id", default="")
    parser.add_argument("--curriculum-id", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.node_id = (args.node_id or "").strip()
    args.curriculum_id = (args.curriculum_id or "").strip()
    if not args.node_id and not args.curriculum_id:
        parser.error("Требуется хотя бы один из --node-id / --curriculum-id")
    return args


def find_grounding_lock_keys(node_id: str, curriculum_id: str) -> list[str]:
    if not redis_enabled():
        return []
    r = get_redis()
    nid, cid = node_id.strip(), curriculum_id.strip()
    if cid and nid:
        key = _lock_key(cid, nid)
        return [key] if r.exists(key) else []
    if cid:
        return list(r.scan_iter(match=f"ke:lock:node_ground:{_escape_glob(cid)}:*"))
    return list(r.scan_iter(match=f"ke:lock:node_ground:*:{_escape_glob(nid)}"))


def cleanup_redis(node_id: str, curriculum_id: str, *, dry_run: bool) -> dict:
    report: dict = {
        "backend": "redis",
        "applicable": True,
        "keys_found": [],
        "keys_deleted": 0,
    }
    if not redis_enabled():
        report["applicable"] = False
        print("· Redis: не сконфигурирован (REDIS_URL/KE_USE_REDIS) — пропуск")
        return report

    keys = find_grounding_lock_keys(node_id, curriculum_id)
    report["keys_found"] = keys
    if not keys:
        print("· Redis: подходящих ke:lock:node_ground:* ключей не найдено")
        return report

    for k in keys:
        verb = "[dry-run] would delete" if dry_run else "deleting"
        print(f"  {verb} {k}")
    if not dry_run:
        r = get_redis()
        for k in keys:
            r.delete(k)
        report["keys_deleted"] = len(keys)
    return report


def _get_qdrant_store():
    from knowledge_engine.config import QDRANT_URL

    if not (QDRANT_URL or "").strip():
        return None
    from knowledge_engine.services.qdrant_vector_store import QdrantVectorStore

    store = QdrantVectorStore()
    return store if store.enabled else None


def _urls_for_scope(node_id: str, curriculum_id: str) -> list[str]:
    """Resolve node_id/curriculum_id → source URLs.

    For a specific node, reuses clear_node_sources._collect_node_source_urls
    (registry-resolved mapped_source_ids + node.resource_urls + node.source_ref)
    rather than the registry alone — a node's mapped_source_ids can reference
    ids that were never (or no longer are) present in
    curriculum_sources_registry, in which case its actual URLs live only on
    resource_urls/source_ref. Registry-only resolution silently found 0 URLs
    for such nodes (real bug: Qdrant cleanup was a no-op for them even though
    LanceDB cleanup via clear_node_sources correctly found their URLs).
    Without a node_id, falls back to every URL in the curriculum's full source
    registry (no single node's resource_urls/source_ref to fall back to)."""
    from knowledge_engine.services.skill_tree_store import get_curriculum_graph

    graph = get_curriculum_graph(curriculum_id)
    if not graph:
        return []
    nid = (node_id or "").strip()
    if nid:
        from knowledge_engine.scripts.clear_node_sources import (
            _collect_node_source_urls,
        )

        _, urls = _collect_node_source_urls(graph, nid)
        return sorted(set(urls))

    from knowledge_engine.src.curriculum.source_registry import registry_index

    entries = list(registry_index(graph).values())
    return sorted(
        {str(e.get("url") or "").strip() for e in entries if (e.get("url") or "").strip()}
    )


async def _delete_qdrant_scope(store, urls: list[str]) -> int:
    """Single event loop for the whole batch — avoids a fresh AsyncQdrantClient
    reconnect per delete call (asyncio.run() per call would otherwise create
    N+1 separate connections; see QdrantVectorStore._get_client loop-identity
    tracking)."""
    from knowledge_engine.db.knowledge_atoms_schema import KNOWLEDGE_ATOMS_TABLE
    from knowledge_engine.db.rag_chunks_schema import RAG_CHUNKS_TABLE
    from knowledge_engine.services.vector_store import TABLE_NAME as DOCUMENT_SUMMARIES_TABLE
    from knowledge_engine.services.vector_store import VectorStore

    deleted = 0
    for u in urls:
        doc_id = VectorStore.doc_id_for_url(u)
        await store.delete_by_field(DOCUMENT_SUMMARIES_TABLE, "url", u)
        await store.delete_by_field(RAG_CHUNKS_TABLE, "doc_id", doc_id)
        await store.delete_by_field(KNOWLEDGE_ATOMS_TABLE, "doc_id", doc_id)
        deleted += 1
    return deleted


def delete_qdrant_urls(urls: list[str]) -> int:
    """Delete document_summaries/rag_chunks/knowledge_atoms Qdrant points for
    each already-confirmed-orphaned URL. No-op (returns 0) when QDRANT_URL is
    unset or `urls` is empty.

    Reused by callers outside this CLI's own node-scoped path — library GC
    (sync_curriculum_library_sources.apply_sync) and the standalone orphan
    sweep (sweep_orphaned_knowledge_atoms.py) previously deleted these same
    URLs from LanceDB only, leaving Qdrant Cloud counts untouched forever for
    any URL that became orphaned via their (broader, cross-curriculum) GC
    logic rather than this file's single-node URL resolution."""
    store = _get_qdrant_store()
    if store is None or not urls:
        return 0
    return asyncio.run(_delete_qdrant_scope(store, urls))


def cleanup_qdrant(node_id: str, curriculum_id: str, *, dry_run: bool) -> dict:
    report: dict = {
        "backend": "qdrant",
        "applicable": True,
        "urls_found": [],
        "urls_deleted": 0,
    }
    cid = (curriculum_id or "").strip()
    nid = (node_id or "").strip()
    if nid and not cid:
        report["applicable"] = False
        print(
            "· Qdrant: --node-id без --curriculum-id — неизвестно, чей граф "
            "куррикулума смотреть за источниками ноды, пропуск"
        )
        return report

    store = _get_qdrant_store()
    if store is None:
        report["applicable"] = False
        print("· Qdrant: QDRANT_URL не сконфигурирован — пропуск")
        return report

    urls = _urls_for_scope(nid, cid)
    report["urls_found"] = urls
    if not urls:
        scope = f"node_id={nid!r}" if nid else "весь curriculum_sources_registry"
        print(f"· Qdrant: в графе curriculum_id={cid!r} источников для {scope} не найдено")
        return report

    from knowledge_engine.services.vector_store import VectorStore

    for u in urls:
        verb = "[dry-run] would delete" if dry_run else "deleting"
        print(f"  {verb} url={u[:70]} doc_id={VectorStore.doc_id_for_url(u)[:12]}…")
    if not dry_run:
        report["urls_deleted"] = asyncio.run(_delete_qdrant_scope(store, urls))
    return report


def cleanup_gemini_cache(node_id: str, curriculum_id: str, *, dry_run: bool) -> dict:
    _ = (node_id, curriculum_id, dry_run)
    print(
        "· Gemini Cache: CacheMetadata (cloud_cache_manager.py) не хранит node_id/"
        "curriculum_id (display_name — hash digest / session_id) — очистка неприменима, пропуск"
    )
    return {"backend": "gemini_cache", "applicable": False}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(
        f"=== CLEANUP CLOUD RESOURCES | node_id={args.node_id!r} "
        f"curriculum_id={args.curriculum_id!r} dry_run={args.dry_run} ==="
    )

    redis_report = cleanup_redis(args.node_id, args.curriculum_id, dry_run=args.dry_run)
    qdrant_report = cleanup_qdrant(args.node_id, args.curriculum_id, dry_run=args.dry_run)
    cleanup_gemini_cache(args.node_id, args.curriculum_id, dry_run=args.dry_run)

    print("\n=== SUMMARY ===")
    print(
        f"Redis: found={len(redis_report['keys_found'])} "
        f"deleted={redis_report['keys_deleted']} dry_run={args.dry_run}"
    )
    if qdrant_report["applicable"]:
        print(
            f"Qdrant: urls_found={len(qdrant_report['urls_found'])} "
            f"urls_deleted={qdrant_report['urls_deleted']} dry_run={args.dry_run} "
            "(document_summaries by url, rag_chunks/knowledge_atoms by doc_id)"
        )
    else:
        print("Qdrant: not applicable (see reason above)")
    print("Gemini Cache: not applicable (no node/curriculum linkage in CacheMetadata today)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
