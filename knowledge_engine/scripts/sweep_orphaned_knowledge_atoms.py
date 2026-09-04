"""One-off sweep: purge local LanceDB ``knowledge_atoms`` rows AND live Qdrant
Cloud points (document_summaries/rag_chunks/knowledge_atoms) whose URL is no
longer referenced by ANY saved curriculum.

Context: ``knowledge_atoms`` writes moved to Qdrant-only (see
``VectorStore.upsert_knowledge_atoms``), but pre-migration rows were left
behind in LanceDB with no cleanup path. ``clear_node_sources.apply_clear``
and ``sync_curriculum_library_sources.apply_sync`` were since wired up to
also call ``VectorStore.delete_knowledge_atoms_for_urls`` — but those two
entry points only clean rows for URLs that pass through a NODE-scoped clear
or a CURRICULUM-scoped orphan GC. Rows belonging to nodes/curricula that are
never re-cleared stay orphaned forever. This script is a one-off global sweep
for exactly that backlog (confirmed 184 rows in this table as of 2026-08-27,
found while auditing why ``clear_node_data.py`` seemed to leave data behind).

The candidate URL set is NOT just the local Lance table's rows: those are only
the pre-migration backlog, and shrink over time as this script (and node/
library GC) clear them — as of 2026-08-27 it's down to a handful of rows,
while the LIVE Qdrant ``knowledge_atoms`` collection (all post-migration
writes) can independently hold hundreds of points for URLs that never had a
local Lance row at all. A sweep scoped to Lance rows only can never see, let
alone clean, that live collection's own orphans. So candidates are the UNION
of local Lance URLs and Qdrant's own distinct ``url`` payload values for the
knowledge_atoms collection (via ``QdrantVectorStore.distinct_field_values``).

"Live" (kept) is the same notion ``sync_curriculum_library_sources.py`` uses
for its own per-curriculum orphan GC — the union, across EVERY saved
curriculum, of: every node's ``resource_urls`` / ``learning_resources`` /
``source_ref``, AND every ``curriculum_sources_registry`` entry's url (a
registry entry not yet attached to any node is still "claimed", not
orphaned). A knowledge_atoms row whose URL is in neither set, for no
curriculum, is safe to delete.

Dry-run by default.

Usage:
  PYTHONPATH=. ./.venv/bin/python knowledge_engine/scripts/sweep_orphaned_knowledge_atoms.py
  PYTHONPATH=. ./.venv/bin/python knowledge_engine/scripts/sweep_orphaned_knowledge_atoms.py --apply
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("KE_TRACE_STDOUT", "1")

from knowledge_engine.db.knowledge_atoms_schema import COL_DOC_ID, COL_URL
from knowledge_engine.scripts.sync_curriculum_library_sources import (
    _collect_node_references,
    _url_key,
)
from knowledge_engine.services.skill_tree_store import (
    get_curriculum_graph,
    list_curriculum_summaries,
)
from knowledge_engine.services.vector_store import VectorStore


def _all_live_urls() -> set[str]:
    """Every URL still claimed by any node OR registry entry, across every
    saved curriculum — same notion sync_curriculum_library_sources.py uses
    per-curriculum, just without excluding one."""
    live: set[str] = set()
    for row in list_curriculum_summaries():
        curriculum_id = str(row.get("curriculum_id") or "").strip()
        if not curriculum_id:
            continue
        graph = get_curriculum_graph(curriculum_id) or {}
        _, node_urls = _collect_node_references(graph)
        live.update(node_urls)
        for entry in graph.get("curriculum_sources_registry") or []:
            if isinstance(entry, dict):
                key = _url_key(entry.get("url"))
                if key:
                    live.add(key)
    return live


def _qdrant_knowledge_atom_urls() -> set[str]:
    """Distinct ``url`` payload values in the LIVE Qdrant knowledge_atoms
    collection. Empty set (safe no-op) when QDRANT_URL isn't configured."""
    from knowledge_engine.config import QDRANT_URL

    if not (QDRANT_URL or "").strip():
        return set()
    import asyncio

    from knowledge_engine.db.knowledge_atoms_schema import KNOWLEDGE_ATOMS_TABLE
    from knowledge_engine.services.qdrant_vector_store import QdrantVectorStore

    store = QdrantVectorStore()
    if not store.enabled:
        return set()
    return asyncio.run(store.distinct_field_values(KNOWLEDGE_ATOMS_TABLE, "url"))


def build_plan() -> dict[str, Any]:
    store = VectorStore()
    table = store._knowledge_atoms_table()
    table_present = table is not None

    by_url: dict[str, dict[str, Any]] = {}
    total_rows = 0
    if table_present:
        rows = table.to_arrow().to_pylist()
        total_rows = len(rows)
        for r in rows:
            raw_url = str(r.get(COL_URL) or "").strip()
            key = _url_key(raw_url)
            if not key:
                continue
            bucket = by_url.setdefault(
                key, {"url": raw_url, "doc_ids": set(), "row_count": 0}
            )
            bucket["row_count"] += 1
            did = str(r.get(COL_DOC_ID) or "").strip()
            if did:
                bucket["doc_ids"].add(did)

    qdrant_only_urls = 0
    for raw_url in _qdrant_knowledge_atom_urls():
        key = _url_key(raw_url)
        if not key or key in by_url:
            continue
        by_url[key] = {"url": raw_url, "doc_ids": set(), "row_count": 0}
        qdrant_only_urls += 1

    if not by_url:
        return {
            "table_present": table_present,
            "total_rows": total_rows,
            "total_distinct_urls": 0,
            "qdrant_only_urls": qdrant_only_urls,
            "_orphaned": {},
            "_kept": {},
            "orphaned_rows": 0,
            "kept_urls": [],
        }

    live = _all_live_urls()
    orphaned = {k: v for k, v in by_url.items() if k not in live}
    kept = {k: v for k, v in by_url.items() if k in live}

    return {
        "table_present": table_present,
        "total_rows": total_rows,
        "total_distinct_urls": len(by_url),
        "qdrant_only_urls": qdrant_only_urls,
        "_orphaned": orphaned,
        "_kept": kept,
        "orphaned_rows": sum(v["row_count"] for v in orphaned.values()),
        "kept_urls": sorted(kept.keys()),
    }


def apply_sweep() -> dict[str, Any]:
    plan = build_plan()
    orphaned = plan.pop("_orphaned")
    plan.pop("_kept", None)
    urls_to_delete = [v["url"] for v in orphaned.values()]
    store = VectorStore()
    removed = store.delete_knowledge_atoms_for_urls(urls_to_delete)

    from knowledge_engine.scripts.cleanup_cloud_resources import delete_qdrant_urls

    qdrant_removed = delete_qdrant_urls(urls_to_delete)
    return {
        **plan,
        "applied": True,
        "urls_deleted": len(urls_to_delete),
        "removed": removed,
        "qdrant_removed": qdrant_removed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Execute (default: dry-run plan only)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.apply:
        report = apply_sweep()
    else:
        plan = build_plan()
        orphaned = plan.pop("_orphaned", {})
        plan.pop("_kept", None)
        plan["applied"] = False
        plan["orphaned_url_count"] = len(orphaned)
        plan["orphaned_url_list"] = sorted(
            f"{v['url']} (rows={v['row_count']})" for v in orphaned.values()
        )
        report = plan

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0

    print(f"=== SWEEP ORPHANED knowledge_atoms | apply={args.apply} ===")
    if not report.get("table_present"):
        print("local knowledge_atoms LanceDB table does not exist — scanning Qdrant only")
    print(f"local Lance rows: {report.get('total_rows')}")
    print(
        f"total distinct urls: {report.get('total_distinct_urls')} "
        f"(qdrant-only, no local row: {report.get('qdrant_only_urls')})"
    )
    if not args.apply:
        for line in report.get("orphaned_url_list") or []:
            print(f"  [dry-run] would delete: {line}")
        print(
            f"orphaned urls (candidates): {report.get('orphaned_url_count')} "
            f"(local rows among them: {report.get('orphaned_rows')})"
        )
        print(f"kept urls (still referenced): {len(report.get('kept_urls') or [])}")
        print("re-run with --apply to execute")
    else:
        print(f"urls deleted: {report.get('urls_deleted')}")
        print(f"local LanceDB rows removed: {report.get('removed')}")
        print(f"Qdrant points removed (document_summaries+rag_chunks+knowledge_atoms): {report.get('qdrant_removed')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
