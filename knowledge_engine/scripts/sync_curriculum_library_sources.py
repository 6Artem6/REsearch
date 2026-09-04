"""Prune course-library sources no longer referenced by any curriculum node.

The script is deliberately dry-run by default.  It keeps a registry entry when
either its ``source_id`` or its URL is still referenced from a node, rebuilds
the legacy ``route_sources`` mirror, and aligns existing node sessions with the
node's canonical ``mapped_source_ids``.

LanceDB is global, not curriculum-scoped.  Therefore document material is
deleted only when the orphaned URL is unused by every saved curriculum.

Examples:
  PYTHONPATH=. .venv/bin/python -m knowledge_engine.scripts.sync_curriculum_library_sources \
    --curriculum agentic_systems_architecture

  PYTHONPATH=. .venv/bin/python -m knowledge_engine.scripts.sync_curriculum_library_sources \
    --curriculum agentic_systems_architecture --apply
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

from knowledge_engine.services.lecture_rag_source_scope import (
    normalize_lecture_source_url,
)
from knowledge_engine.services.node_source_registry import build_session_source_registry
from knowledge_engine.services.skill_tree_store import (
    get_curriculum_graph,
    get_curriculum_meta,
    list_curriculum_summaries,
    save_curriculum_record,
)
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.src.node_deep_dive.session_store import (
    _load_all,
    _session_key,
    get_session,
    save_session,
)
from knowledge_engine.src.node_deep_dive.tutor_source_citations import (
    retarget_content_source_anchors,
    retarget_dialog_history_source_anchors,
    scrub_content_references,
)


def _url_key(url: object) -> str:
    value = str(url or "").strip()
    if not value.startswith("http"):
        return ""
    return normalize_lecture_source_url(value)


def _node_material_urls(node: dict[str, Any]) -> set[str]:
    urls: set[str] = set()

    def add(value: object) -> None:
        key = _url_key(value)
        if key:
            urls.add(key)

    for raw in node.get("resource_urls") or []:
        add(raw)
    for row in node.get("learning_resources") or []:
        if isinstance(row, dict):
            add(row.get("url"))
    source_ref = node.get("source_ref")
    if isinstance(source_ref, dict):
        add(source_ref.get("url"))
    return urls


def _node_source_ids(node: dict[str, Any]) -> set[str]:
    ids = {
        str(source_id).strip()
        for source_id in (node.get("mapped_source_ids") or [])
        if str(source_id).strip()
    }
    primary = str(node.get("primary_source_id") or "").strip()
    if primary:
        ids.add(primary)
    source_ref = node.get("source_ref")
    if isinstance(source_ref, dict):
        ref_id = str(source_ref.get("source_id") or "").strip()
        if ref_id:
            ids.add(ref_id)
    return ids


def _collect_node_references(graph: dict[str, Any]) -> tuple[set[str], set[str]]:
    source_ids: set[str] = set()
    urls: set[str] = set()
    for node in graph.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        source_ids.update(_node_source_ids(node))
        urls.update(_node_material_urls(node))
    return source_ids, urls


def _all_curriculum_urls(*, exclude_curriculum_id: str) -> set[str]:
    """URLs still owned by other saved curricula; protects global LanceDB rows."""
    urls: set[str] = set()
    for row in list_curriculum_summaries():
        curriculum_id = str(row.get("curriculum_id") or "").strip()
        if not curriculum_id or curriculum_id == exclude_curriculum_id:
            continue
        graph = get_curriculum_graph(curriculum_id) or {}
        _, node_urls = _collect_node_references(graph)
        urls.update(node_urls)
        for entry in graph.get("curriculum_sources_registry") or []:
            if isinstance(entry, dict):
                key = _url_key(entry.get("url"))
                if key:
                    urls.add(key)
    return urls


def _session_sync_plan(
    curriculum_id: str,
    graph: dict[str, Any],
) -> list[dict[str, Any]]:
    blobs = _load_all()
    out: list[dict[str, Any]] = []
    for node in graph.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("node_id") or "").strip()
        if not node_id:
            continue
        key = _session_key(curriculum_id, node_id)
        blob = blobs.get(key)
        if not isinstance(blob, dict):
            continue
        old_registry = list(blob.get("source_registry") or [])
        mapped = [
            str(source_id).strip()
            for source_id in (node.get("mapped_source_ids") or [])
            if str(source_id).strip()
        ]
        target_registry = build_session_source_registry(curriculum_id, mapped)
        old_urls = {
            _url_key(row.get("url"))
            for row in old_registry
            if isinstance(row, dict) and _url_key(row.get("url"))
        }
        new_urls = {
            _url_key(row.get("url"))
            for row in target_registry
            if isinstance(row, dict) and _url_key(row.get("url"))
        }
        if old_urls != new_urls:
            out.append(
                {
                    "node_id": node_id,
                    "session_key": key,
                    "mapped_source_ids": mapped,
                    "old_registry": old_registry,
                    "target_registry": target_registry,
                    "urls_removed": sorted(old_urls - new_urls),
                    "urls_added": sorted(new_urls - old_urls),
                }
            )
    return out


def build_plan(curriculum_id: str) -> dict[str, Any]:
    curriculum_id = (curriculum_id or "").strip()
    graph = get_curriculum_graph(curriculum_id)
    if not graph:
        return {"error": f"no graph for curriculum_id={curriculum_id}"}

    referenced_ids, node_urls = _collect_node_references(graph)
    registry = [
        entry
        for entry in (graph.get("curriculum_sources_registry") or [])
        if isinstance(entry, dict)
    ]
    keep: list[dict[str, Any]] = []
    orphaned: list[dict[str, Any]] = []
    for entry in registry:
        source_id = str(entry.get("source_id") or "").strip()
        url = _url_key(entry.get("url"))
        if source_id in referenced_ids or (url and url in node_urls):
            keep.append(entry)
        else:
            orphaned.append(entry)

    other_curriculum_urls = _all_curriculum_urls(exclude_curriculum_id=curriculum_id)
    delete_from_lancedb: list[str] = []
    retained_global_urls: list[str] = []
    for entry in orphaned:
        url = _url_key(entry.get("url"))
        if not url:
            continue
        if url in other_curriculum_urls:
            retained_global_urls.append(url)
        else:
            delete_from_lancedb.append(url)

    sessions = _session_sync_plan(curriculum_id, graph)
    return {
        "curriculum_id": curriculum_id,
        "registry_before": len(registry),
        "registry_after": len(keep),
        "referenced_source_ids": sorted(referenced_ids),
        "node_material_urls": sorted(node_urls),
        "orphaned_registry_entries": [
            {
                "source_id": str(entry.get("source_id") or ""),
                "url": _url_key(entry.get("url")),
                "title": str(entry.get("title") or ""),
            }
            for entry in orphaned
        ],
        "orphaned_urls_delete_from_lancedb": sorted(set(delete_from_lancedb)),
        "orphaned_urls_kept_for_other_curricula": sorted(set(retained_global_urls)),
        "sessions_to_sync": [
            {
                "node_id": row["node_id"],
                "urls_removed": row["urls_removed"],
                "urls_added": row["urls_added"],
            }
            for row in sessions
        ],
        "_kept_registry": keep,
        "_session_sync": sessions,
    }


def apply_sync(curriculum_id: str) -> dict[str, Any]:
    plan = build_plan(curriculum_id)
    if plan.get("error"):
        return plan

    curriculum_id = plan["curriculum_id"]
    graph = get_curriculum_graph(curriculum_id) or {}
    graph["curriculum_sources_registry"] = list(plan["_kept_registry"])
    meta = get_curriculum_meta(curriculum_id) or {}
    save_curriculum_record(
        graph,
        target_goal=str(meta.get("target_goal") or graph.get("description") or ""),
        generation_mode=str(meta.get("generation_mode") or "fast"),
        depth_level=str(meta.get("depth_level") or "Standard"),
        user_level=str(meta.get("user_level") or "Intermediate/Advanced"),
        source_policy=str(meta.get("source_policy") or "") or None,
    )

    session_count = 0
    for row in plan["_session_sync"]:
        session = get_session(curriculum_id, row["node_id"])
        old_registry = row["old_registry"]
        target_registry = row["target_registry"]
        content = retarget_content_source_anchors(
            scrub_content_references(session.content, target_registry),
            old_registry,
            target_registry,
        )
        history = retarget_dialog_history_source_anchors(
            session.history,
            old_registry,
            target_registry,
        )
        blob = _load_all().get(row["session_key"]) or {}
        save_session(
            curriculum_id,
            row["node_id"],
            session.node_status,
            content,
            history,
            rag_fact_labels=list(blob.get("rag_fact_labels") or []),
            memory=session.memory,
            source_registry=target_registry,
        )
        session_count += 1

    store = VectorStore()
    urls_to_delete = list(plan["orphaned_urls_delete_from_lancedb"])

    from knowledge_engine.scripts.cleanup_cloud_resources import delete_qdrant_urls
    from knowledge_engine.services.article_diagram_store import (
        delete_diagrams_for_urls,
    )
    from knowledge_engine.services.article_ingestion.figure_registry_service import (
        delete_figure_registry_for_urls,
    )

    return {
        **plan,
        "applied": True,
        "sessions_synced": session_count,
        "lance_rag_chunks_removed": store.delete_rag_chunks_for_urls(urls_to_delete),
        "lance_summaries_removed": store.delete_summaries_for_urls(urls_to_delete),
        "lance_knowledge_atoms_removed": store.delete_knowledge_atoms_for_urls(
            urls_to_delete
        ),
        "qdrant_urls_removed": delete_qdrant_urls(urls_to_delete),
        "diagrams_removed": delete_diagrams_for_urls(urls_to_delete),
        "figure_registry_removed": delete_figure_registry_for_urls(urls_to_delete),
    }


def _public_report(plan: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in plan.items() if not key.startswith("_")}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Synchronize course library with current node source mappings"
    )
    parser.add_argument("--curriculum", required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write graph/session changes and remove globally unreferenced LanceDB URLs",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    plan = apply_sync(args.curriculum) if args.apply else build_plan(args.curriculum)
    if plan.get("error"):
        print(f"ERROR: {plan['error']}")
        return 2
    report = _public_report(plan)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    print(f"curriculum={report['curriculum_id']}")
    print(
        f"course library: {report['registry_before']} → {report['registry_after']} "
        f"| orphaned={len(report['orphaned_registry_entries'])}"
    )
    for row in report["orphaned_registry_entries"]:
        print(f"  - {row['source_id']}: {row['title']} | {row['url']}")
    print(f"sessions to sync={len(report['sessions_to_sync'])}")
    print(
        "LanceDB URL delete candidates="
        f"{len(report['orphaned_urls_delete_from_lancedb'])}"
    )
    if report["orphaned_urls_kept_for_other_curricula"]:
        print(
            "LanceDB retained (used by another curriculum)="
            f"{len(report['orphaned_urls_kept_for_other_curricula'])}"
        )
    if not args.apply:
        print("dry-run; re-run with --apply to write changes")
    else:
        print(
            f"applied sessions={report['sessions_synced']} "
            f"rag_chunks={report['lance_rag_chunks_removed']} "
            f"summaries={report['lance_summaries_removed']} "
            f"knowledge_atoms={report['lance_knowledge_atoms_removed']} "
            f"qdrant_urls={report['qdrant_urls_removed']} "
            f"diagrams={report['diagrams_removed']} "
            f"figure_registry={report['figure_registry_removed']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
