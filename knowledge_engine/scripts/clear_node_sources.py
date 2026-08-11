"""Clear grounded sources for one curriculum node (graph + LanceDB + session + optional blocklist).

Example (re-ground test for anti-bot / Exa highlights fallback):
  python knowledge_engine/scripts/clear_node_sources.py \\
    --curriculum agentic_systems_architecture \\
    --node governed_agent_pipelines \\
    --apply
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

from knowledge_engine.db.domain_blocklist import (
    extract_domain_from_url,
    get_blocked_domains,
    remove_blocked_domain,
)
from knowledge_engine.services.node_session_reset import (
    reset_node_deep_dive_persistence,
)
from knowledge_engine.services.skill_tree_store import (
    get_curriculum_graph,
    get_curriculum_meta,
    save_curriculum_record,
)
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.src.curriculum.source_registry import resolve_sources_for_node


def _collect_node_source_urls(
    graph: dict[str, Any], node_id: str
) -> tuple[list[str], list[str]]:
    nodes = graph.get("nodes") or []
    node_raw = next(
        (n for n in nodes if isinstance(n, dict) and str(n.get("node_id")) == node_id),
        None,
    )
    if not node_raw:
        return [], []
    mapped = [
        str(x).strip()
        for x in (node_raw.get("mapped_source_ids") or [])
        if str(x).strip()
    ]
    urls: list[str] = []
    seen: set[str] = set()
    for row in resolve_sources_for_node(graph, node_id, mapped):
        u = (row.get("url") or "").strip()
        if not u.startswith("http"):
            continue
        key = u.rstrip("/").lower()
        if key in seen:
            continue
        seen.add(key)
        urls.append(u)
    for u in node_raw.get("resource_urls") or []:
        raw = str(u).strip()
        if raw.startswith("http") and raw.rstrip("/").lower() not in seen:
            seen.add(raw.rstrip("/").lower())
            urls.append(raw)
    ref = node_raw.get("source_ref")
    if isinstance(ref, dict):
        ru = str(ref.get("url") or "").strip()
        if ru.startswith("http") and ru.rstrip("/").lower() not in seen:
            urls.append(ru)
    return mapped, urls


def _scrub_registry_extracts(
    graph: dict[str, Any],
    source_ids: list[str],
) -> int:
    if not source_ids:
        return 0
    want = {s.strip() for s in source_ids if s.strip()}
    n = 0
    for key in ("curriculum_sources_registry", "route_sources"):
        entries = graph.get(key) or []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            sid = str(entry.get("source_id") or "").strip()
            if sid not in want:
                continue
            if entry.get("key_extracts") or entry.get("snippet"):
                entry["key_extracts"] = []
                entry["snippet"] = ""
                n += 1
    return n


def build_plan(
    curriculum_id: str,
    node_id: str,
    *,
    clear_blocklist: bool,
) -> dict[str, Any]:
    graph = get_curriculum_graph(curriculum_id)
    if not graph:
        return {"error": f"no graph for curriculum_id={curriculum_id}"}
    mapped, urls = _collect_node_source_urls(graph, node_id)
    domains = sorted(
        {extract_domain_from_url(u) for u in urls if extract_domain_from_url(u)}
    )
    blocked_now = get_blocked_domains()
    domains_to_unblock = [d for d in domains if d in blocked_now]
    return {
        "curriculum_id": curriculum_id,
        "node_id": node_id,
        "mapped_source_ids": mapped,
        "urls": urls,
        "domains": domains,
        "blocklist_clear_domains": domains_to_unblock if clear_blocklist else [],
    }


def apply_clear(
    curriculum_id: str,
    node_id: str,
    *,
    clear_blocklist: bool,
    scrub_registry: bool,
) -> dict[str, Any]:
    plan = build_plan(curriculum_id, node_id, clear_blocklist=clear_blocklist)
    if plan.get("error"):
        return plan

    urls = list(plan["urls"])
    mapped = list(plan["mapped_source_ids"])
    store = VectorStore()
    rag_removed = store.delete_rag_chunks_for_urls(urls)
    summary_removed = store.delete_summaries_for_urls(urls)

    graph = get_curriculum_graph(curriculum_id) or {}
    registry_scrubbed = 0
    if scrub_registry and mapped:
        registry_scrubbed = _scrub_registry_extracts(graph, mapped)

    node_patch = {
        "mapped_source_ids": [],
        "primary_source_id": "",
        "resource_urls": [],
        "source_ref": None,
        "grounding_status": "pending_grounding",
        "learning_resources": [],
    }
    patched = False
    for raw in graph.get("nodes") or []:
        if not isinstance(raw, dict):
            continue
        if str(raw.get("node_id") or "").strip() != node_id:
            continue
        raw.update(node_patch)
        patched = True
        break

    if patched or registry_scrubbed:
        meta = get_curriculum_meta(curriculum_id) or {}
        save_curriculum_record(
            graph,
            target_goal=str(meta.get("target_goal") or graph.get("description") or ""),
            generation_mode=str(meta.get("generation_mode") or "fast"),
            depth_level=str(meta.get("depth_level") or "Standard"),
            user_level=str(meta.get("user_level") or "Intermediate/Advanced"),
            source_policy=str(meta.get("source_policy") or "") or None,
        )

    session_reset = reset_node_deep_dive_persistence(curriculum_id, node_id)

    unblocked: list[str] = []
    if clear_blocklist:
        for dom in plan.get("blocklist_clear_domains") or []:
            if remove_blocked_domain(dom):
                unblocked.append(dom)

    return {
        **plan,
        "applied": True,
        "lance_rag_chunks_removed": rag_removed,
        "lance_summaries_removed": summary_removed,
        "registry_entries_scrubbed": registry_scrubbed,
        "graph_node_patched": patched,
        "session_reset": session_reset,
        "blocklist_domains_removed": unblocked,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Clear collected sources for one node")
    parser.add_argument("--curriculum", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Execute deletes (default: dry-run plan only)",
    )
    parser.add_argument(
        "--no-clear-blocklist",
        action="store_true",
        help="Do not remove domains from SQLite blocklist for cleared URLs",
    )
    parser.add_argument(
        "--no-scrub-registry",
        action="store_true",
        help="Keep key_extracts/snippet on curriculum_sources_registry entries",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    clear_blocklist = not args.no_clear_blocklist
    scrub_registry = not args.no_scrub_registry

    if args.apply:
        report = apply_clear(
            args.curriculum.strip(),
            args.node.strip(),
            clear_blocklist=clear_blocklist,
            scrub_registry=scrub_registry,
        )
    else:
        report = build_plan(
            args.curriculum.strip(),
            args.node.strip(),
            clear_blocklist=clear_blocklist,
        )
        report["applied"] = False
        report["hint"] = "Re-run with --apply to execute"

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"curriculum={report.get('curriculum_id')} node={report.get('node_id')}")
        if report.get("error"):
            print(f"ERROR: {report['error']}")
            return 2
        print(f"mapped_source_ids={report.get('mapped_source_ids')}")
        print(f"urls ({len(report.get('urls') or [])}):")
        for u in report.get("urls") or []:
            print(f"  - {u}")
        if report.get("blocklist_clear_domains"):
            print(f"blocklist would clear: {report['blocklist_clear_domains']}")
        if report.get("applied"):
            print(
                f"Lance rag_chunks removed={report.get('lance_rag_chunks_removed')} "
                f"summaries={report.get('lance_summaries_removed')}"
            )
            print(
                f"registry scrubbed={report.get('registry_entries_scrubbed')} "
                f"node_patched={report.get('graph_node_patched')} "
                f"session_reset={report.get('session_reset')}"
            )
            print(f"blocklist removed={report.get('blocklist_domains_removed')}")
        else:
            print(report.get("hint", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
