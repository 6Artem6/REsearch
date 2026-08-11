"""Align node session source_registry with graph mapped_source_ids (fix stale «Источники в материале»).

When you remove src_* from «Адресация ноды», the JSON session may still list old [S1]…
entries until registry is rebuilt from the current mapped list.

Example:
  PYTHONPATH=. .venv/bin/python -m knowledge_engine.scripts.sync_node_session_sources \\
    --curriculum agentic_systems_architecture \\
    --node governed_agent_pipelines

  PYTHONPATH=. .venv/bin/python -m knowledge_engine.scripts.sync_node_session_sources \\
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

from knowledge_engine.services.node_source_registry import build_session_source_registry
from knowledge_engine.services.skill_tree_store import (
    get_curriculum_graph,
    get_curriculum_meta,
    save_curriculum_record,
)
from knowledge_engine.src.curriculum.source_registry import resolve_sources_for_node
from knowledge_engine.src.node_deep_dive.session_store import (
    _STORE_PATH,
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
from knowledge_engine.utils.link_sanitizer import normalize_lecture_url


def _norm_url(url: str) -> str:
    return normalize_lecture_url((url or "").strip())


def _registry_urls(registry: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for e in registry or []:
        if not isinstance(e, dict):
            continue
        u = _norm_url(str(e.get("url") or ""))
        if u:
            out.add(u)
    return out


def _registry_course_ids(registry: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for e in registry or []:
        if not isinstance(e, dict):
            continue
        cid = str(e.get("course_source_id") or "").strip()
        if cid:
            out.add(cid)
    return out


def _node_mapped_ids(graph: dict[str, Any], node_id: str) -> list[str]:
    for raw in graph.get("nodes") or []:
        if not isinstance(raw, dict):
            continue
        if str(raw.get("node_id") or "").strip() == node_id:
            return [
                str(x).strip()
                for x in (raw.get("mapped_source_ids") or [])
                if str(x).strip()
            ]
    return []


def build_plan(curriculum_id: str, node_id: str) -> dict[str, Any]:
    cid = curriculum_id.strip()
    nid = node_id.strip()
    graph = get_curriculum_graph(cid)
    if not graph:
        return {"error": f"no graph for curriculum_id={cid}"}

    mapped = _node_mapped_ids(graph, nid)
    target_registry = build_session_source_registry(cid, mapped)

    key = _session_key(cid, nid)
    blob = _load_all().get(key) or {}
    old_registry = list(blob.get("source_registry") or [])

    old_urls = _registry_urls(old_registry)
    new_urls = _registry_urls(target_registry)
    removed_urls = sorted(old_urls - new_urls)
    added_urls = sorted(new_urls - old_urls)

    old_cids = _registry_course_ids(old_registry)
    _registry_course_ids(target_registry)
    mapped_set = set(mapped)
    intersection_ids = sorted(mapped_set & old_cids)
    only_in_mapped = sorted(mapped_set - old_cids)
    only_in_session = sorted(old_cids - mapped_set)

    session = get_session(cid, nid)
    content_before = session.content
    content_after = retarget_content_source_anchors(
        scrub_content_references(content_before, target_registry),
        old_registry,
        target_registry,
    )
    refs_removed = len(content_before.references or []) - len(
        content_after.references or []
    )

    resolved = resolve_sources_for_node(graph, nid, mapped)
    resource_urls = [
        str(r.get("url") or "").strip()
        for r in resolved
        if str(r.get("url") or "").startswith("http")
    ]

    return {
        "curriculum_id": cid,
        "node_id": nid,
        "session_key": key,
        "session_file": str(_STORE_PATH),
        "has_session": bool(blob),
        "mapped_source_ids": mapped,
        "target_registry_count": len(target_registry),
        "old_registry_count": len(old_registry),
        "intersection_course_source_ids": intersection_ids,
        "only_in_mapped_not_session": only_in_mapped,
        "only_in_session_not_mapped": only_in_session,
        "urls_removed_from_material": removed_urls,
        "urls_added_to_material": added_urls,
        "content_references_removed": refs_removed,
        "resource_urls_after_sync": resource_urls,
        "target_registry": target_registry,
        "target_registry_preview": [
            {
                "id": e.get("id"),
                "course_source_id": e.get("course_source_id"),
                "url": e.get("url"),
            }
            for e in target_registry
        ],
    }


def apply_sync(curriculum_id: str, node_id: str) -> dict[str, Any]:
    plan = build_plan(curriculum_id, node_id)
    if plan.get("error"):
        return plan

    cid = plan["curriculum_id"]
    nid = plan["node_id"]
    if not plan.get("has_session"):
        plan["applied"] = False
        plan["hint"] = "no session blob — open node or run init first"
        return plan

    target_registry = list(plan["target_registry"])
    session = get_session(cid, nid)
    blob = _load_all().get(plan["session_key"]) or {}
    old_registry = list(blob.get("source_registry") or [])
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
    rag_labels = list(blob.get("rag_fact_labels") or [])

    save_session(
        cid,
        nid,
        session.node_status,
        content,
        history,
        rag_fact_labels=rag_labels,
        memory=session.memory,
        source_registry=target_registry,
    )

    graph = get_curriculum_graph(cid) or {}
    patched = False
    for raw in graph.get("nodes") or []:
        if not isinstance(raw, dict):
            continue
        if str(raw.get("node_id") or "").strip() != nid:
            continue
        raw["resource_urls"] = list(plan.get("resource_urls_after_sync") or [])
        if raw.get("primary_source_id") and raw["primary_source_id"] not in (
            plan.get("mapped_source_ids") or []
        ):
            mids = plan.get("mapped_source_ids") or []
            raw["primary_source_id"] = mids[0] if mids else ""
        patched = True
        break

    if patched:
        meta = get_curriculum_meta(cid) or {}
        save_curriculum_record(
            graph,
            target_goal=str(meta.get("target_goal") or graph.get("description") or ""),
            generation_mode=str(meta.get("generation_mode") or "fast"),
            depth_level=str(meta.get("depth_level") or "Standard"),
            user_level=str(meta.get("user_level") or "Intermediate/Advanced"),
            source_policy=str(meta.get("source_policy") or "") or None,
        )

    plan["applied"] = True
    plan["graph_resource_urls_patched"] = patched
    plan.pop("target_registry", None)
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync session source_registry with mapped_source_ids",
    )
    parser.add_argument("--curriculum", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write session JSON + scrub content references",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.apply:
        report = apply_sync(args.curriculum, args.node)
    else:
        report = build_plan(args.curriculum, args.node)
        report["applied"] = False
        report["hint"] = "Re-run with --apply to update session"
        report.pop("target_registry", None)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        if report.get("error"):
            print(f"ERROR: {report['error']}")
            return 2
        print(f"curriculum={report['curriculum_id']} node={report['node_id']}")
        print(
            f"session={report.get('session_key')} has_session={report.get('has_session')}"
        )
        print(f"mapped_source_ids={report.get('mapped_source_ids')}")
        print(
            f"registry old={report.get('old_registry_count')} "
            f"→ target={report.get('target_registry_count')}"
        )
        print(f"intersection src ids={report.get('intersection_course_source_ids')}")
        if report.get("only_in_session_not_mapped"):
            print(
                f"stale in session (not mapped): {report['only_in_session_not_mapped']}"
            )
        if report.get("urls_removed_from_material"):
            print("URLs removed from material:")
            for u in report["urls_removed_from_material"]:
                print(f"  - {u}")
        preview = report.get("target_registry_preview") or []
        if preview:
            print("target SOURCE REGISTRY (canonical 1 URL per src_*):")
            for row in preview:
                print(
                    f"  [{row.get('id')}] {row.get('course_source_id')} "
                    f"{row.get('url')}"
                )
        if report.get("content_references_removed"):
            print(f"content.references removed={report['content_references_removed']}")
        if report.get("applied"):
            print(
                f"applied graph_resource_urls={report.get('graph_resource_urls_patched')}"
            )
        else:
            print(report.get("hint", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
