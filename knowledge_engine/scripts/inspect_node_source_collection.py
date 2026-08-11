"""Inspect what is known/collected for a curriculum node source (graph + LanceDB + fetch probe).

Example:
  python knowledge_engine/scripts/inspect_node_source_collection.py \\
    --curriculum agentic_systems_architecture \\
    --node governed_agent_pipelines \\
    --source-id src_9 \\
    --url 'https://openreview.net/forum?id=zS5eRqW2QQ' \\
    --probe-fetch
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

os.environ.setdefault("KE_TRACE_STDOUT", "0")

from knowledge_engine.config import LANCE_DB_PATH
from knowledge_engine.services.lecture_rag_source_scope import (
    collect_mapped_source_urls,
    mapped_doc_ids_for_node,
)
from knowledge_engine.services.skill_tree_store import (
    get_curriculum_graph,
    get_curriculum_meta,
)
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.src.curriculum.source_registry import (
    registry_index,
    resolve_sources_for_node,
)
from knowledge_engine.src.node_deep_dive.schemas import NodeDataInput
from knowledge_engine.src.node_deep_dive.session_store import (
    get_node_statuses_for_curriculum,
)


def _norm_url(url: str) -> str:
    return (url or "").strip().rstrip("/").lower()


def _node_input_from_dict(
    node_dict: dict[str, Any], nid: str, mapped_ids: list[str]
) -> NodeDataInput:
    concepts = [
        str(c).strip() for c in (node_dict.get("core_concepts") or []) if str(c).strip()
    ]
    if not concepts:
        concepts = ["topic"]
    return NodeDataInput(
        node_id=nid,
        title=str(node_dict.get("title") or nid),
        layer=str(node_dict.get("layer") or "foundation"),
        category=str(node_dict.get("category") or ""),
        brief_summary=str(node_dict.get("brief_summary") or ""),
        core_concepts=concepts,
        mapped_source_ids=mapped_ids,
    )


def _registry_entry(idx: dict[str, dict[str, Any]], sid: str) -> dict[str, Any] | None:
    e = idx.get((sid or "").strip())
    return e if isinstance(e, dict) else None


def _probe_url(url: str, timeout: float) -> dict[str, Any]:
    import httpx

    out: dict[str, Any] = {"url": url, "ok": False}
    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; REsearch-knowledge-engine/1.0; +inspect)"
                ),
            },
        ) as client:
            resp = client.get(url)
        body = (resp.text or "")[:4000]
        low = body.lower()
        blocked_markers = (
            "verifying your browser",
            "cloudflare",
            "cf-browser-verification",
            "access denied",
            "captcha",
            "robot check",
        )
        blocked = any(m in low for m in blocked_markers)
        textish = sum(c.isalnum() for c in body)
        out.update(
            {
                "http_status": resp.status_code,
                "content_type": resp.headers.get("content-type", ""),
                "body_chars": len(body),
                "alnum_chars": textish,
                "likely_blocked": blocked,
                "ok": resp.status_code == 200 and not blocked and textish > 200,
            }
        )
        if blocked:
            out["block_hint"] = "anti-bot / browser verification page detected"
    except Exception as exc:
        out["error"] = str(exc)
    return out


def _smart_fetch_probe(url: str) -> dict[str, Any]:
    from knowledge_engine.services.web_extract import smart_fetch_page_text

    out: dict[str, Any] = {"url": url}
    try:
        text, method = smart_fetch_page_text(url)
        t = (text or "").strip()
        out["method"] = method
        out["text_chars"] = len(t)
        out["text_words"] = len(t.split())
        out["ok"] = len(t) >= 200
        if not out["ok"]:
            out["hint"] = "smart_fetch returned <200 chars — snippet-only or blocked"
    except Exception as exc:
        out["error"] = str(exc)
        out["ok"] = False
    return out


def _lance_report(urls: list[str]) -> dict[str, Any]:
    store = VectorStore()
    vs = VectorStore()
    per_url: dict[str, Any] = {}
    for raw in urls:
        u = (raw or "").strip()
        if not u.startswith("http"):
            continue
        did = VectorStore.doc_id_for_url(u)
        summaries = store.fetch_summaries_by_urls([u], limit=1)
        chunks = vs.fetch_rag_chunks_by_doc_id(did)
        chunk_chars = sum(len(str(c.get("chunk_text") or "")) for c in chunks)
        summary = summaries[0] if summaries else None
        per_url[u] = {
            "doc_id": did,
            "document_summary_in_lancedb": summary is not None,
            "summary_title": (summary.title or "")[:200] if summary else "",
            "summary_takeaways": len(summary.key_takeaways or []) if summary else 0,
            "rag_chunk_count": len(chunks),
            "rag_chunk_text_chars": chunk_chars,
            "rag_ok": len(chunks) > 0 and chunk_chars >= 200,
        }
    tables = []
    try:
        db = vs._db  # noqa: SLF001 — diagnostic script
        tables = list(db.list_tables())
    except Exception as exc:
        tables = [f"error: {exc}"]
    return {
        "lance_db_path": str(LANCE_DB_PATH),
        "tables": tables,
        "by_url": per_url,
    }


def build_report(
    *,
    curriculum_id: str,
    node_id: str,
    source_id: str | None,
    url_override: str | None,
    probe_fetch: bool,
    probe_smart: bool,
    timeout: float,
) -> dict[str, Any]:
    cid = (curriculum_id or "").strip()
    nid = (node_id or "").strip()
    graph = get_curriculum_graph(cid) or {}
    meta = get_curriculum_meta(cid) or {}
    idx = registry_index(graph)

    node_dict = next(
        (n for n in (graph.get("nodes") or []) if str(n.get("node_id")) == nid),
        None,
    )
    mapped_ids = (
        [
            str(x).strip()
            for x in (node_dict.get("mapped_source_ids") or [])
            if str(x).strip()
        ]
        if isinstance(node_dict, dict)
        else []
    )

    sid = (source_id or "").strip()
    reg_entry = _registry_entry(idx, sid) if sid else None
    reg_url = str(reg_entry.get("url") or "").strip() if reg_entry else ""

    urls_to_check: list[str] = []
    if url_override and url_override.startswith("http"):
        urls_to_check.append(url_override.strip())
    if reg_url.startswith("http") and _norm_url(reg_url) not in {
        _norm_url(u) for u in urls_to_check
    }:
        urls_to_check.append(reg_url)

    if node_dict and isinstance(node_dict, dict):
        node_input = _node_input_from_dict(node_dict, nid, mapped_ids)
        mapped_urls = collect_mapped_source_urls(cid, node_input)
        for u in mapped_urls:
            if _norm_url(u) not in {_norm_url(x) for x in urls_to_check}:
                urls_to_check.append(u)

    resolved = resolve_sources_for_node(graph, nid, mapped_ids) if node_dict else []

    report: dict[str, Any] = {
        "curriculum_id": cid,
        "node_id": nid,
        "graph_found": bool(graph),
        "target_goal": str(meta.get("target_goal") or graph.get("description") or ""),
        "node": {
            "title": node_dict.get("title") if node_dict else None,
            "grounding_status": (
                node_dict.get("grounding_status") if node_dict else None
            ),
            "mapped_source_ids": mapped_ids,
            "resolved_mapped_sources": resolved,
        },
        "source_id": sid or None,
        "registry_entry": reg_entry,
        "urls_checked": urls_to_check,
        "mapped_doc_ids": (
            mapped_doc_ids_for_node(
                cid, _node_input_from_dict(node_dict, nid, mapped_ids)
            )
            if node_dict
            else []
        ),
        "session_status": get_node_statuses_for_curriculum(cid).get(nid),
        "lance": _lance_report(urls_to_check),
    }

    if probe_fetch:
        report["http_probe"] = [_probe_url(u, timeout) for u in urls_to_check]
    if probe_smart:
        report["smart_fetch_probe"] = [_smart_fetch_probe(u) for u in urls_to_check]

    hints: list[str] = []
    if not graph:
        hints.append(
            "curriculum graph not in skill_tree_store — проверьте .runs/skill_tree_curricula.json"
        )
    if sid and not reg_entry:
        hints.append(f"source_id {sid} не найден в curriculum_sources_registry")
    if reg_entry and url_override and _norm_url(reg_url) != _norm_url(url_override):
        hints.append(
            "URL в registry отличается от --url (проверьте актуальность src_* в графе)"
        )
    for u, row in report["lance"]["by_url"].items():
        if not row["document_summary_in_lancedb"] and not row["rag_ok"]:
            hints.append(
                f"LanceDB пуст для {u} — summarizer/ingest не сохранил контент"
            )
        elif row["document_summary_in_lancedb"] and not row["rag_ok"]:
            hints.append(
                f"summary есть, но rag_chunks слабые для {u} — проверьте ingest_document_summary"
            )
    if probe_fetch:
        for p in report.get("http_probe") or []:
            if p.get("likely_blocked"):
                hints.append(
                    f"HTTP probe: возможная блокировка для {p.get('url')} "
                    "(OpenReview часто даёт browser verification)"
                )
    report["hints"] = hints
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect collected sources for a curriculum node"
    )
    parser.add_argument("--curriculum", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--source-id", default="", help="e.g. src_9")
    parser.add_argument(
        "--url", default="", help="optional URL to cross-check LanceDB doc_id"
    )
    parser.add_argument(
        "--probe-fetch",
        action="store_true",
        help="GET URL via httpx (detect anti-bot pages)",
    )
    parser.add_argument(
        "--probe-smart",
        action="store_true",
        help="Run smart_fetch_page_text (same path as academic ingest)",
    )
    parser.add_argument("--timeout", type=float, default=25.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = build_report(
        curriculum_id=args.curriculum,
        node_id=args.node,
        source_id=args.source_id or None,
        url_override=(args.url or "").strip() or None,
        probe_fetch=args.probe_fetch,
        probe_smart=args.probe_smart,
        timeout=args.timeout,
    )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"curriculum={report['curriculum_id']} node={report['node_id']}")
        print(
            f"graph_found={report['graph_found']} session={report.get('session_status')}"
        )
        print(f"mapped_source_ids={report['node']['mapped_source_ids']}")
        if report.get("registry_entry"):
            e = report["registry_entry"]
            print(
                f"registry[{report['source_id']}]: {e.get('title', '')[:80]} | {e.get('url')}"
            )
        print("--- LanceDB ---")
        for u, row in report["lance"]["by_url"].items():
            print(
                f"  {u}\n"
                f"    doc_id={row['doc_id']} summary={row['document_summary_in_lancedb']} "
                f"chunks={row['rag_chunk_count']} chunk_chars={row['rag_chunk_text_chars']}"
            )
        if report.get("http_probe"):
            print("--- HTTP probe ---")
            for p in report["http_probe"]:
                print(f"  {p}")
        if report.get("smart_fetch_probe"):
            print("--- smart_fetch ---")
            for p in report["smart_fetch_probe"]:
                print(f"  {p}")
        if report["hints"]:
            print("--- hints ---")
            for h in report["hints"]:
                print(f"  • {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
