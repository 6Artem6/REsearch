"""Backfill: регистрирует в curriculum_sources_registry URL, которые уже
есть у ноды в resource_urls (и в document_summaries), но не привязаны через
mapped_source_ids — иначе tutor_source_citations.coerce_references_to_registry
видит пустой registry и отбрасывает ВСЕ references/used_sources безусловно
(``if not registry: return []``), лекция цитирует голыми [n] вместо [Sn], в
правой панели и под ответом — пусто.

Разовая причина: до фикса persist_verified_external_sources_to_node писала
только document_summaries + resource_urls, но не curriculum_sources_registry
+ mapped_source_ids (второй слой добавлен позже). Этот скрипт закрывает уже
накопленный разрыв для нод, обработанных ДО фикса; после фикса новый разрыв
не должен появляться (написан unit-тест на регресс).

Только URL, для которых уже есть document_summaries (Qdrant) — считаются
источником; URL без summary просто пропускаются (не выдумываем title/snippet
из воздуха, ничего с ними не делаем).

Example:
  PYTHONPATH=. .venv/bin/python -m knowledge_engine.scripts.backfill_verified_source_registry \\
    --curriculum indexes_and_data_structures --node b_tree_indexes

  PYTHONPATH=. .venv/bin/python -m knowledge_engine.scripts.backfill_verified_source_registry \\
    --curriculum indexes_and_data_structures --node b_tree_indexes --apply

  # Прогнать по ВСЕМ куррикулумам/нодам:
  PYTHONPATH=. .venv/bin/python -m knowledge_engine.scripts.backfill_verified_source_registry --apply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("KE_TRACE_STDOUT", "1")

from knowledge_engine.config import CURRICULUM_DEEP_NODE_MAX_HITS
from knowledge_engine.services.skill_tree_store import (
    get_curriculum_graph,
    list_curriculum_summaries,
    patch_curriculum_graph_node,
    patch_curriculum_sources_registry,
)
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.src.curriculum.source_registry import registry_index


def _norm_url(url: str) -> str:
    return (url or "").strip().rstrip("/").lower()


def _registered_urls(graph: dict[str, Any], mapped_source_ids: list[str]) -> set[str]:
    idx = registry_index(graph)
    out: set[str] = set()
    for sid in mapped_source_ids:
        entry = idx.get((sid or "").strip())
        if isinstance(entry, dict):
            url = str(entry.get("url") or "").strip()
            if url:
                out.add(_norm_url(url))
    return out


def _nodes_to_scan(
    curriculum_id: str | None, node_id: str | None
) -> list[tuple[str, str]]:
    """[(curriculum_id, node_id), ...] с непустым resource_urls."""
    out: list[tuple[str, str]] = []
    cids = (
        [curriculum_id.strip()]
        if curriculum_id
        else [
            str(row.get("curriculum_id") or "").strip()
            for row in list_curriculum_summaries()
        ]
    )
    for cid in cids:
        if not cid:
            continue
        graph = get_curriculum_graph(cid) or {}
        for raw in graph.get("nodes") or []:
            nid = str(raw.get("node_id") or "").strip()
            if not nid:
                continue
            if node_id and nid != node_id.strip():
                continue
            if raw.get("resource_urls"):
                out.append((cid, nid))
    return out


async def _summary_for_url(store: VectorStore, url: str) -> dict[str, str] | None:
    rows = await store.fetch_summaries_by_urls([url], limit=1)
    if not rows:
        return None
    ds = rows[0]
    takeaways = list(ds.key_takeaways or [])
    snippet = (takeaways[0] if takeaways else (ds.executive_summary or "")).strip()
    return {
        "title": (ds.title or url)[:400],
        "whitelist_domain": "",
        "source_type": "verified_external",
        "url": url[:2000],
        "why_read": snippet[:800],
        "snippet": snippet[:1200],
        "key_extracts": [],
        "source_tier": "exa",
    }


async def _plan_for_node(
    store: VectorStore, curriculum_id: str, node_id: str, graph: dict[str, Any]
) -> dict[str, Any]:
    node = next(
        (n for n in graph.get("nodes") or [] if str(n.get("node_id")) == node_id),
        {},
    )
    resource_urls = [str(u).strip() for u in (node.get("resource_urls") or []) if u]
    mapped_ids = [str(x).strip() for x in (node.get("mapped_source_ids") or []) if x]
    registered = _registered_urls(graph, mapped_ids)

    orphaned = [u for u in resource_urls if _norm_url(u) not in registered]
    entries: list[dict[str, str]] = []
    skipped_no_summary: list[str] = []
    for url in orphaned:
        entry = await _summary_for_url(store, url)
        if entry is None:
            skipped_no_summary.append(url)
            continue
        entries.append(entry)

    return {
        "curriculum_id": curriculum_id,
        "node_id": node_id,
        "resource_urls": len(resource_urls),
        "already_registered": len(resource_urls) - len(orphaned),
        "orphaned_urls": len(orphaned),
        "backfillable": len(entries),
        "skipped_no_summary": skipped_no_summary,
        "_entries": entries,
    }


async def run(
    *, curriculum_id: str | None, node_id: str | None, apply: bool
) -> list[dict[str, Any]]:
    store = VectorStore()
    plans: list[dict[str, Any]] = []
    graph_cache: dict[str, dict[str, Any]] = {}
    for cid, nid in _nodes_to_scan(curriculum_id, node_id):
        if cid not in graph_cache:
            graph_cache[cid] = get_curriculum_graph(cid) or {}
        plan = await _plan_for_node(store, cid, nid, graph_cache[cid])
        if plan["backfillable"] == 0:
            plans.append(plan)
            continue

        if apply:
            new_ids = patch_curriculum_sources_registry(cid, plan["_entries"])
            graph_cache[cid] = get_curriculum_graph(cid) or {}
            node = next(
                (
                    n
                    for n in graph_cache[cid].get("nodes") or []
                    if str(n.get("node_id")) == nid
                ),
                {},
            )
            existing_mapped = [
                str(x).strip() for x in (node.get("mapped_source_ids") or []) if x
            ]
            merged = list(existing_mapped)
            seen = set(merged)
            for sid in new_ids:
                if sid not in seen:
                    seen.add(sid)
                    merged.append(sid)
            patch_curriculum_graph_node(
                cid,
                nid,
                {"mapped_source_ids": merged[:CURRICULUM_DEEP_NODE_MAX_HITS]},
            )
            plan["applied"] = True
            plan["new_source_ids"] = new_ids
            graph_cache[cid] = get_curriculum_graph(cid) or {}
        plans.append(plan)
    return plans


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--curriculum", default="", help="curriculum_id (пусто — все)")
    parser.add_argument("--node", default="", help="node_id (требует --curriculum)")
    parser.add_argument("--apply", action="store_true", help="Применить (по умолчанию dry-run)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.node and not args.curriculum:
        parser.error("--node требует --curriculum")

    plans = asyncio.run(
        run(
            curriculum_id=args.curriculum or None,
            node_id=args.node or None,
            apply=args.apply,
        )
    )
    for p in plans:
        p.pop("_entries", None)

    if args.json:
        print(json.dumps(plans, ensure_ascii=False, indent=2))
        return 0

    print(f"=== BACKFILL verified_source_registry | apply={args.apply} ===")
    total_backfillable = sum(p["backfillable"] for p in plans)
    for p in plans:
        if p["orphaned_urls"] == 0:
            continue
        line = (
            f"  {p['curriculum_id']}/{p['node_id']} | resource_urls={p['resource_urls']} "
            f"already_registered={p['already_registered']} orphaned={p['orphaned_urls']} "
            f"backfillable={p['backfillable']}"
        )
        if p["skipped_no_summary"]:
            line += f" skipped_no_summary={len(p['skipped_no_summary'])}"
        if p.get("applied"):
            line += f" → mapped_source_ids+={p['new_source_ids']}"
        print(line)
    print(f"\nnodes scanned: {len(plans)} | total backfillable URLs: {total_backfillable}")
    if not args.apply and total_backfillable:
        print("re-run with --apply to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
