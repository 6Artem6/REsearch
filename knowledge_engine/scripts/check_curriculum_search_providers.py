"""Проверка поисковиков curriculum: arXiv, SS, SearXNG, CSE (опц.), DDGS (опц.)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import httpx

from knowledge_engine.config import (
    CURRICULUM_GOOGLE_CSE_ENABLED,
    CURRICULUM_PRACTICAL_DDGS_ENABLED,
    SEARXNG_ENABLED,
    SEMANTIC_SCHOLAR_API_KEY,
)
from knowledge_engine.services.search.searxng_health import check_searxng
from knowledge_engine.src.curriculum.practical_searxng_search import collect_searxng_practical_rows
from knowledge_engine.src.curriculum.search_query_builder import build_search_queries
from knowledge_engine.src.retrieval.semantic_scholar import (
    search_arxiv_fallback,
    search_semantic_scholar,
)

_ARXIV_PROBE_URL = (
    "https://export.arxiv.org/api/query?search_query=all:kafka&max_results=1"
)


async def _probe_arxiv() -> dict:
    try:
        async with httpx.AsyncClient(timeout=25.0, follow_redirects=True) as client:
            resp = await client.get(_ARXIV_PROBE_URL)
            body = (resp.text or "")[:500]
            ok = resp.status_code == 200 and "<feed" in body
            return {
                "ok": ok,
                "http": resp.status_code,
                "atom": "<feed" in body,
            }
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}


async def _probe_semantic_scholar(query: str) -> dict:
    papers = await search_semantic_scholar(query, limit=2, ignore_enabled_flag=True)
    return {
        "ok": len(papers) > 0,
        "papers": len(papers),
        "samples": [p.title[:60] for p in papers[:2]],
    }


async def _probe_searxng_practical(goal: str) -> dict:
    if not SEARXNG_ENABLED:
        return {"ok": False, "skipped": "SEARXNG_ENABLED=false"}
    ok_health, msg = check_searxng()
    rows = await collect_searxng_practical_rows(goal, limit=4)
    return {
        "ok": ok_health and len(rows) > 0,
        "health": msg,
        "hits": len(rows),
        "urls": [r["url"][:70] for r in rows[:3]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Curriculum search providers probe")
    parser.add_argument(
        "--goal",
        default="kafka replication consensus",
        help="Тестовая цель для практического/академического query",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    goal = (args.goal or "").strip()
    if len(goal) < 8:
        print("goal too short", file=sys.stderr)
        return 2

    built = build_search_queries(goal)
    academic_q = built.academic_query

    async def run() -> dict:
        arxiv_api = await _probe_arxiv()
        arxiv_search = await search_arxiv_fallback(academic_q, limit=2)
        ss = await _probe_semantic_scholar(academic_q)
        searxng = await _probe_searxng_practical(goal)
        return {
            "goal": goal,
            "queries": {
                "academic": academic_q,
                "practical_sample": built.practical_query[:120],
            },
            "arxiv": {
                "api_probe": arxiv_api,
                "search_papers": len(arxiv_search),
                "ok": arxiv_api.get("ok") and len(arxiv_search) > 0,
            },
            "semantic_scholar": {
                **ss,
                "key_set": bool(SEMANTIC_SCHOLAR_API_KEY),
            },
            "searxng": searxng,
            "google_cse": {
                "enabled": CURRICULUM_GOOGLE_CSE_ENABLED,
                "used_in_pipeline": CURRICULUM_GOOGLE_CSE_ENABLED,
                "ok": not CURRICULUM_GOOGLE_CSE_ENABLED,
                "note": "disabled by default (billing); enable CURRICULUM_GOOGLE_CSE_ENABLED",
            },
            "ddgs": {
                "enabled": CURRICULUM_PRACTICAL_DDGS_ENABLED,
                "used_in_pipeline": CURRICULUM_PRACTICAL_DDGS_ENABLED,
                "ok": True,
                "note": "off by default; SearXNG primary",
            },
            "pipeline_ready": {
                "academic": ss.get("ok") or len(arxiv_search) > 0,
                "practical": searxng.get("ok"),
            },
        }

    report = asyncio.run(run())

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"goal: {goal}")
        print(
            f"arXiv: api={report['arxiv']['api_probe']} "
            f"search_papers={report['arxiv']['search_papers']}"
        )
        print(f"Semantic Scholar: {report['semantic_scholar']}")
        print(f"SearXNG: {report['searxng']}")
        print(f"Google CSE: {report['google_cse']}")
        print(f"DDGS: {report['ddgs']}")
        print(f"pipeline_ready: {report['pipeline_ready']}")

    academic_ok = report["pipeline_ready"]["academic"]
    practical_ok = report["pipeline_ready"]["practical"]
    if academic_ok and practical_ok:
        print("\n✓ Можно проверять в деле: smoke + curriculum generate")
        print("  python -m knowledge_engine.scripts.smoke_curriculum_sources --with-collect --policy hybrid")
        return 0
    if academic_ok or practical_ok:
        print("\n~ Частично: можно generate с academic_only или practical_only")
        return 0
    print("\n✗ Ни академика ни практика — поднять SearXNG / SS / arXiv")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
