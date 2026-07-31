"""Smoke: curriculum source APIs (SS / arXiv / CSE / DDGS) без полной генерации графа."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import httpx

from knowledge_engine.config import (
    CURRICULUM_GOOGLE_CSE_ENABLED,
    GOOGLE_CSE_API_KEY,
    GOOGLE_CSE_ID,
    SEMANTIC_SCHOLAR_API_KEY,
)
from knowledge_engine.services.curriculum_api_quota_store import get_quota_summary
from knowledge_engine.src.curriculum.academic_source_fetch import fetch_academic_sources
from knowledge_engine.src.curriculum.practical_source_fetch import fetch_practical_sources
from knowledge_engine.src.curriculum.search_query_builder import build_search_queries
from knowledge_engine.src.curriculum.source_material_pipeline import collect_sources_by_policy
from knowledge_engine.src.retrieval.semantic_scholar import (
    _SS_SEARCH_URL,
    get_semantic_scholar_paper_by_id,
    search_arxiv_fallback,
    search_semantic_scholar,
)


async def _probe_ss_paper_endpoint(paper_id: str) -> dict:
    status, data = await get_semantic_scholar_paper_by_id(paper_id)
    title = ""
    if data:
        title = str(data.get("title") or "")[:80]
    return {"http": status, "ok": status == 200, "title": title}


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke curriculum source collectors")
    parser.add_argument(
        "--goal",
        default="distributed consensus replication latency reduction",
        help="Цель / expansion vector для запросов",
    )
    parser.add_argument(
        "--policy",
        default="hybrid",
        choices=("hybrid", "practical_only", "academic_only"),
        help="collect_sources_by_policy",
    )
    parser.add_argument(
        "--with-collect",
        action="store_true",
        help="collect_sources_by_policy (отключает Gemini web/grounding по умолчанию)",
    )
    parser.add_argument(
        "--with-playwright",
        action="store_true",
        help="Не отключать Gemini web при --with-collect",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    goal = (args.goal or "").strip()
    if len(goal) < 8:
        print("goal too short", file=sys.stderr)
        return 2

    report: dict = {
        "goal": goal,
        "env": {
            "google_cse_enabled": CURRICULUM_GOOGLE_CSE_ENABLED,
            "google_cse_configured": bool(GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID),
            "semantic_scholar_key_set": bool(SEMANTIC_SCHOLAR_API_KEY),
        },
        "queries": {},
        "semantic_scholar": {},
        "arxiv": {},
        "practical": {},
        "academic_fetch": {},
        "collect_policy": {},
        "quota": get_quota_summary(),
    }

    built = build_search_queries(goal)
    report["queries"] = {
        "academic_query": built.academic_query,
        "practical_query": built.practical_query,
        "keywords": list(built.keywords),
    }

    async def run_async() -> None:
        papers = await search_semantic_scholar(
            built.academic_query,
            limit=3,
            ignore_enabled_flag=True,
        )
        report["semantic_scholar"]["search_papers"] = len(papers)
        report["semantic_scholar"]["samples"] = [
            {
                "title": p.title[:70],
                "source": p.source,
                "has_tldr": bool(p.tldr),
                "abstract_len": len(p.abstract or ""),
            }
            for p in papers[:3]
        ]
        if papers and papers[0].paper_id:
            pid = papers[0].paper_id
            report["semantic_scholar"]["paper_id_probe"] = await _probe_ss_paper_endpoint(pid)
        else:
            async with httpx.AsyncClient(timeout=20.0) as client:
                from knowledge_engine.src.retrieval.semantic_scholar_rate_limit import (
                    acquire_semantic_scholar_slot_async,
                )

                await acquire_semantic_scholar_slot_async()
                r = await client.get(
                    _SS_SEARCH_URL,
                    params={
                        "query": built.academic_query[:80],
                        "limit": 1,
                        "fields": "title,paperId",
                    },
                )
                report["semantic_scholar"]["search_http"] = r.status_code

        arxiv = await search_arxiv_fallback(built.academic_query, limit=2)
        report["arxiv"]["papers"] = len(arxiv)
        report["arxiv"]["samples"] = [p.title[:70] for p in arxiv[:2]]

    asyncio.run(run_async())

    pr = fetch_practical_sources(goal, max_hits=4)
    report["practical"]["hits"] = len(pr)
    report["practical"]["tiers"] = [h.source_tier for h in pr]
    report["practical"]["urls"] = [h.url[:90] for h in pr[:4]]

    ac = fetch_academic_sources(goal)
    report["academic_fetch"]["hits"] = len(ac)
    report["academic_fetch"]["samples"] = [
        {
            "tier": h.source_tier,
            "title": h.title[:60],
            "extracts": len(h.key_extracts),
        }
        for h in ac[:5]
    ]

    if args.with_collect:
        if not args.with_playwright:
            os.environ["CURRICULUM_GEMINI_WEB_HARVEST_ENABLED"] = "false"
            os.environ["CURRICULUM_GEMINI_GROUNDING_ENABLED"] = "false"
        hits = collect_sources_by_policy(
            goal,
            source_policy=args.policy,
            context_vector=goal,
        )
        report["collect_policy"] = {
            "policy": args.policy,
            "total": len(hits),
            "by_tier": {},
        }
        for h in hits:
            t = h.source_tier or "?"
            report["collect_policy"]["by_tier"][t] = (
                report["collect_policy"]["by_tier"].get(t, 0) + 1
            )

    report["quota"] = get_quota_summary()

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"goal: {goal}")
        print(f"CSE enabled: {report['env']['google_cse_enabled']} configured: {report['env']['google_cse_configured']}")
        print(f"SS key set: {report['env']['semantic_scholar_key_set']} (search works without)")
        print(f"queries: academic={built.academic_query[:70]}…")
        print(
            f"SS search: {report['semantic_scholar'].get('search_papers')} papers | "
            f"probe: {report['semantic_scholar'].get('paper_id_probe')}"
        )
        print(f"arXiv: {report['arxiv'].get('papers')} papers")
        print(f"practical fetch: {report['practical']['hits']} ({report['practical']['tiers']})")
        print(f"academic fetch: {report['academic_fetch']['hits']}")
        if report.get("collect_policy"):
            print(f"collect {args.policy}: {report['collect_policy']}")
        print("quota:", report["quota"].get("google_cse"), report["quota"].get("semantic_scholar"))

    ok = (
        report["semantic_scholar"].get("search_papers", 0) > 0
        or report["arxiv"].get("papers", 0) > 0
        or report["practical"]["hits"] > 0
        or report["academic_fetch"]["hits"] > 0
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
