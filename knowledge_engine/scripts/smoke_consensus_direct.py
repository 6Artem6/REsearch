#!/usr/bin/env python3
"""Smoke + timing: fast warmup + cached Consensus Direct search."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

os.environ.setdefault("KE_TRACE_STDOUT", "1")

from knowledge_engine.services.search.consensus_direct_client import (
    ConsensusDirectClient,
    shutdown_consensus_direct_client,
)
from knowledge_engine.services.search.consensus_session_manager import (
    get_consensus_session_manager,
    shutdown_consensus_session_manager,
)


async def _run(query: str) -> dict:
    await shutdown_consensus_direct_client()
    await shutdown_consensus_session_manager()

    client = ConsensusDirectClient()
    out: dict = {"query": query}
    try:
        mgr = await get_consensus_session_manager()

        t0 = time.perf_counter()
        sess = await mgr.get_active_session(force=True)
        warmup_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        papers = await client.search_papers(query, limit=20)
        search1_ms = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        papers2 = await client.search_papers(query + " survey", limit=10)
        search2_ms = (time.perf_counter() - t2) * 1000.0

        t3 = time.perf_counter()
        cached = await mgr.get_active_session()
        cache_hit_ms = (time.perf_counter() - t3) * 1000.0

        out.update(
            {
                "ok": bool(papers) and bool(papers2),
                "warmup_ms": round(warmup_ms, 1),
                "search1_ms": round(search1_ms, 1),
                "search1_papers": len(papers),
                "search2_ms": round(search2_ms, 1),
                "search2_papers": len(papers2),
                "cache_hit_ms": round(cache_hit_ms, 1),
                "cache_age_sec": round(cached.age_sec, 2),
                "cookie_count": len(sess.cookies_dict),
                "cookie_header_len": len(
                    "; ".join(f"{k}={v}" for k, v in sess.cookies_dict.items())
                ),
                "titles": [p.title for p in papers[:5]],
            }
        )
    finally:
        await client.close()
        await shutdown_consensus_direct_client()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        default="retrieval augmented generation vector database",
    )
    args = parser.parse_args()
    result = asyncio.run(_run(args.query))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
