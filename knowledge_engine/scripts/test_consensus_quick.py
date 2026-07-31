"""Smoke: Consensus /quick/?q= + модал Find papers + Load more (headed Playwright)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

os.environ.setdefault("KE_TRACE_STDOUT", "1")

from knowledge_engine.config import (
    CONSENSUS_USE_QUICK_PAPER_SEARCH,
    BROWSER_PROFILE_PATH,
)
from knowledge_engine.src.retrieval.consensus_session import ConsensusSessionManager


async def main_async(query: str, headless: bool) -> dict:
    if not CONSENSUS_USE_QUICK_PAPER_SEARCH:
        print("WARN: CONSENSUS_USE_QUICK_PAPER_SEARCH=false — включите для этого теста")
    mgr = ConsensusSessionManager(headless=headless)
    out: dict = {
        "profile": str(BROWSER_PROFILE_PATH),
        "quick_mode": CONSENSUS_USE_QUICK_PAPER_SEARCH,
        "query": query,
    }
    try:
        await mgr.start()
        turn = await mgr.send_message(query)
        papers = turn.papers or []
        out["papers"] = len(papers)
        out["response_len"] = len(turn.raw_text or "")
        out["sample_papers"] = [
            {
                "title": (p.title or "")[:120],
                "url": (p.source_url or "")[:120],
                "abstract_len": len((p.abstract or p.tldr or "")),
            }
            for p in papers[:12]
        ]
        out["response_preview"] = (turn.raw_text or "")[:1500]
        out["ok"] = True
    except Exception as exc:
        out["ok"] = False
        out["error"] = str(exc)
        raise
    finally:
        await mgr.close()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Consensus quick paper search")
    parser.add_argument(
        "--query",
        default="Efficient lightweight local RAG with vector databases",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = asyncio.run(main_async(args.query.strip(), args.headless))
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
