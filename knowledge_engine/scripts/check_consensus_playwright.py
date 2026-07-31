"""Диагностика Consensus Playwright: селекторы, submit, сбор ответа (без v0.8 analyze)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KE_TRACE_STDOUT", "1")

from knowledge_engine.config import CONSENSUS_BROWSER_HEADLESS, BROWSER_PROFILE_PATH, PACKAGE_ROOT
from knowledge_engine.src.retrieval.consensus_session import ConsensusSessionManager


def _profile_locked() -> bool:
    lock = BROWSER_PROFILE_PATH / "Default" / "SingletonLock"
    return lock.is_file()


async def _run(args: argparse.Namespace) -> dict:
    if _profile_locked():
        print(
            "WARN: SingletonLock в profile — другой Chromium уже держит profile. "
            "Остановите make dev / закройте окно Consensus.",
            file=sys.stderr,
        )
    mgr = ConsensusSessionManager(headless=args.headless)
    out: dict = {"profile": str(BROWSER_PROFILE_PATH), "headless": args.headless}
    try:
        await mgr.start()
        page = mgr.page
        assert page is not None
        probe = await page.evaluate(
            """() => {
                const testids = [...document.querySelectorAll('[data-testid]')]
                    .map((e) => e.getAttribute('data-testid'))
                    .filter(Boolean);
                return {
                    url: location.href,
                    title: document.title,
                    testids: testids.slice(0, 120),
                    search_button: !!document.querySelector('button[data-testid="search-button"]'),
                    new_thread_input: !!document.querySelector('[data-testid="new-thread-input"]'),
                };
            }"""
        )
        out["probe_after_start"] = probe
        if args.send:
            q = args.query.strip()
            turn = await mgr.send_message(q)
            out["send"] = {
                "query_len": len(q),
                "response_len": len(turn.raw_text or ""),
                "papers": len(turn.papers),
                "response_preview": (turn.raw_text or "")[:2000],
            }
        out["ok"] = True
    except Exception as exc:
        out["ok"] = False
        out["error"] = str(exc)
        raise
    finally:
        await mgr.close()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Consensus.app Playwright session")
    parser.add_argument(
        "--query",
        default="Efficient lightweight local RAG with vector databases",
        help="Тестовый academic query для --send",
    )
    parser.add_argument(
        "--send",
        action="store_true",
        help="Отправить запрос и ждать ответ (может занять 2–5 мин)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Headless (default: headed, как CONSENSUS_BROWSER_HEADLESS)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not args.headless:
        args.headless = CONSENSUS_BROWSER_HEADLESS

    try:
        result = asyncio.run(_run(args))
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
