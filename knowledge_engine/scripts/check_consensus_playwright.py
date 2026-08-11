"""Диагностика Consensus Playwright: селекторы, submit, сбор ответа (без v0.8 analyze).

Опционально пишет HAR + verbose JSON traffic для reverse-engineering API:
  PYTHONPATH=. python -m knowledge_engine.scripts.check_consensus_playwright \\
    --send --record-har --query "retrieval augmented generation"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KE_TRACE_STDOUT", "1")

from knowledge_engine.config import (
    BROWSER_PROFILE_PATH,
    CONSENSUS_BROWSER_HEADLESS,
    CONSENSUS_HAR_PATH,
)
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
    har_path = None
    if args.record_har:
        har_path = str(Path(args.har_path).expanduser())
        print(f"HAR → {har_path}", flush=True)
    mgr = ConsensusSessionManager(
        headless=args.headless,
        record_har_path=har_path,
        log_json_traffic=True if args.record_har or args.log_json else None,
    )
    out: dict = {
        "profile": str(BROWSER_PROFILE_PATH),
        "headless": args.headless,
        "har_path": har_path,
    }
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
        out["json_traffic"] = getattr(mgr, "_json_traffic_log", [])[:80]
        out["ok"] = True
    except Exception as exc:
        out["ok"] = False
        out["error"] = str(exc)
        raise
    finally:
        await mgr.close()
        if har_path and Path(har_path).is_file():
            out["har_bytes"] = Path(har_path).stat().st_size
            print(f"HAR written: {har_path} ({out['har_bytes']} bytes)", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe Consensus.app Playwright session"
    )
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
    parser.add_argument(
        "--record-har",
        action="store_true",
        help="Записать network HAR (record_har_content=embed) при закрытии context",
    )
    parser.add_argument(
        "--har-path",
        default=str(CONSENSUS_HAR_PATH),
        help=f"Путь HAR (default: {CONSENSUS_HAR_PATH})",
    )
    parser.add_argument(
        "--log-json",
        action="store_true",
        help="Логировать все HTTP 200 application/json (URL/method/preview)",
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
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
