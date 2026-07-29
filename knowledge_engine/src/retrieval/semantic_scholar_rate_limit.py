"""Semantic Scholar: 1 успешный запрос/сек суммарно на все endpoints."""

from __future__ import annotations

import asyncio
import threading
import time

from knowledge_engine.config import SEMANTIC_SCHOLAR_MIN_INTERVAL_SEC

_lock = threading.Lock()
_last_request_mono = 0.0


def _compute_delay() -> float:
    global _last_request_mono
    with _lock:
        now = time.monotonic()
        delay = max(0.0, SEMANTIC_SCHOLAR_MIN_INTERVAL_SEC - (now - _last_request_mono))
        return delay


def _mark_request() -> None:
    global _last_request_mono
    with _lock:
        _last_request_mono = time.monotonic()


def acquire_semantic_scholar_slot() -> None:
    """Sync: ждать слот перед любым SS HTTP."""
    delay = _compute_delay()
    if delay > 0:
        time.sleep(delay)
    _mark_request()


async def acquire_semantic_scholar_slot_async() -> None:
    """Async: ждать слот перед любым SS HTTP."""
    delay = _compute_delay()
    if delay > 0:
        await asyncio.sleep(delay)
    _mark_request()
