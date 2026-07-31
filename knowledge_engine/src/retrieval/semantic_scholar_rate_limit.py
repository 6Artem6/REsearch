"""Semantic Scholar: 1 успешный запрос/сек суммарно на все endpoints."""

from __future__ import annotations

import asyncio
import threading
import time

from knowledge_engine.config import SEMANTIC_SCHOLAR_MIN_INTERVAL_SEC

_lock = threading.Lock()
_async_lock: asyncio.Lock | None = None
_last_request_mono = 0.0


def _get_async_lock() -> asyncio.Lock:
    global _async_lock
    if _async_lock is None:
        _async_lock = asyncio.Lock()
    return _async_lock


def _compute_delay() -> float:
    global _last_request_mono
    with _lock:
        now = time.monotonic()
        return max(0.0, SEMANTIC_SCHOLAR_MIN_INTERVAL_SEC - (now - _last_request_mono))


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
    """Async: ждать слот перед любым SS HTTP (без гонок между корутинами)."""
    lock = _get_async_lock()
    async with lock:
        with _lock:
            now = time.monotonic()
            delay = max(
                0.0, SEMANTIC_SCHOLAR_MIN_INTERVAL_SEC - (now - _last_request_mono)
            )
        if delay > 0:
            await asyncio.sleep(delay)
        with _lock:
            _last_request_mono = time.monotonic()


async def semantic_scholar_pause_before_retry_async(min_wait_sec: float) -> None:
    """
    Пауза после 429: минимум min_wait_sec (обычно 1–1.5s) и соблюдение MIN_INTERVAL.
  """
    wait = max(min_wait_sec, SEMANTIC_SCHOLAR_MIN_INTERVAL_SEC)
    lock = _get_async_lock()
    async with lock:
        with _lock:
            now = time.monotonic()
            delay = max(0.0, wait - (now - _last_request_mono))
        if delay > 0:
            await asyncio.sleep(delay)
        with _lock:
            _last_request_mono = time.monotonic()
