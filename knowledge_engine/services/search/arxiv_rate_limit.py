"""arXiv API: strict ≥ ARXIV_MIN_INTERVAL_SEC between requests across processes.

Official guidance: incorporate a ~3s delay between successive API calls.
Local enforcement uses ARXIV_MIN_INTERVAL_SEC (default 3.25) via a
cross-process file lock so API + worker cannot race.
"""

from __future__ import annotations

import asyncio
import fcntl
import os
import threading
import time
from pathlib import Path

from knowledge_engine.config import ARXIV_MIN_INTERVAL_SEC, PACKAGE_ROOT

_LOCK_PATH: Path = (PACKAGE_ROOT / ".runs" / "arxiv_rate_lock").resolve()

_thread_lock = threading.Lock()
_async_lock: asyncio.Lock | None = None
_async_lock_loop: asyncio.AbstractEventLoop | None = None


def _get_async_lock() -> asyncio.Lock:
    global _async_lock, _async_lock_loop
    loop = asyncio.get_running_loop()
    if _async_lock is None or _async_lock_loop is not loop:
        _async_lock = asyncio.Lock()
        _async_lock_loop = loop
    return _async_lock


def _min_interval() -> float:
    return max(0.0, float(ARXIV_MIN_INTERVAL_SEC))


def _acquire_cross_process_slot(*, min_wait_sec: float = 0.0) -> None:
    """
    Exclusive flock + wall-clock timestamp in .runs/.

    Guarantees: next mark is at least max(min_wait_sec, MIN_INTERVAL since last mark)
    across all processes sharing PACKAGE_ROOT.
    """
    interval = _min_interval()
    min_wait = max(0.0, float(min_wait_sec))
    _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_LOCK_PATH, "a+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            raw = (f.read() or "").strip()
            last = float(raw) if raw else 0.0
            now = time.time()
            delay_from_last = max(0.0, interval - (now - last)) if last > 0.0 else 0.0
            delay = max(min_wait, delay_from_last)
            if delay > 0.0:
                time.sleep(delay)
            stamped = time.time()
            f.seek(0)
            f.truncate()
            f.write(f"{stamped:.6f}\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def acquire_arxiv_slot() -> None:
    """Sync: wait for the global arXiv slot before any HTTP."""
    with _thread_lock:
        _acquire_cross_process_slot(min_wait_sec=0.0)


async def acquire_arxiv_slot_async() -> None:
    """Async: wait for the global arXiv slot (no in-process coroutine races)."""
    lock = _get_async_lock()
    async with lock:
        await asyncio.to_thread(_acquire_cross_process_slot, min_wait_sec=0.0)


async def arxiv_pause_before_retry_async(min_wait_sec: float) -> None:
    """
    Pause after 503/429/403: at least min_wait_sec and still respect MIN_INTERVAL.
    """
    wait = max(float(min_wait_sec), _min_interval())
    lock = _get_async_lock()
    async with lock:
        await asyncio.to_thread(_acquire_cross_process_slot, min_wait_sec=wait)
