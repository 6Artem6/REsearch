"""arXiv API: one in-flight request at a time, ≥ ARXIV_MIN_INTERVAL_SEC apart.

Official guidance (info.arxiv.org/help/api/user-manual.html): incorporate a
~3 second delay between successive API calls. Local default is 3.25s.

The exclusive gate holds a cross-process flock for the entire HTTP round-trip
(including retries), so concurrent curriculum producers cannot overlap arXiv
calls. Non-arXiv providers keep their own concurrency.
"""

from __future__ import annotations

import asyncio
import fcntl
import os
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Iterator

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


def _begin_exclusive_slot(*, min_wait_sec: float = 0.0):
    """
    flock + wait until max(min_wait, MIN_INTERVAL since last stamp).
    Returns an open locked file; caller must _end_exclusive_slot(f).
    """
    interval = _min_interval()
    min_wait = max(0.0, float(min_wait_sec))
    _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    f = open(_LOCK_PATH, "a+", encoding="utf-8")
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
        return f
    except Exception:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        finally:
            f.close()
        raise


def _end_exclusive_slot(f) -> None:
    """Stamp completion time, unlock, close."""
    try:
        stamped = time.time()
        f.seek(0)
        f.truncate()
        f.write(f"{stamped:.6f}\n")
        f.flush()
        os.fsync(f.fileno())
    finally:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        finally:
            f.close()


@contextmanager
def arxiv_request_exclusive(*, min_wait_sec: float = 0.0) -> Iterator[None]:
    """Hold the global arXiv slot for one request (sync)."""
    with _thread_lock:
        f = _begin_exclusive_slot(min_wait_sec=min_wait_sec)
        try:
            yield
        finally:
            _end_exclusive_slot(f)


@asynccontextmanager
async def arxiv_request_exclusive_async(*, min_wait_sec: float = 0.0):
    """Hold the global arXiv slot for one request (async, cross-process)."""
    lock = _get_async_lock()
    await lock.acquire()
    f = None
    try:
        f = await asyncio.to_thread(_begin_exclusive_slot, min_wait_sec=min_wait_sec)
        yield
    finally:
        if f is not None:
            await asyncio.to_thread(_end_exclusive_slot, f)
        lock.release()


def acquire_arxiv_slot() -> None:
    """Sync: claim and release a slot (interval spacing only; prefer exclusive)."""
    with arxiv_request_exclusive():
        pass


async def acquire_arxiv_slot_async() -> None:
    """Async: claim and release a slot (interval spacing only; prefer exclusive)."""
    async with arxiv_request_exclusive_async():
        pass


async def arxiv_pause_before_retry_async(min_wait_sec: float) -> None:
    """
    Pause after 503/429/403 when not already inside an exclusive request.
    Prefer sleeping inside arxiv_request_exclusive_async instead.
    """
    wait = max(float(min_wait_sec), _min_interval())
    async with arxiv_request_exclusive_async(min_wait_sec=wait):
        pass
