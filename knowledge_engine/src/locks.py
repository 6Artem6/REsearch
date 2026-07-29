"""Knowledge Engine v0.7 — staged UMA resource locking (Ollama ↔ LanceDB)."""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any, AsyncIterator, Callable, TypeVar

# asyncio.Lock deadlock в worker: asyncio.run + sync Gemini/LanceDB в одном процессе.
_uma_thread_lock = threading.RLock()
# Legacy export (threading lock, не asyncio.Lock).
uma_resource_lock = _uma_thread_lock

F = TypeVar("F", bound=Callable[..., Any])


@asynccontextmanager
async def staged_uma_lock() -> AsyncIterator[None]:
    """
    Serialize heavy Ollama 7B inference and LanceDB index/search.
    Rule: never call Ollama inside LanceDB blocks without this lock.
    """
    _uma_thread_lock.acquire()
    try:
        yield
    finally:
        _uma_thread_lock.release()


def staged_uma_lock_decorator(fn: F) -> F:
    """Async decorator wrapping ``staged_uma_lock``."""

    if asyncio.iscoroutinefunction(fn):

        @wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            async with staged_uma_lock():
                return await fn(*args, **kwargs)

        return async_wrapper  # type: ignore[return-value]

    @wraps(fn)
    async def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        async with staged_uma_lock():
            return fn(*args, **kwargs)

    return sync_wrapper  # type: ignore[return-value]


async def run_under_uma_lock(
    sync_fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Run sync Ollama/LanceDB work inside the global UMA lock (from async code)."""
    async with staged_uma_lock():
        return sync_fn(*args, **kwargs)
