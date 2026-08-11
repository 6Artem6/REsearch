"""Named thread pools for blocking sync work in async pipelines.

``asyncio.to_thread`` uses a single small default executor. Long LanceDB collect,
CE/MMR rerank, and Gemini HTTP in threads can exhaust it and stall later steps
(e.g. dense lecture after RAG) even when the event loop is idle.

Use :func:`run_blocking` / :func:`run_blocking_timed` with an appropriate pool
instead of ``asyncio.to_thread`` in request-scoped orchestration code.
"""

from __future__ import annotations

import asyncio
import functools
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

_REGISTRY: dict[str, ThreadPoolExecutor] = {}


def _executor(key: str, default_workers: int, prefix: str) -> ThreadPoolExecutor:
    if key not in _REGISTRY:
        env_key = f"KE_POOL_{key.upper()}_WORKERS"
        workers = int(os.getenv(env_key, str(default_workers)))
        workers = max(1, min(workers, 32))
        _REGISTRY[key] = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix=prefix,
        )
    return _REGISTRY[key]


def pool_light() -> ThreadPoolExecutor:
    """Fast sync prep (graph enrich, small file/JSON reads)."""
    return _executor("light", 8, "ke-pool-light")


def pool_rag_io() -> ThreadPoolExecutor:
    """LanceDB / hybrid document collect (often 5–90s)."""
    return _executor("rag_io", 4, "ke-pool-rag-io")


def pool_rag_ce() -> ThreadPoolExecutor:
    """Cross-encoder rerank and embedding MMR."""
    return _executor("rag_ce", 2, "ke-pool-rag-ce")


def pool_llm_sync() -> ThreadPoolExecutor:
    """Blocking Gemini: step_analysis, tutor chat, fact_manifest в pipeline."""
    return _executor("llm_sync", 3, "ke-pool-llm")


def pool_llm_lecture() -> ThreadPoolExecutor:
    """Только dense_material / догенерация лекции (изолировано от step pipeline)."""
    return _executor("llm_lecture", 2, "ke-pool-llm-lecture")


def pool_net_sync() -> ThreadPoolExecutor:
    """Sync HTTP SDKs (Exa, etc.)."""
    return _executor("net_sync", 4, "ke-pool-net")


async def run_blocking(
    executor: ThreadPoolExecutor,
    fn: Callable[P, R],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> R:
    loop = asyncio.get_running_loop()
    if kwargs:
        fn = functools.partial(fn, **kwargs)
    return await loop.run_in_executor(executor, fn, *args)


async def run_blocking_timed(
    executor: ThreadPoolExecutor,
    timeout_sec: float,
    fn: Callable[P, R],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> R:
    loop = asyncio.get_running_loop()
    if kwargs:
        fn = functools.partial(fn, **kwargs)
    fut = loop.run_in_executor(executor, fn, *args)
    return await asyncio.wait_for(fut, timeout=timeout_sec)
