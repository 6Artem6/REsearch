"""Один lazy-grounding на (curriculum_id, node_id) — повторный init ждёт, не перезапускает поиск."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from knowledge_engine.config import KE_NODE_DIVE_INIT_GROUNDING_MIN_TIMEOUT_SEC
from knowledge_engine.services.redis_client import get_redis, redis_enabled
from knowledge_engine.ui.run_log import trace


def _lock_key(curriculum_id: str, node_id: str) -> str:
    return f"ke:lock:node_ground:{curriculum_id}:{node_id}"


@asynccontextmanager
async def node_grounding_lock(
    curriculum_id: str,
    node_id: str,
    *,
    wait_sec: float | None = None,
) -> AsyncIterator[bool]:
    """Redis lock: второй init ждёт первого. True = держим lock и можно искать."""
    timeout = (
        wait_sec
        if wait_sec is not None
        else KE_NODE_DIVE_INIT_GROUNDING_MIN_TIMEOUT_SEC
    )
    cid = str(curriculum_id or "").strip()
    nid = str(node_id or "").strip()
    if not cid or not nid or not redis_enabled():
        yield True
        return

    r = get_redis()
    lock_ttl = int(max(timeout, 60)) + 120
    lock = r.lock(_lock_key(cid, nid), timeout=lock_ttl, blocking_timeout=0)
    deadline = time.monotonic() + timeout
    acquired = False
    last_wait_log = 0.0
    while time.monotonic() < deadline:
        if lock.acquire(blocking=False):
            acquired = True
            break
        now = time.monotonic()
        if now - last_wait_log >= 15.0:
            left = max(0.0, deadline - now)
            ttl = -1
            try:
                ttl = int(r.ttl(_lock_key(cid, nid)))
            except Exception:
                pass
            trace(
                f"NODE_DIVE lazy grounding … waiting lock | {cid}/{nid} "
                f"ttl≈{ttl}s left_wait={left:.0f}s"
            )
            last_wait_log = now
        await asyncio.sleep(0.5)

    if not acquired:
        trace(
            f"NODE_DIVE lazy grounding ⊘ | {cid}/{nid} "
            f"grounding lock wait>{timeout:.0f}s — skip duplicate search"
        )
        yield False
        return

    try:
        trace(f"NODE_DIVE lazy grounding lock ✓ | {cid}/{nid}")
        yield True
    finally:
        try:
            lock.release()
        except Exception:
            pass


def force_release_node_grounding_lock(curriculum_id: str, node_id: str) -> bool:
    """Drop stale Redis lock (cancel/kill left it held). Returns True if deleted."""
    cid = str(curriculum_id or "").strip()
    nid = str(node_id or "").strip()
    if not cid or not nid or not redis_enabled():
        return False
    try:
        return bool(get_redis().delete(_lock_key(cid, nid)))
    except Exception:
        return False
