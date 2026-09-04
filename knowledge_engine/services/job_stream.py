"""Replay-safe event log for worker→API SSE (chat-stream, explain-stream).

API never runs the engine: it tails this log. Redis List when Redis is on,
otherwise append-only JSONL under ``.runs/job_streams/``.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator

from knowledge_engine.config import PACKAGE_ROOT
from knowledge_engine.services.redis_client import get_redis, redis_enabled
from knowledge_engine.services.work_job_store import WorkJobStatus, work_job_store

_STREAM_DIR = (PACKAGE_ROOT / ".runs" / "job_streams").resolve()
_RKEY = "ke:jobstream:"
_TTL_SEC = 1800
_SENTINEL = frozenset({"complete", "error"})


def _redis_key(job_id: str) -> str:
    return f"{_RKEY}{job_id}"


def _jsonl_path(job_id: str):
    return _STREAM_DIR / f"{job_id}.jsonl"


def append_job_stream_event(job_id: str, event: dict[str, Any]) -> None:
    jid = (job_id or "").strip()
    if not jid:
        return
    line = json.dumps(event, ensure_ascii=False)
    if redis_enabled():
        r = get_redis()
        key = _redis_key(jid)
        r.rpush(key, line)
        r.expire(key, _TTL_SEC)
        return
    path = _jsonl_path(jid)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def read_job_stream_events(job_id: str, start: int = 0) -> list[dict[str, Any]]:
    jid = (job_id or "").strip()
    if not jid:
        return []
    idx = max(0, int(start))
    raw_lines: list[str] = []
    if redis_enabled():
        rows = get_redis().lrange(_redis_key(jid), idx, -1) or []
        raw_lines = [str(x) for x in rows]
    else:
        path = _jsonl_path(jid)
        if path.is_file():
            all_lines = path.read_text(encoding="utf-8").splitlines()
            raw_lines = all_lines[idx:]
    out: list[dict[str, Any]] = []
    for raw in raw_lines:
        try:
            item = json.loads(raw)
        except Exception:
            continue
        if isinstance(item, dict):
            out.append(item)
    return out


async def iter_job_stream_events(
    job_id: str,
    *,
    timeout_sec: float,
    poll_sec: float = 0.05,
) -> AsyncIterator[dict[str, Any]]:
    """Yield worker SSE payloads until complete/error, job fail, or timeout."""
    jid = (job_id or "").strip()
    deadline = time.monotonic() + max(1.0, float(timeout_sec))
    idx = 0
    while True:
        batch = read_job_stream_events(jid, idx)
        for evt in batch:
            idx += 1
            yield evt
            if str(evt.get("type") or "") in _SENTINEL:
                return
        job = work_job_store.get(jid)
        if job is not None and job.status == WorkJobStatus.FAILED:
            yield {"type": "error", "detail": job.error or "worker job failed"}
            return
        if (
            job is not None
            and job.status == WorkJobStatus.COMPLETED
            and idx == 0
            and job.result is not None
        ):
            yield {"type": "complete", "result": job.result}
            return
        if time.monotonic() >= deadline:
            yield {"type": "error", "detail": "worker stream timeout"}
            return
        await asyncio.sleep(max(0.02, float(poll_sec)))
