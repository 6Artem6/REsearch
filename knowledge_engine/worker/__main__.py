"""KE Worker: Gemini, анализ, v07, Skill Tree — отдельно от HTTP API."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import knowledge_engine.config as cfg
from knowledge_engine.services.redis_client import (
    get_redis_pubsub_client,
    redis_enabled,
    reset_redis_pubsub_client,
)
from knowledge_engine.services.redis_worker_dispatch import dispatch_task_message
from knowledge_engine.services.work_handlers import (
    process_pending_analysis_jobs,
    process_pending_v07_run,
)
from knowledge_engine.services.work_job_store import (
    recover_stale_running_work_jobs,
    write_worker_heartbeat,
    work_job_store,
)
from knowledge_engine.services.gemini_quota_store import clear_stale_quota_blocks
from knowledge_engine.services.worker_busy import worker_busy_scope
from knowledge_engine.ui.run_log import trace

_job_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ke-worker-job")


def _poll_sec() -> float:
    return cfg.KE_WORKER_POLL_SEC


def _heartbeat_sec() -> float:
    return cfg.KE_WORKER_HEARTBEAT_SEC


def _dispatch_safe(data: dict) -> None:
    try:
        with worker_busy_scope(str(data.get("type") or "task")):
            dispatch_task_message(data)
    except Exception as exc:
        trace(f"WORKER ✗ task dispatch | {exc}")


def _run_poll_loop() -> None:
    pid = os.getpid()
    trace(f"WORKER ▶ poll mode pid={pid}")
    last_hb = 0.0
    while True:
        now = time.monotonic()
        if now - last_hb >= _heartbeat_sec():
            write_worker_heartbeat(pid)
            last_hb = now
        did = False
        job = work_job_store.claim_next_pending()
        if job:
            did = True
            _job_executor.submit(
                _dispatch_safe,
                {"type": "work_job", "id": job.id},
            )
        with worker_busy_scope("poll"):
            if process_pending_analysis_jobs():
                did = True
            elif process_pending_v07_run():
                did = True
        if not did:
            time.sleep(_poll_sec())


def _subscribe_pubsub():
    import redis

    client = get_redis_pubsub_client()
    pubsub = client.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe(cfg.KE_TASKS_CHANNEL)
    return pubsub


def _run_redis_pubsub_loop() -> None:
    import redis

    pid = os.getpid()
    trace(f"WORKER ▶ redis pub/sub pid={pid} channel={cfg.KE_TASKS_CHANNEL}")
    pubsub = _subscribe_pubsub()
    last_hb = 0.0
    while True:
        now = time.monotonic()
        if now - last_hb >= _heartbeat_sec():
            write_worker_heartbeat(pid)
            last_hb = now
        try:
            msg = pubsub.get_message(timeout=1.0)
        except (TimeoutError, redis.exceptions.TimeoutError, redis.exceptions.ConnectionError) as exc:
            trace(f"WORKER redis pubsub read — reconnect | {exc}")
            try:
                pubsub.close()
            except Exception:
                pass
            reset_redis_pubsub_client()
            pubsub = _subscribe_pubsub()
            continue
        if msg and msg.get("type") == "message":
            try:
                data = json.loads(msg["data"])
                _job_executor.submit(_dispatch_safe, data)
            except Exception as exc:
                trace(f"WORKER ✗ bad task message | {exc}")
        else:
            time.sleep(_poll_sec())


def main() -> None:
    cleared = clear_stale_quota_blocks()
    if cleared:
        trace(f"WORKER ▶ quota store cleared {cleared} stale block(s)")
    n = recover_stale_running_work_jobs()
    if n:
        trace(f"WORKER ▶ recovered {n} stale running work job(s)")
    if redis_enabled():
        _run_redis_pubsub_loop()
    else:
        _run_poll_loop()


if __name__ == "__main__":
    main()
