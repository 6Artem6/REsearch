"""KE Worker: Gemini, анализ, v07, Skill Tree — отдельно от HTTP API."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import knowledge_engine.config as cfg
from knowledge_engine.services.gemini_quota_store import clear_stale_quota_blocks
from knowledge_engine.services.redis_client import (
    get_redis_pubsub_client,
    redis_enabled,
    reset_redis_command_client,
    reset_redis_pubsub_client,
)
from knowledge_engine.services.redis_worker_dispatch import dispatch_task_message
from knowledge_engine.services.work_handlers import (
    process_pending_analysis_jobs,
    process_pending_v07_run,
)
from knowledge_engine.services.work_job_store import (
    recover_stale_running_work_jobs,
    republish_pending_work_jobs,
    requeue_running_work_jobs_on_startup,
    work_job_store,
    write_worker_heartbeat,
)
from knowledge_engine.services.worker_busy import (
    clear_worker_busy_file,
    worker_busy_scope,
)
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
    client = get_redis_pubsub_client()
    pubsub = client.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe(cfg.KE_TASKS_CHANNEL)
    return pubsub


def _safe_redis_command(label: str, fn, default=None):
    """Redis command failures must not terminate the long-running worker."""
    import redis

    errors = (
        redis.exceptions.TimeoutError,
        redis.exceptions.ConnectionError,
        OSError,
    )
    try:
        return fn()
    except errors as exc:
        trace(f"WORKER redis command {label} — reconnect | {exc}")
        reset_redis_command_client()
        try:
            return fn()
        except errors as exc2:
            trace(f"WORKER redis command {label} — failed after reconnect | {exc2}")
            reset_redis_command_client()
            return default


def _reconnect_pubsub(pubsub):
    """Close an invalid subscription and retry connection without killing worker."""
    import redis

    if pubsub is not None:
        try:
            pubsub.close()
        except Exception:
            pass
    reset_redis_pubsub_client()
    while True:
        try:
            return _subscribe_pubsub()
        except (
            redis.exceptions.TimeoutError,
            redis.exceptions.ConnectionError,
            OSError,
        ) as exc:
            trace(f"WORKER redis pubsub connect — retry | {exc}")
            time.sleep(max(1.0, _poll_sec()))


def _run_redis_pubsub_loop() -> None:
    import redis

    pid = os.getpid()
    trace(f"WORKER ▶ redis pub/sub pid={pid} channel={cfg.KE_TASKS_CHANNEL}")
    pubsub = _reconnect_pubsub(None)
    last_hb = 0.0
    # Pub/sub is not durable: reclaim pending jobs left after restart / missed notify.
    drain_every = max(5.0, _poll_sec() * 10.0)
    last_drain = 0.0
    while True:
        now = time.monotonic()
        if now - last_hb >= _heartbeat_sec():
            _safe_redis_command("heartbeat", lambda: write_worker_heartbeat(pid))
            last_hb = time.monotonic()
        if now - last_drain >= drain_every:
            n_pending = _safe_redis_command(
                "republish pending",
                republish_pending_work_jobs,
                default=0,
            )
            if n_pending:
                trace(f"WORKER ↻ drain pending | n={n_pending}")
            # Stamp AFTER the call so a slow timeout does not immediately re-enter drain.
            last_drain = time.monotonic()
        try:
            msg = pubsub.get_message(timeout=1.0)
        except (
            redis.exceptions.TimeoutError,
            redis.exceptions.ConnectionError,
            OSError,
        ) as exc:
            trace(f"WORKER redis pubsub read — reconnect | {exc}")
            pubsub = _reconnect_pubsub(pubsub)
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
    # Drop stale busy from a killed process so watch/reload is not stuck forever.
    if clear_worker_busy_file():
        trace("WORKER ▶ cleared stale worker_dev_busy.json")
    # Previous worker is gone: RUNNING jobs are orphans — requeue, don't wait 2h.
    n_requeue = requeue_running_work_jobs_on_startup()
    if n_requeue:
        trace(f"WORKER ▶ startup requeue running→pending | n={n_requeue}")
    n = recover_stale_running_work_jobs()
    if n:
        trace(f"WORKER ▶ recovered {n} stale running work job(s)")
    if redis_enabled():
        n_pending = _safe_redis_command(
            "startup republish pending",
            republish_pending_work_jobs,
            default=0,
        )
        if n_pending:
            trace(f"WORKER ▶ startup republish pending | n={n_pending}")
        _run_redis_pubsub_loop()
    else:
        _run_poll_loop()


if __name__ == "__main__":
    main()
