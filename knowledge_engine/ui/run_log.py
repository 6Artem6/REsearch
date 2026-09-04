"""Файловый trace прогона: фазы графа, Gemma Cloud, статусы (дополняет Rich Live)."""

from __future__ import annotations

import logging
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from knowledge_engine.config import PACKAGE_ROOT

RUN_LOG_DIR: Path = (PACKAGE_ROOT / ".runs").resolve()

_lock = Lock()
_log_path: Path | None = None
_log_redis_id: str | None = None
_node_stack: list[tuple[str, float]] = []

_trace_q: queue.SimpleQueue[_TraceJob | None] = queue.SimpleQueue()
_redis_q: queue.SimpleQueue[tuple[str, str] | None] = queue.SimpleQueue()
_trace_worker_lock = threading.Lock()
_trace_worker_started = False
_redis_worker_started = False
_trace_stdout: bool | None = None


@dataclass(frozen=True)
class _TraceJob:
    message: str
    line: str
    payload: str
    multiline: bool
    stdout: bool
    log_id: str | None
    use_redis: bool
    skip_file: bool
    path: Path | None
    caller_file: str
    caller_line: int
    caller_func: str


def _ke_trace_stdout() -> bool:
    global _trace_stdout
    if _trace_stdout is None:
        from knowledge_engine.config import KE_TRACE_STDOUT

        _trace_stdout = bool(KE_TRACE_STDOUT)
    return _trace_stdout


def _redis_logs_on() -> bool:
    from knowledge_engine.services.redis_run_log import redis_logs_enabled

    return redis_logs_enabled()


def _slug(text: str, max_len: int = 36) -> str:
    s = re.sub(r"[^\w\-]+", "-", text.strip(), flags=re.UNICODE)
    s = re.sub(r"-+", "-", s).strip("-")
    return (s[:max_len] or "run").lower()


def _redis_worker_loop() -> None:
    from knowledge_engine.services.redis_run_log import append_line

    while True:
        item = _redis_q.get()
        if item is None:
            break
        log_id, payload = item
        try:
            append_line(log_id, payload)
        except Exception:
            pass


def _ensure_redis_worker() -> None:
    global _redis_worker_started
    if _redis_worker_started:
        return
    with _trace_worker_lock:
        if _redis_worker_started:
            return
        threading.Thread(
            target=_redis_worker_loop,
            name="ke-trace-redis",
            daemon=True,
        ).start()
        _redis_worker_started = True


def _trace_worker_loop() -> None:
    from knowledge_engine.logging_setup import trace_mirror_logger

    while True:
        job = _trace_q.get()
        if job is None:
            break
        # Re-checked per job (cheap — trace_mirror_logger() caches internally);
        # keeps this responsive to LOG_TO_FILE being set after the worker starts.
        # The whole block is guarded: an exception here (e.g. trace_mirror_logger()
        # failing to create its log dir) must never kill this single worker thread —
        # that would silently stop ALL trace() output (file, stdout, Redis), not
        # just the mirror write, since every sink is served by this one loop.
        try:
            std_logger = trace_mirror_logger()
            if std_logger is not None:
                # Attribute the record to the real trace() call site (captured on
                # the calling thread) instead of this worker loop's own frame —
                # stacklevel can't do it here since emission happens on a
                # different thread than the call, with its own stack.
                record = std_logger.makeRecord(
                    std_logger.name,
                    logging.INFO,
                    job.caller_file,
                    job.caller_line,
                    job.message,
                    (),
                    None,
                    func=job.caller_func,
                )
                std_logger.handle(record)
        except Exception:
            pass
        try:
            if job.stdout:
                if job.multiline:
                    print(job.message, flush=True)
                else:
                    print(job.line, flush=True)
            if job.use_redis and job.log_id:
                _ensure_redis_worker()
                try:
                    _redis_q.put_nowait((job.log_id, job.payload))
                except Exception:
                    pass
            if job.path is None or job.skip_file:
                continue
            if not _lock.acquire(timeout=0.5):
                continue
            try:
                with job.path.open("a", encoding="utf-8") as f:
                    if job.multiline:
                        f.write(job.message + "\n")
                    else:
                        f.write(job.line + "\n")
            finally:
                _lock.release()
        except Exception:
            pass


def _ensure_trace_worker() -> None:
    global _trace_worker_started
    if _trace_worker_started:
        return
    with _trace_worker_lock:
        if _trace_worker_started:
            return
        threading.Thread(
            target=_trace_worker_loop,
            name="ke-trace-worker",
            daemon=True,
        ).start()
        _trace_worker_started = True


def init_run_log(title: str) -> Path:
    """Создать лог прогона (Redis list или файл .runs)."""
    global _log_path, _node_stack, _log_redis_id
    from knowledge_engine.ui.logger import begin_run_tracking

    begin_run_tracking()
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_id = f"{stamp}-{_slug(title)}"
    _log_redis_id = log_id
    _log_path = RUN_LOG_DIR / f"{log_id}.log"
    _node_stack = []
    header = (
        f"Knowledge Engine run — {datetime.now().isoformat(timespec='seconds')}\n"
        f"Task: {title}\n"
        f"Log id: {log_id}\n"
        "---\n"
    )
    from knowledge_engine.services.redis_run_log import init_log, redis_logs_enabled

    if redis_logs_enabled():
        init_log(log_id, header)
    else:
        _log_path.write_text(header, encoding="utf-8")
    return _log_path


def get_run_log_path() -> Path | None:
    return _log_path


def trace(message: str) -> None:
    """SimpleQueue + один worker; только put_nowait на вызывающем потоке."""
    _ensure_trace_worker()
    line = f"{datetime.now().strftime('%H:%M:%S')} | {message}"
    multiline = "\n" in message
    payload = line if not multiline else message
    log_id = _log_redis_id
    path = _log_path
    use_redis = bool(log_id and _redis_logs_on())
    skip_file = use_redis and not multiline
    # Captured here (on the caller's thread/stack) — the worker thread that
    # eventually logs this has its own stack, so stacklevel can't reach back
    # to the real trace(...) call site from there.
    caller = sys._getframe(1)
    _trace_q.put_nowait(
        _TraceJob(
            message=message,
            line=line,
            payload=payload,
            multiline=multiline,
            stdout=_ke_trace_stdout(),
            log_id=log_id,
            use_redis=use_redis,
            skip_file=skip_file,
            path=path,
            caller_file=caller.f_code.co_filename,
            caller_line=caller.f_lineno,
            caller_func=caller.f_code.co_name,
        )
    )


def node_start(node_id: str) -> None:
    from knowledge_engine.ui.logger import on_node_start

    _node_stack.append((node_id, time.monotonic()))
    on_node_start(node_id)


def node_end(node_id: str, detail: str = "") -> None:
    from knowledge_engine.ui.logger import on_node_end

    elapsed = 0.0
    if _node_stack and _node_stack[-1][0] == node_id:
        _, t0 = _node_stack.pop()
        elapsed = time.monotonic() - t0
    on_node_end(node_id, elapsed, detail)


def gemma_cloud_invoke(llm: Any, messages: list[Any], label: str) -> Any:
    from knowledge_engine.ui.logger import on_gemma_cloud_end, on_gemma_cloud_start

    model = getattr(llm, "model", None) or getattr(llm, "model_name", "gemma-cloud")
    on_gemma_cloud_start(str(model), label)
    t0 = time.monotonic()
    try:
        result = llm.invoke(messages)
        from knowledge_engine.config import KE_LLM_FULL_TRACE
        from knowledge_engine.ui.llm_trace import trace_llm_messages

        if KE_LLM_FULL_TRACE:
            raw = result
            if hasattr(result, "model_dump_json"):
                raw = result.model_dump_json(indent=2)
            elif hasattr(result, "content"):
                raw = (
                    result.content
                    if isinstance(result.content, str)
                    else str(result.content)
                )
            else:
                raw = str(result)
            trace_llm_messages(label, messages, raw, model=str(model))
        on_gemma_cloud_end(str(model), label, time.monotonic() - t0, ok=True)
        return result
    except Exception as exc:
        from knowledge_engine.ui.errors import format_error_with_cause

        on_gemma_cloud_end(str(model), label, time.monotonic() - t0, ok=False)
        trace(f"GEMMA_CLOUD ✗ {model} | {label} — {format_error_with_cause(exc)}")
        raise


# Historical name — same Gemma Cloud path.
ollama_invoke = gemma_cloud_invoke


_ensure_trace_worker()
