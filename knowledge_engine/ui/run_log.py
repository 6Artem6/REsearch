"""Файловый trace прогона: фазы графа, Ollama, статусы (дополняет Rich Live)."""

from __future__ import annotations

import re
import time
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


def _slug(text: str, max_len: int = 36) -> str:
    s = re.sub(r"[^\w\-]+", "-", text.strip(), flags=re.UNICODE)
    s = re.sub(r"-+", "-", s).strip("-")
    return (s[:max_len] or "run").lower()


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
    from knowledge_engine.services.redis_run_log import (
        init_log,
        redis_logs_enabled,
    )

    if redis_logs_enabled():
        init_log(log_id, header)
    else:
        _log_path.write_text(header, encoding="utf-8")
    return _log_path


def get_run_log_path() -> Path | None:
    return _log_path


def trace(message: str) -> None:
    """Redis или файл .runs; при KE_TRACE_STDOUT — stdout."""
    from knowledge_engine.config import KE_TRACE_STDOUT
    from knowledge_engine.services.redis_run_log import (
        append_line,
        redis_logs_enabled,
    )

    line = f"{datetime.now().strftime('%H:%M:%S')} | {message}"
    if "\n" in message:
        if KE_TRACE_STDOUT:
            print(message, flush=True)
    elif KE_TRACE_STDOUT:
        print(line, flush=True)
    if _log_redis_id and redis_logs_enabled():
        append_line(_log_redis_id, line if "\n" not in message else message)
    path = _log_path
    if path is None:
        return
    if redis_logs_enabled() and "\n" not in message:
        return
    with _lock:
        with path.open("a", encoding="utf-8") as f:
            if "\n" in message:
                f.write(message + "\n")
            else:
                f.write(line + "\n")


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


def ollama_invoke(llm: Any, messages: list[Any], label: str) -> Any:
    from knowledge_engine.ui.logger import on_ollama_end, on_ollama_start

    model = getattr(llm, "model", None) or getattr(llm, "model_name", "ollama")
    on_ollama_start(str(model), label)
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
                raw = result.content if isinstance(result.content, str) else str(
                    result.content
                )
            else:
                raw = str(result)
            trace_llm_messages(label, messages, raw, model=str(model))
        on_ollama_end(str(model), label, time.monotonic() - t0, ok=True)
        return result
    except Exception as exc:
        from knowledge_engine.ui.errors import format_error_with_cause

        on_ollama_end(str(model), label, time.monotonic() - t0, ok=False)
        trace(f"OLLAMA ✗ {model} | {label} — {format_error_with_cause(exc)}")
        raise
