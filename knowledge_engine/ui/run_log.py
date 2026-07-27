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
_node_stack: list[tuple[str, float]] = []


def _slug(text: str, max_len: int = 36) -> str:
    s = re.sub(r"[^\w\-]+", "-", text.strip(), flags=re.UNICODE)
    s = re.sub(r"-+", "-", s).strip("-")
    return (s[:max_len] or "run").lower()


def init_run_log(title: str) -> Path:
    """Создать новый лог-файл для прогона analyze."""
    global _log_path, _node_stack
    from knowledge_engine.ui.logger import begin_run_tracking

    begin_run_tracking()
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    _log_path = RUN_LOG_DIR / f"{stamp}-{_slug(title)}.log"
    _node_stack = []
    header = (
        f"Knowledge Engine run — {datetime.now().isoformat(timespec='seconds')}\n"
        f"Task: {title}\n"
        f"Log file: {_log_path}\n"
        "---\n"
    )
    _log_path.write_text(header, encoding="utf-8")
    return _log_path


def get_run_log_path() -> Path | None:
    return _log_path


def trace(message: str) -> None:
    """Файл .runs; при KE_TRACE_STDOUT — также stdout (docker logs)."""
    from knowledge_engine.config import KE_TRACE_STDOUT

    line = f"{datetime.now().strftime('%H:%M:%S')} | {message}"
    if KE_TRACE_STDOUT:
        print(line, flush=True)
    path = _log_path
    if path is None:
        return
    with _lock:
        with path.open("a", encoding="utf-8") as f:
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
        on_ollama_end(str(model), label, time.monotonic() - t0, ok=True)
        return result
    except Exception as exc:
        from knowledge_engine.ui.errors import format_error_with_cause

        on_ollama_end(str(model), label, time.monotonic() - t0, ok=False)
        trace(f"OLLAMA ✗ {model} | {label} — {format_error_with_cause(exc)}")
        raise
