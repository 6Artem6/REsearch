"""Rich Live + потоковый вывод токенов Ollama."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Iterator, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

_console = Console()
_live: Optional[Live] = None
_status_lines: list[str] = []
_stream_buffer: str = ""
_lock = threading.Lock()

_run_start: Optional[float] = None
_current_phase: str = "—"
_last_refresh: float = 0.0
_MIN_REFRESH_INTERVAL = 0.2  # не дёргать Live чаще (влияние на CPU)
_node_seconds: dict[str, float] = {}
_ollama_seconds: float = 0.0
_ollama_call_count: int = 0


def _elapsed_prefix() -> str:
    if _run_start is None:
        return "[--:--]"
    sec = int(time.monotonic() - _run_start)
    return f"[{sec // 60:02d}:{sec % 60:02d}]"


def begin_run_tracking() -> None:
    """Сброс таймера и фазы (вызывать вместе с init_run_log)."""
    global _run_start, _current_phase, _stream_buffer
    global _node_seconds, _ollama_seconds, _ollama_call_count
    _run_start = time.monotonic()
    _current_phase = "запуск"
    _node_seconds = {}
    _ollama_seconds = 0.0
    _ollama_call_count = 0
    with _lock:
        _status_lines.clear()
        _stream_buffer = ""


def set_phase(phase: str) -> None:
    global _current_phase
    _current_phase = phase
    _refresh_live(force=True)


def _panel_title() -> str:
    elapsed = _elapsed_prefix()
    return f"Knowledge Engine 0.2  │  {elapsed}  │  {_current_phase}"


def _renderable() -> Panel:
    with _lock:
        body = Group(
            Text("\n".join(_status_lines[-18:]), style="cyan"),
            Text(_stream_buffer[-2000:], style="dim white"),
        )
    return Panel(body, title=_panel_title(), border_style="blue")


def _refresh_live(force: bool = False) -> None:
    global _last_refresh
    if _live is None:
        return
    now = time.monotonic()
    if not force and (now - _last_refresh) < _MIN_REFRESH_INTERVAL:
        return
    _last_refresh = now
    _live.update(_renderable(), refresh=True)


def _append_status_line(line: str, permanent_console: bool = False) -> None:
    from knowledge_engine.config import KE_LOG_PLAIN, KE_TRACE_STDOUT

    with _lock:
        _status_lines.append(line)
    to_console = permanent_console or KE_LOG_PLAIN or KE_TRACE_STDOUT
    if to_console and _live is None:
        _console.print(line, highlight=False)
    elif to_console and (_live is not None) and (KE_LOG_PLAIN or KE_TRACE_STDOUT):
        _console.print(line, highlight=False)
    _refresh_live(force=permanent_console)


def set_status(message: str) -> None:
    """Строка в Live-панель, файл .runs и префикс времени."""
    from knowledge_engine.ui.run_log import trace

    trace(f"STATUS | {message}")
    _append_status_line(f"{_elapsed_prefix()} │ {message}")


def on_node_start(node_id: str) -> None:
    from knowledge_engine.ui.run_log import trace

    set_phase(node_id)
    trace(f"NODE ▶ {node_id}")
    _append_status_line(
        f"{_elapsed_prefix()} │ ▶ NODE  {node_id}", permanent_console=True
    )


def on_node_end(node_id: str, elapsed: float, detail: str = "") -> None:
    from knowledge_engine.ui.run_log import trace

    _node_seconds[node_id] = _node_seconds.get(node_id, 0.0) + elapsed
    extra = f" — {detail}" if detail else ""
    trace(f"NODE ✓ {node_id} ({elapsed:.1f}s){extra}")
    line = f"{_elapsed_prefix()} │ ✓ NODE  {node_id} ({elapsed:.1f}s){extra}"
    _append_status_line(line, permanent_console=True)
    set_phase("ожидание…")


def on_ollama_start(model: str, label: str) -> None:
    from knowledge_engine.ui.run_log import trace

    short = label if len(label) <= 56 else label[:53] + "…"
    set_phase(f"Ollama {model}")
    trace(f"OLLAMA ▶ {model} | {label}")
    _append_status_line(f"{_elapsed_prefix()} │ ▶ OLLAMA  {model}  {short}")


def on_ollama_end(model: str, label: str, elapsed: float, ok: bool = True) -> None:
    global _ollama_seconds, _ollama_call_count
    _ollama_seconds += elapsed
    _ollama_call_count += 1
    from knowledge_engine.ui.run_log import trace

    short = label if len(label) <= 48 else label[:45] + "…"
    mark = "✓" if ok else "✗"
    trace(f"OLLAMA {mark} {model} | {label} ({elapsed:.1f}s)")
    _append_status_line(
        f"{_elapsed_prefix()} │ {mark} OLLAMA  {model}  {short} ({elapsed:.1f}s)",
        permanent_console=not ok,
    )


def append_stream_token(token: str) -> None:
    global _stream_buffer
    with _lock:
        _stream_buffer += token
    _refresh_live()


def clear_stream() -> None:
    global _stream_buffer
    with _lock:
        _stream_buffer = ""


@contextmanager
def live_session() -> Iterator[None]:
    """Контекст Rich Live; в Docker (KE_LOG_PLAIN) — только строки в консоль."""
    global _live
    from knowledge_engine.config import KE_LOG_PLAIN

    if _run_start is None:
        begin_run_tracking()
    if KE_LOG_PLAIN:
        yield
        return
    _live = Live(_renderable(), console=_console, refresh_per_second=6)
    with _live:
        yield
    _live = None


def get_console() -> Console:
    return _console


def print_timing_summary() -> None:
    """Сводка времени в консоль и trace-лог."""
    from knowledge_engine.ui.run_log import trace

    if _run_start is None:
        return
    total = time.monotonic() - _run_start
    lines = [
        f"Итого прогона: {total:.1f}s ({total / 60:.1f} min)",
        f"Ollama: {_ollama_seconds:.1f}s ({_ollama_call_count} вызовов)",
    ]
    for name, sec in sorted(_node_seconds.items(), key=lambda x: -x[1]):
        lines.append(f"  узел {name}: {sec:.1f}s")
    other = max(0.0, total - _ollama_seconds - sum(_node_seconds.values()))
    if other > 5:
        lines.append(f"  сеть/Playwright/прочее (оценка): {other:.1f}s")
    block = "\n".join(lines)
    trace("TIMING | " + block.replace("\n", " | "))
    _console.print()
    _console.print("[bold]Время по этапам[/bold]")
    for line in lines:
        _console.print(f"  {line}")
