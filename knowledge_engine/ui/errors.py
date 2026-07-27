"""Форматирование ошибок с путём и строкой для консоли и trace-лога."""

from __future__ import annotations

import traceback
from pathlib import Path

from knowledge_engine.config import PACKAGE_ROOT

_REPO_ROOT = PACKAGE_ROOT.parent


def format_error_location(exc: BaseException) -> str:
    """Одна строка: тип, сообщение, файл:строка, функция."""
    tb = exc.__traceback__
    if tb is None:
        return f"{type(exc).__name__}: {exc}"

    frames = traceback.extract_tb(tb)
    if not frames:
        return f"{type(exc).__name__}: {exc}"

    frame = frames[-1]
    path = Path(frame.filename)
    try:
        path_str = str(path.relative_to(_REPO_ROOT))
    except ValueError:
        path_str = str(path)

    return (
        f"{type(exc).__name__}: {exc} " f"[{path_str}:{frame.lineno} in {frame.name}]"
    )


def format_error_with_cause(exc: BaseException) -> str:
    base = format_error_location(exc)
    if exc.__cause__ is not None:
        base += f" ← {format_error_location(exc.__cause__)}"
    return base


def trace_exception(exc: BaseException, prefix: str = "") -> str:
    """Записать в run log и вернуть строку для UI."""
    from knowledge_engine.ui.run_log import trace

    detail = format_error_with_cause(exc)
    tag = f"ERROR {prefix}" if prefix else "ERROR"
    trace(f"{tag} | {detail}")
    tb_tail = traceback.format_exc().strip().splitlines()
    if len(tb_tail) > 1:
        for line in tb_tail[-4:]:
            trace(f"  {line}")
    return detail
