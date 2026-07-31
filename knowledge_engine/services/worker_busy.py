"""Сигнал для dev worker watch: не перезапускать worker во время задачи."""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from knowledge_engine.config import PACKAGE_ROOT

_BUSY_PATH = (PACKAGE_ROOT / ".runs" / "worker_dev_busy.json").resolve()
_lock = threading.Lock()
_active = 0


def _write_state() -> None:
    _BUSY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _active <= 0:
        try:
            _BUSY_PATH.unlink(missing_ok=True)
        except OSError:
            pass
        return
    payload = {
        "busy": True,
        "count": _active,
        "pid": os.getpid(),
        "updated_at": time.time(),
    }
    _BUSY_PATH.write_text(json.dumps(payload), encoding="utf-8")


@contextmanager
def worker_busy_scope(label: str = "") -> Iterator[None]:
    global _active
    with _lock:
        _active += 1
        _write_state()
    try:
        yield
    finally:
        with _lock:
            _active = max(0, _active - 1)
            _write_state()


def worker_busy_for_reload() -> bool:
    """True — dev watch должен отложить reload (другой процесс читает файл)."""
    try:
        if _BUSY_PATH.is_file():
            data = json.loads(_BUSY_PATH.read_text(encoding="utf-8"))
            if int(data.get("count") or 0) > 0:
                return True
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        pass
    try:
        from knowledge_engine.services.work_job_store import count_running_work_jobs

        return count_running_work_jobs() > 0
    except Exception:
        return False
