"""Auto-restart knowledge_engine.worker on Python changes (make dev)."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KE = ROOT / "knowledge_engine"

# Совпадает с uvicorn reload_dirs в api/__main__.py + worker
WATCH_DIRS: tuple[Path, ...] = tuple(
    p
    for p in (
        KE / "api",
        KE / "graph",
        KE / "services",
        KE / "nodes",
        KE / "ui",
        KE / "src",
        KE / "worker",
    )
    if p.is_dir()
)
WATCH_FILES: tuple[Path, ...] = tuple(
    p for p in (KE / "config.py", KE / "__init__.py") if p.is_file()
)


def _should_reload(paths: set[str]) -> bool:
    for raw in paths:
        p = raw.replace("\\", "/")
        if not p.endswith(".py"):
            continue
        if "__pycache__" in p or "/.runs/" in p:
            continue
        return True
    return False


def _worker_busy() -> bool:
    os.environ.setdefault("PYTHONPATH", str(ROOT))
    from knowledge_engine.services.worker_busy import worker_busy_for_reload

    return worker_busy_for_reload()


def _wait_until_idle(poll_sec: float = 0.5) -> None:
    announced = False
    while _worker_busy():
        if not announced:
            print(
                "[ke-worker-watch] reload отложен — worker выполняет задачу…",
                flush=True,
            )
            announced = True
        time.sleep(poll_sec)


def main() -> None:
    os.chdir(ROOT)
    os.environ.setdefault("PYTHONPATH", str(ROOT))

    from knowledge_engine.config import (
        KE_WORKER_RELOAD_DEBOUNCE_SEC,
        KE_WORKER_STOP_TIMEOUT_SEC,
    )
    from watchfiles import watch

    proc: subprocess.Popen | None = None
    min_gap = KE_WORKER_RELOAD_DEBOUNCE_SEC
    reload_pending = False
    pending_sample = ""

    def stop() -> None:
        nonlocal proc
        if proc is None or proc.poll() is not None:
            proc = None
            return
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=KE_WORKER_STOP_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        proc = None

    def start() -> None:
        nonlocal proc
        stop()
        proc = subprocess.Popen(
            [sys.executable, "-m", "knowledge_engine.worker"],
            cwd=ROOT,
            env=os.environ.copy(),
        )
        print(f"[ke-worker-watch] worker pid={proc.pid}", flush=True)

    start()
    last_restart = 0.0

    watch_args: list[Path] = list(WATCH_DIRS) + list(WATCH_FILES)
    if not watch_args:
        print("[ke-worker-watch] no watch paths", flush=True)
        return

    def _on_term(*_args: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _on_term)

    try:
        for changes in watch(
            *watch_args,
            debounce=int(min_gap * 1000),
            step=500,
            rust_timeout=500,
        ):
            if changes:
                paths = {str(path) for _kind, path in changes}
                if _should_reload(paths):
                    reload_pending = True
                    pending_sample = next(iter(paths))

            if not reload_pending:
                continue

            now = time.monotonic()
            if now - last_restart < min_gap:
                continue

            if _worker_busy():
                _wait_until_idle()
            print(f"[ke-worker-watch] reload ▶ {pending_sample}", flush=True)
            start()
            reload_pending = False
            pending_sample = ""
            last_restart = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        stop()


if __name__ == "__main__":
    main()
