"""Доступность Ollama на хосте и keep_alive для router-моделей."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import time

import httpx

from knowledge_engine.config import OLLAMA_AUTO_START, OLLAMA_BASE_URL


def _base_url() -> str:
    return (OLLAMA_BASE_URL or "http://localhost:11434").rstrip("/")


async def ollama_tags_reachable(timeout_sec: float = 2.0) -> bool:
    url = f"{_base_url()}/api/tags"
    try:
        timeout = httpx.Timeout(timeout_sec)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            return resp.status_code == 200
    except Exception:
        return False


def _spawn_ollama_serve() -> bool:
    if not shutil.which("ollama"):
        return False
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return True


async def ensure_ollama_server(
    *,
    wait_sec: float = 20.0,
    poll_interval: float = 0.4,
) -> bool:
    """Если OLLAMA_AUTO_START=1 — запустить ollama serve и ждать /api/tags."""
    if await ollama_tags_reachable():
        return True
    if not OLLAMA_AUTO_START:
        return False
    if not _spawn_ollama_serve():
        return False
    deadline = time.monotonic() + wait_sec
    while time.monotonic() < deadline:
        if await ollama_tags_reachable():
            return True
        await asyncio.sleep(poll_interval)
    return False


async def ollama_touch_keep_alive(
    model: str,
    keep_alive: str,
    *,
    timeout_sec: float = 90.0,
) -> None:
    """
    Загрузить модель в память по запросу и продлить keep_alive (например 5m).
    Лёгкий generate; не ждём полный ответ — только установка keep_alive.
    """
    name = (model or "").strip()
    alive = (keep_alive or "").strip()
    if not name or not alive:
        return
    url = f"{_base_url()}/api/generate"
    payload = {
        "model": name,
        "prompt": " ",
        "stream": False,
        "keep_alive": alive,
        "options": {"num_predict": 1},
    }
    timeout = httpx.Timeout(timeout_sec)
    async with httpx.AsyncClient(timeout=timeout) as client:
        await client.post(url, json=payload)
