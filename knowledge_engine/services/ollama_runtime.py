"""Ollama host helpers — disabled. LLM SSOT is Gemma Cloud."""

from __future__ import annotations


async def ollama_tags_reachable(timeout_sec: float = 2.0) -> bool:
    _ = timeout_sec
    return False


async def ensure_ollama_server(
    *,
    wait_sec: float = 20.0,
    poll_interval: float = 0.4,
) -> bool:
    _ = (wait_sec, poll_interval)
    return False


async def ollama_touch_keep_alive(
    model: str,
    keep_alive: str,
    *,
    timeout_sec: float = 90.0,
) -> None:
    _ = (model, keep_alive, timeout_sec)
    return None
