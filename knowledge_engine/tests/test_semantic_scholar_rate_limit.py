"""Cross-process Semantic Scholar rate lock (≥ MIN_INTERVAL between acquires)."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from knowledge_engine.config import SEMANTIC_SCHOLAR_MIN_INTERVAL_SEC
from knowledge_engine.src.retrieval import semantic_scholar_rate_limit as ss_rl


@pytest.fixture()
def isolated_ss_lock(tmp_path, monkeypatch):
    path = tmp_path / "semantic_scholar_rate_lock"
    monkeypatch.setattr(ss_rl, "_LOCK_PATH", path)
    # Keep production default unless test overrides
    return path


def test_sync_acquires_respect_min_interval(isolated_ss_lock, monkeypatch):
    monkeypatch.setattr(ss_rl, "SEMANTIC_SCHOLAR_MIN_INTERVAL_SEC", 0.15)
    stamps: list[float] = []

    def _one() -> None:
        ss_rl.acquire_semantic_scholar_slot()
        stamps.append(time.time())

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: _one(), range(4)))

    stamps.sort()
    gaps = [stamps[i] - stamps[i - 1] for i in range(1, len(stamps))]
    assert all(g >= 0.14 for g in gaps), gaps


def test_async_acquires_respect_min_interval(isolated_ss_lock, monkeypatch):
    monkeypatch.setattr(ss_rl, "SEMANTIC_SCHOLAR_MIN_INTERVAL_SEC", 0.15)

    async def _run() -> list[float]:
        stamps: list[float] = []

        async def _one() -> None:
            await ss_rl.acquire_semantic_scholar_slot_async()
            stamps.append(time.time())

        await asyncio.gather(*[_one() for _ in range(4)])
        return stamps

    stamps = sorted(asyncio.run(_run()))
    gaps = [stamps[i] - stamps[i - 1] for i in range(1, len(stamps))]
    assert all(g >= 0.14 for g in gaps), gaps


def test_default_interval_is_at_least_one_second():
    assert SEMANTIC_SCHOLAR_MIN_INTERVAL_SEC >= 1.0
