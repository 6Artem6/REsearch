"""Gemma: параллельная попытка при зависании, лимит попыток, circuit breaker.

Живой случай: gemma-4-31b-it висел 57 с и отвечал 500/503, запрос держался
на зависшей попытке, а следующие запросы снова шли в него. Теперь:
нет ответа за ``hedge_after`` → новая попытка параллельно (если у модели
есть свободный слот), всего не больше ``max_attempts``; все провалились →
модель «открыта», пул сначала берёт исправные модели."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from knowledge_engine.src.adapters.llm_providers import gemma_client as gc


class _Limiter:
    def __init__(self, free: bool = True) -> None:
        self.free = free
        self.calls = 0

    async def try_acquire(self, *args, **kwargs) -> bool:
        self.calls += 1
        return self.free


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(gc, "_retry_backoff_sec", lambda exc, attempt: 0.0)
    gc._circuit_open_until.clear()
    yield
    gc._circuit_open_until.clear()


def _script(*steps):
    """Попытка i выполняет steps[i]: ("ok", delay) | ("500", delay)."""
    calls: list[int] = []

    async def attempt():
        i = len(calls)
        calls.append(i)
        kind, delay = steps[i]
        await asyncio.sleep(delay)
        if kind == "ok":
            return f"resp{i}"
        raise gc.GemmaTransientError("HTTP 500")

    return attempt, calls


def _run(attempt, limiter, **kw):
    return gc.post_with_hedging(
        attempt,
        model="gemma-31b",
        label="t",
        limiter=limiter,
        est_tokens=100,
        max_attempts=kw.get("max_attempts", 3),
        hedge_after_sec=kw.get("hedge_after", 0.05),
        slot_wait_sec=0,
    )


@pytest.mark.anyio
async def test_hung_attempt_gets_parallel_attempt_first_answer_wins() -> None:
    attempt, calls = _script(("ok", 5.0), ("ok", 0.01))
    out = await asyncio.wait_for(_run(attempt, _Limiter()), timeout=2)
    assert out == "resp1"  # зависшую попытку 0 не ждали
    assert calls == [0, 1]


@pytest.mark.anyio
async def test_no_free_slot_means_no_parallel_attempt() -> None:
    attempt, calls = _script(("ok", 0.2), ("ok", 0.01))
    limiter = _Limiter(free=False)
    out = await _run(attempt, limiter)
    assert out == "resp0"
    assert calls == [0]
    assert limiter.calls >= 1  # слот спрашивали, но его не было


@pytest.mark.anyio
async def test_at_most_max_attempts_then_circuit_opens() -> None:
    attempt, calls = _script(("500", 0), ("500", 0), ("500", 0), ("ok", 0))
    with pytest.raises(gc.GemmaTransientError):
        await _run(attempt, _Limiter(), max_attempts=3)
    assert calls == [0, 1, 2]
    assert gc.gemma_circuit_open("gemma-31b")


@pytest.mark.anyio
async def test_success_after_failure_closes_circuit() -> None:
    gc._open_circuit("gemma-31b", "test")
    attempt, calls = _script(("500", 0), ("ok", 0))
    assert await _run(attempt, _Limiter()) == "resp1"
    assert not gc.gemma_circuit_open("gemma-31b")


@pytest.mark.anyio
async def test_preacquired_path_without_limiter_only_retries_on_failure() -> None:
    """MAP-волны: слоты уже распределены — параллельных попыток нет,
    повтор после отказа остаётся."""
    attempt, calls = _script(("ok", 0.15), ("ok", 0.0))
    assert await _run(attempt, None) == "resp0"
    assert calls == [0]
    attempt, calls = _script(("500", 0), ("ok", 0))
    assert await _run(attempt, None) == "resp1"


class _Out(BaseModel):
    ok: bool


@pytest.mark.anyio
async def test_pool_sends_to_healthy_model_while_primary_circuit_open() -> None:
    def slot(label, model):
        client = AsyncMock()
        client.complete_structured = AsyncMock(return_value=_Out(ok=True))
        return gc.GemmaModelSlot(
            label=label, model=model, client=client, limiter=_Limiter()
        )

    primary, fallback = slot("primary", "gemma-31b"), slot("fallback", "gemma-26b")
    gc._open_circuit("gemma-31b", "test")
    pool = gc.RateLimitedLLMClient(slots=[primary, fallback])

    assert await pool.complete_structured("s", "p", _Out, label="t") == _Out(ok=True)
    primary.client.complete_structured.assert_not_called()
    fallback.client.complete_structured.assert_awaited_once()


class _SlowLimiter:
    """Слот освобождается через ``delay`` (новая минута окна)."""

    def __init__(self, delay: float) -> None:
        self.delay = delay

    async def try_acquire(self, *args, **kwargs) -> bool:
        await asyncio.sleep(self.delay)
        return True


@pytest.mark.anyio
async def test_answer_during_slot_wait_wins_without_extra_attempt() -> None:
    """Ожидание слота не блокирует: исходная попытка ответила, пока ждали
    новую минуту, — она и побеждает, параллельная не запускается."""
    attempt, calls = _script(("ok", 0.15), ("ok", 0.0))
    out = await asyncio.wait_for(_run(attempt, _SlowLimiter(1.0)), timeout=0.8)
    assert out == "resp0"
    assert calls == [0]


@pytest.mark.anyio
async def test_pool_fail_fast_retry_gives_up_and_opens_circuit() -> None:
    """retry_slot_wait_sec=0 (у пула есть fallback): слота прямо сейчас нет —
    не ждём минуту ради отказавшей модели, сдаёмся после 1 попытки."""
    attempt, calls = _script(("500", 0), ("ok", 0))
    with pytest.raises(gc.GemmaTransientError):
        await gc.post_with_hedging(
            attempt,
            model="gemma-31b",
            label="t",
            limiter=_Limiter(free=False),
            est_tokens=100,
            retry_slot_wait_sec=0,
        )
    assert calls == [0]
    assert gc.gemma_circuit_open("gemma-31b")


def test_pool_retry_wait_zero_only_when_alternative_exists() -> None:
    def slot(label, model):
        return gc.GemmaModelSlot(
            label=label, model=model, client=AsyncMock(), limiter=_Limiter()
        )

    pool = gc.RateLimitedLLMClient(
        slots=[slot("primary", "gemma-31b"), slot("fallback", "gemma-26b")]
    )
    assert pool._retry_slot_wait("gemma-31b", {"gemma-31b"}) == 0.0
    assert pool._retry_slot_wait("gemma-26b", {"gemma-31b", "gemma-26b"}) is None
