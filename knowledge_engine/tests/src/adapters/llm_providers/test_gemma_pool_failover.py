"""RateLimitedLLMClient: отказ primary (5xx/перегрузка после его ретраев →
``None``) переводит запрос на fallback-модель, а не повторяет primary.

Раньше цикл failover звал ``_pick_slot()``, который всегда начинал с
primary: при свободных лимитах primary выбирался снова, fallback
включался только по исчерпанию лимитов/429 (живой случай: gemma-4-31b-it
отвечал 500/503, gemma-4-26b-a4b-it был доступен)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from knowledge_engine.src.adapters.llm_providers.gemma_client import (
    GemmaModelSlot,
    GemmaRateLimitError,
    RateLimitedLLMClient,
)


class _Out(BaseModel):
    ok: bool


class _FreeLimiter:
    async def try_acquire(self, *args, **kwargs) -> bool:
        return True


def _slot(label: str, model: str, result) -> GemmaModelSlot:
    client = AsyncMock()
    if isinstance(result, BaseException):
        client.complete_structured = AsyncMock(side_effect=result)
    else:
        client.complete_structured = AsyncMock(return_value=result)
    return GemmaModelSlot(
        label=label, model=model, client=client, limiter=_FreeLimiter()
    )


@pytest.mark.anyio
async def test_primary_failure_falls_back_once() -> None:
    primary = _slot("primary", "gemma-31b", None)
    fallback = _slot("fallback", "gemma-26b", _Out(ok=True))
    pool = RateLimitedLLMClient(slots=[primary, fallback])

    out = await pool.complete_structured("sys", "prompt", _Out, label="t")

    assert out == _Out(ok=True)
    primary.client.complete_structured.assert_awaited_once()
    fallback.client.complete_structured.assert_awaited_once()


@pytest.mark.anyio
async def test_all_models_fail_returns_none_without_repeats() -> None:
    primary = _slot("primary", "gemma-31b", None)
    fallback = _slot("fallback", "gemma-26b", GemmaRateLimitError())
    pool = RateLimitedLLMClient(slots=[primary, fallback])

    assert await pool.complete_structured("sys", "prompt", _Out, label="t") is None
    primary.client.complete_structured.assert_awaited_once()
    fallback.client.complete_structured.assert_awaited_once()
