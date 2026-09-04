"""AsyncRateLimiter REDUCE priority reserve + RateLimitedLLMClient dual-basket
proportional wave splitting (DUAL-BASKET BALANCING & SWITCH TO SLIDING WINDOW
PACING task)."""

from __future__ import annotations

import asyncio

from knowledge_engine.services.llm.gemma_client import (
    GemmaCloudClient,
    GemmaModelSlot,
    RateLimitedLLMClient,
)
from knowledge_engine.services.llm.rate_limiter import AsyncRateLimiter


def test_priority_reserves_headroom_for_reduce() -> None:
    """A non-priority (MAP) caller must be capped below the real TPM/RPM
    ceiling; a priority (REDUCE) caller must still see the full ceiling even
    when non-priority usage has already filled the reserved-away slice."""
    lim = AsyncRateLimiter(
        max_rpm=100, max_tpm=1000, max_rpd=10000, reduce_reserved_ratio=0.2
    )

    async def _run() -> None:
        # Non-priority effective TPM ceiling is 1000 * (1 - 0.2) = 800.
        ok_map = await lim.try_acquire(800, max_wait=0.0)
        assert ok_map
        blocked_map = await lim.try_acquire(1, max_wait=0.0)
        assert not blocked_map
        # A priority (REDUCE) caller can still use the reserved 200 tokens
        # of headroom that MAP was denied.
        ok_reduce = await lim.try_acquire(150, max_wait=0.0, priority=True)
        assert ok_reduce

    asyncio.run(_run())


def test_record_429_spike_blocks_next_try_acquire_on_same_limiter() -> None:
    """Реальный 429 от API должен заставить ЭТОТ лимитер сам увидеть
    отсутствие headroom для следующего вызова — до фикса 429 обновлял
    только отдельный (и обходимый для MAP/REDUCE) GemmaTokenBudgetManager,
    так что собственный учёт slot.limiter не узнавал об отказе и снова
    говорил "ok" на следующий try_acquire той же модели."""
    lim = AsyncRateLimiter(max_rpm=100, max_tpm=1000, max_rpd=10000)

    async def _run() -> None:
        ok_before = await lim.try_acquire(1, max_wait=0.0)
        assert ok_before
        await lim.record_429_spike(model="gemma-4-31b-it")
        blocked_after = await lim.try_acquire(1, max_wait=0.0)
        assert not blocked_after

    asyncio.run(_run())


def test_pick_slot_fails_over_after_429_spike_recorded() -> None:
    """End-to-end: как только на лимитере слота записан 429,
    RateLimitedLLMClient._pick_slot должен переключиться на следующий слот,
    а не выбирать тот же только что отклонённый на следующем вызове."""
    primary = _make_slot("primary", "gemma-4-31b-it", max_tpm=1000)
    fallback = _make_slot("fallback", "gemma-4-26b-a4b-it", max_tpm=1000)
    client = RateLimitedLLMClient(slots=[primary, fallback])

    async def _run() -> None:
        picked_before = await client._pick_slot(10)
        assert picked_before is primary
        await primary.limiter.record_429_spike(model=primary.model)
        picked_after = await client._pick_slot(10)
        assert picked_after is fallback

    asyncio.run(_run())


def test_no_reserve_means_priority_is_a_no_op() -> None:
    lim = AsyncRateLimiter(
        max_rpm=100, max_tpm=1000, max_rpd=10000, reduce_reserved_ratio=0.0
    )

    async def _run() -> None:
        ok = await lim.try_acquire(1000, max_wait=0.0)
        assert ok
        blocked = await lim.try_acquire(1, max_wait=0.0, priority=True)
        assert not blocked

    asyncio.run(_run())


def _make_slot(label: str, model: str, *, max_tpm: int) -> GemmaModelSlot:
    return GemmaModelSlot(
        label=label,
        model=model,
        client=GemmaCloudClient(api_key="test", model=model),
        limiter=AsyncRateLimiter(max_rpm=1000, max_tpm=max_tpm, max_rpd=100000),
    )


def test_acquire_parallel_wave_splits_proportionally_across_slots() -> None:
    """A wave that doesn't fit on the primary slot's free TPM must spill the
    remainder onto the next slot instantly, not wait for the primary."""
    primary = _make_slot("primary", "gemma-4-31b-it", max_tpm=100)
    fallback = _make_slot("fallback", "gemma-4-26b-a4b-it", max_tpm=1000)
    client = RateLimitedLLMClient(slots=[primary, fallback])

    async def _run() -> None:
        # 4 requests of 40 tokens each: primary (100 TPM) fits only 2,
        # the remaining 2 must land on fallback.
        plan = await client.acquire_parallel_wave([40, 40, 40, 40], max_parallel=4)
        assert sum(k for _slot, k, _reserved, _event in plan) == 4
        assert len(plan) == 2
        slot0, k0, _r0, _e0 = plan[0]
        slot1, k1, _r1, _e1 = plan[1]
        assert slot0 is primary
        assert k0 == 2
        assert slot1 is fallback
        assert k1 == 2

    asyncio.run(_run())


def test_acquire_parallel_wave_single_slot_when_all_fit() -> None:
    primary = _make_slot("primary", "gemma-4-31b-it", max_tpm=1000)
    fallback = _make_slot("fallback", "gemma-4-26b-a4b-it", max_tpm=1000)
    client = RateLimitedLLMClient(slots=[primary, fallback])

    async def _run() -> None:
        plan = await client.acquire_parallel_wave([40, 40], max_parallel=4)
        assert len(plan) == 1
        slot0, k0, _r0, _e0 = plan[0]
        assert slot0 is primary
        assert k0 == 2

    asyncio.run(_run())
