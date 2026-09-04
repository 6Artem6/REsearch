"""Unit tests for GemmaTokenBudgetManager sliding window."""

import asyncio

from knowledge_engine.services.gemma_rate_limiter import (
    FLASH_OVERFLOW_WAIT_SEC,
    GemmaTokenBudgetManager,
    count_prompt_tokens,
)


def test_acquire_budget_oversized_single_request_overflows_immediately() -> None:
    """Regression: a request whose own token estimate exceeds max_tpm must
    overflow to Flash-Lite right away, not hang. _required_wait used to return
    a flat 0.25s wait for this case forever — since a request bigger than the
    whole budget can never be admitted to _history, that branch re-triggered
    on every retry, live-locking acquire_budget (confirmed: a real 17781-token
    MAP call against max_tpm=14400 hung a live process for 5+ minutes)."""
    mgr = GemmaTokenBudgetManager(max_tpm=14400, max_rpm=27, window_sec=60.0)

    async def _run() -> None:
        result = await asyncio.wait_for(mgr.acquire_budget(17781), timeout=2.0)
        assert not result.acquired
        assert result.overflow_to_flash
        assert result.projected_wait_sec > FLASH_OVERFLOW_WAIT_SEC
        # Must not have been silently admitted despite exceeding the budget.
        assert mgr.tpm_used == 0

    asyncio.run(_run())


def test_acquire_budget_oversized_request_with_nonempty_history_still_overflows() -> None:
    """Same oversized-request guard, but with prior usage already recorded —
    must hit the new early-return, not fall through to the oldest-entry-expiry
    branch (which would compute a finite, eventually-satisfiable wait for an
    unsatisfiable request)."""
    mgr = GemmaTokenBudgetManager(max_tpm=100, max_rpm=50, window_sec=60.0)

    async def _run() -> None:
        await mgr.acquire_budget(10, max_wait_for_overflow=60.0)
        result = await asyncio.wait_for(mgr.acquire_budget(150), timeout=2.0)
        assert result.overflow_to_flash
        assert result.projected_wait_sec > FLASH_OVERFLOW_WAIT_SEC

    asyncio.run(_run())


def test_acquire_budget_rpm_cap() -> None:
    mgr = GemmaTokenBudgetManager(max_tpm=100000, max_rpm=3, window_sec=60.0)

    async def _run() -> None:
        for _ in range(3):
            r = await mgr.acquire_budget(10, max_wait_for_overflow=60.0)
            assert r.acquired
            assert not r.overflow_to_flash
        overflow = await mgr.acquire_budget(10, max_wait_for_overflow=0.05)
        assert not overflow.acquired
        assert overflow.overflow_to_flash
        assert overflow.projected_wait_sec > 0

    asyncio.run(_run())


def test_acquire_budget_tpm_sliding_window() -> None:
    mgr = GemmaTokenBudgetManager(max_tpm=100, max_rpm=100, window_sec=0.15)

    async def _run() -> None:
        await mgr.acquire_budget(60, max_wait_for_overflow=60.0)
        await mgr.acquire_budget(40, max_wait_for_overflow=60.0)
        assert mgr.tpm_used == 100
        blocked = await mgr.acquire_budget(5, max_wait_for_overflow=0.02)
        assert blocked.overflow_to_flash
        await asyncio.sleep(0.16)
        ok = await mgr.acquire_budget(50, max_wait_for_overflow=60.0)
        assert ok.acquired
        assert mgr.tpm_used == 50

    asyncio.run(_run())


def test_record_429_spike_blocks_burst() -> None:
    mgr = GemmaTokenBudgetManager(max_tpm=200, max_rpm=50, window_sec=60.0)

    async def _run() -> None:
        await mgr.record_429_spike(180)
        overflow = await mgr.acquire_budget(30, max_wait_for_overflow=0.05)
        assert overflow.overflow_to_flash

    asyncio.run(_run())


def test_count_prompt_tokens_includes_output_reserve() -> None:
    n = count_prompt_tokens("system", "hello world", max_output_tokens=1000)
    assert n >= 1001


def test_reconcile_actual_updates_last_entry() -> None:
    mgr = GemmaTokenBudgetManager(max_tpm=10000, max_rpm=50, window_sec=60.0)

    async def _run() -> None:
        await mgr.acquire_budget(500, max_wait_for_overflow=60.0)
        await mgr.reconcile_actual(1200)
        assert mgr.tpm_used == 1200

    asyncio.run(_run())
