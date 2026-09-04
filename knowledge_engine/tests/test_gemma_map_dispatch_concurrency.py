"""BLOG_SPATIAL MAP dispatch: a fast model's slot must not sit idle waiting
for a slow model's call from the SAME wave before the next wave is even
requested.

`_run_gemma_map_waves`/`_dispatch_group` in blog_spatial_summarizer.py are
private closures nested inside `map_reduce_jobs_pooled_async` — not
independently importable. This test reproduces the exact control-flow
pattern actually shipped there (per-(slot,k)-group background task,
non-blocking outer loop, reap-on-next-iteration) against real
`AsyncRateLimiter` instances, so it validates the real fix's structural
properties: reservation/reconciliation correctness, no lost exceptions, and
— the actual bug — that a fast slot's next wave isn't gated on a slow
slot's in-flight call finishing first.

Timing is asserted by EVENT ORDER (asyncio.Event gates), never by wall-clock
sleeps + cancellation — the latter proved flaky under pytest's own event-loop
overhead (observed both patterns draining all work well inside the intended
cutoff in CI-like conditions)."""

from __future__ import annotations

from typing import Awaitable, Callable

import asyncio

import pytest

from knowledge_engine.services.llm.rate_limiter import AsyncRateLimiter


class _FakeSlot:
    def __init__(self, name: str, dispatch_one: Callable[[int], Awaitable[int]]) -> None:
        self.model = name
        self.limiter = AsyncRateLimiter(max_rpm=1000, max_tpm=10_000_000)
        self.dispatch_one = dispatch_one


async def _dispatch_group(
    slot: _FakeSlot, items: list[int], reserved_tpm: int, event: list | None = None
) -> None:
    real_usage = await asyncio.gather(*[slot.dispatch_one(est) for est in items])
    await slot.limiter.reconcile_batch_total(sum(real_usage), event=event)


def _take_one_per_slot(
    batch: list[int], slots: list[_FakeSlot]
) -> tuple[list[tuple[_FakeSlot, list[int]]], list[int]]:
    groups: list[tuple[_FakeSlot, list[int]]] = []
    for slot in slots:
        if not batch:
            break
        item, batch = batch[:1], batch[1:]
        groups.append((slot, item))
    return groups, batch


async def _run_waves_old_blocking(
    work: list[int], slots: list[_FakeSlot], *, max_parallel: int, n_waves: int
) -> None:
    """Pre-fix control flow: await the WHOLE wave (every slot's group)
    before requesting the next one."""
    pos = 0
    for _ in range(n_waves):
        batch = work[pos : pos + max_parallel]
        groups, _ = _take_one_per_slot(batch, slots)
        if not groups:
            return
        events = {}
        for slot, items in groups:
            _k, ev = await slot.limiter.try_acquire_parallel(
                items, max_parallel=1, model=slot.model
            )
            events[id(slot)] = ev
        await asyncio.gather(  # <-- the bug: blocks the WHOLE wave
            *[
                _dispatch_group(slot, items, sum(items), events[id(slot)])
                for slot, items in groups
            ]
        )
        pos += max_parallel


async def _run_waves_new_nonblocking(
    work: list[int], slots: list[_FakeSlot], *, max_parallel: int, n_waves: int
) -> None:
    """Post-fix control flow: each (slot, k) group is its own background
    task; the loop does not wait for any group before admitting the next."""
    pos = 0
    in_flight: list[asyncio.Task] = []
    for _ in range(n_waves):
        batch = work[pos : pos + max_parallel]
        groups, _ = _take_one_per_slot(batch, slots)
        if not groups:
            break
        for slot, items in groups:
            _k, ev = await slot.limiter.try_acquire_parallel(
                items, max_parallel=1, model=slot.model
            )
            in_flight.append(
                asyncio.create_task(_dispatch_group(slot, items, sum(items), ev))
            )
        pos += max_parallel
        still_running = []
        for t in in_flight:
            if t.done():
                t.result()
            else:
                still_running.append(t)
        in_flight = still_running
    if in_flight:
        await asyncio.gather(*in_flight)


def test_old_pattern_blocks_wave_two_admission_on_slow_group() -> None:
    """Baseline proving the bug actually exists in the old shape: wave 2's
    admission (here, simply reaching the point of dispatching wave 2's fast
    call) never happens until wave 1's slow group signals completion."""

    async def _run() -> list[str]:
        order: list[str] = []
        slow_gate = asyncio.Event()

        async def fast_dispatch(est: int) -> int:
            order.append(f"fast_call_{est}")
            return est

        async def slow_dispatch(est: int) -> int:
            order.append("slow_start")
            await slow_gate.wait()
            order.append("slow_end")
            return est

        fast = _FakeSlot("fast", fast_dispatch)
        slow = _FakeSlot("slow", slow_dispatch)
        work = [1, 2, 3, 4]  # wave1=(fast:1, slow:2), wave2=(fast:3, slow:4)

        task = asyncio.create_task(
            _run_waves_old_blocking(work, [fast, slow], max_parallel=2, n_waves=2)
        )
        await asyncio.sleep(0.01)  # let wave 1 start and reach slow_gate.wait()
        assert "slow_start" in order
        assert "fast_call_3" not in order  # wave 2 must NOT have started yet

        slow_gate.set()
        await task
        return order

    order = asyncio.run(_run())
    assert order.index("slow_end") < order.index("fast_call_3")


def test_new_pattern_admits_wave_two_without_waiting_for_slow_group() -> None:
    """The actual fix: wave 2's fast call starts BEFORE wave 1's slow group
    finishes — the fast slot is never held hostage by the slow one."""

    async def _run() -> list[str]:
        order: list[str] = []
        slow_gate = asyncio.Event()

        async def fast_dispatch(est: int) -> int:
            order.append(f"fast_call_{est}")
            return est

        async def slow_dispatch(est: int) -> int:
            order.append("slow_start")
            await slow_gate.wait()
            order.append("slow_end")
            return est

        fast = _FakeSlot("fast", fast_dispatch)
        slow = _FakeSlot("slow", slow_dispatch)
        work = [1, 2, 3, 4]  # wave1=(fast:1, slow:2), wave2=(fast:3, slow:4)

        task = asyncio.create_task(
            _run_waves_new_nonblocking(work, [fast, slow], max_parallel=2, n_waves=2)
        )
        # Let both waves get ADMITTED — the loop no longer waits on slow's
        # group to finish before moving to wave 2 (only real TPM/RPM headroom
        # gates admission, and the fake limiter here has effectively none).
        await asyncio.sleep(0.01)
        assert "slow_start" in order
        assert "fast_call_3" in order  # wave 2's fast call already ran!
        assert "slow_end" not in order  # slow group is still gated

        slow_gate.set()
        await task
        return order

    order = asyncio.run(_run())
    # Wave 2's fast call completed strictly BEFORE wave 1's slow group did —
    # the inverse of the old pattern's guaranteed ordering.
    assert order.index("fast_call_3") < order.index("slow_end")


def test_new_dispatch_reconciles_every_item_exactly_once() -> None:
    """No item's usage is lost or double-counted across background groups —
    the sliding-window ledger ends up with exactly the real total, not the
    inflated pre-call reservation."""

    async def instant(est: int) -> int:
        return est

    async def _run() -> AsyncRateLimiter:
        slot = _FakeSlot("solo", instant)
        work = [50, 60, 70, 80]
        await _run_waves_new_nonblocking(work, [slot], max_parallel=1, n_waves=4)
        return slot.limiter

    limiter = asyncio.run(_run())
    snap = limiter.snapshot()
    assert snap.tpm_used == 50 + 60 + 70 + 80


def test_reconcile_does_not_corrupt_a_later_still_pending_reservation() -> None:
    """DEEP AUDIT FIX regression: two groups admitted back-to-back on the
    SAME slot's limiter (wave A's slow group is still in flight when wave
    B's fast group is admitted and reconciles first). Reconciling by deque
    POSITION ("touch token_events[-1]") corrupts wave B's still-correct
    entry once wave A's slow group finally completes and reconciles too —
    at that moment wave B's entry is the last one in the deque, so wave A's
    real usage gets written into wave B's slot instead of its own, and wave
    A's own entry is left stuck at its stale pre-call ESTIMATE forever.
    Reconciling by the returned event reference must keep both figures
    correct regardless of completion order."""

    async def _run() -> AsyncRateLimiter:
        order: list[str] = []
        slow_gate = asyncio.Event()

        async def slow_dispatch(_est: int) -> int:
            order.append("slow_start")
            await slow_gate.wait()
            order.append("slow_end")
            return 77  # real usage != reserved estimate (100)

        async def fast_dispatch(_est: int) -> int:
            order.append("fast_done")
            return 55  # real usage != reserved estimate (50)

        async def dispatch_one_group(
            fn, est: int, event: list | None, limiter: AsyncRateLimiter
        ) -> None:
            real = await fn(est)
            await limiter.reconcile_batch_total(real, event=event)

        limiter = AsyncRateLimiter(max_rpm=1000, max_tpm=10_000_000)

        k_a, event_a = await limiter.try_acquire_parallel(
            [100], max_parallel=1, model="solo"
        )
        assert k_a == 1
        task_a = asyncio.create_task(
            dispatch_one_group(slow_dispatch, 100, event_a, limiter)
        )
        await asyncio.sleep(0)  # let task_a run up to slow_gate.wait()

        # Wave B admitted on the SAME limiter while wave A is still pending.
        k_b, event_b = await limiter.try_acquire_parallel(
            [50], max_parallel=1, model="solo"
        )
        assert k_b == 1
        assert "slow_start" in order
        await dispatch_one_group(fast_dispatch, 50, event_b, limiter)  # finishes first
        assert "fast_done" in order
        assert "slow_end" not in order

        slow_gate.set()
        await task_a
        return limiter

    limiter = asyncio.run(_run())
    snap = limiter.snapshot()
    # Real usage: 77 (slow) + 55 (fast) = 132. The bug this replaces would
    # leave wave A's entry stuck at its estimate (100) and stomp wave B's
    # entry with wave A's real usage (77), giving 100 + 77 = 177.
    assert snap.tpm_used == 77 + 55


def test_new_dispatch_propagates_group_exceptions() -> None:
    """A failing dispatch group must not vanish silently — the loop's reap
    step (or the final gather) must surface it."""

    async def _boom(est: int) -> int:
        raise RuntimeError("simulated dispatch failure")

    async def _run() -> None:
        slot = _FakeSlot("solo", _boom)
        work = [10, 10]
        await _run_waves_new_nonblocking(work, [slot], max_parallel=1, n_waves=2)

    with pytest.raises(RuntimeError, match="simulated dispatch failure"):
        asyncio.run(_run())
