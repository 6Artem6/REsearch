"""Скользящие лимиты RPM / TPM / RPD для внешнего LLM API."""

from __future__ import annotations

import asyncio
import datetime
import time
from collections import deque
from dataclasses import dataclass

from knowledge_engine.config import (
    GEMMA_MAX_RPD,
    GEMMA_MAX_RPM,
    GEMMA_MAX_TPM,
    GEMMA_REDUCE_TPM_RESERVE_RATIO,
)


def wait_for_next_minute_window() -> float:
    """
    Дождаться UTC :00 следующей минуты (сброс TPM в AI Studio / Vertex).
    Возвращает фактическую длительность sleep (сек).
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    sleep_seconds = 60.0 - now.second - (now.microsecond / 1_000_000.0)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds + 0.1)
        return sleep_seconds + 0.1
    return 0.0


async def await_next_minute_window() -> float:
    return await asyncio.to_thread(wait_for_next_minute_window)


@dataclass
class RateLimitSnapshot:
    rpm_used: int
    tpm_used: int
    rpd_used: int
    max_rpm: int
    max_tpm: int
    max_rpd: int
    model: str = ""


class AsyncRateLimiter:
    """60s sliding window для RPM/TPM; 24h для RPD."""

    def __init__(
        self,
        *,
        max_rpm: int = GEMMA_MAX_RPM,
        max_tpm: int = GEMMA_MAX_TPM,
        max_rpd: int = GEMMA_MAX_RPD,
        window_sec: float = 60.0,
        day_sec: float = 86400.0,
        safety_ratio: float = 1.0,
        reduce_reserved_ratio: float = GEMMA_REDUCE_TPM_RESERVE_RATIO,
    ) -> None:
        self._max_rpm = max(1, max_rpm)
        self._max_tpm = max(100, max_tpm)
        self._max_rpd = max(1, max_rpd)
        self._safety_ratio = max(0.5, min(1.0, float(safety_ratio)))
        # Non-priority (MAP) callers are capped below the real ceiling by this
        # fraction, so a priority (REDUCE) caller on the same slot always has
        # room reserved for it instead of queueing behind MAP's own usage.
        self._reduce_reserved_ratio = max(0.0, min(0.5, float(reduce_reserved_ratio)))
        self._window = window_sec
        self._day = day_sec
        self._req_times: deque[float] = deque()
        # Each event is a mutable [timestamp, tokens] pair (not a tuple) so a
        # reservation can be reconciled by DIRECT REFERENCE (see
        # try_acquire_parallel/reconcile_batch_total) instead of by deque
        # position — positional ("last event") reconciliation silently
        # corrupts an unrelated, still in-flight reservation once more than
        # one group can be admitted on this limiter before either completes.
        self._token_events: deque[list] = deque()
        self._day_req_times: deque[float] = deque()
        self._lock = asyncio.Lock()

    def _eff_rpm(self, *, priority: bool = False) -> int:
        base = max(1, int(self._max_rpm * self._safety_ratio))
        if priority or self._reduce_reserved_ratio <= 0:
            return base
        return max(1, int(base * (1.0 - self._reduce_reserved_ratio)))

    def _eff_tpm(self, *, priority: bool = False) -> int:
        base = max(100, int(self._max_tpm * self._safety_ratio))
        if priority or self._reduce_reserved_ratio <= 0:
            return base
        return max(100, int(base * (1.0 - self._reduce_reserved_ratio)))

    def _eff_rpd(self) -> int:
        return max(1, int(self._max_rpd * self._safety_ratio))

    def update_limits(
        self,
        max_rpm: int | None = None,
        max_tpm: int | None = None,
        max_rpd: int | None = None,
        safety_ratio: float | None = None,
    ) -> None:
        if max_rpm is not None:
            self._max_rpm = max(1, int(max_rpm))
        if max_tpm is not None:
            self._max_tpm = max(100, int(max_tpm))
        if max_rpd is not None:
            self._max_rpd = max(1, int(max_rpd))
        if safety_ratio is not None:
            self._safety_ratio = max(0.5, min(1.0, float(safety_ratio)))

    async def reconcile_last_tpm(self, actual_tokens: int) -> None:
        actual = max(1, int(actual_tokens))
        async with self._lock:
            if self._token_events:
                self._token_events[-1][1] = actual

    def snapshot(self, model: str = "") -> RateLimitSnapshot:
        now = time.monotonic()
        wall = time.time()
        self._evict(now, wall)
        tpm = sum(t for _, t in self._token_events)
        return RateLimitSnapshot(
            rpm_used=len(self._req_times),
            tpm_used=tpm,
            rpd_used=len(self._day_req_times),
            # Show effective hard ceiling used by acquire (not the pre-safety raw)
            max_rpm=self._eff_rpm(),
            max_tpm=self._eff_tpm(),
            max_rpd=self._eff_rpd(),
            model=model,
        )

    async def acquire(
        self, estimated_tokens: int, *, model: str = "", priority: bool = False
    ) -> RateLimitSnapshot:
        est = max(1, int(estimated_tokens))
        while True:
            async with self._lock:
                now = time.monotonic()
                wall = time.time()
                self._evict(now, wall)
                wait = self._required_wait(now, wall, est, priority=priority)
                if wait <= 0:
                    self._record(now, wall, est)
                    return self.snapshot(model=model)
            await asyncio.sleep(min(wait, 30.0))

    async def try_acquire(
        self,
        estimated_tokens: int,
        *,
        max_wait: float = 0.0,
        model: str = "",
        priority: bool = False,
    ) -> bool:
        """Забронировать слот; max_wait=0 — без паузы (для failover на другую модель).
        priority=True (REDUCE) sees the FULL ceiling, ignoring the headroom
        reserved away from non-priority (MAP) callers on this same limiter."""
        est = max(1, int(estimated_tokens))
        async with self._lock:
            now = time.monotonic()
            wall = time.time()
            self._evict(now, wall)
            wait = self._required_wait(now, wall, est, priority=priority)
            if wait > max_wait:
                return False
            if wait <= 0:
                self._record(now, wall, est)
                return True
            sleep_for = wait
        from knowledge_engine.ui.run_log import trace

        snap = self.snapshot(model=model)
        trace(
            f"BLOG_SPATIAL gemma slot wait ▶ | model={model or 'gemma'} "
            f"{sleep_for:.1f}s (max_wait={max_wait:.0f}s) "
            f"rpm {snap.rpm_used}/{snap.max_rpm} tpm {snap.tpm_used}/{snap.max_tpm}"
            f"{' priority=reduce' if priority else ''}"
        )
        await asyncio.sleep(min(sleep_for, max_wait if max_wait > 0 else sleep_for))
        async with self._lock:
            now = time.monotonic()
            wall = time.time()
            self._evict(now, wall)
            wait2 = self._required_wait(now, wall, est, priority=priority)
            if wait2 > max_wait:
                return False
            self._record(now, wall, est)
            return True

    def _evict(self, now: float, wall: float) -> None:
        cutoff = now - self._window
        while self._req_times and self._req_times[0] <= cutoff:
            self._req_times.popleft()
        while self._token_events and self._token_events[0][0] <= cutoff:
            self._token_events.popleft()
        day_cutoff = wall - self._day
        while self._day_req_times and self._day_req_times[0] <= day_cutoff:
            self._day_req_times.popleft()

    def _would_exceed(
        self,
        now: float,
        wall: float,
        tpm_add: int,
        extra_rpm: int,
        extra_rpd: int,
        *,
        priority: bool = False,
    ) -> bool:
        if len(self._day_req_times) + extra_rpd > self._eff_rpd():
            return True
        if len(self._req_times) + extra_rpm > self._eff_rpm(priority=priority):
            return True
        tpm = sum(t for _, t in self._token_events)
        if tpm + tpm_add > self._eff_tpm(priority=priority):
            return True
        return False

    async def try_acquire_parallel(
        self,
        estimates: list[int],
        *,
        max_parallel: int,
        model: str = "",
        priority: bool = False,
    ) -> tuple[int, list | None]:
        """
        Атомарно зарезервировать до max_parallel запросов (in+out TPM, RPM, RPD).
        Возвращает (k, event) — k: сколько первых estimates из списка
        поместилось; event: ссылка на добавленную [ts, tokens]-запись в
        _token_events (None, если k == 0), которую вызывающий код обязан
        передать обратно в reconcile_batch_total(event=...), чтобы reconcile
        обновлял ИМЕННО эту резервацию, а не «последнюю на данный момент» —
        под конкурентным диспетчером (несколько групп одновременно в полёте
        на одном лимитере) это НЕ одно и то же.
        """
        if not estimates:
            return 0, None
        cap = max(1, int(max_parallel))
        async with self._lock:
            now = time.monotonic()
            wall = time.time()
            self._evict(now, wall)
            k = 0
            cum_tpm = 0
            for est in estimates[:cap]:
                e = max(1, int(est))
                if self._would_exceed(
                    now, wall, cum_tpm + e, k + 1, k + 1, priority=priority
                ):
                    break
                cum_tpm += e
                k += 1
            if k == 0:
                return 0, None
            for _ in range(k):
                self._req_times.append(now)
                self._day_req_times.append(wall)
            event = [now, cum_tpm]
            self._token_events.append(event)
            return k, event

    def _required_wait(
        self, now: float, wall: float, est: int, *, priority: bool = False
    ) -> float:
        if len(self._day_req_times) >= self._eff_rpd():
            oldest = self._day_req_times[0]
            return max(0.05, oldest + self._day - wall)

        if len(self._req_times) >= self._eff_rpm(priority=priority):
            return max(0.05, self._req_times[0] + self._window - now)

        tpm = sum(t for _, t in self._token_events)
        if tpm + est > self._eff_tpm(priority=priority):
            if not self._token_events:
                return 0.5
            return max(0.05, self._token_events[0][0] + self._window - now)
        return 0.0

    def _record(self, now: float, wall: float, est: int) -> None:
        self._req_times.append(now)
        self._token_events.append([now, est])
        self._day_req_times.append(wall)

    async def record_429_spike(
        self, tokens: int | None = None, *, model: str = ""
    ) -> None:
        """Реальный HTTP 429 от ЭТОЙ модели нужно вернуть в её же скользящее
        окно — иначе try_acquire/_pick_slot видят только собственную
        оптимистичную оценку до вызова, которая не узнала об отказе, и на
        следующий вызов той же модели снова говорят "ok" вместо переключения
        на следующий слот. Зеркалит record_429_spike у GemmaTokenBudgetManager
        (gemma_rate_limiter.py), но на per-model лимитере, который сейчас
        реально гейтит admission для MAP/REDUCE-вызовов (см. заметки BUG
        FIXED в gemma_client.py post_structured*)."""
        spike = max(1, int(tokens)) if tokens is not None else self._eff_tpm()
        now = time.monotonic()
        wall = time.time()
        async with self._lock:
            self._evict(now, wall)
            self._record(now, wall, spike)
        from knowledge_engine.ui.run_log import trace

        trace(f"GEMMA limiter 429 spike recorded | model={model or 'gemma'} est_tokens={spike}")

    async def reconcile_batch_total(
        self, actual_total_tokens: int, *, event: list | None = None
    ) -> None:
        """Подстроить TPM-резервацию пачки (сумма in+out по API) на РЕАЛЬНЫЙ
        расход. ``event`` — ссылка на конкретную запись из try_acquire_parallel
        (обязательна под конкурентным диспетчером — несколько групп могут
        одновременно висеть на одном лимитере, и «последняя запись» к моменту
        завершения ЭТОЙ группы может уже принадлежать ДРУГОЙ, всё ещё
        выполняющейся группе). Без event — легаси-поведение для одиночных
        путей (acquire/try_acquire), где резервация всегда ровно одна.

        AUDIT: logs reserved-vs-real so a live run can PROVE, not guess,
        whether the pre-call estimate systematically undercounts what Gemma
        actually bills — a candidate explanation for TPM observed on the
        GCP dashboard exceeding what the local ledger shows: for a call
        still in flight, the ledger holds the pre-call ESTIMATE, corrected
        to the real usage only here, on completion."""
        actual = max(1, int(actual_total_tokens))
        async with self._lock:
            if event is not None:
                reserved = event[1]
                event[1] = actual
            elif self._token_events:
                reserved = self._token_events[-1][1]
                self._token_events[-1][1] = actual
            else:
                reserved = None
        if reserved is not None and reserved != actual:
            from knowledge_engine.ui.run_log import trace

            delta = actual - reserved
            pct = (delta / reserved * 100.0) if reserved else 0.0
            trace(
                f"BLOG_SPATIAL gemma reconcile | reserved={reserved} real={actual} "
                f"delta={delta:+d} ({pct:+.0f}%)"
            )

    async def wait_for_room(
        self, needed_tokens: int, *, priority: bool = False
    ) -> float:
        """Ждать, пока в 60s окне есть место для needed_tokens (in+out)."""
        est = max(1, int(needed_tokens))
        slept = 0.0
        while True:
            async with self._lock:
                now = time.monotonic()
                wall = time.time()
                self._evict(now, wall)
                wait = self._required_wait(now, wall, est, priority=priority)
                if wait <= 0:
                    return slept
            pause = min(wait, 30.0)
            await asyncio.sleep(pause)
            slept += pause


_global_limiters: dict[str, AsyncRateLimiter] = {}


def get_vlm_gemini_rate_limiter() -> AsyncRateLimiter:
    from knowledge_engine.config import vlm_gemini_rate_limits_live

    rpm, tpm, rpd = vlm_gemini_rate_limits_live()
    lim = get_gemma_rate_limiter(
        slot="vlm_gemini_lite",
        max_rpm=rpm,
        max_tpm=tpm,
        max_rpd=rpd,
    )
    lim.update_limits(max_rpm=rpm, max_tpm=tpm, max_rpd=rpd)
    return lim


def get_gemma_rate_limiter(
    *,
    slot: str = "default",
    max_rpm: int | None = None,
    max_tpm: int | None = None,
    max_rpd: int | None = None,
    safety_ratio: float | None = None,
) -> AsyncRateLimiter:
    from knowledge_engine.config import GEMINI_QUOTA_SAFETY_RATIO

    key = slot
    ratio = GEMINI_QUOTA_SAFETY_RATIO if safety_ratio is None else float(safety_ratio)
    if key not in _global_limiters:
        from knowledge_engine.config import GEMMA_MAX_RPD, GEMMA_MAX_RPM, GEMMA_MAX_TPM

        _global_limiters[key] = AsyncRateLimiter(
            max_rpm=max_rpm if max_rpm is not None else GEMMA_MAX_RPM,
            max_tpm=max_tpm if max_tpm is not None else GEMMA_MAX_TPM,
            max_rpd=max_rpd if max_rpd is not None else GEMMA_MAX_RPD,
            safety_ratio=ratio,
        )
    lim = _global_limiters[key]
    if max_rpm is not None or max_tpm is not None or max_rpd is not None:
        lim.update_limits(
            max_rpm=max_rpm,
            max_tpm=max_tpm,
            max_rpd=max_rpd,
            safety_ratio=ratio if safety_ratio is not None else None,
        )
    elif safety_ratio is not None:
        lim.update_limits(safety_ratio=ratio)
    return lim
