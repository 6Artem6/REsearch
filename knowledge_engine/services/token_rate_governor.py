"""Token & Rate Governor — единая, sync-совместимая точка контроля RPM/TPM
per model, с zero-wait fast path и обратной связью по фактическому usage.

Заменяет в gemini_stateless.py:
- `_rpm_pause_for_model()` — безусловный `time.sleep(60/RPM)` перед КАЖДЫМ
  вызовом, вне зависимости от фактического использования (аудит STEP 1:
  109.4s чистого простоя при факте 6/15 RPM — формула `max(60/rpm, 4.0)`
  не смотрит на реальный sliding window вообще).
- Хардкод `reserve_gemini_minute_slot(model, 800)` — 800 токенов не связаны
  с реальным размером payload; здесь оценка приходит от вызывающего кода
  через единый `fast_tokenizer.token_counter` (см.
  article_ingestion/paragraph_token_splitter.py::estimate_text_tokens).

НЕ заменяет и не трогает:
- `services/llm/rate_limiter.py::AsyncRateLimiter` (Gemma) — это уже
  корректный async sliding-window limiter с тем же zero-wait-if-free
  поведением, dual-basket wave allocation, MAP/REDUCE priority reservation
  и реальным reconcile_batch_total(usage.total_tokens) из ответа API.
  Изначальная посылка аудита («Gemma клиент работал обособленно от
  контроля квот») не подтвердилась при детальном чтении gemma_client.py —
  Gemma-пайплайн уже устроен ЛУЧШЕ старого Gemini-пейсера, просто под
  другим именем/файлом. Замена его на новый generic-класс была бы чистым
  риском регрессии (потеря wave/priority-логики) без выигрыша. Целевые RPM/
  TPM из ТЗ (27/30, 15.2K/16K) применяются к СУЩЕСТВУЮЩЕМУ
  AsyncRateLimiter через его собственный `safety_ratio`, не через этот
  модуль — см. config.GEMMA_GOVERNOR_TARGET_RPM/TPM и их использование в
  `services/llm/rate_limiter.py::get_gemma_rate_limiter`.
- `services/gemini_quota_store.py` — RPD (24h) tracking и
  `filter_models_for_quota`/`model_minute_guard_ok` (soft-cap ранний
  переход по chain ДО начала цикла) остаются как есть; это другая забота
  (какую модель вообще пробовать), не пейсинг внутри одной модели.

Sync, не async: вызывающий код (gemini_stateless.py::_call_with_model_fallback)
сам синхронный и выполняется в worker-потоке через asyncio.to_thread —
блокирующий time.sleep() здесь безопасен, не держит event loop.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

from knowledge_engine.ui.run_log import trace


@dataclass
class GovernorSnapshot:
    model: str
    rpm_used: int
    tpm_used: int
    max_rpm: int
    max_tpm: int


class TokenRateGovernor:
    """60s sliding-window RPM/TPM пейсер для одной модели."""

    def __init__(
        self, model: str, *, max_rpm: int, max_tpm: int, window_sec: float = 60.0
    ) -> None:
        self.model = model
        self.max_rpm = max(1, int(max_rpm))
        self.max_tpm = max(100, int(max_tpm))
        self._window = window_sec
        self._req_times: deque[float] = deque()
        self._token_events: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def _evict(self, now: float) -> None:
        cutoff = now - self._window
        while self._req_times and self._req_times[0] <= cutoff:
            self._req_times.popleft()
        while self._token_events and self._token_events[0][0] <= cutoff:
            self._token_events.popleft()

    def _required_wait(self, now: float, est_tokens: int) -> float:
        """0.0, если бюджет свободен прямо сейчас; иначе — сколько ждать до
        освобождения ближайшего слота (RPM или TPM, смотря что упирается)."""
        waits: list[float] = []
        if len(self._req_times) >= self.max_rpm:
            oldest = self._req_times[0]
            waits.append(max(0.0, (oldest + self._window) - now))
        tpm = sum(t for _, t in self._token_events)
        if tpm + est_tokens > self.max_tpm:
            running = tpm
            freed = False
            for ts, tok in self._token_events:
                running -= tok
                if running + est_tokens <= self.max_tpm:
                    waits.append(max(0.0, (ts + self._window) - now))
                    freed = True
                    break
            if not freed:
                waits.append(self._window)
        return max(waits) if waits else 0.0

    def acquire(self, estimated_tokens: int) -> float:
        """Блокирует (time.sleep), только если бюджет реально исчерпан —
        ровно на нужное время. 0.0, если места хватает прямо сейчас (никаких
        искусственных пауз/джиттеров). Возвращает суммарное время ожидания."""
        est = max(1, int(estimated_tokens))
        total_waited = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                self._evict(now)
                wait = self._required_wait(now, est)
                if wait <= 0:
                    self._req_times.append(now)
                    self._token_events.append((now, est))
                    return total_waited
            sleep_for = min(wait, 30.0) + 0.05
            time.sleep(sleep_for)
            total_waited += sleep_for

    def confirm(self, actual_tokens: int) -> None:
        """Примирить последнюю резервацию TPM с фактическим usage_metadata
        из ответа API — реальный feedback loop (отсутствовал у старого
        пейсера полностью, см. аудит STEP 1)."""
        if actual_tokens is None or actual_tokens <= 0:
            return
        actual = max(1, int(actual_tokens))
        est: int | None = None
        with self._lock:
            if self._token_events:
                ts, est = self._token_events[-1]
                self._token_events[-1] = (ts, actual)
        if est is not None:
            trace(
                f"[GOVERNOR FEEDBACK] model={self.model} est={est} "
                f"actual={actual} diff={actual - est}"
            )

    def snapshot(self) -> GovernorSnapshot:
        with self._lock:
            now = time.monotonic()
            self._evict(now)
            tpm = sum(t for _, t in self._token_events)
            return GovernorSnapshot(
                model=self.model,
                rpm_used=len(self._req_times),
                tpm_used=tpm,
                max_rpm=self.max_rpm,
                max_tpm=self.max_tpm,
            )


class AdaptiveTokenRateLimiter(TokenRateGovernor):
    """``TokenRateGovernor`` + cache-aware adaptive TPM budgeting.

    Gemini's context caching (explicit cached_content + implicit prefix-cache
    reuse) means the REAL billed/uncached cost of a call is often far below
    the raw prompt-token estimate a local tokenizer produces before the call
    — the base governor's fixed estimate-then-confirm sliding window has no
    way to anticipate this, so it reserves the full cold-cache size on every
    request and can starve the free-tier TPM budget on tokens Gemini never
    actually re-billed. ``effective_cache_factor`` is an EMA of
    ``actual_uncached_prompt_tokens / estimated_prompt_tokens`` across calls
    to this model, applied to the PROMPT portion of future pre-call estimates
    (never to the output-token estimate — caching never discounts generation)
    so the reservation converges toward the model's real cache-hit rate
    instead of always assuming a cold cache.

    Reuses the base class's sliding-window bucket/eviction/lock as-is
    (``acquire``/``confirm``) — this only adds the pre-call cache adjustment
    and the post-call EMA update on top, it does not reimplement RPM/TPM
    tracking.
    """

    def __init__(
        self, model: str, *, max_rpm: int, max_tpm: int, window_sec: float = 60.0
    ) -> None:
        super().__init__(model, max_rpm=max_rpm, max_tpm=max_tpm, window_sec=window_sec)
        self.effective_cache_factor: float = 1.0  # консервативный старт — холодный кэш

    def reserve_tokens(
        self,
        estimated_prompt_tokens: int,
        estimated_max_output_tokens: int = 0,
        *,
        is_warmup: bool = False,
    ) -> float:
        """FAST PRE-CHECK: cache-adjusted prompt estimate + raw output
        estimate, reserved from the sliding window. ``is_warmup=True``
        bypasses the bucket entirely — прогревочные вызовы не должны
        конкурировать за TPM с реальным пользовательским трафиком."""
        if is_warmup:
            return 0.0
        projected_tpm_cost = max(
            1,
            round(max(0, int(estimated_prompt_tokens)) * self.effective_cache_factor)
            + max(0, int(estimated_max_output_tokens)),
        )
        return self.acquire(projected_tpm_cost)

    def reconcile_tokens(
        self,
        *,
        estimated_prompt_tokens: int,
        actual_prompt_tokens: int,
        cached_content_tokens: int = 0,
        actual_completion_tokens: int = 0,
        is_warmup: bool = False,
    ) -> None:
        """POST-RESPONSE RECONCILIATION: replace the projected reservation
        with the REAL cost from Gemini's own ``usage_metadata`` — an
        over-estimate (``delta = projected - real > 0``) is freed back into
        the TPM window immediately via ``confirm()``'s existing replace
        semantics. Then updates the cache-hit EMA for the NEXT
        ``reserve_tokens()`` call on this model."""
        if is_warmup:
            return
        real_uncached_prompt = max(
            0, int(actual_prompt_tokens) - max(0, int(cached_content_tokens))
        )
        real_tpm_cost = max(1, real_uncached_prompt + max(0, int(actual_completion_tokens)))
        self.confirm(real_tpm_cost)
        est = max(0, int(estimated_prompt_tokens))
        if est > 0:
            new_alpha = real_uncached_prompt / est
            old_factor = self.effective_cache_factor
            self.effective_cache_factor = 0.8 * old_factor + 0.2 * new_alpha
            trace(
                f"[ADAPTIVE_LIMITER] model={self.model} alpha={new_alpha:.3f} "
                f"effective_cache_factor {old_factor:.3f}→{self.effective_cache_factor:.3f}"
            )


_governors: dict[str, AdaptiveTokenRateLimiter] = {}
_registry_lock = threading.Lock()


def _target_limits_for_model(model: str) -> tuple[int, int]:
    from knowledge_engine.config import (
        GEMINI_GOVERNOR_TARGET_RPM,
        GEMINI_GOVERNOR_TARGET_TPM,
        GEMMA_GOVERNOR_TARGET_RPM,
        GEMMA_GOVERNOR_TARGET_TPM,
    )

    m = (model or "").lower()
    if "gemma" in m:
        return GEMMA_GOVERNOR_TARGET_RPM, GEMMA_GOVERNOR_TARGET_TPM
    if "flash-lite" in m or "flash_lite" in m:
        return GEMINI_GOVERNOR_TARGET_RPM, GEMINI_GOVERNOR_TARGET_TPM
    from knowledge_engine.services.gemini_stateless import (
        default_rpm_limit_for_model,
        default_tpm_limit_for_model,
    )

    return default_rpm_limit_for_model(model), default_tpm_limit_for_model(model)


def get_governor(model: str) -> AdaptiveTokenRateLimiter:
    m = (model or "").strip()
    with _registry_lock:
        g = _governors.get(m)
        if g is None:
            rpm, tpm = _target_limits_for_model(m)
            g = AdaptiveTokenRateLimiter(m, max_rpm=rpm, max_tpm=tpm)
            _governors[m] = g
        return g


def reset_governors_for_tests() -> None:
    """Тестовый хелпер — очищает registry между тестами (иначе sliding-window
    состояние одной модели протекает между test-кейсами)."""
    with _registry_lock:
        _governors.clear()
