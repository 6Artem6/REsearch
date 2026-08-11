"""90% Gemma 4 sliding-window TPM/RPM budget (14.4k TPM / 27 RPM per 60s)."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel

from knowledge_engine.config import (
    GEMINI_LITE_MODEL,
    GEMMA_MAP_MAX_OUTPUT_TOKENS,
)
from knowledge_engine.services.article_ingestion.paragraph_token_splitter import (
    estimate_text_tokens,
)
from knowledge_engine.services.llm.gemma_client import _gemma_user_content
from knowledge_engine.ui.run_log import trace

T = TypeVar("T", bound=BaseModel)

# 90% of Gemma 4 published limits (16k TPM / 30 RPM).
DEFAULT_MAX_TPM = 14400
DEFAULT_MAX_RPM = 27
DEFAULT_WINDOW_SEC = 60.0
DEFAULT_OUTPUT_RESERVE = 1000
FLASH_OVERFLOW_WAIT_SEC = 10.0


@dataclass(frozen=True)
class BudgetAcquireResult:
    acquired: bool
    overflow_to_flash: bool = False
    projected_wait_sec: float = 0.0


def count_prompt_tokens(
    system: str,
    prompt: str,
    *,
    schema: type[BaseModel] | None = None,
    max_output_tokens: int | None = None,
) -> int:
    """Input tokens (local estimate) + reserved output budget."""
    out_cap = (
        max_output_tokens if max_output_tokens is not None else DEFAULT_OUTPUT_RESERVE
    )
    user_prompt = _gemma_user_content(prompt)
    inp = estimate_text_tokens(f"{system}\n{user_prompt}")
    return max(1, inp + out_cap)


class GemmaTokenBudgetManager:
    def __init__(
        self,
        *,
        max_tpm: int = DEFAULT_MAX_TPM,
        max_rpm: int = DEFAULT_MAX_RPM,
        window_sec: float = DEFAULT_WINDOW_SEC,
        overflow_wait_sec: float = FLASH_OVERFLOW_WAIT_SEC,
    ) -> None:
        self._max_tpm = max(100, int(max_tpm))
        self._max_rpm = max(1, int(max_rpm))
        self._window = float(window_sec)
        self._overflow_wait_sec = float(overflow_wait_sec)
        self._history: deque[tuple[float, int]] = deque()
        self._lock = asyncio.Lock()

    @property
    def max_tpm(self) -> int:
        return self._max_tpm

    @property
    def tpm_used(self) -> int:
        self._evict()
        return sum(t for _, t in self._history)

    @property
    def rpm_used(self) -> int:
        self._evict()
        return len(self._history)

    def _evict(self, now: float | None = None) -> None:
        wall = now if now is not None else time.time()
        cutoff = wall - self._window
        while self._history and self._history[0][0] <= cutoff:
            self._history.popleft()

    def _required_wait(self, tokens: int, now: float) -> float:
        self._evict(now)
        if len(self._history) >= self._max_rpm:
            oldest = self._history[0][0]
            return max(0.05, oldest + self._window - now)
        tpm = sum(t for _, t in self._history)
        if tpm + tokens > self._max_tpm:
            if not self._history:
                return 0.25
            oldest = self._history[0][0]
            return max(0.05, oldest + self._window - now)
        return 0.0

    def projected_wait(self, tokens: int) -> float:
        return self._required_wait(max(1, int(tokens)), time.time())

    async def acquire_budget(
        self,
        tokens: int,
        *,
        max_wait_for_overflow: float | None = None,
    ) -> BudgetAcquireResult:
        """Reserve TPM/RPM slot; overflow to Flash if wait would exceed threshold."""
        est = max(1, int(tokens))
        overflow_cap = (
            max_wait_for_overflow
            if max_wait_for_overflow is not None
            else self._overflow_wait_sec
        )
        while True:
            async with self._lock:
                now = time.time()
                wait = self._required_wait(est, now)
                if wait > overflow_cap:
                    trace(
                        f"GEMMA budget overflow ▶ Flash-Lite | wait={wait:.1f}s "
                        f"tpm={self.tpm_used}/{self._max_tpm} rpm={self.rpm_used}/{self._max_rpm}"
                    )
                    return BudgetAcquireResult(
                        acquired=False,
                        overflow_to_flash=True,
                        projected_wait_sec=wait,
                    )
                if wait <= 0:
                    self._history.append((now, est))
                    return BudgetAcquireResult(acquired=True)
            await asyncio.sleep(min(wait, overflow_cap))

    async def acquire_budget_blocking(
        self,
        tokens: int,
        *,
        max_wait_sec: float = 600.0,
    ) -> None:
        """Ждать окно TPM/RPM без overflow на Flash (строго Gemma 4)."""
        est = max(1, int(tokens))
        deadline = time.monotonic() + max_wait_sec
        while time.monotonic() < deadline:
            async with self._lock:
                now = time.time()
                wait = self._required_wait(est, now)
                if wait <= 0:
                    self._history.append((now, est))
                    return
            await asyncio.sleep(min(wait, 30.0, max(0.05, deadline - time.monotonic())))
        raise TimeoutError(
            f"Gemma budget wait exceeded {max_wait_sec:.0f}s for {est} tokens"
        )

    async def record_429_spike(self, tokens: int | None = None) -> None:
        """Treat 429 as full-window pressure to avoid immediate retry storm."""
        spike = tokens if tokens is not None else self._max_tpm
        async with self._lock:
            self._history.append((time.time(), max(1, int(spike))))
        trace(f"GEMMA budget 429 spike recorded | est_tokens={spike}")

    async def reconcile_actual(self, actual_total_tokens: int) -> None:
        """Adjust last reservation to API usage total (prompt+completion)."""
        actual = max(1, int(actual_total_tokens))
        async with self._lock:
            if self._history:
                ts, _old = self._history[-1]
                self._history[-1] = (ts, actual)


_budget_manager: GemmaTokenBudgetManager | None = None


def get_gemma_token_budget_manager() -> GemmaTokenBudgetManager:
    global _budget_manager
    if _budget_manager is None:
        from knowledge_engine.config import (
            GEMMA_BUDGET_MAX_RPM,
            GEMMA_BUDGET_MAX_TPM,
            GEMMA_BUDGET_OVERFLOW_WAIT_SEC,
        )

        _budget_manager = GemmaTokenBudgetManager(
            max_tpm=GEMMA_BUDGET_MAX_TPM,
            max_rpm=GEMMA_BUDGET_MAX_RPM,
            overflow_wait_sec=GEMMA_BUDGET_OVERFLOW_WAIT_SEC,
        )
    return _budget_manager


async def complete_structured_gemini_flash_async(
    system: str,
    prompt: str,
    schema: type[T],
    *,
    label: str = "gemini_flash_overflow",
    max_output_tokens: int | None = None,
) -> T | None:
    """Lite-tier fallback when Gemma budget/parse fails (not full Flash reasoner)."""
    from knowledge_engine.src.analytics.gemini_v07 import run_gemini_lite_structured

    cap = (
        max_output_tokens
        if max_output_tokens is not None
        else GEMMA_MAP_MAX_OUTPUT_TOKENS
    )
    trace(
        f"GEMMA budget Flash-Lite fallback ▶ | model={GEMINI_LITE_MODEL} "
        f"label={label} max_out={cap}"
    )

    def _run() -> T:
        return run_gemini_lite_structured(
            system,
            (prompt or "").strip(),
            global_anchor="",
            response_schema=schema,
            label=label,
        )

    try:
        return await asyncio.to_thread(_run)
    except Exception as exc:
        trace(f"GEMMA budget Flash-Lite fallback ✗ | {label} | {exc}")
        return None
