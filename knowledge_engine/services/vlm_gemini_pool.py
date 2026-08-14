"""Единый VLM-пул: несколько Flash Lite моделей, лимиты per-model, round-robin при квоте."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from knowledge_engine.config import (
    refresh_vlm_gemini_env_from_dotenv,
    vlm_gemini_model_live,
    vlm_gemini_quota_track_live,
    vlm_gemini_rate_limits_live,
)
from knowledge_engine.services.gemini_quota_store import (
    filter_models_for_quota,
    model_usable,
    set_model_daily_limit_rpd,
)
from knowledge_engine.services.gemini_stateless import gemini_lite_model_chain
from knowledge_engine.services.llm.rate_limiter import (
    AsyncRateLimiter,
    get_gemma_rate_limiter,
)
from knowledge_engine.ui.run_log import trace


@dataclass
class VlmModelSlot:
    model: str
    limiter: AsyncRateLimiter


def _dedupe_models(names: list[str]) -> list[str]:
    out: list[str] = []
    for n in names:
        m = (n or "").strip()
        if m and m not in out:
            out.append(m)
    return out


def resolve_vlm_pool_model_ids() -> list[str]:
    refresh_vlm_gemini_env_from_dotenv()
    import os

    raw = (os.getenv("VLM_GEMINI_MODELS") or "").strip()
    if raw:
        chain = _dedupe_models([p.strip() for p in raw.split(",")])
    else:
        chain = _dedupe_models(list(gemini_lite_model_chain(vlm_gemini_model_live())))
    rpm, tpm, rpd = vlm_gemini_rate_limits_live()
    for m in chain:
        set_model_daily_limit_rpd(m, rpd)
    if vlm_gemini_quota_track_live():
        filtered = filter_models_for_quota(chain)
        if len(filtered) < len(chain):
            trace(
                f"VLM pool quota | {' → '.join(chain[:5])} "
                f"→ usable {' → '.join(filtered[:5]) or '(none)'}"
            )
        if chain and not filtered:
            trace(
                "VLM skipped due to local quota guard | "
                "all pool models filtered (no VLM API retries)"
            )
        chain = filtered
    return chain


class VlmGeminiPool:
    """Список моделей как один пул: RPM/TPM/RPD на каждую, round-robin + failover."""

    def __init__(self, model_ids: list[str]) -> None:
        self._model_ids = list(model_ids)
        self._slots: list[VlmModelSlot] = []
        self._rr = 0
        self._rebuild_slots()

    @property
    def model_ids(self) -> list[str]:
        return list(self._model_ids)

    def label_chain(self) -> str:
        return " ⇄ ".join(self._model_ids[:6])

    def refresh_models(self, model_ids: list[str]) -> None:
        ids = _dedupe_models(model_ids)
        if ids == self._model_ids:
            self._sync_limiter_limits()
            return
        self._model_ids = ids
        self._rebuild_slots()

    def _sync_limiter_limits(self) -> None:
        rpm, tpm, rpd = vlm_gemini_rate_limits_live()
        for slot in self._slots:
            # Hard RPM ceiling (no extra 0.9 shrink) — refuse >14/min for Lite
            slot.limiter.update_limits(
                max_rpm=rpm, max_tpm=tpm, max_rpd=rpd, safety_ratio=1.0
            )

    def _rebuild_slots(self) -> None:
        rpm, tpm, rpd = vlm_gemini_rate_limits_live()
        self._slots = []
        for m in self._model_ids:
            lim = get_gemma_rate_limiter(
                slot=f"vlm:{m}",
                max_rpm=rpm,
                max_tpm=tpm,
                max_rpd=rpd,
                safety_ratio=1.0,
            )
            lim.update_limits(max_rpm=rpm, max_tpm=tpm, max_rpd=rpd, safety_ratio=1.0)
            self._slots.append(VlmModelSlot(model=m, limiter=lim))

    def _usable_slots_ordered(self) -> list[VlmModelSlot]:
        if not self._slots:
            return []
        if not vlm_gemini_quota_track_live():
            n = len(self._slots)
            return [self._slots[(self._rr + i) % n] for i in range(n)]
        ok_slots: list[VlmModelSlot] = []
        for slot in self._slots:
            usable, _ = model_usable(slot.model)
            if usable:
                ok_slots.append(slot)
        if not ok_slots:
            return []
        n = len(ok_slots)
        start = self._rr % n
        return [ok_slots[(start + i) % n] for i in range(n)]

    async def acquire_slot(self, est_tokens: int) -> VlmModelSlot | None:
        est = max(1, int(est_tokens))
        ordered = self._usable_slots_ordered()
        if not ordered:
            return None

        for slot in ordered:
            if await slot.limiter.try_acquire(est, max_wait=0.0, model=slot.model):
                self._rr += 1
                snap = slot.limiter.snapshot(model=slot.model)
                trace(
                    f"VLM slot ✓ | {slot.model} "
                    f"rpm {snap.rpm_used}/{snap.max_rpm} "
                    f"tpm {snap.tpm_used}/{snap.max_tpm} "
                    f"rpd {snap.rpd_used}/{snap.max_rpd}"
                )
                return slot

        # Квота на всех — ждём окно RPM/TPM, перебираем по кругу
        for slot in ordered:
            pause = await slot.limiter.wait_for_room(est)
            if pause >= 0.5:
                trace(
                    f"VLM pool ⏳ | {slot.model} pause {pause:.1f}s "
                    f"(лимит RPM/TPM, пробуем другие)"
                )
            if await slot.limiter.try_acquire(est, max_wait=0.0, model=slot.model):
                self._rr += 1
                return slot

        # Последний шанс: blocking acquire на первом доступном
        for slot in ordered:
            await slot.limiter.acquire(est, model=slot.model)
            self._rr += 1
            return slot
        return None


_pool: VlmGeminiPool | None = None
_pool_lock = asyncio.Lock()


async def get_vlm_gemini_pool() -> VlmGeminiPool:
    global _pool
    ids = resolve_vlm_pool_model_ids()
    async with _pool_lock:
        if _pool is None:
            _pool = VlmGeminiPool(ids)
        else:
            _pool.refresh_models(ids)
        return _pool


def get_vlm_gemini_pool_sync() -> VlmGeminiPool:
    ids = resolve_vlm_pool_model_ids()
    global _pool
    if _pool is None or _pool.model_ids != ids:
        _pool = VlmGeminiPool(ids)
    else:
        _pool.refresh_models(ids)
    return _pool
