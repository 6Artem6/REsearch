"""Gemma 4 cloud API + rate-limited pool (primary / fallback models)."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt
from tenacity.wait import wait_base

from knowledge_engine.config import (
    GEMMA_API_BASE,
    GEMMA_API_KEY,
    GEMMA_API_TIMEOUT_SEC,
    GEMMA_BUDGET_MAX_RPM,
    GEMMA_BUDGET_MAX_TPM,
    GEMMA_FALLBACK_MAX_RPD,
    GEMMA_FALLBACK_MAX_RPM,
    GEMMA_FALLBACK_MAX_TPM,
    GEMMA_FALLBACK_MAX_WAIT_SEC,
    GEMMA_FALLBACK_MODEL,
    GEMMA_MAX_RPD,
    GEMMA_MAX_RPM,
    GEMMA_MAX_TPM,
    GEMMA_PRIMARY_MAX_RPD,
    GEMMA_PRIMARY_MAX_RPM,
    GEMMA_PRIMARY_MAX_TPM,
    GEMMA_PRIMARY_MODEL,
    GEMMA_QUOTA_SHARED,
)
from knowledge_engine.services.article_ingestion.paragraph_token_splitter import (
    estimate_text_tokens,
)
from knowledge_engine.services.llm.rate_limiter import (
    AsyncRateLimiter,
    get_gemma_rate_limiter,
)
from knowledge_engine.ui.run_log import trace

logger = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.I)
_THOUGHT_BLOCK_RE = re.compile(r"<thought\b[^>]*>.*?</thought>", re.I | re.DOTALL)
_THOUGHT_UNCLOSED_RE = re.compile(r"<thought\b[^>]*>[\s\S]*$", re.I)
_MD_FENCE_OPEN_RE = re.compile(r"^```(?:json)?\s*", re.I)
_MD_FENCE_CLOSE_RE = re.compile(r"\s*```$", re.I)

T = TypeVar("T", bound=BaseModel)


class GemmaRateLimitError(Exception):
    def __init__(self, retry_after: float | None = None) -> None:
        self.retry_after = retry_after
        super().__init__("rate limited")


@dataclass
class GemmaModelSlot:
    label: str
    model: str
    client: GemmaCloudClient
    limiter: AsyncRateLimiter
    failover_max_wait_sec: float = 0.0


def resolve_gemma_map_max_output_tokens(input_tokens: int | None = None) -> int:
    """MAP completion cap — always 4096 (no input-size / model branching)."""
    _ = input_tokens
    return 4096


def _strip_gemma_thought_wrapper(text: str) -> str:
    """Remove Gemma ``<thought>…</thought>`` (closed or truncated) before JSON."""
    t = (text or "").strip()
    if not t:
        return t
    t = _THOUGHT_BLOCK_RE.sub("", t).strip()
    # Unclosed <thought>: keep from first `{` if JSON already started, else drop.
    if re.search(r"<thought\b", t, re.I):
        brace = t.find("{")
        if brace >= 0:
            t = t[brace:]
        else:
            t = _THOUGHT_UNCLOSED_RE.sub("", t).strip()
    brace = t.find("{")
    if brace > 0:
        t = t[brace:]
    return t.strip()


def _strip_markdown_fences(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    m = _JSON_FENCE_RE.search(t)
    if m:
        return m.group(1).strip()
    t = _MD_FENCE_OPEN_RE.sub("", t)
    t = _MD_FENCE_CLOSE_RE.sub("", t)
    return t.strip()


def loads_json_lenient(raw: str) -> object:
    """``json.loads`` then ``json_repair.loads`` for truncated / bad escapes."""
    text = _strip_markdown_fences(_strip_gemma_thought_wrapper((raw or "").strip()))
    if not text:
        raise ValueError("empty JSON payload after thought/fence cleanup")
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        import json_repair

        return json_repair.loads(text)
    except Exception as exc:
        raise ValueError(f"json_repair failed: {exc}") from exc


def _parse_structured(raw: str, schema: type[T]) -> T | None:
    """Parse Gemma JSON into ``schema``; log full raw + ValidationError details on failure."""
    schema_name = getattr(schema, "__name__", str(schema))
    try:
        data = loads_json_lenient(raw)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.error(
            "Gemma schema parse failed | schema=%s | json_error=%s | raw_response=%s",
            schema_name,
            exc,
            raw,
        )
        trace(
            f"BLOG_SPATIAL parse ✗ | schema={schema_name} json_error={exc} "
            f"raw_len={len(raw or '')}"
        )
        return None
    except Exception as exc:
        logger.error(
            "Gemma schema parse failed | schema=%s | error=%s | raw_response=%s",
            schema_name,
            exc,
            raw,
        )
        return None

    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        logger.error(
            "Gemma schema parse failed | schema=%s | validation_errors=%s | raw_response=%s",
            schema_name,
            exc.errors(),
            raw,
        )
        trace(
            f"BLOG_SPATIAL parse ✗ | schema={schema_name} "
            f"validation_errors={exc.errors()!r} raw_len={len(raw or '')}"
        )
        return None
    except Exception as exc:
        logger.error(
            "Gemma schema parse failed | schema=%s | error=%s | raw_response=%s",
            schema_name,
            exc,
            raw,
        )
        return None


class _GemmaRetryWait(wait_base):
    def __call__(self, retry_state: object) -> float:
        outcome = getattr(retry_state, "outcome", None)
        exc = outcome.exception() if outcome else None
        if isinstance(exc, GemmaRateLimitError) and exc.retry_after is not None:
            return min(float(exc.retry_after), 120.0)
        attempt = int(getattr(retry_state, "attempt_number", 1))
        return min(2**attempt, 60.0)


_GEMMA_JSON_USER_TAIL = (
    "CRITICAL: DO NOT output <thought> tags or any reasoning steps. "
    "Start your response IMMEDIATELY with the open curly bracket `{` and "
    "output ONLY pure, valid JSON (no markdown fences).\n"
    "Ensure all backslashes inside JSON strings (e.g. LaTeX math or quotes) "
    "are properly escaped with double backslashes (`\\\\`)."
)


def _gemma_user_content(prompt: str) -> str:
    body = (prompt or "").strip()
    if not body:
        return _GEMMA_JSON_USER_TAIL
    return f"{body}\n\n{_GEMMA_JSON_USER_TAIL}"


class GemmaCloudClient:
    def __init__(
        self,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_sec: float | None = None,
    ) -> None:
        self._base = (api_base or GEMMA_API_BASE).rstrip("/")
        self._api_key = (api_key or GEMMA_API_KEY).strip()
        self._model = (model or GEMMA_PRIMARY_MODEL).strip()
        self._timeout = timeout_sec or GEMMA_API_TIMEOUT_SEC

    @property
    def model(self) -> str:
        return self._model

    @property
    def max_output_tokens(self) -> int:
        return resolve_gemma_map_max_output_tokens()

    def estimate_input_tokens(
        self, system: str, prompt: str, schema: type[BaseModel]
    ) -> int:
        user_prompt = _gemma_user_content(prompt)
        return estimate_text_tokens(f"{system}\n{user_prompt}")

    def estimate_request_tokens(
        self,
        system: str,
        prompt: str,
        schema: type[BaseModel] | None = None,
        *,
        max_output_tokens: int | None = None,
    ) -> int:
        if schema is not None:
            inp = self.estimate_input_tokens(system, prompt, schema)
        else:
            inp = estimate_text_tokens(f"{system}\n{prompt}")
        out_cap = (
            max_output_tokens
            if max_output_tokens is not None
            else resolve_gemma_map_max_output_tokens(inp)
        )
        return inp + out_cap

    def estimate_budget(
        self, system: str, prompt: str, schema: type[BaseModel]
    ) -> tuple[int, int, int]:
        inp = self.estimate_input_tokens(system, prompt, schema)
        out_cap = resolve_gemma_map_max_output_tokens(inp)
        return inp, out_cap, inp + out_cap

    async def complete_structured(
        self,
        system: str,
        prompt: str,
        schema: type[T],
        *,
        label: str = "gemma",
        client: httpx.AsyncClient | None = None,
        limiter: AsyncRateLimiter | None = None,
        reconcile_tpm: bool = True,
        max_tokens: int | None = None,
        use_token_budget: bool = True,
    ) -> T | None:
        if not self._api_key:
            trace(f"BLOG_SPATIAL {label} ✗ | GEMINI_API_KEY empty (Gemma cloud)")
            return None

        inp_est = self.estimate_input_tokens(system, prompt, schema)
        out_tokens = (
            max_tokens
            if max_tokens is not None
            else resolve_gemma_map_max_output_tokens(inp_est)
        )

        budget = None
        est_total = 0
        if use_token_budget:
            from knowledge_engine.services.gemma_rate_limiter import (
                complete_structured_gemini_flash_async,
                count_prompt_tokens,
                get_gemma_token_budget_manager,
            )

            budget = get_gemma_token_budget_manager()
            est_total = count_prompt_tokens(
                system,
                prompt,
                schema=schema,
                max_output_tokens=out_tokens,
            )
            acquire = await budget.acquire_budget(est_total)
            if acquire.overflow_to_flash:
                return await complete_structured_gemini_flash_async(
                    system,
                    prompt,
                    schema,
                    label=f"{label}/flash_overflow",
                    max_output_tokens=out_tokens,
                )

        user_prompt = _gemma_user_content(prompt)

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": out_tokens,
            "response_format": {"type": "json_object"},
        }
        url = f"{self._base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        @retry(
            retry=retry_if_exception_type(GemmaRateLimitError),
            stop=stop_after_attempt(5),
            wait=_GemmaRetryWait(),
            reraise=True,
        )
        async def _post_once() -> httpx.Response:
            if client is not None:
                resp = await client.post(url, json=payload, headers=headers)
            else:
                timeout = httpx.Timeout(self._timeout)
                async with httpx.AsyncClient(timeout=timeout) as ephemeral:
                    resp = await ephemeral.post(url, json=payload, headers=headers)
            if resp.status_code == 429:
                retry_after: float | None = None
                ra = resp.headers.get("Retry-After")
                if ra:
                    try:
                        retry_after = float(ra)
                    except ValueError:
                        retry_after = None
                raise GemmaRateLimitError(retry_after)
            resp.raise_for_status()
            return resp

        try:
            resp = await _post_once()
        except GemmaRateLimitError:
            if budget is not None:
                await budget.record_429_spike(est_total)
                from knowledge_engine.services.gemma_rate_limiter import (
                    complete_structured_gemini_flash_async,
                )

                flash = await complete_structured_gemini_flash_async(
                    system,
                    prompt,
                    schema,
                    label=f"{label}/flash_429",
                    max_output_tokens=out_tokens,
                )
                if flash is not None:
                    return flash
            raise
        except Exception as exc:
            trace(f"BLOG_SPATIAL {label} ✗ | {exc}")
            return None

        usage_total = 0
        parsed: T | None = None
        for parse_try in range(2):
            if parse_try == 1:
                trace(
                    f"BLOG_SPATIAL {label} ↻ | schema parse retry (Gemma attempt 2/2)"
                )
                try:
                    resp = await _post_once()
                except GemmaRateLimitError:
                    from knowledge_engine.services.gemma_rate_limiter import (
                        complete_structured_gemini_flash_async,
                    )

                    flash = await complete_structured_gemini_flash_async(
                        system,
                        prompt,
                        schema,
                        label=f"{label}/flash_429",
                        max_output_tokens=out_tokens,
                    )
                    if flash is not None:
                        return flash
                    return None
                except Exception as exc:
                    trace(f"BLOG_SPATIAL {label} ✗ | parse retry HTTP | {exc}")
                    break

            try:
                data = resp.json()
                usage = data.get("usage") or {}
                total = usage.get("total_tokens")
                if total is None:
                    pt = int(usage.get("prompt_tokens") or 0)
                    ct = int(usage.get("completion_tokens") or 0)
                    if pt or ct:
                        total = pt + ct
                usage_total += int(total or 0)
                content = (
                    (data.get("choices") or [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
            except Exception as exc:
                trace(f"BLOG_SPATIAL {label} ✗ | bad response JSON | {exc}")
                parsed = None
                continue

            parsed = _parse_structured(str(content), schema)
            if parsed is not None:
                break
            trace(f"BLOG_SPATIAL {label} ✗ | schema parse failed")

        if parsed is None:
            from knowledge_engine.services.gemma_rate_limiter import (
                complete_structured_gemini_flash_async,
            )

            trace(
                f"BLOG_SPATIAL {label} ▶ | Gemma parse exhausted — Gemini Flash-Lite fallback"
            )
            flash = await complete_structured_gemini_flash_async(
                system,
                prompt,
                schema,
                label=f"{label}/flash_parse",
                max_output_tokens=out_tokens,
            )
            if flash is not None:
                return flash

        if budget is not None and usage_total > 0:
            await budget.reconcile_actual(usage_total)
        if limiter is not None and reconcile_tpm and usage_total > 0:
            await limiter.reconcile_batch_total(usage_total)
        return parsed


def build_gemma_model_slots(
    *, map_parallel_streams: bool = False
) -> list[GemmaModelSlot]:
    slots: list[GemmaModelSlot] = []
    shared_limiter: AsyncRateLimiter | None = None
    use_shared = GEMMA_QUOTA_SHARED and not map_parallel_streams
    if use_shared:
        shared_limiter = get_gemma_rate_limiter(
            slot="gemma:shared",
            max_rpm=GEMMA_BUDGET_MAX_RPM,
            max_tpm=GEMMA_BUDGET_MAX_TPM,
            max_rpd=GEMMA_MAX_RPD,
        )

    def _limiter_for(
        label: str, rpm: int, tpm: int, rpd: int, model: str
    ) -> AsyncRateLimiter:
        if shared_limiter is not None:
            return shared_limiter
        return get_gemma_rate_limiter(
            slot=f"gemma:{model}",
            max_rpm=rpm,
            max_tpm=tpm,
            max_rpd=rpd,
        )

    primary = (GEMMA_PRIMARY_MODEL or "").strip()
    if primary:
        slots.append(
            GemmaModelSlot(
                label="primary",
                model=primary,
                client=GemmaCloudClient(model=primary),
                limiter=_limiter_for(
                    "primary",
                    GEMMA_PRIMARY_MAX_RPM,
                    GEMMA_PRIMARY_MAX_TPM,
                    GEMMA_PRIMARY_MAX_RPD,
                    primary,
                ),
                failover_max_wait_sec=0.0,
            )
        )
    fallback = (GEMMA_FALLBACK_MODEL or "").strip()
    if fallback and fallback != primary:
        slots.append(
            GemmaModelSlot(
                label="fallback",
                model=fallback,
                client=GemmaCloudClient(model=fallback),
                limiter=_limiter_for(
                    "fallback",
                    GEMMA_FALLBACK_MAX_RPM,
                    GEMMA_FALLBACK_MAX_TPM,
                    GEMMA_FALLBACK_MAX_RPD,
                    fallback,
                ),
                failover_max_wait_sec=GEMMA_FALLBACK_MAX_WAIT_SEC,
            )
        )
    if shared_limiter is not None and slots:
        trace(
            f"BLOG_SPATIAL gemma quota | shared pool "
            f"rpm={GEMMA_MAX_RPM} tpm={GEMMA_MAX_TPM} rpd={GEMMA_MAX_RPD} "
            f"primary={primary or '—'} fallback={fallback if fallback != primary else '—'}"
        )
    elif map_parallel_streams and len(slots) > 1:
        trace(
            "BLOG_SPATIAL gemma quota | unified MAP pool → "
            f"{slots[0].model} + {slots[1].model} "
            f"({GEMMA_PRIMARY_MAX_TPM}+{GEMMA_FALLBACK_MAX_TPM} TPM/min) "
            f"| fixed-minute pacing"
        )
    return slots


class RateLimitedLLMClient:
    """Пул Gemma: primary → fallback при исчерпании лимитов или 429."""

    def __init__(
        self,
        slots: list[GemmaModelSlot] | None = None,
        *,
        map_parallel_streams: bool = False,
    ) -> None:
        self._slots = slots or build_gemma_model_slots(
            map_parallel_streams=map_parallel_streams
        )
        if not self._slots:
            trace("BLOG_SPATIAL gemma ⊘ | no GEMMA_PRIMARY_MODEL configured")

    @property
    def model_labels(self) -> str:
        return " → ".join(s.model for s in self._slots)

    def primary_limiter(self) -> AsyncRateLimiter | None:
        return self._slots[0].limiter if self._slots else None

    @property
    def model_slots(self) -> list[GemmaModelSlot]:
        return self._slots

    def estimate_request_tokens(
        self,
        system: str,
        prompt: str,
        schema: type[BaseModel] | None = None,
        *,
        max_output_tokens: int | None = None,
    ) -> int:
        if not self._slots:
            inp = estimate_text_tokens(f"{system}\n{prompt}")
            cap = (
                max_output_tokens
                if max_output_tokens is not None
                else resolve_gemma_map_max_output_tokens(inp)
            )
            return inp + cap
        return self._slots[0].client.estimate_request_tokens(
            system,
            prompt,
            schema=schema,
            max_output_tokens=max_output_tokens,
        )

    def estimate_budget(
        self, system: str, prompt: str, schema: type[BaseModel]
    ) -> tuple[int, int, int]:
        if not self._slots:
            inp = estimate_text_tokens(f"{system}\n{prompt}")
            cap = resolve_gemma_map_max_output_tokens(inp)
            return inp, cap, inp + cap
        return self._slots[0].client.estimate_budget(system, prompt, schema)

    async def acquire_parallel_wave(
        self,
        token_estimates: list[int],
        *,
        max_parallel: int,
    ) -> tuple[GemmaModelSlot | None, int, int]:
        """
        Зарезервировать до max_parallel запросов на общем/первом доступном лимитере.
        Возвращает (slot, k, reserved_tpm).
        """
        if not self._slots or not token_estimates:
            return None, 0, 0
        ests = [max(1, int(e)) for e in token_estimates[: max(1, max_parallel)]]
        seen_limiters: set[int] = set()
        for slot in self._slots:
            lid = id(slot.limiter)
            if lid in seen_limiters:
                continue
            seen_limiters.add(lid)
            k = await slot.limiter.try_acquire_parallel(
                ests,
                max_parallel=max_parallel,
                model=slot.model,
            )
            if k > 0:
                reserved = sum(ests[:k])
                snap = slot.limiter.snapshot(model=slot.model)
                trace(
                    f"BLOG_SPATIAL gemma wave reserve ✓ | {slot.model} ×{k} "
                    f"est_tpm={reserved} rpm {snap.rpm_used}/{snap.max_rpm} "
                    f"tpm {snap.tpm_used}/{snap.max_tpm}"
                )
                return slot, k, reserved
            trace(
                f"BLOG_SPATIAL gemma limits ⊘ | {slot.label} {slot.model} "
                "wave full — try next slot or wait"
            )
        lim = self._slots[0].limiter
        await lim.wait_for_room(ests[0])
        return await self.acquire_parallel_wave(
            token_estimates, max_parallel=max_parallel
        )

    async def post_structured_preacquired(
        self,
        slot: GemmaModelSlot,
        system: str,
        prompt: str,
        schema: type[T],
        *,
        label: str = "gemma",
        client: httpx.AsyncClient | None = None,
        max_tokens: int | None = None,
    ) -> tuple[T | None, int]:
        """HTTP без повторного acquire (после acquire_parallel_wave). usage_total для reconcile."""
        try:
            out = await slot.client.complete_structured(
                system,
                prompt,
                schema,
                label=f"{label}/{slot.model}",
                client=client,
                limiter=None,
                reconcile_tpm=False,
                max_tokens=max_tokens,
            )
        except GemmaRateLimitError:
            trace(f"BLOG_SPATIAL gemma 429 | {slot.model} (preacquired wave)")
            return None, 0
        if out is None:
            return None, 0
        est = self.estimate_request_tokens(
            system,
            prompt,
            schema=schema,
            max_output_tokens=max_tokens,
        )
        return out, est

    async def post_structured_with_failover(
        self,
        system: str,
        prompt: str,
        schema: type[T],
        *,
        label: str = "gemma",
        client: httpx.AsyncClient | None = None,
        max_tokens: int | None = None,
        prefer_slot: GemmaModelSlot | None = None,
    ) -> T | None:
        """Одиночный запрос (REDUCE): primary → fallback, с acquire."""
        if prefer_slot is not None:
            out, _ = await self.post_structured_preacquired(
                prefer_slot,
                system,
                prompt,
                schema,
                label=label,
                client=client,
                max_tokens=max_tokens,
            )
            if out is not None:
                return out
        return await self.post_structured(
            system,
            prompt,
            schema,
            label=label,
            client=client,
            slot=prefer_slot,
            max_tokens=max_tokens,
        )

    async def reconcile_batch_usage(
        self,
        slot: GemmaModelSlot,
        actual_totals: list[int],
        reserved_tpm: int,
    ) -> None:
        total = sum(int(x) for x in actual_totals if x)
        if total > 0:
            await slot.limiter.reconcile_batch_total(total)
        elif reserved_tpm > 0:
            await slot.limiter.reconcile_batch_total(reserved_tpm)

    async def _pick_slot(self, estimated_tokens: int) -> GemmaModelSlot | None:
        for slot in self._slots:
            ok = await slot.limiter.try_acquire(
                estimated_tokens,
                max_wait=slot.failover_max_wait_sec,
                model=slot.model,
            )
            if ok:
                if slot.label == "fallback":
                    trace(
                        f"BLOG_SPATIAL gemma failover ✓ | using {slot.model} "
                        "(primary limits exhausted)"
                    )
                return slot
            trace(
                f"BLOG_SPATIAL gemma limits ⊘ | {slot.label} {slot.model} "
                "busy — try next slot"
            )
        return None

    async def post_structured(
        self,
        system: str,
        prompt: str,
        schema: type[T],
        *,
        label: str = "gemma",
        client: httpx.AsyncClient | None = None,
        slot: GemmaModelSlot | None = None,
        reconcile_tpm: bool = True,
        max_tokens: int | None = None,
    ) -> T | None:
        if slot is not None:
            try:
                out = await slot.client.complete_structured(
                    system,
                    prompt,
                    schema,
                    label=f"{label}/{slot.model}",
                    client=client,
                    limiter=slot.limiter if reconcile_tpm else None,
                    reconcile_tpm=reconcile_tpm,
                    max_tokens=max_tokens,
                )
                if out is not None:
                    return out
            except GemmaRateLimitError:
                trace(
                    f"BLOG_SPATIAL gemma 429 | {slot.model} — try fallback if available"
                )
        est = self.estimate_request_tokens(
            system,
            prompt,
            schema=schema,
            max_output_tokens=max_tokens,
        )
        for s in self._slots:
            if slot is not None and s.model == slot.model:
                continue
            picked = await self._pick_slot(est)
            if picked is None:
                continue
            s = picked
            try:
                out = await s.client.complete_structured(
                    system,
                    prompt,
                    schema,
                    label=f"{label}/{s.model}",
                    client=client,
                    limiter=s.limiter,
                    max_tokens=max_tokens,
                )
                if out is not None:
                    return out
            except GemmaRateLimitError:
                trace(f"BLOG_SPATIAL gemma 429 | {s.model} — try fallback if available")
                continue
        return None

    async def complete_structured(
        self,
        system: str,
        prompt: str,
        schema: type[T],
        *,
        label: str = "gemma",
        client: httpx.AsyncClient | None = None,
    ) -> T | None:
        return await self.post_structured(
            system,
            prompt,
            schema,
            label=label,
            client=client,
        )
