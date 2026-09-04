"""Gemma 4 cloud API + rate-limited pool (primary / fallback models)."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Callable, TypeVar, get_origin

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
from knowledge_engine.services.llm.rate_limiter import (
    AsyncRateLimiter,
    get_gemma_rate_limiter,
)
from knowledge_engine.ui.run_log import trace

logger = logging.getLogger(__name__)


def _estimate_text_tokens(text: str) -> int:
    from knowledge_engine.services.article_ingestion.paragraph_token_splitter import (
        estimate_text_tokens,
    )

    return estimate_text_tokens(text)

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


class GemmaTransientError(Exception):
    """Retryable cloud failure: HTTP 5xx, timeouts, transport errors."""


@dataclass
class GemmaModelSlot:
    label: str
    model: str
    client: GemmaCloudClient
    limiter: AsyncRateLimiter
    failover_max_wait_sec: float = 0.0


def resolve_gemma_map_max_output_tokens(
    input_tokens: int | None = None, *, projected_out: int | None = None
) -> int:
    """MAP completion cap — from config.GEMMA_MAP_MAX_OUTPUT_TOKENS by
    default (``input_tokens`` kept for call-site compatibility; no
    input-size branching on its own — see BUG FIXED note below).

    ``projected_out``: an ALREADY content-aware output estimate from the
    caller (e.g. ``dynamic_target_facts()`` in blog_spatial_summarizer.py —
    proportional to how many knowledge_atoms a window can realistically
    need, not just its input size). When given, the cap becomes
    ``min(GEMMA_MAP_MAX_OUTPUT_TOKENS, projected_out + margin)`` instead of
    the flat ceiling — this is Step 1 of the "почему MAP-вызовы 35-93s+"
    audit: `dynamic_target_facts()` already computed this number and logged
    it, but nothing fed it back into the actual API `max_tokens`, so every
    window — regardless of how little it had to say — was allowed to
    generate up to the full flat cap, and a model that used a good chunk of
    that budget took proportionally longer with no benefit (window content
    doesn't need more than its own fact budget). Callers that don't have a
    per-window estimate (e.g. dynamic_target_facts()'s own internal call to
    get ITS ceiling) keep getting the flat, static cap — unchanged.

    BUG FIXED (pre-existing, kept for history): this used to hardcode 4096
    unconditionally, silently ignoring GEMMA_MAP_MAX_OUTPUT_TOKENS from
    config/.env entirely. Every MAP call in this codebase's whole session
    history ran with max_tokens=4096 regardless of what config said — the
    config layer itself was wired correctly (config.py already does
    int(os.getenv("GEMMA_MAP_MAX_OUTPUT_TOKENS", ...))), the disconnect was
    only here.
    """
    from knowledge_engine.config import GEMMA_MAP_MAX_OUTPUT_TOKENS

    _ = input_tokens
    if projected_out is not None and projected_out > 0:
        # Margin covers window_role + JSON structure overhead beyond the
        # raw knowledge_atoms text that projected_out itself estimates.
        margin = max(200, int(projected_out * 0.3))
        return max(256, min(GEMMA_MAP_MAX_OUTPUT_TOKENS, projected_out + margin))
    return GEMMA_MAP_MAX_OUTPUT_TOKENS


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


def _is_thought_only_empty_response(content: str) -> bool:
    """True when the model emitted nothing but a <thought> block (or empty
    text) — no JSON at all, open or closed. Retrying the identical prompt
    against the identical model reproduces this same non-answer far more
    often than it fixes it (it's a model behavior pattern, not a truncation
    or escaping glitch); observed repeatedly for ConsensusBatchResponse in
    production (perf_debug.log: json_error=empty JSON payload after
    thought/fence cleanup). Skipping the same-prompt retry for this specific
    case avoids paying for a doomed extra ~20-60s HTTP round-trip before
    falling through to the Gemini Flash-Lite fallback."""
    stripped = _strip_markdown_fences(_strip_gemma_thought_wrapper((content or "").strip()))
    return not stripped


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


def _wrap_bare_list_for_schema(data: object, schema: type[T]) -> object:
    """Gemma sometimes emits a bare JSON array instead of the wrapper object the
    schema expects; if the schema has exactly one list-typed field, wrap into it."""
    if not isinstance(data, list):
        return data
    list_fields = [
        name
        for name, info in schema.model_fields.items()
        if get_origin(info.annotation) is list
    ]
    if len(list_fields) == 1:
        return {list_fields[0]: data}
    return data


def _drop_truncated_list_tail(
    data: object, errors: list[dict]
) -> tuple[dict, int] | None:
    """Gemma's max_tokens cutoff often lands mid-object inside a trailing
    list item (e.g. knowledge_atoms[5] missing its 'statement' field) —
    the rest of the response is valid JSON, only the tail item is broken.
    Truncating the offending list(s) at the first bad index and re-validating
    salvages the response without paying for a full extra HTTP round-trip
    (schema parse retry costs another 20-70s Gemma call). Returns
    (repaired_data, dropped_count) or None if no error matches this shape."""
    if not isinstance(data, dict):
        return None
    truncate_at: dict[str, int] = {}
    for err in errors:
        loc = err.get("loc") or ()
        if len(loc) < 2:
            continue
        field, idx = loc[0], loc[1]
        if not isinstance(field, str) or not isinstance(idx, int):
            continue
        if not isinstance(data.get(field), list):
            continue
        truncate_at[field] = min(truncate_at.get(field, idx), idx)
    if not truncate_at:
        return None
    repaired = dict(data)
    dropped = 0
    for field, idx in truncate_at.items():
        original = data[field]
        repaired[field] = original[:idx]
        dropped += len(original) - idx
    if dropped <= 0:
        return None
    return repaired, dropped


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
        wrapped = _wrap_bare_list_for_schema(data, schema)
        if wrapped is not data:
            try:
                return schema.model_validate(wrapped)
            except ValidationError:
                pass
        repair = _drop_truncated_list_tail(data, exc.errors())
        if repair is not None:
            repaired_data, dropped = repair
            try:
                result = schema.model_validate(repaired_data)
                trace(
                    f"BLOG_SPATIAL parse ⚠ | schema={schema_name} "
                    f"repaired truncated tail — dropped {dropped} malformed "
                    "trailing item(s), kept the rest"
                )
                return result
            except ValidationError:
                pass
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
        return _estimate_text_tokens(f"{system}\n{user_prompt}")

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
            inp = _estimate_text_tokens(f"{system}\n{prompt}")
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
        on_usage: Callable[[int], None] | None = None,
    ) -> T | None:
        """``on_usage`` — вызывается с РЕАЛЬНЫМ ``usage_total`` из ответа API
        (не оценкой) прямо перед возвратом, если он есть. Нужен вызывающим,
        которые не передают ``limiter`` (preacquired-путь — реконсиляция
        нескольких элементов волны на один слот должна суммировать их
        реальные usage, а не полагаться на одно значение через
        ``reconcile_batch_total``, которое просто заменяет последнюю запись
        и не подходит, когда k>1 элементов делят одну резервацию слота)."""
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
            retry=retry_if_exception_type((GemmaRateLimitError, GemmaTransientError)),
            stop=stop_after_attempt(3),
            wait=_GemmaRetryWait(),
            reraise=True,
        )
        async def _post_once() -> httpx.Response:
            t0 = time.monotonic()
            trace(f"GEMMA HTTP ▶ {label} | model={self._model}")
            try:
                if client is not None:
                    resp = await client.post(url, json=payload, headers=headers)
                else:
                    timeout = httpx.Timeout(self._timeout)
                    async with httpx.AsyncClient(timeout=timeout) as ephemeral:
                        resp = await ephemeral.post(url, json=payload, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                trace(
                    f"GEMMA HTTP ✗ {label} | {time.monotonic() - t0:.2f}s | "
                    f"{type(exc).__name__}: {exc}"
                )
                raise GemmaTransientError(str(exc)) from exc
            elapsed = time.monotonic() - t0
            if resp.status_code == 429:
                trace(f"GEMMA HTTP ✗ {label} | {elapsed:.2f}s | HTTP 429")
                retry_after: float | None = None
                ra = resp.headers.get("Retry-After")
                if ra:
                    try:
                        retry_after = float(ra)
                    except ValueError:
                        retry_after = None
                raise GemmaRateLimitError(retry_after)
            if resp.status_code in (500, 502, 503, 504):
                trace(f"GEMMA HTTP ✗ {label} | {elapsed:.2f}s | HTTP {resp.status_code}")
                raise GemmaTransientError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            trace(f"GEMMA HTTP ✓ {label} | {elapsed:.2f}s")
            return resp

        try:
            resp = await _post_once()
        except GemmaRateLimitError:
            if limiter is not None:
                # Возвращаем реальный 429 в admission-лимитер ЭТОЙ модели —
                # budget.record_429_spike ниже обновляет только отдельный,
                # теперь обходимый (use_token_budget=False для MAP/REDUCE)
                # глобальный GemmaTokenBudgetManager; без этого следующий
                # _pick_slot/try_acquire на той же модели видит устаревшее
                # "ok" и не переключается на fallback.
                await limiter.record_429_spike(est_total or None, model=self._model)
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
                    if limiter is not None:
                        await limiter.record_429_spike(est_total or None, model=self._model)
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
            if parse_try == 0 and _is_thought_only_empty_response(str(content)):
                trace(
                    f"BLOG_SPATIAL {label} ⊘ | empty thought-only response — "
                    "skip same-prompt retry, go straight to fallback"
                )
                break

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
        if on_usage is not None and usage_total > 0:
            on_usage(usage_total)
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
            inp = _estimate_text_tokens(f"{system}\n{prompt}")
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
            inp = _estimate_text_tokens(f"{system}\n{prompt}")
            cap = resolve_gemma_map_max_output_tokens(inp)
            return inp, cap, inp + cap
        return self._slots[0].client.estimate_budget(system, prompt, schema)

    async def acquire_parallel_wave(
        self,
        token_estimates: list[int],
        *,
        max_parallel: int,
    ) -> list[tuple[GemmaModelSlot, int, int, list | None]]:
        """
        Dual-basket: распределить волну между слотами пропорционально их
        ТЕКУЩЕЙ свободной TPM/RPM-емкости в скользящем окне. Часть волны,
        помещающаяся на первом слоте (обычно 31B primary), резервируется
        там; остаток, не поместившийся из-за исчерпания лимита первого
        слота, мгновенно (без ожидания) пробуется на следующем слоте пула
        (обычно 26B fallback), и так далее.

        Возвращает план [(slot, k, reserved_tpm, event), ...]: первые k
        записей исходного token_estimates (в порядке появления) уходят на
        первый slot плана, следующие k — на второй, и т.д. ``event`` —
        ссылка на резервацию этой группы в лимитере слота; вызывающий код
        обязан передать её обратно в reconcile (см. reconcile_batch_usage) —
        под конкурентным диспетчером несколько групп могут одновременно
        висеть на одном лимитере. Пустой список — ничего не поместилось ни
        на одном слоте (вызывающий код должен подождать и повторить попытку).
        """
        if not self._slots or not token_estimates:
            return []
        remaining = [max(1, int(e)) for e in token_estimates[: max(1, max_parallel)]]
        budget = max(1, max_parallel)
        plan: list[tuple[GemmaModelSlot, int, int, list | None]] = []
        seen_limiters: set[int] = set()
        for slot in self._slots:
            if not remaining or budget <= 0:
                break
            lid = id(slot.limiter)
            if lid in seen_limiters:
                continue
            seen_limiters.add(lid)
            k, event = await slot.limiter.try_acquire_parallel(
                remaining,
                max_parallel=budget,
                model=slot.model,
            )
            if k <= 0:
                trace(
                    f"BLOG_SPATIAL gemma limits ⊘ | {slot.label} {slot.model} "
                    "wave full — try next slot"
                )
                continue
            reserved = sum(remaining[:k])
            snap = slot.limiter.snapshot(model=slot.model)
            trace(
                f"BLOG_SPATIAL gemma wave reserve ✓ | {slot.model} ×{k} "
                f"est_tpm={reserved} rpm {snap.rpm_used}/{snap.max_rpm} "
                f"tpm {snap.tpm_used}/{snap.max_tpm}"
            )
            plan.append((slot, k, reserved, event))
            remaining = remaining[k:]
            budget -= k

        if not plan:
            lim = self._slots[0].limiter
            await lim.wait_for_room(max(1, int(token_estimates[0])))
            return await self.acquire_parallel_wave(
                token_estimates, max_parallel=max_parallel
            )
        return plan

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
        """HTTP без повторного acquire (после acquire_parallel_wave). usage_total для reconcile.

        BUG FIXED: ``complete_structured()`` defaults to ``use_token_budget=
        True`` — without an explicit override here it ran a SECOND admission
        check, ``GemmaTokenBudgetManager.acquire_budget()``, on top of the
        one ``acquire_parallel_wave`` already did. That manager is a single
        GLOBAL, per-process budget (``GEMMA_BUDGET_MAX_TPM/RPM``, default =
        ``GEMMA_GOVERNOR_TARGET_TPM/RPM`` = 15200/27 — sized for ONE model)
        shared across BOTH primary and fallback slots, so once their
        COMBINED usage crossed ~15200 TPM it started diverting MAP calls to
        Gemini Flash-Lite (``flash_overflow``) even while each individual
        Gemma model still had its own independent headroom free — this is
        exactly why the AI Studio dashboard showed both models stuck at
        roughly 40-50% of their own 16K TPM ceiling instead of each filling
        up close to it. This call already went through the dual-basket
        per-model wave admission — it must not re-acquire against a
        differently-scoped budget on top of that.
        """
        real_usage: list[int] = []
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
                use_token_budget=False,
                on_usage=real_usage.append,
            )
        except GemmaRateLimitError:
            trace(f"BLOG_SPATIAL gemma 429 | {slot.model} (preacquired wave)")
            return None, 0
        if out is None:
            return None, 0
        if real_usage:
            # Реальный usage_total из ответа API — не оценка. Не зовём
            # limiter.reconcile_batch_total() напрямую здесь (limiter=None
            # выше): когда k>1 элементов волны делят ОДНУ резервацию слота
            # (см. try_acquire_parallel — один общий token_event на всю
            # волну), несколько параллельных вызовов reconcile_batch_total
            # затёрли бы друг друга (replace, не sum). Вызывающий
            # (acquire_parallel_wave-волна) суммирует real_usage всех k
            # элементов слота и реконсилирует ОДНИМ вызовом — см.
            # reconcile_batch_usage в blog_spatial_summarizer.py.
            return out, real_usage[0]
        # Без usage в ответе (не должно случаться в норме) — деградируем на
        # оценку, чтобы вызывающий всё равно получил разумное число.
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
        priority: bool = True,
    ) -> T | None:
        """Одиночный запрос (REDUCE): primary → fallback, с acquire.
        priority=True (default) — REDUCE sees each slot's full TPM/RPM
        ceiling, ignoring the headroom reserved away from MAP callers, so
        document completion doesn't queue behind MAP's own usage."""
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
            priority=priority,
        )

    async def reconcile_batch_usage(
        self,
        slot: GemmaModelSlot,
        actual_totals: list[int],
        reserved_tpm: int,
        *,
        event: list | None = None,
    ) -> None:
        total = sum(int(x) for x in actual_totals if x)
        if total > 0:
            await slot.limiter.reconcile_batch_total(total, event=event)
        elif reserved_tpm > 0:
            await slot.limiter.reconcile_batch_total(reserved_tpm, event=event)

    async def _pick_slot(
        self, estimated_tokens: int, *, priority: bool = False
    ) -> GemmaModelSlot | None:
        for slot in self._slots:
            ok = await slot.limiter.try_acquire(
                estimated_tokens,
                max_wait=slot.failover_max_wait_sec,
                model=slot.model,
                priority=priority,
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
        priority: bool = False,
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
                    # BUG FIXED: same double-gate as post_structured_preacquired
                    # (see its docstring) — this slot was already acquired via
                    # a real per-model check (prefer_slot's own admission, or
                    # the fallback-loop's _pick_slot below); the GLOBAL,
                    # single-process GemmaTokenBudgetManager check inside
                    # complete_structured() must not run a second time on top
                    # of that, or it caps combined primary+fallback usage at
                    # ~15200 TPM (one model's worth) instead of each model's
                    # own independent ~16K ceiling.
                    use_token_budget=False,
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
            picked = await self._pick_slot(est, priority=priority)
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
                    # BUG FIXED: _pick_slot() above already ran the real
                    # per-model admission check (slot.limiter.try_acquire) —
                    # see the note on the prefer_slot branch above for why a
                    # second, globally-shared GemmaTokenBudgetManager gate
                    # here silently halved combined Gemma throughput.
                    use_token_budget=False,
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
