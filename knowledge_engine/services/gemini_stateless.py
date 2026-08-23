"""Stateless Gemini через google-genai SDK (без Playwright chat history)."""

from __future__ import annotations

import json
import random
import re
import time
import warnings
from typing import Any, Callable, Type, TypeVar, Union

from pydantic import BaseModel

from knowledge_engine.config import (
    CURRICULUM_GEMINI_GROUNDING_FALLBACK_MODELS,
    CURRICULUM_GEMINI_GROUNDING_MODEL,
    GEMINI_API_KEY,
    GEMINI_API_KEYS,
    GEMINI_API_TIMEOUT_SEC,
    GEMINI_CLIENT,
    GEMINI_GROUNDING_ENABLED,
    GEMINI_GROUNDING_MODEL,
    GEMINI_LITE_FALLBACK_MODELS,
    GEMINI_LITE_MODEL,
    GEMINI_PROBE_BEFORE_USE,
    GEMINI_PROBE_TIMEOUT_SEC,
    GEMINI_REASONER_FALLBACK_MODELS,
    GEMINI_RETRY_BACKOFF_SEC,
    GEMINI_RPM_JITTER_SEC,
    GEMINI_RPM_PAUSE_SEC,
    GEMINI_TUTOR_MODEL,
    SKIP_GEMINI,
    _normalize_grounding_model_id,
)
from knowledge_engine.services.gemini_json_stream import (
    extract_clean_json,
    structured_stream_text_field,
)
from knowledge_engine.ui.run_log import trace

T = TypeVar("T", bound=BaseModel)


def estimate_llm_tokens(text: str, model_name: str = "") -> int:
    """Оценка токенов для логов и session traces (offline fast_tokenizer when vocabs exist)."""
    from knowledge_engine.src.utils.fast_tokenizer import token_counter

    alias = (model_name or "").strip() or "gemini"
    return token_counter.count_tokens(text or "", alias)


def finish_reason_from_gemini_response(response: Any) -> str:
    if response is None:
        return ""
    cands = getattr(response, "candidates", None) or []
    if not cands:
        return ""
    fr = getattr(cands[0], "finish_reason", None)
    if fr is None:
        return ""
    return str(fr).strip()


def warn_if_gemini_truncated(
    label: str,
    finish_reason: str,
    *,
    output_len: int = 0,
) -> None:
    fr = (finish_reason or "").upper().replace(" ", "_")
    if not fr:
        return
    if "MAX" in fr and "TOKEN" in fr:
        trace(
            f"[GEMINI TRUNCATED: Output exceeded max_tokens] {label} | "
            f"finish_reason={finish_reason} | output_len={output_len}"
        )


def trace_gemini_io_sizes(
    label: str,
    *,
    system_instruction: str = "",
    user_payload: str = "",
    output_text: str = "",
    max_output_tokens: int | None = None,
    finish_reason: str = "",
    model_name: str = "",
) -> None:
    sys_len = len(system_instruction or "")
    user_len = len(user_payload or "")
    in_len = sys_len + user_len
    out_len = len(output_text or "")
    trace(
        f"GEMINI IO | {label} | input_prompt_len={in_len} sym "
        f"(system={sys_len} user={user_len}) est_in_tokens≈{estimate_llm_tokens(system_instruction, model_name) + estimate_llm_tokens(user_payload, model_name)} "
        f"| output_len={out_len} est_out_tokens≈{estimate_llm_tokens(output_text, model_name)} "
        f"| max_output_tokens={max_output_tokens if max_output_tokens is not None else 'sdk_default'}"
        + (f" | finish_reason={finish_reason}" if finish_reason else "")
    )
    warn_if_gemini_truncated(label, finish_reason, output_len=out_len)
    try:
        from knowledge_engine.services.session_prompt_trace import (
            consume_pending_trace_output,
        )

        consume_pending_trace_output(
            output_text,
            finish_reason=finish_reason,
            model_name=model_name,
        )
    except Exception:
        pass


# Optional import for chat-isolated calls (node deep dive).
ChatSessionManagerType = Any


class GeminiUnavailableError(RuntimeError):
    """Нет ключа, SDK или явный SKIP_GEMINI."""


class GeminiQuotaExhaustedError(RuntimeError):
    """Все модели в цепочке исчерпали квоту."""


_clients_by_key_timeout: dict[tuple[str, int], Any] = {}


def gemini_api_key_pool() -> list[str]:
    return [k for k in GEMINI_API_KEYS if (k or "").strip()]


def _make_client(timeout_sec: float, api_key: str | None = None) -> Any:
    from google import genai
    from google.genai import types

    key = (api_key or GEMINI_API_KEY or "").strip()
    if not key:
        raise GeminiUnavailableError("Пустой GEMINI API key")
    timeout_ms = max(1, int(timeout_sec * 1000))
    cache_key = (key, timeout_ms)
    if cache_key in _clients_by_key_timeout:
        return _clients_by_key_timeout[cache_key]
    client = genai.Client(
        api_key=key,
        http_options=types.HttpOptions(timeout=timeout_ms),
    )
    _clients_by_key_timeout[cache_key] = client
    return client


def _client_for_api_key(api_key: str, timeout_sec: float | None = None) -> Any:
    t = float(timeout_sec if timeout_sec is not None else GEMINI_API_TIMEOUT_SEC)
    return _make_client(t, api_key=api_key)


def is_gemini_available() -> bool:
    return bool(gemini_api_key_pool()) and not SKIP_GEMINI and GEMINI_CLIENT is not None


def _client(timeout_sec: float | None = None) -> Any:
    if not is_gemini_available():
        raise GeminiUnavailableError(
            "Gemini недоступен: задайте GEMINI_API_KEY, установите google-genai "
            "или не используйте SKIP_GEMINI=true."
        )
    t = float(timeout_sec if timeout_sec is not None else GEMINI_API_TIMEOUT_SEC)
    if abs(t - GEMINI_API_TIMEOUT_SEC) < 0.01 and GEMINI_CLIENT is not None:
        return GEMINI_CLIENT
    return _make_client(t)


_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_RETRYABLE_MARKERS = (
    "RESOURCE_EXHAUSTED",
    "429",
    "503",
    "502",
    "500",
    "504",
    "UNAVAILABLE",
    "INTERNAL",
    "READTIMEOUT",
    "CONNECTTIMEOUT",
    "WRITETIMEOUT",
    "POOLTIMEOUT",
    "TIMED OUT",
)
_HTTP_TIMEOUT_TYPE_NAMES = frozenset(
    {
        "ReadTimeout",
        "ConnectTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "TimeoutException",
        "TimeoutError",
    }
)

_QUOTA_MARKERS = (
    "quota exceeded",
    "generate_content_free_tier",
    "generaterequestsperday",
    "free_tier_requests",
    "exceeded your current quota",
)


def _gemini_error_blob(exc: BaseException) -> str:
    parts: list[str] = [str(exc)]
    for attr in ("message", "response", "details"):
        val = getattr(exc, attr, None)
        if val is not None:
            parts.append(str(val))
    return "\n".join(parts)


def _google_retry_delay_sec(exc: BaseException) -> float | None:
    """Пауза из ответа API: «Please retry in …s» или RetryInfo retryDelay."""
    blob = _gemini_error_blob(exc)
    m = re.search(r"retry in (\d+(?:\.\d+)?)\s*s", blob, re.I)
    if not m:
        m = re.search(
            r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)\s*s", blob, re.I
        )
    if not m:
        return None
    return min(float(m.group(1)) + 0.5, 180.0)


def _lite_eval_fast_retry_label(label: str) -> bool:
    low = (label or "").lower()
    return "lite" in low or "lite_batch" in low


def _lite_eval_retry_wait_sec(
    google_wait: float | None,
    attempt: int,
) -> tuple[float | None, str]:
    import random

    if google_wait is not None:
        capped = min(max(google_wait, 1.0), 3.0)
        return capped, "lite-fast"
    if attempt < 1:
        return 1.0 + random.random() * 2.0, "lite-fast"
    return None, ""


def _is_daily_per_model_quota(exc: BaseException) -> bool:
    """Free tier: quotaId GenerateRequestsPerDayPerProjectPerModel — ждать минуту не поможет."""
    blob = _gemini_error_blob(exc).lower().replace("_", "").replace("-", "")
    return (
        "requestsperdayperprojectpermodel" in blob or "generaterequestsperday" in blob
    )


def _is_hard_quota_exhausted(exc: BaseException) -> bool:
    """
    Дневная / жёсткая квота → перейти на следующую модель в chain.
    RPM (есть retry in …s) → False, остаёмся на модели и ждём.
    Generic 429 без per-day / limit:0 → False (backoff на той же модели).
    """
    if _google_retry_delay_sec(exc) is not None:
        return False
    msg = _gemini_error_blob(exc).lower()
    if "limit: 0" in msg:
        return True
    if "perday" in msg or "per_day" in msg or "generaterequestsperday" in msg:
        return True
    if "perminute" in msg or "per_minute" in msg or "per minute" in msg:
        return False
    code = _extract_status_code(exc)
    if code == 429 or "resource_exhausted" in msg:
        return False
    if not any(marker in msg for marker in _QUOTA_MARKERS):
        return False
    return True


def _is_heavy_flash_model(model: str) -> bool:
    m = (model or "").lower()
    if "flash-lite" in m or "flash_lite" in m:
        return False
    if "gemma" in m:
        return False
    if "3.6-flash" in m or "3.5-flash" in m:
        return True
    if "flash-preview" in m:
        return True
    return False


def _dedupe_model_chain(names: tuple[str, ...] | list[str]) -> list[str]:
    chain: list[str] = []
    for name in names:
        m = (name or "").strip()
        if m and m not in chain:
            chain.append(m)
    return chain


def _is_gemma_model_id(model: str) -> bool:
    return "gemma" in (model or "").strip().lower()


def _strip_gemma_from_chain(models: list[str]) -> list[str]:
    return [m for m in models if m and not _is_gemma_model_id(m)]


def gemini_lite_model_chain(primary: str | None = None) -> list[str]:
    """Lite tier + Flash-Lite fallbacks only (no Gemma)."""
    chain = _dedupe_model_chain(
        (
            (primary or GEMINI_LITE_MODEL),
            GEMINI_LITE_MODEL,
            *GEMINI_LITE_FALLBACK_MODELS,
        )
    )
    return _strip_gemma_from_chain(chain)


def gemini_reasoner_model_chain(primary: str | None = None) -> list[str]:
    """Graph construction only: GEMINI_REASONER_MODEL + GEMINI_REASONER_FALLBACK_MODELS."""
    from knowledge_engine.config import GEMINI_REASONER_MODEL

    return _dedupe_model_chain(
        (
            (primary or GEMINI_REASONER_MODEL),
            GEMINI_REASONER_MODEL,
            *GEMINI_REASONER_FALLBACK_MODELS,
        )
    )


def _is_search_grounding_model(model: str) -> bool:
    """Google Search tool: flash-lite / 2.x; не reasoner-tier 3.5/3.6."""
    m = (model or "").strip().lower()
    if not m.startswith("gemini-"):
        return False
    blocked = (
        "gemini-3.6",
        "gemini-3.5-flash-preview",
        "gemini-3-flash-preview",
        "gemini-3.1-pro",
        "gemini-3-pro",
        "-pro",
    )
    if any(b in m for b in blocked):
        return False
    if "gemini-3.5-flash" in m and "lite" not in m:
        return False
    return True


def default_tpm_limit_for_model(model: str) -> int:
    from knowledge_engine.config import GEMINI_FLASH_LITE_MAX_TPM

    m = (model or "").lower()
    if "gemma" in m:
        return 16000
    if "flash-lite" in m or "flash_lite" in m:
        return max(100, int(GEMINI_FLASH_LITE_MAX_TPM))
    if "3.6-flash" in m or "3.5-flash" in m:
        return 250000
    if "flash-preview" in m:
        return 250000
    return 60000


def default_rpm_limit_for_model(model: str) -> int:
    """Hard per-model RPM ceiling shared by VLM + tutor + overflow + curriculum."""
    from knowledge_engine.config import (
        GEMINI_FLASH_LITE_MAX_RPM,
        GEMMA_MAX_RPM,
    )

    m = (model or "").lower()
    if "gemma" in m:
        return max(1, int(GEMMA_MAX_RPM))
    if "flash-lite" in m or "flash_lite" in m:
        return max(1, int(GEMINI_FLASH_LITE_MAX_RPM))
    if "3.6-flash" in m or "3.5-flash" in m:
        return 5
    return 10


def default_rpd_limit_for_model(model: str) -> int:
    from knowledge_engine.services.gemini_quota_store import default_daily_limit_rpd

    return default_daily_limit_rpd(model)


def curriculum_grounding_model_chain(primary: str | None = None) -> list[str]:
    """Search grounding tier — disabled unless GEMINI_GROUNDING_ENABLED."""
    if not GEMINI_GROUNDING_ENABLED:
        return []
    raw = _dedupe_model_chain(
        (
            (primary or CURRICULUM_GEMINI_GROUNDING_MODEL),
            CURRICULUM_GEMINI_GROUNDING_MODEL,
            GEMINI_GROUNDING_MODEL,
            *CURRICULUM_GEMINI_GROUNDING_FALLBACK_MODELS,
        )
    )
    raw = [m for m in raw if (m or "").strip()]
    if not raw:
        return []
    for i, name in enumerate(raw):
        fixed = _normalize_grounding_model_id(name)
        if fixed != (name or "").strip():
            trace(
                f"CURRICULUM grounding model alias | {name} → {fixed} "
                f"(в API нет «{name}», часто HTTP 404)"
            )
            raw[i] = fixed
    filtered = [m for m in raw if _is_search_grounding_model(m)]
    if not filtered:
        return []
    return filtered


def gemini_model_chain() -> list[str]:
    return gemini_reasoner_model_chain()


def gemini_model_chain_for(primary: str) -> list[str]:
    p = (primary or "").strip()
    if _is_heavy_flash_model(p):
        return gemini_reasoner_model_chain(p)
    return gemini_lite_model_chain(p)


def gemini_tutor_model_chain(primary: str | None = None) -> list[str]:
    return gemini_lite_model_chain(primary or GEMINI_TUTOR_MODEL)


def gemini_min_interval_sec(model: str) -> float:
    rpm = max(1, default_rpm_limit_for_model(model))
    return max(60.0 / rpm, 4.0)


def _sleep_with_jitter(seconds: float) -> None:
    time.sleep(max(0.0, seconds) + random.uniform(0, GEMINI_RPM_JITTER_SEC))


def _rpm_pause_for_model(model: str) -> None:
    m = (model or "").strip()
    if not m:
        return
    base = max(GEMINI_RPM_PAUSE_SEC, gemini_min_interval_sec(m))
    wait = base + random.uniform(0, GEMINI_RPM_JITTER_SEC)
    if wait > GEMINI_RPM_PAUSE_SEC + 0.1:
        trace(
            f"GEMINI RPM spacing {wait:.1f}s | model={m} "
            f"(ориентир {default_rpm_limit_for_model(m)} RPM)"
        )
    time.sleep(wait)


def _extract_status_code(exc: BaseException) -> int | None:
    for attr in ("status_code", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    msg = str(exc)
    m = re.search(r"\b(429|5\d{2})\b", msg)
    if m:
        return int(m.group(1))
    return None


def _is_http_timeout(exc: BaseException) -> bool:
    """True for httpx/httpcore/SDK read/connect timeouts (transient; try next model)."""
    cur: BaseException | None = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if type(cur).__name__ in _HTTP_TIMEOUT_TYPE_NAMES:
            return True
        cur = cur.__cause__ or cur.__context__
    msg = str(exc).upper()
    return any(
        marker in msg
        for marker in (
            "READTIMEOUT",
            "CONNECTTIMEOUT",
            "WRITETIMEOUT",
            "THE READ OPERATION TIMED OUT",
            "THE WRITE OPERATION TIMED OUT",
        )
    )


def _is_retryable(exc: BaseException) -> bool:
    if _is_hard_quota_exhausted(exc):
        return False
    if _is_http_timeout(exc):
        return True
    if _google_retry_delay_sec(exc) is not None:
        return True
    code = _extract_status_code(exc)
    if code is not None and code in _RETRYABLE_STATUS:
        return True
    msg = str(exc).upper()
    return any(marker in msg for marker in _RETRYABLE_MARKERS)


def _generate_once(
    model: str,
    combined_user: str,
    system_instruction: str,
    response_schema: Type[T] | None,
    log_label: str = "",
    http_timeout_sec: float | None = None,
    stream_callback: Callable[[str], None] | None = None,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    from google.genai import types

    config_kwargs: dict[str, Any] = {
        "system_instruction": system_instruction,
    }
    if response_schema is not None:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = response_schema
    if max_output_tokens is not None:
        config_kwargs["max_output_tokens"] = int(max_output_tokens)
    if temperature is not None:
        config_kwargs["temperature"] = float(temperature)

    lab = (log_label or "generate_content").strip()
    tout = float(
        http_timeout_sec if http_timeout_sec is not None else GEMINI_API_TIMEOUT_SEC
    )
    try:
        from knowledge_engine.services.session_prompt_trace import (
            write_stateless_gemini_prompt_trace,
        )

        write_stateless_gemini_prompt_trace(
            trace_label=lab,
            model_name=model,
            system_instruction=system_instruction,
            user_payload=combined_user,
        )
    except Exception:
        pass
    trace(
        f"GEMINI HTTP ▶ {lab} | model={model} | "
        f"лимит HTTP={tout:.0f}s (не пауза; нормально ~2–20s) | payload≈{len(combined_user)} sym"
        + (" | stream" if stream_callback is not None else "")
    )
    t0 = time.perf_counter()
    client = _client(tout)
    finish_reason = ""
    if stream_callback is not None:
        chat = client.chats.create(
            model=model,
            config=types.GenerateContentConfig(**config_kwargs),
            history=[],
        )
        cum_text = ""
        stream = chat.send_message_stream((combined_user or "").strip())
        last_chunk: Any = None
        for chunk in stream:
            last_chunk = chunk
            piece = getattr(chunk, "text", None) or ""
            if not piece:
                continue
            if cum_text and piece.startswith(cum_text):
                delta_raw = piece[len(cum_text) :]
                cum_text = piece
            else:
                delta_raw = piece
                cum_text += piece
            if delta_raw:
                stream_callback(delta_raw)
        text = cum_text.strip()
        finish_reason = finish_reason_from_gemini_response(last_chunk)
    else:
        response = client.models.generate_content(
            model=model,
            contents=combined_user,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        text = (response.text or "").strip()
        finish_reason = finish_reason_from_gemini_response(response)
    elapsed = time.perf_counter() - t0
    if not text:
        raise RuntimeError("Gemini stateless: пустой ответ")
    from knowledge_engine.ui.llm_trace import trace_llm_exchange

    trace_llm_exchange(lab, system_instruction, combined_user, text, model=model)
    trace_gemini_io_sizes(
        lab,
        system_instruction=system_instruction,
        user_payload=combined_user,
        output_text=text,
        max_output_tokens=max_output_tokens,
        finish_reason=finish_reason,
        model_name=model,
    )
    trace(
        f"GEMINI HTTP ✓ {lab} | model={model} | {elapsed:.1f}s | ответ {len(text)} sym"
    )
    return text


def _rpm_pause() -> None:
    _rpm_pause_for_model(GEMINI_LITE_MODEL)


def _generate_multimodal_once(
    client: Any,
    model: str,
    combined_user: str,
    system_instruction: str,
    response_schema: Type[T] | None,
    image_parts: list[tuple[bytes, str]],
) -> str:
    from google.genai import types

    parts: list[Any] = [types.Part.from_text(text=combined_user)]
    for raw, mime in image_parts:
        parts.append(types.Part.from_bytes(data=raw, mime_type=mime))

    config_kwargs: dict[str, Any] = {"system_instruction": system_instruction}
    if response_schema is not None:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = response_schema

    response = client.models.generate_content(
        model=model,
        contents=types.Content(role="user", parts=parts),
        config=types.GenerateContentConfig(**config_kwargs),
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini multimodal: пустой ответ")
    return text


def _parse_structured(
    text: str,
    response_schema: Type[T] | None,
    label: str,
) -> Union[str, T]:
    if response_schema is None:
        return text
    cleaned = extract_clean_json(text)
    try:
        data = json.loads(cleaned)
        return response_schema.model_validate(data)
    except Exception as exc:
        raise RuntimeError(f"Gemini JSON не прошёл валидацию ({label}): {exc}") from exc


def _combine_anchor(global_anchor: str, body: str) -> str:
    return (
        f"GLOBAL ANCHOR (задача и контекст, не игнорировать):\n{global_anchor.strip()}\n\n"
        f"{body.strip()}"
    )


def probe_gemini_model(model: str, label: str = "probe") -> tuple[bool, str]:
    """Минимальный запрос: модель отвечает? Не трогает chat history."""
    m = (model or "").strip()
    if not m:
        return False, "empty model id"
    t0 = time.perf_counter()
    trace(f"GEMINI probe ▶ {m} | {label}")
    try:
        _generate_once(
            m,
            "ping",
            "Reply with exactly one word: OK",
            None,
            f"probe/{label}",
            GEMINI_PROBE_TIMEOUT_SEC,
        )
        trace(f"GEMINI probe ✓ {m} | {label} | {time.perf_counter() - t0:.1f}s")
        return True, ""
    except Exception as exc:
        err = f"{type(exc).__name__}: {str(exc)[:240]}"
        trace(f"GEMINI probe ✗ {m} | {label} | {err}")
        return False, err


def _reorder_models_after_probe(
    model_list: list[str],
    label: str,
    track_quota: bool,
    record_gemini_error: Callable[[str, BaseException], None] | None,
) -> list[str]:
    if not GEMINI_PROBE_BEFORE_USE or len(model_list) <= 1:
        if not GEMINI_PROBE_BEFORE_USE and len(model_list) > 1:
            trace(
                f"GEMINI probe off | {label} | локальный RPD + fallback при ошибке API"
            )
        return model_list
    for i, m in enumerate(model_list):
        ok, err = probe_gemini_model(m, label)
        if ok:
            if track_quota and record_gemini_error is not None:
                from knowledge_engine.services.gemini_quota_store import (
                    record_gemini_success,
                )

                record_gemini_success(m)
            if i > 0:
                trace(
                    f"GEMINI probe | {label} | переключение на {m} "
                    f"(в chain не отвечает: {model_list[0]})"
                )
            return [m] + [x for x in model_list if x != m]
        if track_quota and record_gemini_error is not None:
            record_gemini_error(m, RuntimeError(err or "probe failed"))
    return []


def _sync_chat_session_primary_model(
    chat_mgr: ChatSessionManagerType | None,
    chat_label: str,
    model_list: list[str],
    handoff_summary: str,
    label: str,
) -> None:
    if chat_mgr is None or not chat_label or not model_list:
        return
    primary = model_list[0]
    stored = chat_mgr.resolve_for_model(chat_label, primary, handoff_summary)
    trace(
        f"GEMINI chat bind | {label} | label={chat_label} | "
        f"model={stored.model_name} | context={stored.context_type}"
    )


def _call_with_model_fallback(
    label: str,
    generate_for_model: Callable[[str], str],
    rpm_pause: bool = False,
    models: list[str] | None = None,
    chat_manager: ChatSessionManagerType | None = None,
    chat_label: str = "",
    handoff_summary: str = "",
    session_registry: ChatSessionManagerType | None = None,
) -> str:
    trace(f"GEMINI ▶ chain start | {label}")

    model_list = models or gemini_model_chain()
    trace(
        f"GEMINI chain prep | {label} | requested "
        + " → ".join(model_list[:6])
        + (" …" if len(model_list) > 6 else "")
    )
    from knowledge_engine.services.gemini_quota_store import (
        filter_models_for_quota,
        quota_tracking_enabled,
        record_gemini_error,
        record_gemini_success,
    )

    track_quota = quota_tracking_enabled()
    if track_quota:
        trace(f"GEMINI chain prep | {label} | локальный quota filter…")
        raw_list = list(model_list)
        model_list = filter_models_for_quota(model_list)
        skipped = [m for m in raw_list if m not in model_list]
        if skipped:
            trace(
                f"GEMINI RPD skip | {label} | локальный счётчик / RPM: "
                + ", ".join(skipped)
            )
        if not model_list:
            raise GeminiQuotaExhaustedError(
                f"Локальный счётчик RPD исчерпан для всех моделей в chain ({label}). "
                "Сброс RPD: Pacific midnight (~12:00 UTC+5). "
                "Или удалите knowledge_engine/.runs/gemini_quota_state.json"
            )
    trace(
        f"GEMINI chain prep | {label} | probe reorder (off={not GEMINI_PROBE_BEFORE_USE})…"
    )
    model_list = _reorder_models_after_probe(
        model_list,
        label,
        track_quota,
        record_gemini_error if track_quota else None,
    )
    if not model_list:
        raise GeminiUnavailableError(
            f"Нет доступных моделей в chain ({label}). "
            "Проверьте GEMINI_API_KEY, квоты и GEMINI_LITE_FALLBACK_MODELS."
        )
    trace(
        f"GEMINI chain ready | {label} | "
        + " → ".join(model_list[:6])
        + (" …" if len(model_list) > 6 else "")
    )
    reg = session_registry if session_registry is not None else chat_manager
    trace(f"GEMINI chain prep | {label} | chat session bind…")
    _sync_chat_session_primary_model(
        reg, chat_label, model_list, handoff_summary, label
    )
    delays = list(GEMINI_RETRY_BACKOFF_SEC)
    last_exc: BaseException | None = None
    quota_models: list[str] = []

    trace(
        f"GEMINI chain | {label} | "
        + " → ".join(model_list[:5])
        + (" …" if len(model_list) > 5 else "")
    )

    for model_index, model in enumerate(model_list):
        if (
            chat_manager is not None
            and chat_label
            and model_index > 0
            and model_list[model_index - 1] != model
        ):
            chat_manager.create_new_session(
                model,
                chat_label,
                handoff_summary,
                "Summary",
            )
        trace(f"GEMINI stateless ▶ {label} | model={model}")
        if rpm_pause or GEMINI_RPM_PAUSE_SEC > 0:
            _rpm_pause_for_model(model)
        attempt = 0
        rpm_waits = 0
        while True:
            reserved_minute = False
            if track_quota:
                from knowledge_engine.services.gemini_quota_store import (
                    reserve_gemini_minute_slot,
                )

                if not reserve_gemini_minute_slot(model, 800):
                    quota_models.append(model)
                    trace(
                        f"GEMINI RPM hard_cap skip | {label} | model={model} "
                        f"— next in chain (≤{default_rpm_limit_for_model(model)}/min)"
                    )
                    break
                reserved_minute = True
            try:
                text = generate_for_model(model)
                if model_index > 0:
                    trace(
                        f"GEMINI fallback ✓ {label} | succeeded on {model} "
                        f"(after {model_list[0]})"
                    )
                trace(f"GEMINI stateless ✓ {label} | model={model} | {len(text)} sym")
                if track_quota:
                    record_gemini_success(
                        model, minute_already_reserved=reserved_minute
                    )
                return text
            except Exception as exc:
                if track_quota:
                    record_gemini_error(model, exc)
                last_exc = exc
                if not _is_retryable(exc):
                    raise

                google_wait = _google_retry_delay_sec(exc)
                daily_quota = _is_daily_per_model_quota(exc)
                lite_fast = _lite_eval_fast_retry_label(label)

                if _is_hard_quota_exhausted(exc):
                    quota_models.append(model)
                    trace(f"GEMINI quota ✗ {model} | {label} — пробуем fallback-модель")
                    break

                if daily_quota and google_wait is not None and rpm_waits >= 1:
                    quota_models.append(model)
                    trace(
                        f"GEMINI daily quota ✗ {model} | {label} "
                        f"(лимит дня, не RPM) — fallback-модель"
                    )
                    break

                # Do not burn multi-minute backoff on the same model after a hard
                # HTTP read/connect timeout — fall through to the next chain model.
                if _is_http_timeout(exc):
                    next_model = (
                        model_list[model_index + 1]
                        if model_index + 1 < len(model_list)
                        else None
                    )
                    trace(
                        f"GEMINI timeout ✗ {model} | {label} | "
                        f"{type(exc).__name__} — "
                        + (
                            f"fallback ▶ {next_model}"
                            if next_model
                            else "нет следующей модели в chain"
                        )
                    )
                    break

                wait: float | None = None
                wait_src = ""
                if lite_fast and not _is_hard_quota_exhausted(exc):
                    wait, wait_src = _lite_eval_retry_wait_sec(google_wait, attempt)
                elif google_wait is not None:
                    wait = google_wait
                    wait_src = "API"
                elif attempt < len(delays):
                    wait = delays[attempt]
                    wait_src = "backoff"

                if wait is not None:
                    if google_wait is not None:
                        rpm_waits += 1
                    trace(
                        f"GEMINI wait {wait:.0f}s ({wait_src}) | "
                        f"{label} | model={model} | причина: {type(exc).__name__}"
                    )
                    _sleep_with_jitter(wait)
                    if lite_fast:
                        if rpm_waits >= 1 or attempt >= 1:
                            trace(
                                f"GEMINI lite fast-fail | {label} | model={model} "
                                "— следующая модель в chain"
                            )
                            break
                        attempt += 1
                        continue
                    if google_wait is not None:
                        continue
                    attempt += 1
                    continue

                next_model = (
                    model_list[model_index + 1]
                    if model_index + 1 < len(model_list)
                    else None
                )
                trace(
                    f"GEMINI overload ✗ {model} | {label} — "
                    + (
                        f"fallback ▶ {next_model}"
                        if next_model
                        else "нет следующей модели в chain"
                    )
                )
                break

    if quota_models:
        chain_note = ", ".join(model_list)
        raise GeminiQuotaExhaustedError(
            f"Жёсткая квота Gemini для моделей: {', '.join(quota_models)} "
            f"(цепочка уже была: {chain_note}). "
            "RPM free tier обычно проходит после паузы 30–60s на той же модели; "
            f"дневная квота — billing или другие модели. ({label})"
        )
    if last_exc:
        raise last_exc
    raise RuntimeError(f"Gemini stateless failed ({label})")


def run_stateless_gemini(
    system_instruction: str,
    user_payload: str,
    global_anchor: str,
    response_schema: Type[T] | None = None,
    label: str = "stateless_gemini",
    rpm_pause: bool = False,
) -> Union[str, T]:
    """
    Один изолированный запрос: global_anchor + system + payload.
    При дневной квоте (429) — fallback на GEMINI_FALLBACK_MODELS без retry на той же модели.
    """
    combined_user = (
        f"GLOBAL ANCHOR (задача и L0-контекст, не игнорировать):\n{global_anchor.strip()}\n\n"
        f"{user_payload.strip()}"
    )

    def _gen(model: str) -> str:
        return _generate_once(
            model, combined_user, system_instruction, response_schema, label
        )

    text = _call_with_model_fallback(label, _gen, rpm_pause=rpm_pause)
    return _parse_structured(text, response_schema, label)


def run_gemini_structured_with_chain(
    primary_model: str,
    system_instruction: str,
    user_payload: str,
    global_anchor: str,
    response_schema: Type[T],
    label: str,
    rpm_pause: bool = False,
    chat_manager: ChatSessionManagerType | None = None,
    chat_label: str = "",
    delta_user_message: str = "",
    handoff_summary: str = "",
    models: list[str] | None = None,
    http_timeout_sec: float | None = None,
    session_registry: ChatSessionManagerType | None = None,
    pinned_context: str = "",
    stream_callback: Callable[[str], None] | None = None,
    stream_text_field: str | None = None,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    prompt_trace: Any | None = None,
    layer1_context: str = "",
    layer2_context: str = "",
    node_session_key: str = "",
) -> T:
    """Structured JSON с retry (503/5xx) и fallback: primary → GEMINI_MODEL → FALLBACKS."""
    static_body = user_payload.strip()
    movable = _combine_anchor(global_anchor, static_body)
    tout = http_timeout_sec
    client = _client(tout)
    model_list = models or gemini_model_chain_for(primary_model)
    lab = (chat_label or label).strip()
    field = (stream_text_field or "").strip() or structured_stream_text_field(
        response_schema
    )

    def _gen(model: str) -> str:
        if chat_manager is not None:
            from knowledge_engine.services.gemini_cache_manager import (
                get_or_create_explicit_cache,
            )

            cache_res = None
            if (layer1_context or "").strip():
                cache_res = get_or_create_explicit_cache(
                    client,
                    model=model,
                    system_instruction=system_instruction,
                    layer1_body=layer1_context,
                    node_session_key=node_session_key,
                    label=lab,
                )
            message, dialog_user, payload_meta = chat_manager.build_user_payload(
                lab,
                pinned_context,
                movable,
                delta_user_message,
                layer1_context=layer1_context,
                layer2_context=layer2_context,
                explicit_cache=cache_res,
            )
            if stream_callback is not None:
                return chat_manager.send_chat_message_stream(
                    client,
                    lab,
                    model,
                    message,
                    system_instruction,
                    response_schema,
                    handoff_summary,
                    stream_callback,
                    record_user_text=dialog_user,
                    stream_text_field=field,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    prompt_trace=prompt_trace,
                    explicit_cache=cache_res,
                    pinned_context=pinned_context,
                    movable_context=movable,
                    delta_user_message=delta_user_message,
                    layer1_context=layer1_context,
                    layer2_context=layer2_context,
                    payload_meta=payload_meta,
                )
            return chat_manager.send_chat_message(
                client,
                lab,
                model,
                message,
                system_instruction,
                response_schema,
                handoff_summary,
                record_user_text=dialog_user,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                prompt_trace=prompt_trace,
                explicit_cache=cache_res,
                pinned_context=pinned_context,
                movable_context=movable,
                delta_user_message=delta_user_message,
                layer1_context=layer1_context,
                layer2_context=layer2_context,
                payload_meta=payload_meta,
            )
        full_context = movable
        if pinned_context.strip():
            full_context = f"{pinned_context.strip()}\n\n{full_context}"
        if delta_user_message.strip():
            full_context = f"{full_context}\n\n{delta_user_message.strip()}"
        field_filter = None
        stream_cb: Callable[[str], None] | None = stream_callback
        if stream_callback is not None and field:
            from knowledge_engine.services.gemini_json_stream import (
                wrap_stream_callback_for_json_field,
            )

            field_filter = wrap_stream_callback_for_json_field(field, stream_callback)
            if field_filter is not None:
                stream_cb = field_filter.feed
        text = _generate_once(
            model,
            full_context,
            system_instruction,
            response_schema,
            label,
            tout,
            stream_callback=stream_cb,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        if field_filter is not None:
            field_filter.flush()
        return text

    text = _call_with_model_fallback(
        label,
        _gen,
        rpm_pause=rpm_pause,
        models=model_list,
        chat_manager=chat_manager,
        chat_label=lab,
        handoff_summary=handoff_summary,
        session_registry=session_registry,
    )
    return _parse_structured(text, response_schema, label)


def run_gemini_text_with_chain(
    primary_model: str,
    system_instruction: str,
    user_payload: str,
    global_anchor: str,
    label: str,
    rpm_pause: bool = False,
    stream_callback: Callable[[str], None] | None = None,
    models: list[str] | None = None,
) -> str:
    """Deprecated internal free-text helper.

    New product features must use ``run_gemini_structured_with_chain`` with an
    explicit Pydantic ``response_schema``.
    """
    warnings.warn(
        "run_gemini_text_with_chain is deprecated; use "
        "run_gemini_structured_with_chain with a Pydantic response_schema",
        DeprecationWarning,
        stacklevel=2,
    )
    combined_user = (
        f"GLOBAL ANCHOR (задача и контекст, не игнорировать):\n{global_anchor.strip()}\n\n"
        f"{user_payload.strip()}"
    )
    model_list = models or gemini_model_chain_for(primary_model)

    def _gen(model: str) -> str:
        return _generate_once(
            model,
            combined_user,
            system_instruction,
            None,
            label,
            stream_callback=stream_callback,
        )

    text = _call_with_model_fallback(
        label, _gen, rpm_pause=rpm_pause, models=model_list
    )
    return text.strip()


def run_stateless_gemini_multimodal(
    system_instruction: str,
    user_payload: str,
    global_anchor: str,
    image_parts: list[tuple[bytes, str]],
    response_schema: Type[T] | None = None,
    label: str = "stateless_gemini_multimodal",
    rpm_pause: bool = True,
    models: list[str] | None = None,
) -> Union[str, T]:
    """Vision/Code: изображения + текст, stateless."""
    combined_user = f"GLOBAL ANCHOR:\n{global_anchor.strip()}\n\n{user_payload.strip()}"
    client = _client()

    def _gen(model: str) -> str:
        return _generate_multimodal_once(
            client,
            model,
            combined_user,
            system_instruction,
            response_schema,
            image_parts,
        )

    trace_prefix = f"{label} | images={len(image_parts)}"
    text = _call_with_model_fallback(
        trace_prefix, _gen, rpm_pause=rpm_pause, models=models
    )
    return _parse_structured(text, response_schema, label)


def global_anchor_from_state(
    original_query: str, constraints: str, l0_summary: str
) -> str:
    parts = [
        f"Задача: {original_query}",
        f"Ограничения: {constraints or '(не указаны)'}",
    ]
    if l0_summary.strip():
        parts.append(f"L0 (мета-карта):\n{l0_summary.strip()}")
    return "\n".join(parts)
