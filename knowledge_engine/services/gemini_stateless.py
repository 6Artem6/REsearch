"""Stateless Gemini через google-genai SDK (без Playwright chat history)."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Type, TypeVar, Union

from pydantic import BaseModel

from knowledge_engine.config import (
    GEMINI_API_KEY,
    GEMINI_CLIENT,
    GEMINI_FALLBACK_MODELS,
    GEMINI_MODEL,
    GEMINI_RETRY_BACKOFF_SEC,
    GEMINI_RPM_PAUSE_SEC,
    SKIP_GEMINI,
)
from knowledge_engine.ui.run_log import trace

T = TypeVar("T", bound=BaseModel)


class GeminiUnavailableError(RuntimeError):
    """Нет ключа, SDK или явный SKIP_GEMINI."""


class GeminiQuotaExhaustedError(RuntimeError):
    """Все модели в цепочке исчерпали квоту."""


def is_gemini_available() -> bool:
    return bool(GEMINI_API_KEY) and not SKIP_GEMINI and GEMINI_CLIENT is not None


def _client() -> Any:
    if not is_gemini_available():
        raise GeminiUnavailableError(
            "Gemini недоступен: задайте GEMINI_API_KEY, установите google-genai "
            "или не используйте SKIP_GEMINI=true."
        )
    return GEMINI_CLIENT


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
    """
    if _google_retry_delay_sec(exc) is not None:
        return False
    msg = _gemini_error_blob(exc).lower()
    if "perday" in msg or "per_day" in msg or "generaterequestsperday" in msg:
        return True
    if not any(marker in msg for marker in _QUOTA_MARKERS):
        return False
    if "perminute" in msg or "per_minute" in msg or "per minute" in msg:
        return False
    code = _extract_status_code(exc)
    return code == 429 or "resource_exhausted" in msg


def gemini_model_chain() -> list[str]:
    chain: list[str] = []
    for name in (GEMINI_MODEL, *GEMINI_FALLBACK_MODELS):
        m = (name or "").strip()
        if m and m not in chain:
            chain.append(m)
    return chain


def gemini_model_chain_for(primary: str) -> list[str]:
    """Primary (LITE/FLASH) → GEMINI_MODEL → GEMINI_FALLBACK_MODELS, без дубликатов."""
    chain: list[str] = []
    for name in (primary, GEMINI_MODEL, *GEMINI_FALLBACK_MODELS):
        m = (name or "").strip()
        if m and m not in chain:
            chain.append(m)
    return chain


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


def _is_retryable(exc: BaseException) -> bool:
    if _is_hard_quota_exhausted(exc):
        return False
    if _google_retry_delay_sec(exc) is not None:
        return True
    code = _extract_status_code(exc)
    if code is not None and code in _RETRYABLE_STATUS:
        return True
    msg = str(exc).upper()
    return any(marker in msg for marker in _RETRYABLE_MARKERS)


def _generate_once(
    client: Any,
    model: str,
    combined_user: str,
    system_instruction: str,
    response_schema: Type[T] | None,
) -> str:
    from google.genai import types

    config_kwargs: dict[str, Any] = {
        "system_instruction": system_instruction,
    }
    if response_schema is not None:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = response_schema

    response = client.models.generate_content(
        model=model,
        contents=combined_user,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Gemini stateless: пустой ответ")
    return text


def _rpm_pause() -> None:
    if GEMINI_RPM_PAUSE_SEC > 0:
        time.sleep(GEMINI_RPM_PAUSE_SEC)


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
    try:
        data = json.loads(text)
        return response_schema.model_validate(data)
    except Exception as exc:
        raise RuntimeError(f"Gemini JSON не прошёл валидацию ({label}): {exc}") from exc


def _call_with_model_fallback(
    label: str,
    generate_for_model: Callable[[str], str],
    rpm_pause: bool = False,
    models: list[str] | None = None,
) -> str:
    if rpm_pause:
        _rpm_pause()

    model_list = models or gemini_model_chain()
    delays = list(GEMINI_RETRY_BACKOFF_SEC)
    last_exc: BaseException | None = None
    quota_models: list[str] = []

    trace(
        f"GEMINI chain | {label} | "
        + " → ".join(model_list[:5])
        + (" …" if len(model_list) > 5 else "")
    )

    for model_index, model in enumerate(model_list):
        trace(f"GEMINI stateless ▶ {label} | model={model}")
        attempt = 0
        rpm_waits = 0
        while True:
            try:
                text = generate_for_model(model)
                if model_index > 0:
                    trace(
                        f"GEMINI fallback ✓ {label} | succeeded on {model} "
                        f"(after {model_list[0]})"
                    )
                trace(f"GEMINI stateless ✓ {label} | model={model} | {len(text)} sym")
                return text
            except Exception as exc:
                last_exc = exc
                if not _is_retryable(exc):
                    raise

                google_wait = _google_retry_delay_sec(exc)
                daily_quota = _is_daily_per_model_quota(exc)

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

                wait: float | None = None
                wait_src = ""
                if google_wait is not None:
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
                        f"{label} | model={model} | {exc}"
                    )
                    time.sleep(wait)
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
    client = _client()

    def _gen(model: str) -> str:
        return _generate_once(
            client, model, combined_user, system_instruction, response_schema
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
) -> T:
    """Structured JSON с retry (503/5xx) и fallback: primary → GEMINI_MODEL → FALLBACKS."""
    combined_user = (
        f"GLOBAL ANCHOR (задача и контекст, не игнорировать):\n{global_anchor.strip()}\n\n"
        f"{user_payload.strip()}"
    )
    client = _client()
    models = gemini_model_chain_for(primary_model)

    def _gen(model: str) -> str:
        return _generate_once(
            client, model, combined_user, system_instruction, response_schema
        )

    text = _call_with_model_fallback(label, _gen, rpm_pause=rpm_pause, models=models)
    return _parse_structured(text, response_schema, label)


def run_gemini_text_with_chain(
    primary_model: str,
    system_instruction: str,
    user_payload: str,
    global_anchor: str,
    label: str,
    rpm_pause: bool = False,
) -> str:
    combined_user = (
        f"GLOBAL ANCHOR (задача и контекст, не игнорировать):\n{global_anchor.strip()}\n\n"
        f"{user_payload.strip()}"
    )
    client = _client()
    models = gemini_model_chain_for(primary_model)

    def _gen(model: str) -> str:
        return _generate_once(client, model, combined_user, system_instruction, None)

    text = _call_with_model_fallback(label, _gen, rpm_pause=rpm_pause, models=models)
    return text.strip()


def run_stateless_gemini_multimodal(
    system_instruction: str,
    user_payload: str,
    global_anchor: str,
    image_parts: list[tuple[bytes, str]],
    response_schema: Type[T] | None = None,
    label: str = "stateless_gemini_multimodal",
    rpm_pause: bool = True,
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
    text = _call_with_model_fallback(trace_prefix, _gen, rpm_pause=rpm_pause)
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
