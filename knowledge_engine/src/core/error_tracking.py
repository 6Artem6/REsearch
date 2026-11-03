"""Ошибки → локальный GlitchTip (Sentry-совместимый) через ``sentry-sdk``.

Без ``KE_SENTRY_DSN`` всё no-op (SDK не инициализируется, функции ничего не
шлют). Трейсинг остаётся за OpenTelemetry/Phoenix (observability.py):
performance-режим Sentry выключен, событие только помечается
``otel.trace_id`` — по нему трейс открывается в Phoenix
(``log_profiler.py --phoenix --trace <id>``), а спан получает
``ke.sentry.event_id`` для обратного перехода.

Три источника событий:

1. **fail-open** — ``trace()`` (run_log.py), вызванный внутри ``except``:
   исключение проглочено, конвейер идёт дальше по fallback. Уровень
   ``warning``, тег ``ke.failopen=true``, текст trace-строки в extra;
   ограничение частоты на ключ (тип исключения + начало сообщения).
2. **падение задачи / API** — ``capture_error`` явно (worker, обработчики
   500/5xx в app.py): уровень ``error``.
3. **logging** — штатная LoggingIntegration SDK: ``logger.exception`` /
   ``logger.error`` становятся событиями.

Одно и то же исключение, попавшее в несколько источников, отсекает
DedupeIntegration SDK (повтор того же объекта исключения подряд).
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from typing import Any

_init_lock = threading.Lock()
_initialized = False

# Не ошибки: штатная отмена/останов процесса.
_IGNORED_TYPES: tuple[type[BaseException], ...] = (
    asyncio.CancelledError,
    KeyboardInterrupt,
    SystemExit,
    GeneratorExit,
)


class _RateLimiter:
    """Не больше ``limit`` событий на ключ за ``window`` секунд.

    Fail-open в цикле по документам (например, один и тот же сбой
    эмбеддинга на 500 статьях) иначе превращается в 500 событий."""

    def __init__(
        self,
        limit: int,
        window: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limit = limit
        self._window = window
        self._clock = clock
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = self._clock()
        with self._lock:
            recent = [t for t in self._hits.get(key, []) if now - t < self._window]
            allowed = len(recent) < self._limit
            if allowed:
                recent.append(now)
            self._hits[key] = recent
            if len(self._hits) > 2000:  # ключи с истёкшим окном
                self._hits = {k: v for k, v in self._hits.items() if v}
            return allowed


_failopen_limiter = _RateLimiter(limit=5, window=60.0)


def init_error_tracking(role: str) -> bool:
    """Поднять sentry-sdk для процесса ``role`` (api / worker).

    Идемпотентно и fail-open. True — отправка событий включена."""
    from knowledge_engine.src.config.settings import (
        KE_SENTRY_CAPTURE_FAILOPEN,
        KE_SENTRY_DSN,
        KE_SENTRY_ENVIRONMENT,
        KE_SENTRY_RELEASE,
    )
    from knowledge_engine.src.core.run_log import set_trace_exception_hook
    from knowledge_engine.src.core.run_log import trace as run_trace

    global _initialized
    if not KE_SENTRY_DSN:
        return False
    with _init_lock:
        if _initialized:
            return True
        try:
            import sentry_sdk

            sentry_sdk.init(
                dsn=KE_SENTRY_DSN,
                environment=KE_SENTRY_ENVIRONMENT,
                release=KE_SENTRY_RELEASE or None,
                server_name=role,
                # Трейсинг — OpenTelemetry → Phoenix; Sentry только ошибки.
                traces_sample_rate=None,
                # Без PII и локальных переменных: в кадрах лежат тексты
                # статей, промпты и ключи API.
                send_default_pii=False,
                include_local_variables=False,
                max_breadcrumbs=50,
                before_send=_before_send,
                integrations=_web_integrations(),
            )
            sentry_sdk.set_tag("ke.role", role)
            _initialized = True
        except Exception as exc:
            run_trace(f"SENTRY ✗ init | role={role} | {type(exc).__name__}: {exc}")
            return False
    if KE_SENTRY_CAPTURE_FAILOPEN:
        set_trace_exception_hook(capture_failopen)
    run_trace(
        f"SENTRY ✓ init | role={role} env={KE_SENTRY_ENVIRONMENT} "
        f"failopen={KE_SENTRY_CAPTURE_FAILOPEN}"
    )
    return True


def _web_integrations() -> list[Any]:
    """HTTPException 5xx отправляют обработчики app.py — с исходной причиной
    (``__cause__``); штатный захват по статус-коду дал бы второе событие
    с обёрткой. Необработанные исключения интеграции ловят как обычно."""
    try:
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration
    except Exception:
        return []
    return [
        StarletteIntegration(failed_request_status_codes=set()),
        FastApiIntegration(failed_request_status_codes=set()),
    ]


def shutdown_error_tracking(timeout: float = 5.0) -> None:
    """Дослать буфер событий (вызывать при остановке процесса)."""
    if not _initialized:
        return
    try:
        import sentry_sdk

        sentry_sdk.flush(timeout=timeout)
    except Exception:
        pass


def _otel_ids() -> tuple[str, str] | None:
    from opentelemetry import trace

    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return None
    return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")


def _before_send(event: dict[str, Any], hint: dict[str, Any]) -> dict[str, Any]:
    """Связать событие с OTel-трейсом (before_send идёт в вызывающем потоке)."""
    try:
        tags = event.setdefault("tags", {})
        if "otel.trace_id" not in tags:
            ids = _otel_ids()
            if ids is not None:
                tags["otel.trace_id"], tags["otel.span_id"] = ids
    except Exception:
        pass
    return event


def _is_ignored(exc: BaseException) -> bool:
    if isinstance(exc, _IGNORED_TYPES):
        return True
    # HTTPException 4xx (404/409/422) — ответ клиенту, не сбой.
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and status < 500


def _mark_span(event_id: str | None) -> None:
    if not event_id:
        return
    try:
        from knowledge_engine.src.core.observability import set_span_attributes

        set_span_attributes(**{"ke.sentry.event_id": event_id})
    except Exception:
        pass


def capture_error(
    exc: BaseException,
    *,
    level: str = "error",
    tags: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    trace_context: dict[str, str] | None = None,
) -> str | None:
    """Явная отправка исключения. ``trace_context`` — W3C carrier
    (``WorkJob.trace_context``), когда спан задачи уже закрыт. Возвращает
    event_id или None (выключено / проигнорировано / ошибка SDK)."""
    if not _initialized or _is_ignored(exc):
        return None
    try:
        import sentry_sdk

        with sentry_sdk.new_scope() as scope:
            scope.set_level(level)
            for key, value in (tags or {}).items():
                if value is not None:
                    scope.set_tag(key, str(value)[:200])
            for key, value in (extra or {}).items():
                scope.set_extra(key, value)
            trace_id = _trace_id_from_carrier(trace_context)
            if trace_id:
                scope.set_tag("otel.trace_id", trace_id)
            event_id = sentry_sdk.capture_exception(exc)
    except Exception:
        return None
    _mark_span(event_id)
    return event_id


def capture_failopen(exc: BaseException, message: str) -> str | None:
    """Хук ``trace()``: проглоченное исключение с текстом trace-строки."""
    if not _initialized or _is_ignored(exc):
        return None
    # HTTPException любого кода — ответ клиенту; 5xx уже отправлены явно
    # из обработчиков app.py (с исходной причиной, а не обёрткой).
    if isinstance(getattr(exc, "status_code", None), int):
        return None
    head = message.split("|", 1)[0].strip()[:80]
    if not _failopen_limiter.allow(f"{type(exc).__name__}:{head}"):
        return None
    return capture_error(
        exc,
        level="warning",
        tags={"ke.failopen": "true", "ke.trace_label": head},
        extra={"trace_message": message[:2000]},
    )


def _trace_id_from_carrier(carrier: dict[str, str] | None) -> str | None:
    """``00-<trace_id>-<span_id>-<flags>`` → trace_id."""
    parts = str((carrier or {}).get("traceparent") or "").split("-")
    return parts[1] if len(parts) == 4 and len(parts[1]) == 32 else None


__all__ = [
    "capture_error",
    "capture_failopen",
    "init_error_tracking",
    "shutdown_error_tracking",
]
