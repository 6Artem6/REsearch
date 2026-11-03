"""OpenTelemetry-трейсинг: API → Redis → worker → LangGraph → LLM.

Экспорт — OTLP/HTTP в локальный Arize Phoenix (см. docs/OBSERVABILITY.md).
Без ``KE_OTEL_ENABLED`` ``init_observability`` ничего не делает, а
``span()``/``traced()`` работают через no-op трейсер OpenTelemetry API —
вызывающий код не ветвится по флагу.

Текстовый ``trace()`` (run_log.py) остаётся основным логом: спаны его
дополняют, а не заменяют (``log_profiler.py`` по-прежнему парсит run log).
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import heapq
import inspect
import json
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode, TracerProvider
from opentelemetry.trace.propagation.tracecontext import (
    TraceContextTextMapPropagator,
)

_TRACER_NAME = "knowledge_engine"
_propagator = TraceContextTextMapPropagator()
_init_lock = threading.Lock()
_initialized_role: str | None = None
# Явный провайдер: init_observability() / use_tracer_provider() в тестах.
# None → глобальный провайдер OpenTelemetry (no-op, пока его никто не задал).
_provider: TracerProvider | None = None
# Отправка span annotations в Phoenix; None — оценки пишутся только в спан.
_annotation_sender: _AnnotationSender | None = None

F = TypeVar("F", bound=Callable[..., Any])

# Атрибуты OpenInference (семантика LLM-спанов, которую понимает Phoenix).
_ATTR_SPAN_KIND = "openinference.span.kind"
_ATTR_MODEL = "llm.model_name"
_ATTR_PROVIDER = "llm.provider"
_ATTR_TOKENS_PROMPT = "llm.token_count.prompt"
_ATTR_TOKENS_COMPLETION = "llm.token_count.completion"
_ATTR_TOKENS_TOTAL = "llm.token_count.total"
_ATTR_INPUT = "input.value"
_ATTR_INPUT_MIME = "input.mime_type"
_ATTR_OUTPUT = "output.value"


def _tracer() -> trace.Tracer:
    provider = _provider or trace.get_tracer_provider()
    return provider.get_tracer(_TRACER_NAME)


def use_tracer_provider(provider: TracerProvider | None) -> None:
    """Подменить провайдер (тесты с InMemorySpanExporter). None — сброс."""
    global _provider
    _provider = provider


def init_observability(role: str) -> bool:
    """Поднять TracerProvider и автоинструментацию для процесса ``role``.

    Идемпотентно: повторный вызов в том же процессе — no-op. Fail-open:
    ошибка экспортёра/инструментации не должна ронять API или worker.
    Возвращает True, если трейсинг реально включён."""
    from knowledge_engine.src.config.settings import (
        KE_OTEL_ANNOTATIONS_ENABLED,
        KE_OTEL_CAPTURE_LLM_IO,
        KE_OTEL_ENABLED,
        KE_OTEL_ENDPOINT,
        KE_OTEL_SERVICE_NAME,
    )
    from knowledge_engine.src.core.run_log import trace as run_trace

    global _initialized_role, _provider, _annotation_sender
    if not KE_OTEL_ENABLED:
        return False
    with _init_lock:
        if _initialized_role is not None:
            return True
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            provider = SdkTracerProvider(
                resource=Resource.create(
                    {
                        "service.name": f"{KE_OTEL_SERVICE_NAME}-{role}",
                        "openinference.project.name": KE_OTEL_SERVICE_NAME,
                    }
                )
            )
            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=KE_OTEL_ENDPOINT))
            )
            trace.set_tracer_provider(provider)
            _provider = provider
            _initialized_role = role
        except Exception as exc:
            run_trace(f"OTEL ✗ init | role={role} | {type(exc).__name__}: {exc}")
            return False
        _instrument_libraries(provider, capture_io=KE_OTEL_CAPTURE_LLM_IO)
        annotations_url = (
            _phoenix_annotations_url(KE_OTEL_ENDPOINT)
            if KE_OTEL_ANNOTATIONS_ENABLED
            else None
        )
        if annotations_url:
            _annotation_sender = _AnnotationSender(_httpx_poster(annotations_url))
        run_trace(
            f"OTEL ✓ init | role={role} endpoint={KE_OTEL_ENDPOINT} "
            f"capture_llm_io={KE_OTEL_CAPTURE_LLM_IO} "
            f"annotations={annotations_url or 'off'}"
        )
        return True


def _instrument_libraries(provider: TracerProvider, *, capture_io: bool) -> None:
    """Gemini (google-genai SDK) и LangGraph (через langchain-core runnables).

    Gemma идёт сырым httpx в OpenAI-совместимый endpoint — её спаны ставит
    ``llm_span`` вручную в gemma_client.py."""
    from knowledge_engine.src.core.run_log import trace as run_trace

    config: Any = None
    try:
        from openinference.instrumentation import TraceConfig

        config = TraceConfig(
            hide_inputs=not capture_io,
            hide_outputs=not capture_io,
        )
    except Exception as exc:
        run_trace(f"OTEL ⚠ TraceConfig skip | {exc}")

    for label, import_path in (
        (
            "google_genai",
            "openinference.instrumentation.google_genai:GoogleGenAIInstrumentor",
        ),
        (
            "langchain",
            "openinference.instrumentation.langchain:LangChainInstrumentor",
        ),
    ):
        module_name, class_name = import_path.split(":")
        try:
            module = __import__(module_name, fromlist=[class_name])
            instrumentor = getattr(module, class_name)()
            kwargs: dict[str, Any] = {"tracer_provider": provider}
            if config is not None:
                kwargs["config"] = config
            instrumentor.instrument(**kwargs)
            run_trace(f"OTEL ✓ instrument | {label}")
        except Exception as exc:
            run_trace(f"OTEL ⚠ instrument skip | {label} | {exc}")


def shutdown_observability() -> None:
    """Сбросить буфер BatchSpanProcessor и очередь оценок (при остановке процесса).

    Порядок важен: сначала спаны уходят в Phoenix, потом оценки к ним —
    иначе Phoenix отвечает 404 на аннотацию к ещё не принятому спану."""
    provider = _provider
    force_flush = getattr(provider, "force_flush", None)
    if callable(force_flush):
        with contextlib.suppress(Exception):
            force_flush()
    if _annotation_sender is not None:
        _annotation_sender.flush(timeout=10.0)
    shutdown = getattr(provider, "shutdown", None)
    if callable(shutdown):
        with contextlib.suppress(Exception):
            shutdown()


@contextlib.contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span]:
    """Спан на блок кода; исключение пишется в спан и пробрасывается дальше."""
    with _tracer().start_as_current_span(
        name, record_exception=False, set_status_on_exception=False
    ) as current:
        # CHAIN — этап конвейера; llm_span() переопределяет на LLM.
        current.set_attribute(_ATTR_SPAN_KIND, "CHAIN")
        set_span_attributes(current, **attributes)
        try:
            yield current
        except BaseException as exc:
            # CancelledError — штатная отмена (fast-return в поиске), не ошибка.
            if not isinstance(exc, asyncio.CancelledError):
                current.record_exception(exc)
                current.set_status(Status(StatusCode.ERROR, str(exc)[:500]))
            raise


def traced(name: str | None = None) -> Callable[[F], F]:
    """Декоратор: спан на каждый вызов sync- или async-функции."""

    def decorator(fn: F) -> F:
        span_name = name or f"{fn.__module__.rsplit('.', 1)[-1]}.{fn.__name__}"
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with span(span_name):
                    return await fn(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with span(span_name):
                return fn(*args, **kwargs)

        return sync_wrapper  # type: ignore[return-value]

    return decorator


def set_span_attributes(target: Span | None = None, **attributes: Any) -> None:
    """Атрибуты на спан (по умолчанию — текущий); None-значения пропускаются."""
    current = target or trace.get_current_span()
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            current.set_attribute(key, value)
        else:
            current.set_attribute(key, str(value))


@contextlib.contextmanager
def llm_span(
    name: str,
    *,
    model: str,
    provider: str,
    label: str = "",
) -> Iterator[Span]:
    """LLM-спан в семантике OpenInference (для вызовов вне автоинструментации)."""
    with span(name, **{"ke.llm.label": label or None}) as current:
        current.set_attribute(_ATTR_SPAN_KIND, "LLM")
        current.set_attribute(_ATTR_MODEL, model)
        current.set_attribute(_ATTR_PROVIDER, provider)
        yield current


def record_llm_io(
    target: Span,
    *,
    messages: list[dict[str, Any]] | None = None,
    output: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
) -> None:
    """Токены — всегда; тексты — только при ``KE_OTEL_CAPTURE_LLM_IO``."""
    from knowledge_engine.src.config.settings import KE_OTEL_CAPTURE_LLM_IO

    for key, value in (
        (_ATTR_TOKENS_PROMPT, prompt_tokens),
        (_ATTR_TOKENS_COMPLETION, completion_tokens),
        (_ATTR_TOKENS_TOTAL, total_tokens),
    ):
        if value is not None:
            target.set_attribute(key, int(value))
    if not KE_OTEL_CAPTURE_LLM_IO:
        return
    if messages is not None:
        target.set_attribute(_ATTR_INPUT, json.dumps(messages, ensure_ascii=False))
        target.set_attribute(_ATTR_INPUT_MIME, "application/json")
    if output is not None:
        target.set_attribute(_ATTR_OUTPUT, output)


def record_evaluation(
    name: str,
    *,
    label: str | None = None,
    score: float | None = None,
    explanation: str | None = None,
    metadata: dict[str, Any] | None = None,
    annotator_kind: str = "CODE",
    target: Span | None = None,
) -> None:
    """Оценка на спан (по умолчанию — текущий): атрибуты ``ke.eval.<name>.*``
    и событие ``ke.evaluation`` всегда; span annotation в Phoenix — если
    включена отправка. ``annotator_kind``: CODE (правило/детерминированный
    код), LLM (вердикт модели), HUMAN (решение пользователя на гейте).

    Без активного спана — no-op: оценке не к чему привязаться."""
    current = target or trace.get_current_span()
    ctx = current.get_span_context()
    if not ctx.is_valid:
        return
    short_explanation = (explanation or "")[:1000] or None
    set_span_attributes(
        current,
        **{
            f"ke.eval.{name}.label": label,
            f"ke.eval.{name}.score": score,
        },
    )
    event_attrs = {
        "ke.eval.name": name,
        "ke.eval.label": label,
        "ke.eval.score": score,
        "ke.eval.explanation": short_explanation,
    }
    current.add_event(
        "ke.evaluation",
        {k: v for k, v in event_attrs.items() if v is not None},
    )
    sender = _annotation_sender
    if sender is None:
        return
    sender.submit(
        {
            "span_id": format(ctx.span_id, "016x"),
            "name": name,
            "annotator_kind": annotator_kind,
            "result": {
                "label": label,
                "score": score,
                "explanation": short_explanation,
            },
            # default=str: в metadata попадают enum/Pydantic-значения.
            "metadata": json.loads(json.dumps(metadata or {}, default=str)),
        }
    )


def _phoenix_annotations_url(otlp_endpoint: str) -> str | None:
    """``http://host:6006/v1/traces`` → ``http://host:6006/v1/span_annotations``.

    Другой endpoint (OTel Collector и т.п.) — не Phoenix, аннотации выключены."""
    suffix = "/v1/traces"
    endpoint = (otlp_endpoint or "").rstrip("/")
    if not endpoint.endswith(suffix):
        return None
    return endpoint[: -len(suffix)] + "/v1/span_annotations?sync=true"


def _httpx_poster(url: str) -> Callable[[list[dict[str, Any]]], int]:
    import httpx

    client = httpx.Client(timeout=5.0)

    def post(batch: list[dict[str, Any]]) -> int:
        return client.post(url, json={"data": batch}).status_code

    return post


class _AnnotationSender:
    """Фоновая отправка span annotations в Phoenix (один daemon-поток).

    Phoenix привязывает аннотацию только к уже принятому спану: в
    sync-режиме отвечает 404, в async — отвечает 200 без проверки и
    применяет с неизвестной задержкой (проверено на version-20.25.0).
    sync + повторы дают подтверждённую доставку.
    BatchSpanProcessor экспортирует спаны с задержкой, поэтому 404 →
    повтор с backoff; после исчерпания попыток оценка остаётся только в
    атрибутах и событии спана. Fail-open: сетевые ошибки не доходят до
    вызывающего кода."""

    _RETRY_DELAYS: tuple[float, ...] = (2.0, 4.0, 8.0, 15.0, 30.0)
    # Оценка пишется внутри ещё открытого спана, а BatchSpanProcessor
    # экспортирует раз в 5 с — раньше отправлять бессмысленно (гарантированный 404).
    _INITIAL_DELAY = 5.0
    # Во время flush спаны уже сброшены, но Phoenix пишет их в БД асинхронно.
    _FLUSH_RETRY_DELAY = 0.2

    def __init__(
        self,
        post: Callable[[list[dict[str, Any]]], int],
        *,
        retry_delays: tuple[float, ...] | None = None,
        initial_delay: float | None = None,
    ) -> None:
        self._post = post
        self._delays = retry_delays or self._RETRY_DELAYS
        self._initial_delay = (
            self._INITIAL_DELAY if initial_delay is None else initial_delay
        )
        self._flushing = False
        # Куча (due_time, seq, attempt, item); seq — стабильный порядок.
        self._heap: list[tuple[float, int, int, dict[str, Any]]] = []
        self._seq = 0
        self._in_flight = 0
        self._cond = threading.Condition()
        self._thread = threading.Thread(
            target=self._run, name="ke-otel-annotations", daemon=True
        )
        self._thread.start()

    def submit(self, item: dict[str, Any]) -> None:
        self._push(item, attempt=0, due=time.monotonic() + self._initial_delay)

    def flush(self, timeout: float) -> bool:
        """Отправить всё сейчас (спаны уже сброшены force_flush). True — пусто."""
        deadline = time.monotonic() + timeout
        with self._cond:
            self._flushing = True
            self._heap = [(0.0, seq, att, it) for _, seq, att, it in self._heap]
            heapq.heapify(self._heap)
            self._cond.notify_all()
            try:
                while self._heap or self._in_flight:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._cond.wait(remaining)
            finally:
                self._flushing = False
        return True

    def _push(self, item: dict[str, Any], *, attempt: int, due: float) -> None:
        with self._cond:
            self._seq += 1
            heapq.heappush(self._heap, (due, self._seq, attempt, item))
            self._cond.notify_all()

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._heap or self._heap[0][0] > time.monotonic():
                    wait = self._heap[0][0] - time.monotonic() if self._heap else None
                    self._cond.wait(wait)
                _, _, attempt, item = heapq.heappop(self._heap)
                self._in_flight += 1
            try:
                self._send(item, attempt)
            finally:
                with self._cond:
                    self._in_flight -= 1
                    self._cond.notify_all()

    def _send(self, item: dict[str, Any], attempt: int) -> None:
        from knowledge_engine.src.core.run_log import trace as run_trace

        try:
            status = self._post([item])
        except Exception as exc:
            status = -1
            error = f"{type(exc).__name__}: {exc}"
        else:
            error = f"HTTP {status}"
        if 200 <= status < 300:
            return
        retryable = status == 404 or status == -1 or status >= 500
        if retryable and attempt < len(self._delays):
            delay = self._FLUSH_RETRY_DELAY if self._flushing else self._delays[attempt]
            self._push(item, attempt=attempt + 1, due=time.monotonic() + delay)
            return
        run_trace(
            f"OTEL ⚠ annotation dropped | name={item.get('name')} "
            f"span={item.get('span_id')} attempts={attempt + 1} | {error}"
        )


def inject_trace_context() -> dict[str, str]:
    """W3C traceparent текущего спана — для передачи через Redis/JSON-хранилище."""
    carrier: dict[str, str] = {}
    _propagator.inject(carrier)
    return carrier


@contextlib.contextmanager
def attached_trace_context(carrier: dict[str, str] | None) -> Iterator[None]:
    """Продолжить трейс из ``inject_trace_context`` (другой процесс/поток)."""
    if not carrier:
        yield
        return
    token = otel_context.attach(_propagator.extract(carrier))
    try:
        yield
    finally:
        otel_context.detach(token)


__all__ = [
    "attached_trace_context",
    "init_observability",
    "inject_trace_context",
    "llm_span",
    "record_evaluation",
    "record_llm_io",
    "set_span_attributes",
    "shutdown_observability",
    "span",
    "traced",
    "use_tracer_provider",
]
