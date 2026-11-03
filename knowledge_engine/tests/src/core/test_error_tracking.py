"""error_tracking.py без сети: sentry-sdk с подставным транспортом вместо
GlitchTip. Проверяет fail-open через trace() в except, игнор отмены и HTTP
4xx, ограничение частоты, связь с OTel-трейсом и дедупликацию."""

from __future__ import annotations

import asyncio

import pytest
import sentry_sdk
from fastapi import HTTPException
from opentelemetry.sdk.trace import TracerProvider
from sentry_sdk.transport import Transport

from knowledge_engine.src.core import error_tracking as et
from knowledge_engine.src.core import observability
from knowledge_engine.src.core.run_log import set_trace_exception_hook, trace


class _CollectingTransport(Transport):
    def __init__(self, options=None) -> None:
        super().__init__(options)
        self.events: list[dict] = []

    def capture_envelope(self, envelope) -> None:
        for item in envelope.items:
            event = item.get_event()
            if event is not None:
                self.events.append(event)


@pytest.fixture
def events(monkeypatch):
    transport = _CollectingTransport()
    sentry_sdk.init(
        dsn="http://public@glitchtip.test/1",
        transport=transport,
        before_send=et._before_send,
        default_integrations=False,
        integrations=[
            sentry_sdk.integrations.dedupe.DedupeIntegration(),
        ],
    )
    monkeypatch.setattr(et, "_initialized", True)
    monkeypatch.setattr(et, "_failopen_limiter", et._RateLimiter(5, 60.0))
    set_trace_exception_hook(et.capture_failopen)
    yield transport.events
    set_trace_exception_hook(None)
    sentry_sdk.get_client().close()


def test_trace_inside_except_sends_failopen_warning(events) -> None:
    try:
        raise ValueError("embedding timed out")
    except ValueError as exc:
        trace(f"PRE_MAP_DEDUP pool_vector ✗ | {exc}")
    trace("PRE_MAP_DEDUP ✓ done")  # вне except — не событие

    [event] = events
    assert event["level"] == "warning"
    assert event["tags"]["ke.failopen"] == "true"
    assert event["tags"]["ke.trace_label"] == "PRE_MAP_DEDUP pool_vector ✗"
    assert "embedding timed out" in event["extra"]["trace_message"]
    assert event["exception"]["values"][0]["type"] == "ValueError"


def test_cancellation_and_http_exceptions_are_ignored(events) -> None:
    for exc in (
        asyncio.CancelledError(),
        HTTPException(status_code=404, detail="nope"),
        HTTPException(status_code=502, detail="already sent explicitly"),
    ):
        try:
            raise exc
        except (asyncio.CancelledError, HTTPException):
            trace("API HTTP | skip")
    assert events == []


def test_failopen_rate_limited_per_key(events) -> None:
    for i in range(8):
        try:
            raise ConnectionError(f"exa 503 #{i}")
        except ConnectionError:
            trace(f"CURRICULUM stream producer queue | #{i}")
    assert len(events) == 5


def test_explicit_error_then_trace_is_deduplicated(events) -> None:
    carrier = {"traceparent": "00-" + "a" * 32 + "-" + "b" * 16 + "-01"}
    try:
        raise RuntimeError("reduce crashed")
    except RuntimeError as exc:
        event_id = et.capture_error(
            exc, tags={"ke.job.kind": "node_grounding"}, trace_context=carrier
        )
        trace(f"WORKER ✗ work_job | {exc}")

    assert event_id
    [event] = events
    assert event["level"] == "error"
    assert event["tags"]["otel.trace_id"] == "a" * 32
    assert event["tags"]["ke.job.kind"] == "node_grounding"


def test_event_links_current_otel_span_both_ways(events) -> None:
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    observability.use_tracer_provider(provider)
    try:
        with observability.span("unit.stage") as current:
            try:
                raise KeyError("missing figure")
            except KeyError:
                trace("VLM figure_registry ✗ | missing")
    finally:
        observability.use_tracer_provider(None)

    [event] = events
    ctx = current.get_span_context()
    assert event["tags"]["otel.trace_id"] == format(ctx.trace_id, "032x")
    [finished] = exporter.get_finished_spans()
    assert finished.attributes["ke.sentry.event_id"] == event["event_id"]


def test_rate_limiter_window_expires() -> None:
    now = [0.0]
    limiter = et._RateLimiter(2, 10.0, clock=lambda: now[0])
    assert limiter.allow("k") and limiter.allow("k")
    assert not limiter.allow("k")
    now[0] = 11.0
    assert limiter.allow("k")


def test_disabled_without_dsn(monkeypatch) -> None:
    from knowledge_engine.src.config import settings

    monkeypatch.setattr(settings, "KE_SENTRY_DSN", "")
    monkeypatch.setattr(et, "_initialized", False)
    assert et.init_error_tracking("unit") is False
    assert et.capture_error(RuntimeError("x")) is None
