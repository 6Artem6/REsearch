"""OpenTelemetry-слой (src/core/observability.py) без сети: InMemorySpanExporter
вместо OTLP. Проверяет спаны sync/async, ошибки и отмену, передачу контекста
API → worker через WorkJob.trace_context и LLM-спан Gemma."""

from __future__ import annotations

import asyncio
import inspect
import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode
from pydantic import BaseModel

from knowledge_engine.src.core import observability
from knowledge_engine.src.core.observability import (
    attached_trace_context,
    inject_trace_context,
    span,
    traced,
)


@pytest.fixture
def exporter():
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    observability.use_tracer_provider(provider)
    yield exp
    observability.use_tracer_provider(None)


def _by_name(exporter: InMemorySpanExporter) -> dict:
    return {s.name: s for s in exporter.get_finished_spans()}


def test_traced_sync_and_async_create_named_spans(exporter) -> None:
    @traced("unit.sync")
    def add(a: int, b: int) -> int:
        return a + b

    @traced("unit.async")
    async def mul(a: int, b: int) -> int:
        return a * b

    assert add(2, 3) == 5
    assert asyncio.run(mul(2, 3)) == 6
    assert inspect.iscoroutinefunction(mul)
    assert {"unit.sync", "unit.async"} <= set(_by_name(exporter))


def test_exception_recorded_and_reraised(exporter) -> None:
    @traced("unit.fail")
    def boom() -> None:
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        boom()
    failed = _by_name(exporter)["unit.fail"]
    assert failed.status.status_code == StatusCode.ERROR
    assert any(e.name == "exception" for e in failed.events)


def test_cancellation_is_not_marked_as_error(exporter) -> None:
    @traced("unit.cancel")
    async def slow() -> None:
        await asyncio.sleep(10)

    async def _run() -> None:
        task = asyncio.create_task(slow())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_run())
    cancelled = _by_name(exporter)["unit.cancel"]
    assert cancelled.status.status_code != StatusCode.ERROR


def test_trace_context_round_trip_links_child_to_parent(exporter) -> None:
    with span("unit.parent") as parent:
        carrier = inject_trace_context()
    assert "traceparent" in carrier

    with attached_trace_context(carrier):
        with span("unit.child"):
            pass

    spans = _by_name(exporter)
    child = spans["unit.child"]
    assert child.context.trace_id == parent.get_span_context().trace_id
    assert child.parent.span_id == parent.get_span_context().span_id


def test_empty_carrier_starts_new_root(exporter) -> None:
    with attached_trace_context({}):
        with span("unit.root"):
            pass
    assert _by_name(exporter)["unit.root"].parent is None


def test_work_job_trace_context_serialization_is_backward_compatible() -> None:
    from knowledge_engine.src.shared.job_queue.work_job_store import (
        WorkJob,
        WorkJobKind,
        _from_dict,
        _to_dict,
    )

    job = WorkJob(
        id="abc123",
        kind=WorkJobKind.RAG_GATEWAY,
        payload={"op": "query"},
        trace_context={"traceparent": "00-" + "1" * 32 + "-" + "2" * 16 + "-01"},
    )
    restored = _from_dict(_to_dict(job))
    assert restored.trace_context == job.trace_context

    legacy = _to_dict(job)
    legacy.pop("trace_context")
    assert _from_dict(legacy).trace_context == {}


def test_run_work_job_continues_enqueue_trace(exporter, monkeypatch) -> None:
    from knowledge_engine.src.shared.job_queue import work_handlers
    from knowledge_engine.src.shared.job_queue.work_job_store import (
        WorkJob,
        WorkJobKind,
    )

    monkeypatch.setattr(work_handlers, "_dispatch_work_job", lambda job: {"ok": 1})
    with span("api.request") as api_span:
        carrier = inject_trace_context()
    job = WorkJob(
        id="job42",
        kind=WorkJobKind.RAG_GATEWAY,
        payload={},
        trace_context=carrier,
    )

    assert work_handlers.run_work_job(job) == {"ok": 1}
    job_span = _by_name(exporter)["work_job.rag_gateway"]
    assert job_span.context.trace_id == api_span.get_span_context().trace_id
    assert job_span.attributes["ke.job.id"] == "job42"


class _Schema(BaseModel):
    executive_summary: str


class _FakeResponse:
    status_code = 200

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


class _FakeHttpClient:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    async def post(self, url: str, *, json: dict, headers: dict) -> _FakeResponse:
        return _FakeResponse(self._payload)


def test_gemma_call_emits_llm_span_with_tokens(exporter) -> None:
    from knowledge_engine.src.adapters.llm_providers.gemma_client import (
        GemmaCloudClient,
    )

    response = {
        "choices": [{"message": {"content": json.dumps({"executive_summary": "ok"})}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }
    client = GemmaCloudClient(api_key="test-key")
    result = asyncio.run(
        client.complete_structured(
            "system",
            "prompt",
            _Schema,
            client=_FakeHttpClient(response),
            use_token_budget=False,
            label="unit/map",
        )
    )

    assert isinstance(result, _Schema)
    llm = _by_name(exporter)["gemma.chat_completions"]
    assert llm.attributes["openinference.span.kind"] == "LLM"
    assert llm.attributes["llm.token_count.prompt"] == 11
    assert llm.attributes["llm.token_count.completion"] == 7
    assert llm.attributes["llm.token_count.total"] == 18
    assert llm.attributes["ke.llm.label"] == "unit/map"


def test_init_observability_disabled_is_noop(monkeypatch) -> None:
    from knowledge_engine.src.config import settings

    monkeypatch.setattr(settings, "KE_OTEL_ENABLED", False)
    assert observability.init_observability("unit") is False


# --- Оценки (шаг 3): record_evaluation и отправка span annotations ----------


class _CollectingSender:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def submit(self, item: dict) -> None:
        self.items.append(item)


@pytest.fixture
def sender(monkeypatch) -> _CollectingSender:
    collecting = _CollectingSender()
    monkeypatch.setattr(observability, "_annotation_sender", collecting)
    return collecting


def test_record_evaluation_writes_span_and_annotation(exporter, sender) -> None:
    with span("unit.eval") as current:
        observability.record_evaluation(
            "unit_check",
            label="verified",
            score=1.0,
            explanation="ok",
            metadata={"concept_id": "c1"},
            annotator_kind="LLM",
        )
    finished = _by_name(exporter)["unit.eval"]
    assert finished.attributes["ke.eval.unit_check.label"] == "verified"
    assert finished.attributes["ke.eval.unit_check.score"] == 1.0
    assert any(e.name == "ke.evaluation" for e in finished.events)

    [item] = sender.items
    assert item["span_id"] == format(current.get_span_context().span_id, "016x")
    assert item["annotator_kind"] == "LLM"
    assert item["result"] == {"label": "verified", "score": 1.0, "explanation": "ok"}
    assert item["metadata"] == {"concept_id": "c1"}


def test_record_evaluation_without_span_is_noop(sender) -> None:
    observability.record_evaluation("unit_check", label="x")
    assert sender.items == []


def test_annotation_sender_retries_until_span_exported() -> None:
    statuses = [404, 404, 200]
    calls: list[list[dict]] = []

    def post(batch: list[dict]) -> int:
        calls.append(batch)
        return statuses.pop(0)

    s = observability._AnnotationSender(
        post, retry_delays=(0.01, 0.01, 0.01), initial_delay=0
    )
    s.submit({"span_id": "ab", "name": "n"})
    assert s.flush(timeout=2.0)
    assert len(calls) == 3


def test_annotation_sender_drops_after_retries_and_on_client_error() -> None:
    calls: list[int] = []

    def post(batch: list[dict]) -> int:
        calls.append(1)
        return 404 if batch[0]["name"] == "missing" else 422

    s = observability._AnnotationSender(
        post, retry_delays=(0.01, 0.01), initial_delay=0
    )
    s.submit({"span_id": "ab", "name": "missing"})
    s.submit({"span_id": "cd", "name": "invalid"})
    assert s.flush(timeout=2.0)
    # missing: 1 + 2 повтора; invalid (422): без повторов.
    assert len(calls) == 4


def test_annotation_sender_flush_sends_delayed_items_now() -> None:
    calls: list[int] = []
    s = observability._AnnotationSender(
        lambda batch: calls.append(1) or 200, initial_delay=60.0
    )
    s.submit({"span_id": "ab", "name": "n"})
    assert calls == []
    assert s.flush(timeout=2.0)
    assert calls == [1]


def test_phoenix_annotations_url_only_for_phoenix_endpoint() -> None:
    assert (
        observability._phoenix_annotations_url("http://localhost:6006/v1/traces")
        == "http://localhost:6006/v1/span_annotations?sync=true"
    )
    assert observability._phoenix_annotations_url("http://collector:4318/x") is None


def test_gate_decision_scores_approval_rate(exporter, sender) -> None:
    from types import SimpleNamespace

    from knowledge_engine.src.domains.steering.routes import _record_gate_decision

    offered = [
        SimpleNamespace(url=f"https://a/{i}", title=f"T{i}", source_hub="hub")
        for i in range(4)
    ]
    with span("unit.gate"):
        _record_gate_decision(
            "steering_gate1",
            offered=offered,
            approved_urls=["https://a/0", "https://a/2", "https://a/3"],
            target_goal="goal",
        )
    [item] = sender.items
    assert item["annotator_kind"] == "HUMAN"
    assert item["result"]["label"] == "partial"
    assert item["result"]["score"] == 0.75
    assert item["metadata"]["rejected"] == [
        {"url": "https://a/1", "title": "T1", "source": "hub"}
    ]


def test_sub_concept_verdict_and_skip(exporter, sender, monkeypatch) -> None:
    from types import SimpleNamespace

    from knowledge_engine.src.domains.grounding.graph.nodes import (
        sub_concept_eval as node_mod,
    )
    from knowledge_engine.src.domains.grounding.sub_concept_evaluator import (
        mark_evaluator_skipped,
    )

    row = SimpleNamespace(
        id="c1",
        label="Attention",
        status="partial",
        last_accuracy_grade="PARTIAL",
        failed_attempts=1,
        why_passed=True,
        how_passed=False,
        mechanic_passed=False,
        is_extension=False,
    )
    monkeypatch.setattr(node_mod, "find_sub_concept", lambda memory, ref: row)
    memory = SimpleNamespace(
        evaluator_skipped=False,
        last_evaluator_feedback="нет механики",
        last_eval_directive="PUSH",
    )
    req = SimpleNamespace(
        node_data=SimpleNamespace(layer="advanced"),
        interaction_axis="lecture_self_check",
    )
    with span("unit.turn"):
        node_mod._record_sub_concept_verdict(memory, "c1", req)
        mark_evaluator_skipped(memory, "no pending")
        # После пропуска вердикт не дублируется.
        node_mod._record_sub_concept_verdict(memory, "c1", req)

    verdict, skipped = sender.items
    assert verdict["annotator_kind"] == "LLM"
    assert verdict["result"]["label"] == "partial"
    assert verdict["result"]["score"] == 0.5
    assert verdict["metadata"]["directive"] == "PUSH"
    assert skipped["result"] == {
        "label": "skipped",
        "score": None,
        "explanation": "no pending",
    }
