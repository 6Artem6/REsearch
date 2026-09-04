"""Worker→API SSE event log (JSONL backend)."""

from __future__ import annotations

import asyncio

import pytest

from knowledge_engine.services import job_stream
from knowledge_engine.services.job_stream import (
    append_job_stream_event,
    iter_job_stream_events,
    read_job_stream_events,
)
from knowledge_engine.services.work_job_store import (
    WorkJobKind,
    WorkJobStatus,
    work_job_store,
)


@pytest.fixture()
def jsonl_stream(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(job_stream, "redis_enabled", lambda: False)
    monkeypatch.setattr(job_stream, "_STREAM_DIR", tmp_path)
    return tmp_path


def test_jsonl_append_and_read(jsonl_stream) -> None:
    append_job_stream_event("job1", {"type": "token", "text": "a"})
    append_job_stream_event("job1", {"type": "complete", "result": {"ok": True}})
    all_ev = read_job_stream_events("job1")
    assert [e["type"] for e in all_ev] == ["token", "complete"]
    assert read_job_stream_events("job1", 1)[0]["type"] == "complete"


def test_iter_stops_on_complete(jsonl_stream, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "knowledge_engine.services.work_job_store.redis_enabled",
        lambda: False,
    )
    append_job_stream_event("job2", {"type": "token", "text": "x"})
    append_job_stream_event("job2", {"type": "complete", "result": {"n": 1}})

    async def _collect() -> list[str]:
        types: list[str] = []
        async for evt in iter_job_stream_events("job2", timeout_sec=2.0, poll_sec=0.01):
            types.append(str(evt.get("type")))
        return types

    assert asyncio.run(_collect()) == ["token", "complete"]


def test_iter_failed_job_without_events(
    jsonl_stream, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "knowledge_engine.services.work_job_store.redis_enabled",
        lambda: False,
    )
    work_job_store._jobs.clear()
    job = work_job_store.create(WorkJobKind.RAG_GATEWAY, {"op": "query"})
    work_job_store.fail(job.id, "boom")

    async def _collect() -> list[dict]:
        out: list[dict] = []
        async for evt in iter_job_stream_events(job.id, timeout_sec=2.0, poll_sec=0.01):
            out.append(evt)
        return out

    events = asyncio.run(_collect())
    assert events[0]["type"] == "error"
    assert "boom" in str(events[0].get("detail"))
    work_job_store._jobs.clear()
    assert WorkJobStatus.FAILED.value == "failed"
