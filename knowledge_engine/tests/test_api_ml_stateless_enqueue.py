"""API enqueue helpers never execute ML jobs in-process."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from knowledge_engine.api.helpers import work_enqueue
from knowledge_engine.services.work_job_store import work_job_store


@pytest.fixture(autouse=True)
def _local_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "knowledge_engine.services.work_job_store.redis_enabled",
        lambda: False,
    )
    work_job_store._jobs.clear()


def test_dead_worker_is_503_not_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(work_enqueue, "worker_is_alive", lambda: False)
    monkeypatch.setattr(work_enqueue, "KE_WORKER_INLINE_FALLBACK", True)
    called = {"run": False}

    def _boom(*_a, **_k):
        called["run"] = True
        raise AssertionError("API must not run_work_job")

    monkeypatch.setattr(
        "knowledge_engine.services.work_handlers.run_work_job",
        _boom,
    )
    with pytest.raises(HTTPException) as ei:
        work_enqueue.enqueue_rag_gateway({"op": "query", "body": {}})
    assert ei.value.status_code == 503
    assert called["run"] is False


def test_rag_enqueue_creates_job(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(work_enqueue, "worker_is_alive", lambda: True)
    job_id = work_enqueue.enqueue_rag_gateway(
        {"op": "query", "body": {"target_node": "n1"}}
    )
    job = work_job_store.get(job_id)
    assert job is not None
    assert job.kind.value == "rag_gateway"
    assert job.payload["op"] == "query"
