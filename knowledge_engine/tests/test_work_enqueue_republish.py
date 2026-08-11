"""Duplicate pending init must re-notify Redis worker (pub/sub is not durable)."""

from __future__ import annotations

import pytest

from knowledge_engine.api.helpers import work_enqueue
from knowledge_engine.services.work_job_store import (
    WorkJobKind,
    WorkJobStatus,
    work_job_store,
)


@pytest.fixture(autouse=True)
def _local_work_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "knowledge_engine.services.work_job_store.redis_enabled",
        lambda: False,
    )
    work_job_store._jobs.clear()


def test_duplicate_pending_init_republishes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(work_enqueue, "worker_is_alive", lambda: True)
    published: list[str] = []

    def _publish(job_id: str) -> None:
        published.append(job_id)

    monkeypatch.setattr(
        "knowledge_engine.services.redis_tasks.publish_work_job",
        _publish,
    )

    payload = {
        "curriculum_id": "cur_republish",
        "user_action": "init",
        "node_data": {"node_id": "n1"},
    }
    job = work_job_store.create(WorkJobKind.NODE_DEEP_DIVE, payload)
    assert job.status == WorkJobStatus.PENDING

    jid = work_enqueue.enqueue_node_deep_dive(payload)
    assert jid == job.id
    assert published == [job.id]


def test_duplicate_running_init_does_not_republish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(work_enqueue, "worker_is_alive", lambda: True)
    published: list[str] = []
    monkeypatch.setattr(
        "knowledge_engine.services.redis_tasks.publish_work_job",
        lambda job_id: published.append(job_id),
    )

    payload = {
        "curriculum_id": "cur_running",
        "user_action": "init",
        "node_data": {"node_id": "n2"},
    }
    job = work_job_store.create(WorkJobKind.NODE_DEEP_DIVE, payload)
    job.status = WorkJobStatus.RUNNING

    monkeypatch.setattr(
        work_job_store,
        "find_active_node_deep_dive",
        lambda *a, **k: job,
    )

    jid = work_enqueue.enqueue_node_deep_dive(payload)
    assert jid == job.id
    assert published == []


def test_republish_pending_work_jobs_no_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    from knowledge_engine.services import work_job_store as store_mod

    monkeypatch.setattr(store_mod, "redis_enabled", lambda: False)
    assert store_mod.republish_pending_work_jobs() == 0
