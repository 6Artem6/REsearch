"""Коалесинг повторного node init в work queue."""

import pytest

from knowledge_engine.services.work_job_store import (
    WorkJobKind,
    work_job_store,
)


@pytest.fixture(autouse=True)
def _local_work_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "knowledge_engine.services.work_job_store.redis_enabled",
        lambda: False,
    )
    work_job_store._jobs.clear()


def test_find_active_node_deep_dive_pending_init() -> None:
    payload_a = {
        "curriculum_id": "cur_a",
        "user_action": "init",
        "node_data": {"node_id": "node_x"},
    }
    job1 = work_job_store.create(WorkJobKind.NODE_DEEP_DIVE, payload_a)
    found = work_job_store.find_active_node_deep_dive(
        "cur_a", "node_x", user_action="init"
    )
    assert found is not None
    assert found.id == job1.id

    payload_b = {
        "curriculum_id": "cur_a",
        "user_action": "chat",
        "node_data": {"node_id": "node_x"},
    }
    work_job_store.create(WorkJobKind.NODE_DEEP_DIVE, payload_b)
    found_init = work_job_store.find_active_node_deep_dive(
        "cur_a", "node_x", user_action="init"
    )
    assert found_init is not None
    assert found_init.id == job1.id

    work_job_store.complete(job1.id, {"ok": True})
    found_after = work_job_store.find_active_node_deep_dive(
        "cur_a", "node_x", user_action="init"
    )
    assert found_after is None


def test_find_active_node_deep_dive_other_node() -> None:
    job = work_job_store.create(
        WorkJobKind.NODE_DEEP_DIVE,
        {
            "curriculum_id": "cur_b",
            "user_action": "init",
            "node_data": {"node_id": "n1"},
        },
    )
    assert (
        work_job_store.find_active_node_deep_dive("cur_b", "n2", user_action="init")
        is None
    )
    work_job_store.fail(job.id, "test")
    assert (
        work_job_store.find_active_node_deep_dive("cur_b", "n1", user_action="init")
        is None
    )


def test_find_latest_completed_node_deep_dive() -> None:
    payload = {
        "curriculum_id": "cur_c",
        "user_action": "init",
        "node_data": {"node_id": "node_z"},
    }
    job1 = work_job_store.create(WorkJobKind.NODE_DEEP_DIVE, payload)
    work_job_store.complete(job1.id, {"node_id": "node_z", "v": 1})
    job2 = work_job_store.create(WorkJobKind.NODE_DEEP_DIVE, payload)
    work_job_store.complete(job2.id, {"node_id": "node_z", "v": 2})
    latest = work_job_store.find_latest_completed_node_deep_dive(
        "cur_c", "node_z", user_action="init"
    )
    assert latest is not None
    assert latest.id == job2.id
    assert latest.result == {"node_id": "node_z", "v": 2}
    assert (
        work_job_store.find_latest_completed_node_deep_dive(
            "cur_c", "missing", user_action="init"
        )
        is None
    )
