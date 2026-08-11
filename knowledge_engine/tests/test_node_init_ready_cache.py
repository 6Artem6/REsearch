"""POST /node/init returns ready result without re-queue when session/job done."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from knowledge_engine.api.app import create_app
from knowledge_engine.services.work_job_store import (
    WorkJobKind,
    WorkJobStatus,
    work_job_store,
)


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(
        "knowledge_engine.services.work_job_store.redis_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        "knowledge_engine.api.helpers.work_enqueue.worker_is_alive",
        lambda: True,
    )
    work_job_store._jobs.clear()
    return TestClient(create_app())


def _node_body() -> dict:
    return {
        "curriculum_id": "cur_ready",
        "node_data": {
            "node_id": "node_ready",
            "title": "Ready Node",
            "brief_summary": "Summary text for node validation length ok.",
            "core_concepts": ["hooks"],
            "layer": "foundation",
            "category": "agents",
        },
    }


def test_init_returns_completed_job_result_immediately(client: TestClient) -> None:
    body = _node_body()
    job = work_job_store.create(
        WorkJobKind.NODE_DEEP_DIVE,
        {
            "curriculum_id": "cur_ready",
            "user_action": "init",
            "node_data": body["node_data"],
        },
    )
    result = {
        "node_id": "node_ready",
        "node_status": "unexplored",
        "content": {"summary": "", "diagram": "", "diagrams": []},
        "tutor_message": "",
        "history": [],
        "session_key": "cur_ready::node_ready",
    }
    work_job_store.complete(job.id, result)

    with patch(
        "knowledge_engine.api.routes.node_skill._session_init_ready",
        return_value=False,
    ):
        r = client.post("/api/v1/node/init", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["node_id"] == "node_ready"
    assert "job_id" not in data


def test_init_completes_orphan_active_when_session_ready(client: TestClient) -> None:
    body = _node_body()
    orphan = work_job_store.create(
        WorkJobKind.NODE_DEEP_DIVE,
        {
            "curriculum_id": "cur_ready",
            "user_action": "init",
            "node_data": body["node_data"],
        },
    )
    orphan.status = WorkJobStatus.RUNNING
    work_job_store._persist()

    ready_payload = {
        "node_id": "node_ready",
        "node_status": "unexplored",
        "content": {"summary": "ok", "diagram": "", "diagrams": []},
        "tutor_message": "",
        "history": [],
        "session_key": "x",
    }
    with patch(
        "knowledge_engine.api.routes.node_skill._build_init_result_from_session",
        return_value=ready_payload,
    ):
        r = client.post("/api/v1/node/init", json=body)
    assert r.status_code == 200
    assert r.json()["node_id"] == "node_ready"
    refreshed = work_job_store.get(orphan.id)
    assert refreshed is not None
    assert refreshed.status == WorkJobStatus.COMPLETED
    assert refreshed.result == ready_payload
