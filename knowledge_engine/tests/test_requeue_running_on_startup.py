"""Worker startup requeues orphan RUNNING work jobs."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from knowledge_engine.services import work_job_store as store_mod
from knowledge_engine.services.work_job_store import (
    WorkJobKind,
    WorkJobStatus,
    requeue_running_work_jobs_on_startup,
)


class _FakeRedis:
    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        self.published: list[str] = []

    def scan_iter(self, match: str = "*"):
        prefix = match.rstrip("*")
        for k in list(self.kv):
            if k.startswith(prefix):
                yield k

    def get(self, key: str):
        return self.kv.get(key)

    def set(self, key: str, value: str):
        self.kv[key] = value

    def delete(self, key: str):
        return 1 if self.kv.pop(key, None) is not None else 0


def test_requeue_running_work_jobs_on_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeRedis()
    now = datetime.now(timezone.utc).isoformat()
    job = {
        "id": "abc123orphan1",
        "kind": WorkJobKind.NODE_DEEP_DIVE.value,
        "payload": {
            "curriculum_id": "cur",
            "user_action": "init",
            "node_data": {"node_id": "n1"},
        },
        "status": WorkJobStatus.RUNNING.value,
        "created_at": now,
        "updated_at": now,
        "result": None,
        "error": None,
    }
    fake.kv["ke:work:job:abc123orphan1"] = json.dumps(job)
    fake.kv["ke:lock:work:abc123orphan1"] = "token"
    fake.kv["ke:lock:node_ground:cur:n1"] = "token"

    monkeypatch.setattr(store_mod, "redis_enabled", lambda: True)
    monkeypatch.setattr(store_mod, "get_redis", lambda: fake)
    monkeypatch.setattr(
        store_mod,
        "publish_work_job",
        lambda jid: fake.published.append(jid),
    )

    released: list[tuple[str, str]] = []

    def _release(cid: str, nid: str) -> bool:
        released.append((cid, nid))
        return bool(fake.delete(f"ke:lock:node_ground:{cid}:{nid}"))

    monkeypatch.setattr(
        "knowledge_engine.services.node_grounding_lock.force_release_node_grounding_lock",
        _release,
    )
    monkeypatch.setattr(
        "knowledge_engine.services.worker_busy.clear_worker_busy_file",
        lambda: True,
    )

    n = requeue_running_work_jobs_on_startup()
    assert n == 1
    data = json.loads(fake.kv["ke:work:job:abc123orphan1"])
    assert data["status"] == WorkJobStatus.PENDING.value
    assert fake.published == ["abc123orphan1"]
    assert "ke:lock:work:abc123orphan1" not in fake.kv
    assert released == [("cur", "n1")]
