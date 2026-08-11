"""cancel_work_job lock/busy side effects."""

from __future__ import annotations

from pathlib import Path

import pytest

from knowledge_engine.scripts import cancel_work_job as cwj
from knowledge_engine.services import worker_busy
from knowledge_engine.services.work_job_store import (
    WorkJob,
    WorkJobKind,
    force_release_work_claim_lock,
)


def test_clear_worker_busy_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "worker_dev_busy.json"
    path.write_text('{"busy": true, "count": 2}', encoding="utf-8")
    monkeypatch.setattr(worker_busy, "_BUSY_PATH", path)
    monkeypatch.setattr(worker_busy, "_active", 2)
    assert worker_busy.clear_worker_busy_file() is True
    assert worker_busy._active == 0
    assert not path.exists()
    assert worker_busy.clear_worker_busy_file() is False


def test_force_release_work_claim_lock_no_redis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "knowledge_engine.services.work_job_store.redis_enabled",
        lambda: False,
    )
    assert force_release_work_claim_lock("abc") is False


def test_release_locks_for_job_calls_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        cwj,
        "_release_work_claim_lock",
        lambda jid: calls.append(f"work:{jid}") or True,
    )
    monkeypatch.setattr(
        cwj,
        "_release_node_grounding_lock",
        lambda cid, nid: calls.append(f"ground:{cid}/{nid}") or True,
    )

    job = WorkJob(
        id="jobdeadbeef1",
        kind=WorkJobKind.NODE_DEEP_DIVE,
        payload={
            "curriculum_id": "cur_x",
            "user_action": "init",
            "node_data": {"node_id": "node_y"},
        },
    )
    assert cwj.release_locks_for_job(job) == 2
    assert calls == ["work:jobdeadbeef1", "ground:cur_x/node_y"]

    calls.clear()
    job2 = WorkJob(
        id="jobdeadbeef2",
        kind=WorkJobKind.CURRICULUM_GENERATE,
        payload={},
    )
    assert cwj.release_locks_for_job(job2) == 1
    assert calls == ["work:jobdeadbeef2"]
