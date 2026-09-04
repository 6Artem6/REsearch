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


class _FakeRedisConnBreaksMidScan:
    """Simulates a Redis connection dying after the first key — every
    scan_iter key is yielded up front (real redis-py buffers a batch per
    SCAN cursor call too), but .get() only succeeds for the first one."""

    def __init__(self, keys: list[str], *, break_after: int) -> None:
        self._keys = keys
        self._break_after = break_after
        self._get_calls = 0

    def scan_iter(self, match: str):
        return iter(self._keys)

    def get(self, key: str):
        import redis

        self._get_calls += 1
        if self._get_calls > self._break_after:
            raise redis.exceptions.TimeoutError("Timeout reading from socket")
        return b'{"status": "pending", "id": "' + key.encode() + b'"}'


def test_list_pending_work_job_ids_propagates_connection_error_mid_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a broken connection after key 1 of N used to be silently
    swallowed per-key (`except Exception: continue`) and re-attempted on
    every remaining key against the SAME dead socket — N-1 extra, fully
    silent socket_timeout waits before the function finally returned
    whatever partial list it had (confirmed live: a single 'republish
    pending' call took ~16 minutes with the connection down). It must now
    raise on the first connection failure instead of grinding through the
    rest of the keys."""
    from knowledge_engine.services import work_job_store as store_mod

    fake = _FakeRedisConnBreaksMidScan(
        ["ke:job:1", "ke:job:2", "ke:job:3", "ke:job:4"], break_after=1
    )
    monkeypatch.setattr(store_mod, "redis_enabled", lambda: True)
    monkeypatch.setattr(store_mod, "get_redis", lambda: fake)

    import redis

    with pytest.raises(redis.exceptions.TimeoutError):
        store_mod.list_pending_work_job_ids()

    # Only the keys up to (and including) the one that broke the connection
    # were attempted — it did not keep going through the remaining keys.
    assert fake._get_calls == 2


def test_list_pending_work_job_ids_still_skips_one_malformed_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single bad JSON record (not a connection problem) must still be
    skipped, not treated as fatal — only genuine connection errors should
    propagate."""
    from knowledge_engine.services import work_job_store as store_mod

    class _FakeRedisOneBadRecord:
        def scan_iter(self, match: str):
            return iter(["ke:job:bad", "ke:job:good"])

        def get(self, key: str):
            if key == "ke:job:bad":
                return b"not json at all"
            return b'{"status": "pending", "id": "good-id"}'

    monkeypatch.setattr(store_mod, "redis_enabled", lambda: True)
    monkeypatch.setattr(store_mod, "get_redis", lambda: _FakeRedisOneBadRecord())

    assert store_mod.list_pending_work_job_ids() == ["good-id"]
