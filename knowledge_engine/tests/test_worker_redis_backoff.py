"""_safe_redis_command: per-label exponential backoff during a sustained
Redis outage — a persistent failure must not re-attempt (and re-log) on
every single scheduled heartbeat/drain tick."""

from __future__ import annotations

import pytest
import redis

import knowledge_engine.worker.__main__ as worker_main


@pytest.fixture(autouse=True)
def _reset_backoff_state(monkeypatch):
    monkeypatch.setattr(worker_main, "_redis_backoff_until", {})
    monkeypatch.setattr(worker_main, "_redis_backoff_sec", {})
    monkeypatch.setattr(worker_main, "reset_redis_command_client", lambda: None)
    yield


def test_first_failure_sets_minimum_backoff(monkeypatch):
    monkeypatch.setattr(worker_main.time, "monotonic", lambda: 100.0)
    calls = []

    def failing():
        calls.append(1)
        raise redis.exceptions.TimeoutError("boom")

    result = worker_main._safe_redis_command("heartbeat", failing, default="D")

    assert result == "D"
    assert len(calls) == 2  # initial attempt + one reconnect retry
    assert worker_main._redis_backoff_until["heartbeat"] == pytest.approx(
        100.0 + worker_main._REDIS_BACKOFF_MIN_SEC
    )


def test_call_skipped_silently_while_within_backoff_window(monkeypatch):
    t = [100.0]
    monkeypatch.setattr(worker_main.time, "monotonic", lambda: t[0])
    calls = []

    def failing():
        calls.append(1)
        raise redis.exceptions.TimeoutError("boom")

    worker_main._safe_redis_command("heartbeat", failing, default="D")
    assert len(calls) == 2

    t[0] += 1.0  # still well inside the 10s backoff window
    result = worker_main._safe_redis_command("heartbeat", failing, default="D")

    assert result == "D"
    assert len(calls) == 2  # fn was NOT called again — skipped silently


def test_call_resumes_after_backoff_window_expires(monkeypatch):
    t = [100.0]
    monkeypatch.setattr(worker_main.time, "monotonic", lambda: t[0])
    calls = []

    def failing():
        calls.append(1)
        raise redis.exceptions.TimeoutError("boom")

    worker_main._safe_redis_command("heartbeat", failing, default="D")
    assert len(calls) == 2

    t[0] += worker_main._REDIS_BACKOFF_MIN_SEC + 0.01
    worker_main._safe_redis_command("heartbeat", failing, default="D")

    assert len(calls) == 4  # backoff expired — attempted again (2 more calls)


def test_consecutive_failures_escalate_backoff_up_to_cap(monkeypatch):
    t = [0.0]
    monkeypatch.setattr(worker_main.time, "monotonic", lambda: t[0])

    def failing():
        raise redis.exceptions.TimeoutError("boom")

    expected = [10.0, 20.0, 40.0, 60.0, 60.0]
    for exp in expected:
        worker_main._safe_redis_command("republish pending", failing, default=0)
        assert worker_main._redis_backoff_sec["republish pending"] == exp
        # jump past this backoff window to trigger the next failure
        t[0] = worker_main._redis_backoff_until["republish pending"] + 0.01


def test_success_clears_backoff_state(monkeypatch):
    t = [0.0]
    monkeypatch.setattr(worker_main.time, "monotonic", lambda: t[0])

    def failing():
        raise redis.exceptions.TimeoutError("boom")

    worker_main._safe_redis_command("heartbeat", failing, default=None)
    assert "heartbeat" in worker_main._redis_backoff_sec

    t[0] = worker_main._redis_backoff_until["heartbeat"] + 0.01
    result = worker_main._safe_redis_command("heartbeat", lambda: "ok", default=None)

    assert result == "ok"
    assert "heartbeat" not in worker_main._redis_backoff_sec
    assert "heartbeat" not in worker_main._redis_backoff_until


def test_labels_backoff_independently(monkeypatch):
    monkeypatch.setattr(worker_main.time, "monotonic", lambda: 0.0)

    def failing():
        raise redis.exceptions.TimeoutError("boom")

    worker_main._safe_redis_command("heartbeat", failing, default=None)
    calls = []
    worker_main._safe_redis_command("republish pending", lambda: calls.append(1) or "ok")

    assert calls == [1]  # a different label is not affected by heartbeat's backoff
