from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from knowledge_engine.services import redis_client


def test_redis_clients_disable_library_health_ping():
    assert redis_client._redis_client_kwargs(for_pubsub=False)[
        "health_check_interval"
    ] == 0
    assert redis_client._redis_client_kwargs(for_pubsub=True)[
        "health_check_interval"
    ] == 0


def test_failed_initial_ping_does_not_cache_client(monkeypatch):
    class BrokenClient:
        def ping(self):
            raise ConnectionError("redis unavailable")

    fake_redis = SimpleNamespace(
        Redis=SimpleNamespace(from_url=lambda *_args, **_kwargs: BrokenClient())
    )
    monkeypatch.setitem(sys.modules, "redis", fake_redis)
    monkeypatch.setattr(redis_client, "_command_client", None)
    monkeypatch.setattr(redis_client, "redis_enabled", lambda: True)

    with pytest.raises(ConnectionError, match="redis unavailable"):
        redis_client.get_redis()

    assert redis_client._command_client is None


def test_redis_ping_discards_broken_cached_client(monkeypatch):
    class BrokenClient:
        closed = False

        def ping(self):
            raise ConnectionError("stale socket")

        def close(self):
            self.closed = True

    client = BrokenClient()
    monkeypatch.setattr(redis_client, "_command_client", client)
    monkeypatch.setattr(redis_client, "redis_enabled", lambda: True)

    assert redis_client.redis_ping() is False
    assert client.closed is True
    assert redis_client._command_client is None
