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


def test_redis_clients_disable_internal_retry_on_timeout():
    """Regression: worker/__main__.py уже переигрывает команды сам
    (_safe_redis_command / _reconnect_pubsub) — включённый redis-py-шный
    retry_on_timeout молча удваивал время одной залипшей попытки до нашего
    собственного reconnect (см. CRITICAL BUGFIX: COOLDOWN... reconnect-цикл
    на минуты при недоступном Redis)."""
    assert redis_client._redis_client_kwargs(for_pubsub=False)[
        "retry_on_timeout"
    ] is False
    assert redis_client._redis_client_kwargs(for_pubsub=True)[
        "retry_on_timeout"
    ] is False


def test_tcp_keepalive_options_only_uses_constants_present_on_this_platform(
    monkeypatch,
):
    """macOS не имеет TCP_KEEPIDLE — опции должны собираться из того, что
    реально есть в модуле socket, а не падать с AttributeError."""
    import socket as socket_mod

    monkeypatch.delattr(socket_mod, "TCP_KEEPIDLE", raising=False)
    monkeypatch.setattr(socket_mod, "TCP_KEEPINTVL", 257, raising=False)
    monkeypatch.setattr(socket_mod, "TCP_KEEPCNT", 258, raising=False)

    opts = redis_client._tcp_keepalive_options()

    assert opts == {257: 10, 258: 3}


def test_tcp_keepalive_options_include_keepidle_when_available(monkeypatch):
    import socket as socket_mod

    monkeypatch.setattr(socket_mod, "TCP_KEEPIDLE", 4, raising=False)
    monkeypatch.setattr(socket_mod, "TCP_KEEPINTVL", 5, raising=False)
    monkeypatch.setattr(socket_mod, "TCP_KEEPCNT", 6, raising=False)

    opts = redis_client._tcp_keepalive_options()

    assert opts == {4: 30, 5: 10, 6: 3}


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
