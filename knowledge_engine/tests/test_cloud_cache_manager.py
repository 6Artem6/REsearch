"""cloud_cache_manager: Redis-backed hot/cold Gemini cache, полностью на моках (без сети)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from knowledge_engine.services import cloud_cache_manager as ccm
from knowledge_engine.services.cloud_cache_manager import CacheMetadata, CloudCacheManager


class FakeRedis:
    """Минимальный in-memory заменитель redis-py client (decode_responses=True)."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        self._store[key] = value


class NotFoundError(Exception):
    pass


def _fake_gemini_client(*, cache_name: str = "cachedContents/abc123", get_raises: Exception | None = None) -> Any:
    client = MagicMock()
    created = MagicMock()
    created.name = cache_name
    client.caches.create.return_value = created
    if get_raises is not None:
        client.caches.get.side_effect = get_raises
    else:
        client.caches.get.return_value = MagicMock()
    return client


@pytest.fixture
def fake_redis(monkeypatch: pytest.MonkeyPatch) -> FakeRedis:
    store = FakeRedis()
    monkeypatch.setattr(ccm, "redis_enabled", lambda: True)
    monkeypatch.setattr(ccm, "get_redis", lambda: store)
    return store


def test_redis_disabled_returns_none_without_touching_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ccm, "redis_enabled", lambda: False)
    client = _fake_gemini_client()
    mgr = CloudCacheManager()

    cold = mgr.get_or_create_cold_cache(
        client, model="gemini-2.5-flash", system_instruction="sys", preset_body="preset"
    )
    hot = mgr.get_or_create_hot_session_cache(
        client, session_id="sess-1", model="gemini-2.5-flash", node_context="ctx"
    )

    assert cold is None
    assert hot is None
    client.caches.create.assert_not_called()


def test_cold_cache_miss_then_hit(fake_redis: FakeRedis) -> None:
    client = _fake_gemini_client(cache_name="cachedContents/cold-1")
    mgr = CloudCacheManager()

    first = mgr.get_or_create_cold_cache(
        client,
        model="gemini-2.5-flash",
        system_instruction="STATIC SYSTEM",
        preset_body="STATIC PRESET BODY",
    )
    assert isinstance(first, CacheMetadata)
    assert first.cache_name == "cachedContents/cold-1"
    assert first.is_hot is False
    assert first.ttl_seconds == ccm.GEMINI_CLOUD_CACHE_COLD_TTL_SECONDS
    client.caches.create.assert_called_once()

    second = mgr.get_or_create_cold_cache(
        client,
        model="gemini-2.5-flash",
        system_instruction="STATIC SYSTEM",
        preset_body="STATIC PRESET BODY",
    )
    assert second is not None
    assert second.cache_name == "cachedContents/cold-1"
    client.caches.create.assert_called_once()  # not recreated on hit
    client.caches.get.assert_called_once()  # verified against Gemini


def test_hot_cache_miss_then_hit(fake_redis: FakeRedis) -> None:
    client = _fake_gemini_client(cache_name="cachedContents/hot-1")
    mgr = CloudCacheManager()

    first = mgr.get_or_create_hot_session_cache(
        client,
        session_id="session-42",
        model="gemini-2.5-flash",
        node_context="NODE CONTEXT BODY",
    )
    assert first is not None
    assert first.is_hot is True
    assert first.ttl_seconds == ccm.GEMINI_CLOUD_CACHE_HOT_TTL_SECONDS

    second = mgr.get_or_create_hot_session_cache(
        client,
        session_id="session-42",
        model="gemini-2.5-flash",
        node_context="NODE CONTEXT BODY",
    )
    assert second is not None
    assert second.cache_name == "cachedContents/hot-1"
    client.caches.create.assert_called_once()


def test_stale_remote_cache_is_recreated(fake_redis: FakeRedis) -> None:
    client = _fake_gemini_client(cache_name="cachedContents/first")
    mgr = CloudCacheManager()

    mgr.get_or_create_cold_cache(
        client, model="m", system_instruction="s", preset_body="preset"
    )
    client.caches.create.assert_called_once()

    # Gemini забыл кэш (истёк TTL на стороне API) — 404 на верификации.
    client.caches.get.side_effect = NotFoundError("404 Not Found: cachedContents/first")
    client.caches.create.return_value.name = "cachedContents/second"

    refreshed = mgr.get_or_create_cold_cache(
        client, model="m", system_instruction="s", preset_body="preset"
    )
    assert refreshed is not None
    assert refreshed.cache_name == "cachedContents/second"
    assert client.caches.create.call_count == 2


def test_missing_required_args_skip_without_calling_client(fake_redis: FakeRedis) -> None:
    client = _fake_gemini_client()
    mgr = CloudCacheManager()

    assert (
        mgr.get_or_create_cold_cache(client, model="", system_instruction="", preset_body="p")
        is None
    )
    assert (
        mgr.get_or_create_cold_cache(client, model="m", system_instruction="", preset_body="")
        is None
    )
    assert (
        mgr.get_or_create_hot_session_cache(
            client, session_id="", model="m", node_context="ctx"
        )
        is None
    )
    client.caches.create.assert_not_called()


def test_preset_hash_is_stable_and_model_sensitive() -> None:
    a = ccm.preset_hash("gemini-2.5-flash", "sys", "body")
    b = ccm.preset_hash("gemini-2.5-flash", "sys", "body")
    c = ccm.preset_hash("gemini-2.5-pro", "sys", "body")
    assert a == b
    assert a != c
