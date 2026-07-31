"""Redis client (очередь worker, логи)."""

from __future__ import annotations

from typing import Any

import knowledge_engine.config as cfg

_command_client: Any | None = None
_pubsub_client: Any | None = None


def redis_enabled() -> bool:
    return cfg.KE_USE_REDIS and bool(cfg.REDIS_URL)


def _redis_client_kwargs() -> dict[str, Any]:
    return {
        "decode_responses": True,
        "socket_connect_timeout": 5,
        "socket_timeout": cfg.REDIS_SOCKET_TIMEOUT_SEC,
        "retry_on_timeout": True,
        # health_check_interval>0 + next_health_check=0 → PING во время on_connect
        "health_check_interval": 0,
    }


def get_redis() -> Any:
    """Команды (SET/GET/PUBLISH) — отдельный пул, не pub/sub."""
    global _command_client
    if not redis_enabled():
        raise RuntimeError("Redis не настроен (REDIS_URL / KE_USE_REDIS)")
    if _command_client is None:
        import redis

        _command_client = redis.Redis.from_url(
            cfg.REDIS_URL,
            **_redis_client_kwargs(),
        )
        _command_client.ping()
    return _command_client


def get_redis_pubsub_client() -> Any:
    """Отдельный клиент для SUBSCRIBE — не смешивать с get_redis()."""
    global _pubsub_client
    if not redis_enabled():
        raise RuntimeError("Redis не настроен (REDIS_URL / KE_USE_REDIS)")
    if _pubsub_client is None:
        import redis

        _pubsub_client = redis.Redis.from_url(
            cfg.REDIS_URL,
            **_redis_client_kwargs(),
        )
        _pubsub_client.ping()
    return _pubsub_client


def reset_redis_pubsub_client() -> None:
    """После TimeoutError на pub/sub — пересоздать клиент."""
    global _pubsub_client
    if _pubsub_client is not None:
        try:
            _pubsub_client.close()
        except Exception:
            pass
    _pubsub_client = None


def redis_ping() -> bool:
    if not redis_enabled():
        return False
    try:
        get_redis().ping()
        return True
    except Exception:
        return False
