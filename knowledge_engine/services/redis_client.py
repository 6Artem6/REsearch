"""Redis client (очередь worker, логи)."""

from __future__ import annotations

import socket
from typing import Any

import knowledge_engine.config as cfg

_command_client: Any | None = None
_pubsub_client: Any | None = None


def redis_enabled() -> bool:
    return cfg.KE_USE_REDIS and bool(cfg.REDIS_URL)


def _tcp_keepalive_options() -> dict[int, int]:
    """Тюнинг TCP keepalive поверх ``socket_keepalive=True`` — голый флаг
    без опций полагается на дефолты ОС (Linux: 7200s до первого пробника),
    что для долгоживущего воркера с фазами простоя между задачами практически
    не отличается от отсутствия keepalive вообще: idle-сокет, тихо убитый
    NAT/LB/Docker-сетью, обнаруживается только следующей реальной командой,
    которая блокируется на полный ``socket_timeout``. Опции — Linux-специфичны
    (``TCP_KEEPIDLE``/``TCP_KEEPINTVL``/``TCP_KEEPCNT``); на macOS/иных ОС их
    может не быть в модуле ``socket`` — берём только то, что реально есть."""
    opts: dict[int, int] = {}
    idle = getattr(socket, "TCP_KEEPIDLE", None)
    interval = getattr(socket, "TCP_KEEPINTVL", None)
    count = getattr(socket, "TCP_KEEPCNT", None)
    if idle is not None:
        opts[idle] = 30
    if interval is not None:
        opts[interval] = 10
    if count is not None:
        opts[count] = 3
    return opts


def _redis_client_kwargs(*, for_pubsub: bool = False) -> dict[str, Any]:
    """
    Command clients keep TCP alive (idle Redis/Docker often drops quiet sockets).
    Explicit health checks are used instead of redis-py's connection health PING.

    ``retry_on_timeout=False``: и command-, и pubsub-клиент уже переигрываются
    на уровне приложения (``worker/__main__.py::_safe_redis_command`` /
    ``_reconnect_pubsub``) — включённый redis-py-шный внутренний повтор при
    таймауте молча удваивал время одной "залипшей" попытки (до 2×
    ``socket_timeout`` ДО того, как наш собственный reconnect вообще
    срабатывал), что и растягивало каждый цикл переподключения на минуты
    при недоступном/тихо оборванном Redis.
    """
    return {
        "decode_responses": True,
        "socket_connect_timeout": 5,
        "socket_timeout": cfg.REDIS_SOCKET_TIMEOUT_SEC,
        "retry_on_timeout": False,
        "socket_keepalive": True,
        "socket_keepalive_options": _tcp_keepalive_options(),
        # redis-py 5.3.1 can recurse connect → health PING → connect forever
        # while recovering a stale socket. Pub/sub PINGs are invalid as well.
        "health_check_interval": 0,
    }


def get_redis() -> Any:
    """Команды (SET/GET/PUBLISH) — отдельный пул, не pub/sub."""
    global _command_client
    if not redis_enabled():
        raise RuntimeError("Redis не настроен (REDIS_URL / KE_USE_REDIS)")
    if _command_client is None:
        import redis

        client = redis.Redis.from_url(
            cfg.REDIS_URL,
            **_redis_client_kwargs(for_pubsub=False),
        )
        client.ping()
        _command_client = client
    return _command_client


def get_redis_pubsub_client() -> Any:
    """Отдельный клиент для SUBSCRIBE — не смешивать с get_redis()."""
    global _pubsub_client
    if not redis_enabled():
        raise RuntimeError("Redis не настроен (REDIS_URL / KE_USE_REDIS)")
    if _pubsub_client is None:
        import redis

        client = redis.Redis.from_url(
            cfg.REDIS_URL,
            **_redis_client_kwargs(for_pubsub=True),
        )
        client.ping()
        _pubsub_client = client
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


def reset_redis_command_client() -> None:
    """После ошибки команды Redis пересоздать отдельный command-клиент."""
    global _command_client
    if _command_client is not None:
        try:
            _command_client.close()
        except Exception:
            pass
    _command_client = None


def redis_ping() -> bool:
    if not redis_enabled():
        return False
    try:
        get_redis().ping()
        return True
    except Exception:
        reset_redis_command_client()
        return False
