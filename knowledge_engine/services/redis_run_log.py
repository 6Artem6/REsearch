"""Логи прогонов в Redis (list), вместо .runs/*.log."""

from __future__ import annotations

import json
from datetime import datetime

import knowledge_engine.config as cfg
from knowledge_engine.services.redis_client import get_redis, redis_enabled

_LOG_PREFIX = "ke:runlog:"


def redis_logs_enabled() -> bool:
    return redis_enabled() and cfg.KE_REDIS_LOGS


def log_key(log_id: str) -> str:
    return f"{_LOG_PREFIX}{log_id}"


def append_line(log_id: str, line: str) -> None:
    if not redis_logs_enabled():
        return
    r = get_redis()
    key = log_key(log_id)
    r.rpush(key, line)
    max_lines = max(1000, cfg.KE_REDIS_LOG_MAX_LINES)
    r.ltrim(key, -max_lines, -1)


def read_lines(log_id: str, start: int = 0, limit: int = 500) -> list[str]:
    if not redis_logs_enabled():
        return []
    r = get_redis()
    end = start + limit - 1
    return list(r.lrange(log_key(log_id), start, end))


def init_log(log_id: str, header: str) -> None:
    if not redis_logs_enabled():
        return
    r = get_redis()
    key = log_key(log_id)
    r.delete(key)
    r.rpush(key, header)


def write_heartbeat(pid: int) -> None:
    if not redis_enabled():
        return
    try:
        get_redis().setex(
            "ke:worker:heartbeat",
            45,
            json.dumps(
                {
                    "pid": pid,
                    "ts": datetime.now().isoformat(),
                },
                ensure_ascii=False,
            ),
        )
    except Exception:
        pass


def worker_heartbeat_alive(max_age_sec: float = 45.0) -> bool:
    if not redis_enabled():
        return False
    try:
        return bool(get_redis().exists("ke:worker:heartbeat"))
    except Exception:
        return False
