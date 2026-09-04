"""Repository Pattern — общий контракт для Postgres-репозиториев.

См. prompt.txt (Phase 0/1 миграции): "Прямые SQL/Vector-вызовы из графов
LangGraph и хэндлеров запрещены" — бизнес-код (графовые узлы, API-хэндлеры)
работает только через репозитории/сервисы, не через asyncpg/psycopg напрямую.
"""

from __future__ import annotations

from abc import ABC


class BaseRepository(ABC):
    """Пустой пока намеренно — общая точка расширения для будущих сквозных
    забот на уровне репозитория (retry/metrics/tracing), без давления
    придумывать абстрактные методы, которых сейчас только один наследник."""
