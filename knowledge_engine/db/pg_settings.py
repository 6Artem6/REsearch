"""Postgres/pgvector: типизированная конфигурация (Phase 0/1 миграции с LanceDB/Qdrant/SQLite).

Единственное место в проекте, где для НОВОЙ Postgres-инфраструктуры .env читается
через ``pydantic_settings.BaseSettings``, а не через ``os.getenv``. Это намеренное
локальное исключение из конвенции ``config.py`` ("все переменные окружения читаются
здесь") — переписывать ~800 строк существующего config.py на Pydantic Settings не
входило в задачу; вместо этого ``config.py`` ниже реэкспортирует несколько готовых
констант из этого модуля, чтобы остальной код по-прежнему мог делать
``from knowledge_engine.config import POSTGRES_DSN`` и не знать про эту границу.

DSN-формат: ``POSTGRES_DSN`` в .env — "сырой" (без ``+driver``), например
``postgresql://user:pass@host:5432/db``. И ``AsyncPostgresSaver`` (psycopg, async),
и ``asyncpg`` принимают этот формат напрямую. Только SQLAlchemy нужен
диалект-суффикс (``+psycopg`` sync / ``+asyncpg`` async, см. Alembic env.py) —
он выводится программно ниже, чтобы не дублировать DSN в .env три раза.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from knowledge_engine.dotenv_loader import load_dotenv_once

# Тот же лист-модуль, что использует config.py — .env парсится один раз
# (idempotent), но без цикла config.py <-> pg_settings.py (config.py в конце
# импортирует postgres_settings отсюда).
load_dotenv_once()


class PostgresSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    postgres_dsn: str = Field(
        default="postgresql://knowledge_engine:knowledge_engine@localhost:5432/knowledge_engine",
        alias="POSTGRES_DSN",
    )
    vector_embed_dim: int = Field(default=1024, alias="VECTOR_EMBED_DIM")
    vector_hnsw_m: int = Field(default=16, alias="VECTOR_HNSW_M")
    vector_hnsw_ef_construction: int = Field(
        default=64, alias="VECTOR_HNSW_EF_CONSTRUCTION"
    )
    vector_hnsw_ef_search: int = Field(default=100, alias="VECTOR_HNSW_EF_SEARCH")

    @property
    def sqlalchemy_sync_dsn(self) -> str:
        """Для Base.metadata/create_engine (article_diagrams и т.п.) — psycopg3 sync."""
        return self._with_scheme("postgresql+psycopg")

    @property
    def sqlalchemy_async_dsn(self) -> str:
        """Для Alembic env.py (async-режим, см. prompt.txt Phase 0) — asyncpg."""
        return self._with_scheme("postgresql+asyncpg")

    def _with_scheme(self, scheme: str) -> str:
        _, _, rest = self.postgres_dsn.partition("://")
        return f"{scheme}://{rest}"


postgres_settings = PostgresSettings()
