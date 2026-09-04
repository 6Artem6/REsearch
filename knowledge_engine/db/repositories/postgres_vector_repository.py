"""PostgresVectorRepository — асинхронный доступ к pgvector-таблицам через
SQLAlchemy 2.0 Core + pgvector.sqlalchemy (Phase 3, см. prompt.txt).

Переписано с сырых asyncpg SQL-строк (Phase 2) на select()/insert()
Query Builder — убирает целый класс багов с ручной нумерацией $1/$2 (см.
CHANGELOG в git history: WHERE-клауза сталкивалась с LIMIT на одной и той же
позиции) и JSONB/vector кодеки SQLAlchemy обрабатывает сама, без ручных
conn.set_type_codec()/register_vector() (были нужны только для голого asyncpg).

Схема таблиц одинаковая для knowledge_atoms/intent_vectors/edge_case_vectors/
socratic_poles/light_rag_facts/v07_chunks/domain_registry/rag_chunks/
document_summaries: id/embed_model/embedding/payload(JSONB)/created_at —
репозиторий не завязан на конкретную таблицу, бизнес-поля живут в payload.
Таблицы не объявлены как ORM-модели — Alembic-миграции (Phase 1/2) остаются
источником истины схемы; здесь только Core Table-дескрипторы для запросов.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    TIMESTAMP,
    Column,
    MetaData,
    Table,
    Text,
    delete,
    distinct,
    func,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

_TABLE_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
_VECTOR_DIM = 1024  # BAAI/bge-m3 dense embedding dim (см. config.VECTOR_EMBED_DIM)

_metadata = MetaData()
_table_cache: dict[str, Table] = {}


def _validate_table(table: str) -> str:
    """Имена таблиц приходят из внутренних констант, не от пользователя —
    SQLAlchemy Core сам корректно квотирует identifier (в отличие от
    f-string SQL, которым была написана предыдущая версия), но проверка
    остаётся дешёвой защитой от regressions."""
    if not _TABLE_NAME_RE.match(table):
        raise ValueError(f"Небезопасное/некорректное имя таблицы: {table!r}")
    return table


def _get_table(name: str, *, dim: int = _VECTOR_DIM) -> Table:
    """Общий Core Table-дескриптор для любой vector-таблицы единой формы
    (id/embed_model/embedding/payload/created_at) — не ORM-модель, схема
    живёт в Alembic-миграциях, это только query-строитель поверх неё."""
    name = _validate_table(name)
    table = _table_cache.get(name)
    if table is not None:
        return table
    table = Table(
        name,
        _metadata,
        Column("id", UUID(as_uuid=True), primary_key=True),
        Column("embed_model", Text, nullable=False),
        Column("embedding", Vector(dim), nullable=False),
        Column("payload", JSONB, nullable=False),
        Column("created_at", TIMESTAMP(timezone=True)),
    )
    _table_cache[name] = table
    return table


def _to_asyncpg_dsn(dsn: str) -> str:
    """POSTGRES_DSN — "сырой" DSN (см. db/pg_settings.py), без диалект-суффикса.
    SQLAlchemy async engine нужен postgresql+asyncpg://."""
    if dsn.startswith("postgresql+"):
        return dsn
    return dsn.replace("postgresql://", "postgresql+asyncpg://", 1)


def generate_vector_id(doc_id: str, index: int | str, content: str) -> uuid.UUID:
    """Детерминированный UUIDv5 — закрывает баг vector_store.py:412 (там был
    uuid.uuid4(), из-за чего повторный ingest плодил дубли точек вместо upsert)."""
    return uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{index}:{content}")


class PostgresVectorRepository:
    def __init__(self, engine: AsyncEngine, *, default_ef_search: int = 100) -> None:
        self._engine = engine
        self._default_ef_search = default_ef_search

    @classmethod
    async def create(
        cls,
        dsn: str,
        *,
        pool_size: int = 10,
        default_ef_search: int = 100,
    ) -> "PostgresVectorRepository":
        engine = create_async_engine(
            _to_asyncpg_dsn(dsn), pool_size=pool_size, pool_pre_ping=True
        )
        return cls(engine, default_ef_search=default_ef_search)

    async def close(self) -> None:
        await self._engine.dispose()

    async def upsert_documents(
        self,
        table: str,
        rows: list[dict[str, Any]],
        vectors: Sequence[Sequence[float]],
        *,
        embed_model: str,
    ) -> int:
        """rows[i] должен содержать как минимум ключ "id" (см.
        generate_vector_id) — остальные бизнес-поля целиком уходят в payload."""
        tbl = _get_table(table)
        if len(rows) != len(vectors):
            raise ValueError("rows и vectors должны быть одной длины")
        values = [
            {
                "id": row["id"],
                "embed_model": embed_model,
                "embedding": list(vector),
                "payload": row.get("payload") or {},
            }
            for row, vector in zip(rows, vectors)
        ]
        stmt = pg_insert(tbl).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[tbl.c.id],
            set_={
                "embed_model": stmt.excluded.embed_model,
                "embedding": stmt.excluded.embedding,
                "payload": stmt.excluded.payload,
            },
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)
        return len(rows)

    async def search(
        self,
        table: str,
        query_vector: Sequence[float],
        *,
        limit: int = 10,
        ef_search: int | None = None,
        where_payload_eq: dict[str, str] | None = None,
        where_payload_in: dict[str, Sequence[str]] | None = None,
        where_payload_not_in: dict[str, Sequence[str]] | None = None,
    ) -> list[dict[str, Any]]:
        """Prefilter по полям payload (JSONB) через Core .where() — не
        f-string SQL. Ищет косинусной дистанцией (pgvector.sqlalchemy
        comparator), эквивалент .where() у LanceDB / Qdrant FieldCondition."""
        tbl = _get_table(table)
        ef = int(ef_search or self._default_ef_search)
        distance = tbl.c.embedding.cosine_distance(list(query_vector))
        stmt = (
            select(tbl.c.id, tbl.c.payload, distance.label("distance"))
            .order_by(distance)
            .limit(limit)
        )
        for key, value in (where_payload_eq or {}).items():
            stmt = stmt.where(tbl.c.payload[key].astext == value)
        for key, values in (where_payload_in or {}).items():
            stmt = stmt.where(tbl.c.payload[key].astext.in_(list(values)))
        for key, values in (where_payload_not_in or {}).items():
            stmt = stmt.where(tbl.c.payload[key].astext.notin_(list(values)))

        async with self._engine.begin() as conn:
            # SET LOCAL живёт только внутри текущей транзакции (conn.begin())
            # — не протекает на другие запросы того же пулового соединения.
            # Не параметризуется через bind (SET не принимает $N в Postgres) —
            # ef уже приведён к int() выше, инъекции не через что.
            await conn.execute(text(f"SET LOCAL hnsw.ef_search = {ef}"))
            result = await conn.execute(stmt)
            rows = result.mappings().all()
        return [dict(r) for r in rows]

    async def delete_by_field(self, table: str, field: str, value: str) -> int:
        """Аналог vector_store.py delete_by_field — отдельный метод, не
        автоматический шаг перед upsert (см. аудит: раньше это была
        неатомарная пара delete→upsert)."""
        tbl = _get_table(table)
        stmt = delete(tbl).where(tbl.c.payload[field].astext == value)
        async with self._engine.begin() as conn:
            result = await conn.execute(stmt)
        return result.rowcount or 0

    async def count_by_field(self, table: str, field: str, value: str) -> int:
        tbl = _get_table(table)
        stmt = (
            select(func.count())
            .select_from(tbl)
            .where(tbl.c.payload[field].astext == value)
        )
        async with self._engine.connect() as conn:
            return int(await conn.scalar(stmt) or 0)

    async def distinct_field_values(self, table: str, field: str) -> set[str]:
        tbl = _get_table(table)
        col = tbl.c.payload[field].astext
        stmt = select(distinct(col)).where(col.isnot(None))
        async with self._engine.connect() as conn:
            result = await conn.execute(stmt)
            return {row[0] for row in result if row[0]}

    async def fetch_by_field(
        self, table: str, field: str, value: str
    ) -> dict[str, Any] | None:
        tbl = _get_table(table)
        stmt = (
            select(tbl.c.id, tbl.c.payload)
            .where(tbl.c.payload[field].astext == value)
            .limit(1)
        )
        async with self._engine.connect() as conn:
            row = (await conn.execute(stmt)).mappings().first()
        return dict(row) if row else None
