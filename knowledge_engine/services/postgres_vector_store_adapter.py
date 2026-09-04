"""Адаптер PostgresVectorRepository под интерфейс QdrantVectorStore.

Единственная цель — дать vector_store.py переключать бэкенд ОДНОЙ
фабричной функцией (_get_active_vector_store), не трогая ~9 call site'ов
`store.ensure_collection/upsert_documents/search_similar/delete_by_field`
по всему файлу (Repository Pattern за интерфейсом, который уже есть).

_point_id_from_document повторяет приоритетность ключа
qdrant_vector_store._point_id() (id → chunk_id → url, затем uuid5) —
одинаковый детерминированный id для одного документа НЕЗАВИСИМО от
бэкенда важен для backfill-скрипта (scripts/backfill_qdrant_to_postgres.py):
он просто копирует id как есть, не пересчитывает.
"""

from __future__ import annotations

import uuid
from typing import Any

from knowledge_engine.config import EMBED_MODEL
from knowledge_engine.db.repositories.base_vector_repository import (
    BaseVectorRepository,
)
from knowledge_engine.db.repositories.postgres_vector_repository import (
    PostgresVectorRepository,
)


def _point_id_from_document(document: dict[str, Any]) -> uuid.UUID:
    key = str(
        document.get("id") or document.get("chunk_id") or document.get("url") or ""
    ).strip()
    if not key:
        return uuid.uuid4()
    return uuid.uuid5(uuid.NAMESPACE_URL, key)


class PostgresVectorStoreAdapter(BaseVectorRepository):
    """Drop-in замена QdrantVectorStore для vector_store.py (см. модуль)."""

    enabled = True

    def __init__(self, repo: PostgresVectorRepository) -> None:
        self._repo = repo

    async def ensure_collection(self, collection_name: str, size: int) -> bool:
        # Таблица + HNSW-индекс уже созданы Alembic-миграцией — здесь нечего
        # делать (в отличие от Qdrant, где коллекция создаётся лениво в рантайме).
        return True

    async def upsert_documents(
        self,
        collection_name: str,
        documents: list[dict[str, Any]],
        vectors: list[list[float]],
    ) -> bool:
        if not documents or not vectors or len(documents) != len(vectors):
            return False
        rows = [
            {"id": _point_id_from_document(doc), "payload": doc} for doc in documents
        ]
        await self._repo.upsert_documents(
            collection_name, rows, vectors, embed_model=EMBED_MODEL
        )
        return True

    async def search_similar(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        *,
        doc_id_filter: list[str] | None = None,
        exclude_ids_filter: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not query_vector:
            return []
        allow = [d for d in (doc_id_filter or []) if (d or "").strip()]
        excl = [d for d in (exclude_ids_filter or []) if (d or "").strip()]
        results = await self._repo.search(
            collection_name,
            query_vector,
            limit=limit,
            where_payload_in={"doc_id": allow} if allow else None,
            where_payload_not_in={"id": excl} if excl else None,
        )
        out: list[dict[str, Any]] = []
        for r in results:
            payload = dict(r["payload"])
            # cosine distance (0..2) -> similarity score, тот же смысл, что
            # Qdrant COSINE score в search_similar (см. qdrant_vector_store.py).
            payload["_score"] = 1.0 - float(r["distance"])
            payload["_id"] = str(r["id"])
            out.append(payload)
        return out

    async def delete_by_field(
        self, collection_name: str, field: str, value: str
    ) -> bool:
        if not (value or "").strip():
            return False
        await self._repo.delete_by_field(collection_name, field, value)
        return True

    async def count_by_field(self, collection_name: str, field: str, value: str) -> int:
        if not (value or "").strip():
            return 0
        return await self._repo.count_by_field(collection_name, field, value)

    async def distinct_field_values(
        self, collection_name: str, field: str, *, page_size: int = 256
    ) -> set[str]:
        # page_size — только для сигнатурной совместимости с QdrantVectorStore
        # (там нужен для ручного scroll постранично); Postgres делает DISTINCT
        # одним запросом, страницы не нужны.
        del page_size
        return await self._repo.distinct_field_values(collection_name, field)

    async def fetch_by_url(
        self, collection_name: str, url: str
    ) -> dict[str, Any] | None:
        target = (url or "").strip()
        if not target:
            return None
        row = await self._repo.fetch_by_field(collection_name, "url", target)
        if row is None:
            return None
        payload = dict(row["payload"])
        payload["_id"] = str(row["id"])
        return payload
