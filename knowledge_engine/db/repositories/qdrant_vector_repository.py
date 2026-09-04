"""QdrantVectorRepository — тонкая ABC-обёртка над уже рабочим
services/qdrant_vector_store.py::QdrantVectorStore (Phase 3, см. prompt.txt:
"Repository Switcher"). Чистое делегирование, ноль новой логики — Qdrant-клиент
уже написан, протестирован и жив в проде как fallback (config.VECTOR_STORE_BACKEND
== "qdrant"); эта обёртка только формализует его под BaseVectorRepository.
"""

from __future__ import annotations

from typing import Any

from knowledge_engine.db.repositories.base_vector_repository import (
    BaseVectorRepository,
)
from knowledge_engine.services.qdrant_vector_store import QdrantVectorStore


class QdrantVectorRepository(BaseVectorRepository):
    def __init__(self, store: QdrantVectorStore | None = None) -> None:
        self._store = store or QdrantVectorStore()

    @property
    def enabled(self) -> bool:
        return self._store.enabled

    async def ensure_collection(self, collection_name: str, size: int) -> bool:
        return await self._store.ensure_collection(collection_name, size)

    async def upsert_documents(
        self,
        collection_name: str,
        documents: list[dict[str, Any]],
        vectors: list[list[float]],
    ) -> bool:
        return await self._store.upsert_documents(collection_name, documents, vectors)

    async def search_similar(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        *,
        doc_id_filter: list[str] | None = None,
        exclude_ids_filter: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return await self._store.search_similar(
            collection_name,
            query_vector,
            limit,
            doc_id_filter=doc_id_filter,
            exclude_ids_filter=exclude_ids_filter,
        )

    async def delete_by_field(
        self, collection_name: str, field: str, value: str
    ) -> bool:
        return await self._store.delete_by_field(collection_name, field, value)

    async def count_by_field(self, collection_name: str, field: str, value: str) -> int:
        return await self._store.count_by_field(collection_name, field, value)

    async def distinct_field_values(
        self, collection_name: str, field: str, *, page_size: int = 256
    ) -> set[str]:
        return await self._store.distinct_field_values(
            collection_name, field, page_size=page_size
        )

    async def fetch_by_url(
        self, collection_name: str, url: str
    ) -> dict[str, Any] | None:
        return await self._store.fetch_by_url(collection_name, url)
