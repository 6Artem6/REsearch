"""BaseVectorRepository — формальный ABC для switchable vector-бэкендов
(Phase 3, см. prompt.txt: "Repository Switcher").

Форма интерфейса намеренно повторяет то, что vector_store.py реально зовёт
на все ~9 read/write путей (см. Phase 2) — ensure_collection/upsert_documents/
search_similar/delete_by_field/count_by_field/distinct_field_values/
fetch_by_url — а не буквальный "upsert/search/delete_by_field" из тикета:
это не троекратно дублирующая абстракция, а формализация УЖЕ рабочего
переключателя (config.VECTOR_STORE_BACKEND → _get_active_vector_store() в
vector_store.py), чтобы у него была явная типизированная граница.

Наследники:
- services/postgres_vector_store_adapter.py::PostgresVectorStoreAdapter
- db/repositories/qdrant_vector_repository.py::QdrantVectorRepository
  (тонкая обёртка над уже рабочим services/qdrant_vector_store.py —
  переписывать Qdrant-клиент с нуля не было запроса и не было нужды).

LanceDB сознательно НЕ формализована в третий swappable-бэкенд — она живёт
инлайн внутри services/vector_store.py::VectorStore как legacy pre-Qdrant
путь и никогда не была за отдельным интерфейсом; выделение её в
LanceDBVectorRepository без конкретной новой потребности — риск сломать
рабочий код ради симметрии, не запрошено явно.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseVectorRepository(ABC):
    @abstractmethod
    async def ensure_collection(self, collection_name: str, size: int) -> bool: ...

    @abstractmethod
    async def upsert_documents(
        self,
        collection_name: str,
        documents: list[dict[str, Any]],
        vectors: list[list[float]],
    ) -> bool: ...

    @abstractmethod
    async def search_similar(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        *,
        doc_id_filter: list[str] | None = None,
        exclude_ids_filter: list[str] | None = None,
    ) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def delete_by_field(
        self, collection_name: str, field: str, value: str
    ) -> bool: ...

    @abstractmethod
    async def count_by_field(
        self, collection_name: str, field: str, value: str
    ) -> int: ...

    @abstractmethod
    async def distinct_field_values(
        self, collection_name: str, field: str, *, page_size: int = 256
    ) -> set[str]: ...

    @abstractmethod
    async def fetch_by_url(
        self, collection_name: str, url: str
    ) -> dict[str, Any] | None: ...
