"""Backfill Qdrant Cloud -> Postgres/pgvector (Phase 2, см. prompt.txt).

Переносит существующие точки из Qdrant (document_summaries/knowledge_atoms/
rag_chunks — единственные коллекции с реальными данными на момент аудита:
18/185/36 точек) в соответствующие Postgres-таблицы ДО того, как
vector_store.py переключится на PostgresVectorRepository как основной бэкенд
(config.VECTOR_STORE_BACKEND=postgres) — без этого RAG-выдача для уже
проиндексированных курикулумов молча опустеет до повторного ingest.

Идемпотентно: id берётся НАПРЯМУЮ из Qdrant point.id (уже был детерминирован
через _point_id() при записи, кроме knowledge_atoms до фикса — но и там id
просто копируется как есть, не пересчитывается, чтобы точка соответствия
Qdrant<->Postgres была однозначной и стабильной при повторном запуске
скрипта). Повторный запуск — ON CONFLICT DO UPDATE, безопасно перезапускать.

Usage:
    PYTHONPATH=. ./.venv/bin/python -m knowledge_engine.scripts.backfill_qdrant_to_postgres
    PYTHONPATH=. ./.venv/bin/python -m knowledge_engine.scripts.backfill_qdrant_to_postgres --dry-run
"""

from __future__ import annotations

import argparse
import asyncio

from knowledge_engine.config import EMBED_MODEL, POSTGRES_DSN, QDRANT_URL
from knowledge_engine.db.repositories.postgres_vector_repository import (
    PostgresVectorRepository,
)

_COLLECTIONS = ("document_summaries", "knowledge_atoms", "rag_chunks")
_SCROLL_PAGE_SIZE = 128


async def _scroll_all_points(client, collection: str):
    offset = None
    while True:
        points, next_offset = await client.scroll(
            collection_name=collection,
            limit=_SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for p in points:
            yield p
        if not next_offset or not points:
            return
        offset = next_offset


async def backfill_collection(
    repo: PostgresVectorRepository, client, collection: str, *, dry_run: bool
) -> tuple[int, int]:
    """Возвращает (scanned, written)."""
    scanned = 0
    written = 0
    batch_rows: list[dict] = []
    batch_vectors: list[list[float]] = []

    async def _flush() -> None:
        nonlocal written
        if not batch_rows:
            return
        if not dry_run:
            await repo.upsert_documents(
                collection, batch_rows, batch_vectors, embed_model=EMBED_MODEL
            )
        written += len(batch_rows)
        batch_rows.clear()
        batch_vectors.clear()

    async for point in _scroll_all_points(client, collection):
        scanned += 1
        payload = dict(point.payload or {})
        vector = point.vector
        if isinstance(vector, dict):
            # multi-vector коллекции здесь не ожидаются, но на всякий случай
            vector = next(iter(vector.values()), None)
        if not vector:
            continue
        batch_rows.append({"id": point.id, "payload": payload})
        batch_vectors.append(list(vector))
        if len(batch_rows) >= _SCROLL_PAGE_SIZE:
            await _flush()
    await _flush()
    return scanned, written


async def main_async(dry_run: bool) -> None:
    if not (QDRANT_URL or "").strip():
        print("QDRANT_URL не задан — нечего переносить.")
        return

    from qdrant_client import AsyncQdrantClient

    from knowledge_engine.services.qdrant_vector_store import QdrantVectorStore

    qdrant_store = QdrantVectorStore()
    if not qdrant_store.enabled:
        print("QdrantVectorStore.enabled=False — нечего переносить.")
        return
    client: AsyncQdrantClient = qdrant_store._get_client()

    repo = await PostgresVectorRepository.create(POSTGRES_DSN)
    try:
        existing = {c.name for c in (await client.get_collections()).collections}
        total_scanned = 0
        total_written = 0
        for collection in _COLLECTIONS:
            if collection not in existing:
                print(f"{collection}: коллекции нет в Qdrant, пропуск")
                continue
            scanned, written = await backfill_collection(
                repo, client, collection, dry_run=dry_run
            )
            total_scanned += scanned
            total_written += written
            mode = "would write" if dry_run else "written"
            print(f"{collection}: scanned={scanned} {mode}={written}")
        print(f"TOTAL: scanned={total_scanned} written={total_written}")
    finally:
        await repo.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только посчитать точки, ничего не писать в Postgres.",
    )
    args = parser.parse_args()
    asyncio.run(main_async(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
