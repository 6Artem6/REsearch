"""Standalone Qdrant Cloud connection check — creates+deletes a throwaway collection.

Verifies real connectivity via QdrantVectorStore: ping (get_collections), create a
`test_ping` collection, upsert one point, read it back through
QdrantVectorStore.fetch_by_url, then delete the test collection. Never prints the
raw QDRANT_API_KEY.

Usage:
    PYTHONPATH=. ./.venv/bin/python knowledge_engine/scripts/check_qdrant_connection.py
"""

from __future__ import annotations

import asyncio
import sys

from knowledge_engine.config import QDRANT_API_KEY, QDRANT_URL
from knowledge_engine.services.qdrant_vector_store import QdrantVectorStore

TEST_COLLECTION = "test_ping"
TEST_VECTOR = [0.1, 0.2, 0.3, 0.4]
TEST_URL = "https://ke-check-qdrant-connection.invalid/ping"
TEST_DOCUMENT = {
    "chunk_id": "ping-1",
    "url": TEST_URL,
    "title": "ping",
    "chunk_text": "ping",
}


async def main() -> int:
    if not QDRANT_URL:
        print("✗ QDRANT_URL не задан в .env — проверка пропущена.")
        return 1
    print(f"QDRANT_URL configured: {QDRANT_URL[:20]}...(masked)")
    print(f"QDRANT_API_KEY configured: {'yes' if QDRANT_API_KEY else 'no'}")

    store = QdrantVectorStore()
    if not store.enabled:
        print("✗ QdrantVectorStore disabled — проверьте конфиг.")
        return 1

    try:
        client = store._get_client()
    except Exception as exc:
        print(f"✗ client init failed | {type(exc).__name__}: {exc}")
        return 1

    # 1. Ping / healthcheck
    try:
        collections = await client.get_collections()
        names = {c.name for c in collections.collections}
        print(f"✓ ping ok | existing collections: {len(names)}")
    except Exception as exc:
        print(f"✗ ping failed | {type(exc).__name__}: {exc}")
        return 1

    from qdrant_client import models

    # 2. (Re)create the throwaway test collection
    try:
        if TEST_COLLECTION in names:
            await client.delete_collection(TEST_COLLECTION)
        await client.create_collection(
            collection_name=TEST_COLLECTION,
            vectors_config=models.VectorParams(
                size=len(TEST_VECTOR), distance=models.Distance.COSINE
            ),
        )
        print(f"✓ collection '{TEST_COLLECTION}' created")
    except Exception as exc:
        print(f"✗ create_collection failed | {type(exc).__name__}: {exc}")
        return 1

    result = 1
    try:
        # 3. Upsert one test vector via the adapter (not the raw client)
        ok = await store.upsert_documents(TEST_COLLECTION, [TEST_DOCUMENT], [TEST_VECTOR])
        print(f"{'✓' if ok else '✗'} upsert_documents via QdrantVectorStore")
        if not ok:
            return 1

        # 4. Read-through by url via the adapter
        fetched = await store.fetch_by_url(TEST_COLLECTION, TEST_URL)
        if fetched and fetched.get("chunk_id") == "ping-1":
            print(f"✓ fetch_by_url round-trip ok | payload={fetched}")
            result = 0
        else:
            print(f"✗ fetch_by_url round-trip FAILED | got={fetched}")
            result = 1
    finally:
        # 5. Cleanup — never leave the throwaway collection behind
        try:
            await client.delete_collection(TEST_COLLECTION)
            print(f"✓ cleanup: '{TEST_COLLECTION}' deleted")
        except Exception as exc:
            print(
                f"⚠ cleanup failed (test collection may remain) | "
                f"{type(exc).__name__}: {exc}"
            )

    return result


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
