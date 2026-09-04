"""One-time LanceDB → Qdrant Cloud migration for `rag_chunks` and `document_summaries`.

Copies existing rows/vectors as-is (no re-embedding). Skips rows whose stored
embed_model doesn't match the current EMBED_MODEL (see db/embed_model_guard.py —
same guard the live retrieval path uses). Idempotent: point ids are deterministic
(chunk_id/url via qdrant_vector_store._point_id), safe to re-run.

Usage:
    PYTHONPATH=. ./.venv/bin/python knowledge_engine/scripts/migrate_lancedb_to_qdrant.py
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

BATCH_SIZE = 50
# chunk_vector/vector become the Qdrant point's primary vector, so they're excluded
# from payload; doc_meta_vector stays in payload (search_rag_chunk_rows' doc-level
# gate reads it directly, it isn't a named/indexed Qdrant vector).
_PRIMARY_VECTOR_FIELDS = {"vector", "chunk_vector"}


async def _ensure_collection(client: Any, name: str, size: int) -> None:
    from qdrant_client import models

    cols = await client.get_collections()
    names = {c.name for c in cols.collections}
    if name in names:
        print(f"  · Qdrant collection '{name}' already exists")
        return
    await client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
    )
    print(f"  ✓ created Qdrant collection '{name}' (size={size}, distance=COSINE)")


async def _migrate_table(
    store: Any,
    client: Any,
    *,
    lance_rows: list[dict],
    collection_name: str,
    vector_field: str,
    label: str,
) -> dict:
    from knowledge_engine.db.embed_model_guard import row_matches_embed_model

    total = len(lance_rows)
    skipped_no_vector = 0
    skipped_embed_mismatch = 0
    eligible: list[dict] = []
    for row in lance_rows:
        if row.get(vector_field) is None:
            skipped_no_vector += 1
            continue
        if not row_matches_embed_model(row):
            skipped_embed_mismatch += 1
            continue
        eligible.append(row)

    print(
        f"[{label}] total={total} eligible={len(eligible)} "
        f"skipped_no_vector={skipped_no_vector} skipped_embed_mismatch={skipped_embed_mismatch}"
    )

    migrated = 0
    failed_batches = 0
    if eligible:
        dim = len(eligible[0][vector_field])
        await _ensure_collection(client, collection_name, dim)

    for i in range(0, len(eligible), BATCH_SIZE):
        batch = eligible[i : i + BATCH_SIZE]
        docs = [
            {k: v for k, v in row.items() if k not in _PRIMARY_VECTOR_FIELDS}
            for row in batch
        ]
        for doc in docs:
            if isinstance(doc.get("doc_meta_vector"), tuple):
                doc["doc_meta_vector"] = list(doc["doc_meta_vector"])
        vectors = [list(row[vector_field]) for row in batch]
        ok = await store.upsert_documents(collection_name, docs, vectors)
        status = "OK" if ok else "FAILED"
        if ok:
            migrated += len(batch)
        else:
            failed_batches += 1
        print(f"  [{label}] batch {i // BATCH_SIZE + 1} ({len(batch)} rows) → {status}")

    return {
        "table": label,
        "total_rows": total,
        "eligible": len(eligible),
        "migrated": migrated,
        "skipped_no_vector": skipped_no_vector,
        "skipped_embed_mismatch": skipped_embed_mismatch,
        "failed_batches": failed_batches,
    }


async def main() -> int:
    import lancedb

    from knowledge_engine.config import LANCE_DB_PATH, QDRANT_URL
    from knowledge_engine.services.qdrant_vector_store import QdrantVectorStore

    if not QDRANT_URL:
        print("✗ QDRANT_URL не задан в .env — миграция невозможна.")
        return 1

    store = QdrantVectorStore()
    if not store.enabled:
        print("✗ QdrantVectorStore disabled — проверьте конфиг.")
        return 1
    client = store._get_client()

    db = lancedb.connect(str(LANCE_DB_PATH))
    table_names = set(db.table_names())
    print(f"LanceDB path: {LANCE_DB_PATH}")
    print(f"Tables found: {sorted(table_names)}\n")

    targets = [
        ("rag_chunks", "chunk_vector"),
        ("document_summaries", "vector"),
    ]

    reports = []
    for table_name, vector_field in targets:
        if table_name not in table_names:
            print(f"· LanceDB table '{table_name}' not found — skipped\n")
            continue
        tbl = db.open_table(table_name)
        rows = tbl.to_arrow().to_pylist()
        report = await _migrate_table(
            store,
            client,
            lance_rows=rows,
            collection_name=table_name,
            vector_field=vector_field,
            label=table_name,
        )
        reports.append(report)
        print()

    print("=== MIGRATION REPORT ===")
    total_migrated = 0
    for r in reports:
        print(
            f"{r['table']}: total={r['total_rows']} eligible={r['eligible']} "
            f"migrated={r['migrated']} skipped_no_vector={r['skipped_no_vector']} "
            f"skipped_embed_mismatch={r['skipped_embed_mismatch']} "
            f"failed_batches={r['failed_batches']}"
        )
        total_migrated += r["migrated"]
    print(f"TOTAL migrated: {total_migrated}")

    return 0 if all(r["failed_batches"] == 0 for r in reports) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
