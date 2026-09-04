"""Init-container entrypoint (Phase 3, см. prompt.txt): накатывает Alembic
до head и проверяет целостность (vector extension + HNSW-индексы) при
старте docker-compose стека — без ручного вызова alembic. Используется
сервисом `migrator` в docker-compose.yml; зависимые сервисы (`knowledge-api`)
объявлены с `depends_on: migrator: condition: service_completed_successfully`
— ненулевой exit здесь не даёт им подняться на сломанной/недомигрированной БД.

Usage:
    PYTHONPATH=. ./.venv/bin/python -m knowledge_engine.scripts.run_migrations_and_verify
"""

from __future__ import annotations

import asyncio
import subprocess
import sys

# Все 9 pgvector-таблиц, созданных Phase 1/2 миграциями (см.
# knowledge_engine/alembic/versions/) — держится в коде, а не читается из
# alembic history, т.к. это узкая семантическая проверка ("векторные таблицы
# готовы к работе"), не общий "схема выше ревизии X".
_EXPECTED_HNSW_TABLES = (
    "knowledge_atoms",
    "intent_vectors",
    "edge_case_vectors",
    "socratic_poles",
    "light_rag_facts",
    "v07_chunks",
    "domain_registry",
    "rag_chunks",
    "document_summaries",
)


def _run_alembic_upgrade() -> None:
    # sys.executable -m alembic, не голый "alembic" на PATH — надёжнее
    # в контейнере (entrypoint.sh кладёт venv/bin на PATH, но лучше не
    # зависеть от порядка) и при локальном запуске через ./.venv/bin/python.
    print("MIGRATOR ▶ alembic upgrade head", flush=True)
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], check=True)
    print("MIGRATOR ✓ alembic upgrade head", flush=True)


async def _verify_integrity() -> None:
    import asyncpg

    from knowledge_engine.config import POSTGRES_DSN

    conn = await asyncpg.connect(POSTGRES_DSN)
    try:
        has_ext = await conn.fetchval(
            "SELECT count(*) FROM pg_extension WHERE extname = 'vector'"
        )
        if not has_ext:
            raise RuntimeError("extension 'vector' not found in pg_extension")

        missing: list[str] = []
        for table in _EXPECTED_HNSW_TABLES:
            has_hnsw = await conn.fetchval(
                "SELECT count(*) FROM pg_indexes "
                "WHERE tablename = $1 AND indexdef ILIKE '% USING hnsw %'",
                table,
            )
            if not has_hnsw:
                missing.append(table)
        if missing:
            raise RuntimeError(f"missing HNSW index for tables: {missing}")

        print(
            f"MIGRATOR ✓ integrity check passed | "
            f"vector extension ✓ | hnsw tables={len(_EXPECTED_HNSW_TABLES)} ✓",
            flush=True,
        )
    finally:
        await conn.close()


def main() -> None:
    try:
        _run_alembic_upgrade()
        asyncio.run(_verify_integrity())
    except Exception as exc:
        print(f"MIGRATOR ✗ {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
