"""pgvector init

Phase 1 (см. prompt.txt): CREATE EXTENSION vector + первые 3 векторные таблицы
(knowledge_atoms/intent_vectors/edge_case_vectors — остальные LanceDB-таблицы,
найденные в аудите — socratic_poles/light_rag_facts/v07_chunks/domain_registry,
получают ту же форму отдельными миграциями по мере переноса).

Raw SQL (op.execute), не SQLAlchemy ORM: PostgresVectorRepository работает через
asyncpg напрямую (см. db/repositories/postgres_vector_repository.py), поэтому
autogenerate не задействован — этот файл самодостаточен и не читает
config.VECTOR_HNSW_* в рантайме (стабильность миграции: значения здесь всегда
буквальные m=16/ef_construction=64/dim=1024, как в prompt.txt Phase 1).

Revision ID: e6ab12466e89
Revises:
Create Date: 2026-09-03 00:02:59.573603

"""

from collections.abc import Sequence
from typing import Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e6ab12466e89"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# BAAI/bge-m3 dense embedding dim (см. config.EMBED_MODEL/VECTOR_EMBED_DIM).
_EMBED_DIM = 1024
_HNSW_M = 16
_HNSW_EF_CONSTRUCTION = 64

_VECTOR_TABLES = ("knowledge_atoms", "intent_vectors", "edge_case_vectors")


def _create_vector_table(table: str) -> None:
    """Единая форма для всех vector-таблиц: технические колонки (id/embedding/
    embed_model/created_at) фиксированы, остальные бизнес-поля — в payload
    JSONB (см. PostgresVectorRepository — он не завязан на конкретную схему
    таблицы и умеет работать с любой из них)."""
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id          UUID PRIMARY KEY,
            embed_model TEXT NOT NULL,
            embedding   VECTOR({_EMBED_DIM}) NOT NULL,
            payload     JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS {table}_embedding_hnsw
        ON {table} USING hnsw (embedding vector_cosine_ops)
        WITH (m = {_HNSW_M}, ef_construction = {_HNSW_EF_CONSTRUCTION})
        """
    )


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    for table in _VECTOR_TABLES:
        _create_vector_table(table)


def downgrade() -> None:
    for table in _VECTOR_TABLES:
        op.execute(f"DROP TABLE IF EXISTS {table}")
    # CREATE EXTENSION нарочно не откатываем — другие таблицы/расширения могли
    # начать на него полагаться; удаление extension'а на downgrade — отдельное
    # осознанное действие, не побочный эффект отката этой миграции.
