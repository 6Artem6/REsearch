"""remaining lancedb tables to pgvector

Phase 2 (см. prompt.txt): оставшиеся 4 таблицы из аудита ingest/indexing
пайплайна — socratic_poles/light_rag_facts/v07_chunks/domain_registry. Та же
единая форма (id/embed_model/embedding/payload/created_at), что и в
e6ab12466e89 — переиспользуем helper оттуда буквально (Alembic-миграции
самодостаточны по design, поэтому helper продублирован, а не импортирован
из соседнего файла версии).

Revision ID: 2e85226d6a38
Revises: e6ab12466e89
Create Date: 2026-09-03 10:24:37.719797

"""

from collections.abc import Sequence
from typing import Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "2e85226d6a38"
down_revision: Union[str, Sequence[str], None] = "e6ab12466e89"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# BAAI/bge-m3 dense embedding dim (см. config.EMBED_MODEL/VECTOR_EMBED_DIM) —
# та же модель для всех LanceDB-таблиц проекта (см. embed_model_guard.py).
_EMBED_DIM = 1024
_HNSW_M = 16
_HNSW_EF_CONSTRUCTION = 64

_VECTOR_TABLES = (
    "socratic_poles",
    "light_rag_facts",
    "v07_chunks",
    "domain_registry",
)


def _create_vector_table(table: str) -> None:
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
    # CREATE EXTENSION vector уже сделан в e6ab12466e89 — здесь не повторяем.
    for table in _VECTOR_TABLES:
        _create_vector_table(table)


def downgrade() -> None:
    for table in _VECTOR_TABLES:
        op.execute(f"DROP TABLE IF EXISTS {table}")
