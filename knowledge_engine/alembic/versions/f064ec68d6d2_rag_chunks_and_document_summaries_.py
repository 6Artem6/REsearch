"""rag_chunks and document_summaries tables (missed in phase 1)

Исправление своей же ошибки: исходный Phase 1-тикет просил
knowledge_atoms/rag_chunks/intent_vectors, но e6ab12466e89 по факту создал
knowledge_atoms/intent_vectors/edge_case_vectors — rag_chunks и
document_summaries (ровно те, куда vector_store.py реально пишет
save_summary()/_add_rag_chunk_rows(), и где живут 185+36+18 реальных точек
в Qdrant Cloud, см. backfill-скрипт) остались не созданы. edge_case_vectors
не убираю — она нужна отдельно (см. 2e85226d6a38), просто не должна была
попасть в e6ab12466e89 вместо rag_chunks.

Revision ID: f064ec68d6d2
Revises: 2e85226d6a38
Create Date: 2026-09-03 21:05:53.032859

"""

from collections.abc import Sequence
from typing import Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f064ec68d6d2"
down_revision: Union[str, Sequence[str], None] = "2e85226d6a38"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_EMBED_DIM = 1024
_HNSW_M = 16
_HNSW_EF_CONSTRUCTION = 64

_VECTOR_TABLES = ("rag_chunks", "document_summaries")


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
    for table in _VECTOR_TABLES:
        _create_vector_table(table)


def downgrade() -> None:
    for table in _VECTOR_TABLES:
        op.execute(f"DROP TABLE IF EXISTS {table}")
