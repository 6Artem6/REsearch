"""persist_approved_curriculum_hits_to_lancedb_async — регрессия: раньше

вызывался с ``skip_rag_ingest=True``, из-за чего chunk-level RAG-индекс
(``RAG_CHUNKS_TABLE``, LanceDB) никогда не заполнялся для контента,
заземлённого через общий Map-Reduce путь (Node Grounding Gate, Штурвал) —
модель видела только ``[Sn]`` (document-level каталог, Qdrant), но не
``[Rn]`` (chunk-level RAG). См. docs/STEERING_AND_TOPIC_QNA_ROADMAP.md,
"[Rn] RAG-инжест включён при Map-Reduce"."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from knowledge_engine.src.domains.curriculum.curriculum_lancedb_persist import (
    persist_approved_curriculum_hits_to_lancedb_async,
)
from knowledge_engine.src.domains.curriculum.schemas import CurriculumSearchHit


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_persist_calls_save_summary_with_rag_ingest_enabled() -> None:
    hit = CurriculumSearchHit(
        url="https://habr.com/a",
        title="A",
        key_extracts=["extract one is long enough to pass the word-count floor here"],
        source_tier="whitelist_blog",
    )
    with patch(
        "knowledge_engine.src.domains.curriculum.curriculum_lancedb_persist.VectorStore"
    ) as mock_store_cls:
        mock_store = mock_store_cls.return_value
        mock_store.save_summary = AsyncMock(return_value=True)
        saved = await persist_approved_curriculum_hits_to_lancedb_async([hit])

    assert saved == 1
    mock_store.save_summary.assert_awaited_once()
    _, kwargs = mock_store.save_summary.call_args
    assert kwargs.get("skip_rag_ingest") is False
