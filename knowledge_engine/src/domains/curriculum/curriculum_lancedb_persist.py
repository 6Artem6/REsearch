"""LanceDB: сохранять curriculum hits только после Lite batch approve."""

from __future__ import annotations

import asyncio

from knowledge_engine.src.core.run_log import trace
from knowledge_engine.src.domains.curriculum.curriculum_v08_harvest import (
    _deep_extract_blocks,
)
from knowledge_engine.src.domains.curriculum.schemas import CurriculumSearchHit
from knowledge_engine.src.shared.schemas import DocumentSummary
from knowledge_engine.src.shared.vector_store import VectorStore


def _hit_to_document_summary(hit: CurriculumSearchHit) -> DocumentSummary | None:
    url = (hit.url or "").strip()
    if not url.startswith("http"):
        return None
    takeaways = list(hit.key_extracts or [])
    if not takeaways and (hit.snippet or "").strip():
        takeaways = _deep_extract_blocks(
            [], [], [hit.snippet], min_words=40, max_words=220
        )
    if not takeaways:
        return None
    return DocumentSummary(
        title=(hit.title or url)[:400],
        url=url,
        key_takeaways=takeaways[:8],
        failure_modes=[],
        cs_concepts=[],
        diagram_descriptions=[],
    )


async def persist_approved_curriculum_hits_to_lancedb_async(
    hits: list[CurriculumSearchHit],
    *,
    label: str = "batch_approved",
) -> int:
    """После batch_lite_eval: EMBED только для утверждённых hits.

    ``skip_rag_ingest=False`` (по прямому запросу — раньше был True):
    без этого ``[Rn]`` chunk-level цитаты (RAG_CHUNKS_TABLE, LanceDB,
    см. tutor_source_citations.py/retrieve_lecture_rag_context) никогда не
    заполнялись для контента, заземлённого через этот общий Map-Reduce путь
    (Node Grounding Gate/finalize_node_grounding, Штурвал/
    steering_topic_node_service.py, targeted_node_grounding.py) — модель
    видела только [Sn] (document-level каталог, Qdrant). Чанкуется тот же
    текст, что уже идёт в [Sn]-паспорт (title/executive_summary/takeaways —
    см. _summary_document в vector_store.py), НЕ полный исходный текст
    статьи (тот в эту функцию не приходит, только сжатые key_extracts)."""
    if not hits:
        return 0
    store = VectorStore()
    saved = 0
    failed = 0
    for hit in hits:
        ds = _hit_to_document_summary(hit)
        if not ds:
            continue
        ok = await store.save_summary(ds, skip_rag_ingest=False)
        if ok:
            saved += 1
            trace(
                f"CURRICULUM LanceDB persist ✓ | {label} | "
                f"{ds.url[:70]} | takeaways={len(ds.key_takeaways)}"
            )
        else:
            failed += 1
            trace(
                f"CURRICULUM LanceDB persist ✗ | {label} | Qdrant write failed "
                f"(see QDRANT save ✗ / VECTOR_STORE qdrant ... skip above) | "
                f"{ds.url[:70]}"
            )
    if saved:
        trace(f"CURRICULUM LanceDB persist ✓ | {label} | saved={saved}/{len(hits)}")
    if failed:
        trace(f"CURRICULUM LanceDB persist ✗ | {label} | failed={failed}/{len(hits)}")
    return saved


def persist_approved_curriculum_hits_to_lancedb(
    hits: list[CurriculumSearchHit],
    *,
    label: str = "batch_approved",
) -> int:
    """Sync wrapper — legitimate top-level asyncio.run() bridge for callers
    rooted in the synchronous worker process (no event loop at all — see
    knowledge_engine/src/app/worker.py). An already-async caller must await
    persist_approved_curriculum_hits_to_lancedb_async(...) directly."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            persist_approved_curriculum_hits_to_lancedb_async(hits, label=label)
        )
    raise RuntimeError(
        "persist_approved_curriculum_hits_to_lancedb() called from inside a "
        "running event loop — await "
        "persist_approved_curriculum_hits_to_lancedb_async(...) directly instead"
    )
