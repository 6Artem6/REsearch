"""LanceDB: сохранять curriculum hits только после Lite batch approve."""

from __future__ import annotations

from knowledge_engine.schemas import DocumentSummary
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.src.curriculum.curriculum_v08_harvest import _deep_extract_blocks
from knowledge_engine.src.curriculum.schemas import CurriculumSearchHit
from knowledge_engine.ui.run_log import trace


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


def persist_approved_curriculum_hits_to_lancedb(
    hits: list[CurriculumSearchHit],
    *,
    label: str = "batch_approved",
) -> int:
    """После batch_lite_eval: EMBED только для утверждённых hits."""
    if not hits:
        return 0
    store = VectorStore()
    saved = 0
    for hit in hits:
        ds = _hit_to_document_summary(hit)
        if not ds:
            continue
        store.save_summary(ds, skip_rag_ingest=True)
        saved += 1
        trace(
            f"CURRICULUM LanceDB persist ✓ | {label} | "
            f"{ds.url[:70]} | takeaways={len(ds.key_takeaways)}"
        )
    if saved:
        trace(f"CURRICULUM LanceDB persist ✓ | {label} | saved={saved}/{len(hits)}")
    return saved
