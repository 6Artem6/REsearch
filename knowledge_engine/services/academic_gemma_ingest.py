"""Gemma-only Map-Reduce ingest for academic papers (no local Ollama summarizer)."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from knowledge_engine.config import (
    ACADEMIC_INGEST_MAX_BODY_CHARS,
    GEMMA_CONCURRENCY,
    gemma_cloud_api_key_available,
)
from knowledge_engine.schemas import DocumentSummary
from knowledge_engine.services.article_ingestion.blog_spatial_pipeline import (
    _document_summary_from_final,
)
from knowledge_engine.services.article_ingestion.blog_spatial_summarizer import (
    MapReduceArticleJob,
    map_reduce_jobs_pooled_async,
)
from knowledge_engine.services.article_ingestion.paragraph_token_splitter import (
    TokenWindowChunk,
    estimate_text_tokens,
    split_annotated_text_by_tokens,
)
from knowledge_engine.services.vector_store import VectorStore
from knowledge_engine.ui.run_log import trace

_ACADEMIC_INGEST_SEM = asyncio.Semaphore(max(1, GEMMA_CONCURRENCY))


@dataclass
class AcademicGemmaIngestResult:
    summary: DocumentSummary
    map_window_texts: list[str]


def _require_gemma_cloud() -> bool:
    if gemma_cloud_api_key_available():
        return True
    trace(
        "ACADEMIC ingest ⊘ | GEMINI_API_KEY required — "
        "local Ollama summarizer disabled for academic pipeline"
    )
    return False


async def ingest_academic_body_gemma(
    title: str,
    url: str,
    body: str,
    store: VectorStore,
    *,
    target_topic: str = "",
    pdf_bytes: bytes | None = None,
    apply_paper_structure: bool = True,
    gemma_budget_blocking: bool = False,
    budget_wait_sec: float = 600.0,
) -> AcademicGemmaIngestResult | None:
    """Map-reduce via Gemma cloud; persist ALL MAP window bodies + REDUCE summary."""
    if not _require_gemma_cloud():
        return None
    text = (body or "").strip()
    if len(text) < 80:
        return None

    if apply_paper_structure:
        from knowledge_engine.src.parsers.paper_structure_analyzer import (
            is_academic_pdf_url,
            prepare_paper_body_for_gemma_async,
            try_fetch_pdf_bytes_for_url,
        )

        topic = (target_topic or title or url).strip()
        pdf = pdf_bytes
        if pdf is None and is_academic_pdf_url(url):
            pdf = await asyncio.to_thread(try_fetch_pdf_bytes_for_url, url)
        if topic or pdf or is_academic_pdf_url(url):
            text = await prepare_paper_body_for_gemma_async(
                text,
                topic or "scientific paper",
                pdf_bytes=pdf,
                label=url[:48],
                page_url=url,
            )

    if len(text) < 80:
        return None
    clipped = text[:ACADEMIC_INGEST_MAX_BODY_CHARS]
    windows = split_annotated_text_by_tokens(
        clipped,
        title=title or url,
        all_figure_ids=[],
        figure_registry=None,
    )
    if not windows:
        windows = [TokenWindowChunk(window_index=0, body=clipped)]
    windows_body = [w.body for w in windows if (w.body or "").strip()]

    from knowledge_engine.src.services.openalex_evaluator import (
        resolve_source_trust_score,
    )

    trust = resolve_source_trust_score(url)
    job = MapReduceArticleJob(
        job_id=url,
        title=title or url,
        url=url,
        windows=windows,
        all_figure_ids=[],
        figure_registry=None,
        trust_score=trust,
    )

    from knowledge_engine.config import GEMMA_MAP_MAX_OUTPUT_TOKENS
    from knowledge_engine.services.gemma_rate_limiter import (
        get_gemma_token_budget_manager,
    )

    budget_est = estimate_text_tokens(clipped) + GEMMA_MAP_MAX_OUTPUT_TOKENS * max(
        1, len(windows)
    )
    mgr = get_gemma_token_budget_manager()
    if gemma_budget_blocking:
        await mgr.acquire_budget_blocking(budget_est, max_wait_sec=budget_wait_sec)
    else:
        acquire = await mgr.acquire_budget(
            budget_est,
            max_wait_for_overflow=budget_wait_sec,
        )
        if acquire.overflow_to_flash:
            trace(
                f"ACADEMIC ingest ⊘ | Gemma budget overflow — "
                f"strict stream requires blocking wait | {url[:55]}"
            )
            return None

    async with _ACADEMIC_INGEST_SEM:
        trace(
            f"ACADEMIC ingest ▶ | Gemma budget "
            f"tpm≤{get_gemma_token_budget_manager().max_tpm} "
            f"map-reduce pooled"
        )
        pooled = await map_reduce_jobs_pooled_async([job], force_gemma_cloud=True)
    outcome = pooled.get(job.job_id)
    final = outcome.final if outcome else None
    if final is None:
        trace(f"ACADEMIC ingest ✗ | Gemma map-reduce failed | {url[:55]}")
        return None

    summary = _document_summary_from_final(
        final,
        title=title,
        url=url,
        registry=None,
    )
    store.save_summary(summary)
    # Align MAP bodies + window_summary by window_index (skip empty bodies).
    map_texts: list[str] = []
    window_summaries: list[str | None] = []
    map_results = list(outcome.map_results) if outcome is not None else []
    for i, w in enumerate(windows):
        body = (w.body or "").strip()
        if not body:
            continue
        map_texts.append(body)
        m = map_results[i] if i < len(map_results) else None
        if m is None:
            window_summaries.append(None)
        else:
            window_summaries.append((m.window_summary or "").strip() or None)
    n_map = store.upsert_rag_academic_map_windows(
        url,
        title or url,
        map_texts or windows_body,
        summary,
        window_summaries=window_summaries,
    )
    n_atoms = store.upsert_knowledge_atoms(url, list(final.knowledge_atoms or []))
    trace(
        f"ACADEMIC ingest ✓ | Gemma map-reduce | map_windows={len(map_texts or windows_body)} "
        f"rag_rows={n_map} atoms={n_atoms} | {url[:55]}"
    )
    return AcademicGemmaIngestResult(
        summary=summary,
        map_window_texts=map_texts or windows_body,
    )


async def ingest_academic_batch_gemma(
    items: list[tuple[str, str, str]],
    store: VectorStore,
) -> list[AcademicGemmaIngestResult]:
    """Parallel batch (per-item semaphore inside ingest_academic_body_gemma)."""

    async def _one(t: str, u: str, b: str) -> AcademicGemmaIngestResult | None:
        return await ingest_academic_body_gemma(t, u, b, store)

    results = await asyncio.gather(
        *[_one(title, url, body) for title, url, body in items],
        return_exceptions=True,
    )
    out: list[AcademicGemmaIngestResult] = []
    for r in results:
        if isinstance(r, AcademicGemmaIngestResult):
            out.append(r)
        elif isinstance(r, Exception):
            trace(f"ACADEMIC ingest batch skip | {r}")
    return out
