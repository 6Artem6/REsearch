"""Stage 4 — Gemini Lite structured chunk extraction."""

from __future__ import annotations

import uuid
from typing import List

from knowledge_engine.llm_locale import GEMINI_RUSSIAN_ROLE, RUSSIAN_OUTPUT_RULE
from knowledge_engine.src.analytics.gemini_v07 import run_gemini_lite_structured
from knowledge_engine.src.analytics.schemas import ChunkExtractionResult
from knowledge_engine.src.processors.source_anchors import (
    SOURCE_ANCHOR_RETENTION_PROMPT,
)
from knowledge_engine.src.state import ScrapedDocument, StructuredChunk

_MAX_DOC_CHARS = 14_000


def _anchor_from_doc(doc: ScrapedDocument) -> str:
    return (
        f"Документ: {doc.source_url or doc.doc_id}\n"
        f"Тип источника: {doc.source_type}\n"
        f"doc_id: {doc.doc_id}"
    )


def extract_structured_chunks(
    doc: ScrapedDocument,
    global_anchor: str = "",
    source_anchor: str = "",
) -> List[StructuredChunk]:
    """
    Gemini Lite: сущности, code_snippets, concepts, p99_relevance_score.
    Без Ollama; не использует uma_resource_lock (только API).
    """
    text = (doc.raw_markdown or "").strip()
    if len(text) < 80:
        return []

    anchor = global_anchor.strip() or _anchor_from_doc(doc)
    anchor_tag = (
        f"[{source_anchor}]" if source_anchor else "(assign tags from SOURCE REGISTRY)"
    )
    system = (
        f"{GEMINI_RUSSIAN_ROLE} {RUSSIAN_OUTPUT_RULE} "
        "Разбей технический документ на 3–8 атомарных чанков для GraphRAG/LanceDB. "
        "Для каждого чанка: text (до 900 символов), concepts, code_snippets, "
        "p99_relevance_score (0–1, важность для tail latency / p99 / RAM на Apple Silicon). "
        f"Document source anchor: {anchor_tag}. "
        "Every chunk text MUST end with or include inline source tag(s) e.g. [S1]. "
        f"{SOURCE_ANCHOR_RETENTION_PROMPT} "
        "JSON ChunkExtractionResult."
    )
    user = (
        f"SOURCE ANCHOR for this document: {anchor_tag}\n"
        f"Источник: {doc.source_type}\n"
        f"URL: {doc.source_url}\n\n"
        f"--- markdown ---\n{text[:_MAX_DOC_CHARS]}"
    )

    result = run_gemini_lite_structured(
        system,
        user,
        anchor,
        ChunkExtractionResult,
        "chunker / ChunkExtractionResult",
    )

    out: List[StructuredChunk] = []
    for item in result.chunks:
        chunk_text = (item.text or "").strip()
        if len(chunk_text) < 40:
            continue
        out.append(
            StructuredChunk(
                chunk_id=uuid.uuid4().hex[:16],
                doc_id=doc.doc_id,
                text=chunk_text[:4000],
                concepts=[c.strip() for c in item.concepts if c.strip()],
                code_snippets=[s.strip() for s in item.code_snippets if s.strip()][:6],
                p99_relevance_score=float(item.p99_relevance_score),
                source_anchor=source_anchor or "",
            )
        )
    return out
