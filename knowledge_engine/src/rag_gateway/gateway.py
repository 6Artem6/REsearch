"""Directional RAG Gateway — векторный поиск, cross-encoder, опционально Gemma-сжатие фактов."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from difflib import SequenceMatcher

from knowledge_engine.config import (
    KE_RAG_TIMEOUT_SEC,
    RAG_GATEWAY_FINISH_MARGIN_SEC,
    RAG_LATENCY_WARN_MS,
    RAG_RETRIEVAL_PER_DIRECTION,
)
from knowledge_engine.src.locks import run_under_uma_lock
from knowledge_engine.src.memory.light_rag import LightRAG
from knowledge_engine.src.rag_gateway.cross_encoder import score_relevance_pairs
from knowledge_engine.src.rag_gateway.fact_compressor import compress_fact_if_needed
from knowledge_engine.src.rag_gateway.fact_text import FACT_MAX_CHARS
from knowledge_engine.src.rag_gateway.schemas import (
    DirectionalRAGQuery,
    DirectionalRAGResponse,
    RankedMemoryFact,
    SaveUserFactRequest,
)
from knowledge_engine.ui.run_log import trace

_DEDUP_OVERLAP = 0.90


@dataclass
class _Candidate:
    text: str
    direction_label: str
    weight: float


def _text_overlap(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _deduplicate_facts(
    ranked: list[tuple[float, str, str]],
) -> list[tuple[float, str, str]]:
    """Оставить факт с максимальным score при перекрытии > 90%."""
    sorted_rows = sorted(ranked, key=lambda x: x[0], reverse=True)
    kept: list[tuple[float, str, str]] = []
    kept_texts: list[str] = []
    for score, direction, text in sorted_rows:
        if any(_text_overlap(text, prev) >= _DEDUP_OVERLAP for prev in kept_texts):
            continue
        kept.append((score, direction, text))
        kept_texts.append(text)
    return kept


async def query_directional_rag(req: DirectionalRAGQuery) -> DirectionalRAGResponse:
    """
    Модуль 3: векторный поиск → cross-encoder → cutoff → дедуп → сжатие → top-N.
    """
    t0 = time.perf_counter()
    trace(f"RAG_GATEWAY ▶ directional | node={req.target_node}")
    rag = LightRAG()
    per_dir = RAG_RETRIEVAL_PER_DIRECTION
    by_text: dict[str, _Candidate] = {}

    for direction in req.search_directions:
        hits = await rag.vector_search(
            direction.vector_query,
            per_dir,
            kinds=frozenset({"fact"}),
        )
        for _cos, text, _meta in hits:
            if text in by_text:
                cur = by_text[text]
                if direction.weight > cur.weight:
                    cur.weight = direction.weight
                    cur.direction_label = direction.direction_label
            else:
                by_text[text] = _Candidate(
                    text=text,
                    direction_label=direction.direction_label,
                    weight=direction.weight,
                )

    if not by_text:
        elapsed = (time.perf_counter() - t0) * 1000.0
        trace(f"RAG_GATEWAY ⊘ кандидатов нет | {elapsed:.1f}ms")
        return DirectionalRAGResponse(
            target_node=req.target_node,
            total_found=0,
            facts=[],
            latency_ms=round(elapsed, 2),
        )

    texts = list(by_text.keys())
    scores = await run_under_uma_lock(
        score_relevance_pairs,
        req.relevance_criteria,
        texts,
    )

    threshold = req.min_relevance_threshold
    max_facts = req.max_facts
    weighted: list[tuple[float, str, str]] = []
    for text, raw_score in zip(texts, scores):
        cand = by_text[text]
        final = float(raw_score) * float(cand.weight)
        if final >= threshold:
            weighted.append((final, cand.direction_label, text))

    deduped = _deduplicate_facts(weighted)
    top_rows = deduped[:max_facts]
    context_topic = (req.relevance_criteria or req.target_node).strip()[:500]
    rag_deadline = t0 + KE_RAG_TIMEOUT_SEC - RAG_GATEWAY_FINISH_MARGIN_SEC

    async def _compress_for_pipeline(text: str) -> str:
        if len(text) <= FACT_MAX_CHARS:
            return text
        remaining = rag_deadline - time.perf_counter()
        return await compress_fact_if_needed(
            text,
            context_topic,
            gemma_timeout_sec=remaining,
        )

    compressed_facts = await asyncio.gather(
        *[_compress_for_pipeline(text) for _score, _direction, text in top_rows]
    )
    facts_out: list[RankedMemoryFact] = []
    for (score, direction, _text), fact in zip(top_rows, compressed_facts):
        facts_out.append(
            RankedMemoryFact(
                direction=direction,
                fact=fact,
                relevance_score=round(min(1.0, score), 4),
            )
        )

    elapsed = (time.perf_counter() - t0) * 1000.0
    if elapsed > RAG_LATENCY_WARN_MS:
        trace(f"RAG_GATEWAY ⚠ latency {elapsed:.1f}ms > {RAG_LATENCY_WARN_MS}ms")
    trace(
        f"RAG_GATEWAY ✓ facts={len(facts_out)} candidates={len(texts)} "
        f"| {elapsed:.1f}ms"
    )
    return DirectionalRAGResponse(
        target_node=req.target_node,
        total_found=len(facts_out),
        facts=facts_out,
        latency_ms=round(elapsed, 2),
    )


async def save_user_fact(
    fact_text: str,
    category: str,
    node_id: str,
) -> int:
    """Write-интерфейс: атомарная индексация факта в векторную базу."""
    rag = LightRAG()
    return await rag.save_user_fact(fact_text, category, node_id)


async def save_user_fact_request(body: SaveUserFactRequest) -> int:
    return await save_user_fact(body.fact_text, body.category, body.node_id)


# Совместимость с прежним именованием
query_rag_gateway = query_directional_rag
